# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference, PyTorch FlashAttention, and Triton streaming self-attention."""

import math
from collections.abc import Callable

import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.core.attention.kvcache import BlockKVCache


def _allocate_triton_workspace(
    size: int,
    alignment: int,
    stream: int | None,
) -> Tensor:
    """Allocate tensor-descriptor workspace on the active CUDA device."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_allocate_triton_workspace)


class ReferenceSelfAttention(nn.Module):
    """Streaming self-attention implemented with basic PyTorch operations."""

    n_heads: int
    """Number of query, key, and value heads."""

    head_dim: int
    """Feature dimension of each attention head."""

    q_proj: nn.Linear
    """Projection from input features to query heads."""

    k_proj: nn.Linear
    """Projection from input features to key heads."""

    v_proj: nn.Linear
    """Projection from input features to value heads."""

    output_proj: nn.Linear
    """Projection from concatenated attention heads to output features."""

    q_norm: nn.RMSNorm
    """Per-head RMS normalization applied to projected queries."""

    k_norm: nn.RMSNorm
    """Per-head RMS normalization applied to projected keys."""

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
    ) -> None:
        """Initialize the self-attention projections and per-head normalization.

        Args:
            query_dim: Feature dimension of input tokens and projected output.
            n_heads: Number of attention heads.
            head_dim: Feature dimension of each attention head.
        """
        super().__init__()
        inner_dim = n_heads * head_dim

        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Reject context parallelism for this reference implementation."""
        if cp_group is not None:
            raise NotImplementedError(
                "ReferenceSelfAttention does not support context parallelism."
            )

    @staticmethod
    def _apply_rope(x: Tensor, rope_freqs: Tensor) -> Tensor:
        """Apply non-interleaved rotary position embeddings to ``x``.

        Args:
            x: Query or key heads of shape ``[B, L, H, D]``. ``D`` must be even.
            rope_freqs: Rotation angles of shape ``[L, 1, 1, D]``.

        Returns:
            Position-encoded heads with the same shape and dtype as ``x``.
        """
        half_dim = x.shape[-1] // 2
        freqs = rope_freqs[:, 0, 0, :half_dim].unsqueeze(0).unsqueeze(2)
        cos_freqs = torch.cos(freqs).to(dtype=x.dtype)
        sin_freqs = torch.sin(freqs).to(dtype=x.dtype)
        first_half, second_half = x.chunk(2, dim=-1)
        return torch.cat(
            (
                first_half * cos_freqs - second_half * sin_freqs,
                second_half * cos_freqs + first_half * sin_freqs,
            ),
            dim=-1,
        )

    def _apply_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply explicit scaled dot-product attention.

        Args:
            query: Current query heads of shape ``[B, H, L, D]``.
            key: Cached key heads of shape ``[B, H, S, D]``.
            value: Cached value heads of shape ``[B, H, S, D]``.

        Returns:
            Attention output of shape ``[B, H, L, D]``.
        """
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, value)

    def initialize_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BlockKVCache:
        """Initialize the KV cache used by streaming self-attention."""
        total_size = sink_size + window_size
        return BlockKVCache(
            k_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            v_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            seq_dim=-3,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=dtype,
        )

    @nvtx.annotate("self_attention.forward")
    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor,
    ) -> Tensor:
        """Update the KV cache and apply self-attention to ``x``.

        Args:
            x: Current token chunk of shape ``[..., L, D]``.
            kv_cache: Cache prepared for this chunk with
                :meth:`BlockKVCache.before_update`.
            rope_freqs: RoPE angles of shape ``[L, 1, 1, head_dim]``.

        Returns:
            Projected attention output with the same shape as ``x``.
        """
        with nvtx.annotate("self_attention.prepare_shapes"):
            # Preserve all leading batch-like dimensions so the final tensor can
            # be restored after attention flattens them into one batch axis.
            batch_shape = x.shape[:-2]
            batch_size = math.prod(batch_shape)

            # Separate sequence and feature dimensions and verify that the input
            # width equals the concatenated width of all attention heads.
            sequence_length, hidden_dim = x.shape[-2:]
            inner_dim = self.n_heads * self.head_dim
            assert hidden_dim == inner_dim, "n_heads * head_dim must be equal to D"

        with nvtx.annotate("self_attention.qkv_projection"):
            # Every projection is reshaped to the cache's ``[B, L, H, D_head]``
            # layout. RMSNorm is applied independently to each Q/K head.
            head_shape = (
                batch_size,
                sequence_length,
                self.n_heads,
                self.head_dim,
            )
            query = self.q_norm(self.q_proj(x).reshape(head_shape))
            key = self.k_norm(self.k_proj(x).reshape(head_shape))
            value = self.v_proj(x).reshape(head_shape)

        with nvtx.annotate("self_attention.rope_and_cache_update"):
            # Encode current positions in Q/K before storing K. Values carry no
            # positional rotation and are written to the same prepared slots.
            query = self._apply_rope(query, rope_freqs)
            key = self._apply_rope(key, rope_freqs)
            kv_cache.update(key, value)

        with nvtx.annotate("self_attention.attention"):
            # Attention implementations consume ``[B, H, sequence, D_head]``;
            # cached K/V include both historical tokens and the current chunk.
            query = query.transpose(1, 2)
            key = kv_cache.cached_k().transpose(1, 2)
            value = kv_cache.cached_v().transpose(1, 2)

            # Dispatch only the attention calculation to the selected subclass.
            output = self._apply_attention(query, key, value)

        with nvtx.annotate("self_attention.output_projection"):
            # Move heads behind the sequence axis, concatenate their features,
            # and restore the original leading batch-like dimensions.
            output = output.transpose(1, 2).reshape(
                batch_shape + (sequence_length, inner_dim)
            )

            # Mix the concatenated heads back into the model feature space.
            return self.output_proj(output)


class FlashAttnSelfAttention(ReferenceSelfAttention):
    """Streaming self-attention backed by PyTorch SDPA FlashAttention."""

    def _apply_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply scaled dot-product attention with the FlashAttention backend."""
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )


@triton.jit
def _flash_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    query_stride_b,
    query_stride_h,
    query_stride_l,
    query_stride_d,
    key_stride_b,
    key_stride_h,
    key_stride_s,
    key_stride_d,
    value_stride_b,
    value_stride_h,
    value_stride_s,
    value_stride_d,
    output_stride_b,
    output_stride_h,
    output_stride_l,
    output_stride_d,
    num_heads,
    query_length,
    key_length,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Apply non-causal FlashAttention to one query tile.

    The launch grid is ``[B * H, ceil(L / BLOCK_M)]``. Each program streams
    across cached K/V blocks while retaining only its online-softmax state and
    ``[BLOCK_M, BLOCK_D]`` output accumulator.
    """
    # Select one batch/head pair and one query-row tile.
    batch_head = tl.program_id(0)
    query_block = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    # Load Q once; padded feature lanes remain zero throughout both dot products.
    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    query_mask = query_offsets < query_length
    dim_mask = dim_offsets < HEAD_DIM
    query_ptrs = (
        query_ptr
        + batch * query_stride_b
        + head * query_stride_h
        + query_offsets[:, None] * query_stride_l
        + dim_offsets[None, :] * query_stride_d
    )
    query = tl.load(
        query_ptrs,
        mask=query_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    denominator = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Base-2 exponentials are cheaper in Triton; fold log2(e) into the QK scale.
    qk_scale = scale.to(tl.float32) * 1.4426950408889634
    for key_start in range(0, key_length, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < key_length
        key_ptrs = (
            key_ptr
            + batch * key_stride_b
            + head * key_stride_h
            + key_offsets[:, None] * key_stride_s
            + dim_offsets[None, :] * key_stride_d
        )
        value_ptrs = (
            value_ptr
            + batch * value_stride_b
            + head * value_stride_h
            + key_offsets[:, None] * value_stride_s
            + dim_offsets[None, :] * value_stride_d
        )
        key = tl.load(
            key_ptrs,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        value = tl.load(
            value_ptrs,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )

        scores = tl.dot(query, tl.trans(key), input_precision="ieee") * qk_scale
        scores = tl.where(key_mask[None, :], scores, -float("inf"))

        # Merge this block into the running softmax without materializing scores.
        block_max = tl.max(scores, axis=1)
        new_row_max = tl.maximum(row_max, block_max)
        correction = tl.exp2(row_max - new_row_max)
        probabilities = tl.exp2(scores - new_row_max[:, None])
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities.to(value.dtype), value, input_precision="ieee"
        )
        row_max = new_row_max

    output = accumulator / denominator[:, None]
    output_ptrs = (
        output_ptr
        + batch * output_stride_b
        + head * output_stride_h
        + query_offsets[:, None] * output_stride_l
        + dim_offsets[None, :] * output_stride_d
    )
    tl.store(output_ptrs, output, mask=query_mask[:, None] & dim_mask[None, :])


@triton.jit
def _flash_attention_tma_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    query_stride_b,
    query_stride_h,
    query_stride_l,
    query_stride_d,
    key_stride_b,
    key_stride_h,
    key_stride_s,
    key_stride_d,
    value_stride_b,
    value_stride_h,
    value_stride_s,
    value_stride_d,
    output_stride_b,
    output_stride_h,
    output_stride_l,
    output_stride_d,
    batch_size,
    num_heads,
    query_length,
    key_length,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    """Apply non-causal FlashAttention with tensor-memory-accelerator loads."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    query_desc = tl.make_tensor_descriptor(
        query_ptr,
        shape=[batch_size, num_heads, query_length, HEAD_DIM],
        strides=[
            query_stride_b,
            query_stride_h,
            query_stride_l,
            query_stride_d,
        ],
        block_shape=[1, 1, BLOCK_M, HEAD_DIM],
    )
    key_desc = tl.make_tensor_descriptor(
        key_ptr,
        shape=[batch_size, num_heads, key_length, HEAD_DIM],
        strides=[key_stride_b, key_stride_h, key_stride_s, key_stride_d],
        block_shape=[1, 1, BLOCK_N, HEAD_DIM],
    )
    value_desc = tl.make_tensor_descriptor(
        value_ptr,
        shape=[batch_size, num_heads, key_length, HEAD_DIM],
        strides=[
            value_stride_b,
            value_stride_h,
            value_stride_s,
            value_stride_d,
        ],
        block_shape=[1, 1, BLOCK_N, HEAD_DIM],
    )
    output_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[batch_size, num_heads, query_length, HEAD_DIM],
        strides=[
            output_stride_b,
            output_stride_h,
            output_stride_l,
            output_stride_d,
        ],
        block_shape=[1, 1, BLOCK_M, HEAD_DIM],
    )

    query_start = query_block * BLOCK_M
    query = query_desc.load([batch, head, query_start, 0]).reshape((BLOCK_M, HEAD_DIM))
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    denominator = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    key_offsets = tl.arange(0, BLOCK_N)
    qk_scale = scale.to(tl.float32) * 1.4426950408889634

    for key_start in tl.range(
        0,
        key_length,
        BLOCK_N,
        warp_specialize=WARP_SPECIALIZE,
    ):
        key = key_desc.load([batch, head, key_start, 0]).reshape((BLOCK_N, HEAD_DIM))
        scores = tl.dot(query, tl.trans(key), input_precision="ieee")
        scores *= qk_scale
        key_mask = key_start + key_offsets < key_length
        scores = tl.where(key_mask[None, :], scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_row_max = tl.maximum(row_max, block_max)
        correction = tl.exp2(row_max - new_row_max)
        probabilities = tl.exp2(scores - new_row_max[:, None])
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        value = value_desc.load([batch, head, key_start, 0]).reshape(
            (BLOCK_N, HEAD_DIM)
        )
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities.to(value.dtype),
            value,
            input_precision="ieee",
        )
        row_max = new_row_max

    output = accumulator / denominator[:, None]
    output_desc.store(
        [batch, head, query_start, 0],
        output.reshape((1, 1, BLOCK_M, HEAD_DIM)),
    )


@triton.jit
def _qkv_postprocess_cache_kernel(
    qkv_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    key_cache_fp8_ptr,
    value_cache_fp8_ptr,
    query_weight_ptr,
    key_weight_ptr,
    rope_freqs_ptr,
    qkv_stride_b,
    qkv_stride_l,
    qkv_stride_p,
    qkv_stride_h,
    qkv_stride_d,
    query_stride_b,
    query_stride_l,
    query_stride_h,
    query_stride_d,
    key_cache_stride_b,
    key_cache_stride_l,
    key_cache_stride_h,
    key_cache_stride_d,
    value_cache_stride_b,
    value_cache_stride_l,
    value_cache_stride_h,
    value_cache_stride_d,
    rope_stride_l,
    rope_stride_d,
    sequence_length,
    num_heads,
    cache_read_start,
    cache_write_start,
    cache_write_length,
    EPS: tl.constexpr,
    HEAD_DIM_HALF: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Normalize and rotate fused QKV while writing K/V into the cache."""
    batch = tl.program_id(0)
    sequence_block = tl.program_id(1)
    head_block = tl.program_id(2)

    sequence_offsets = sequence_block * BLOCK_S + tl.arange(0, BLOCK_S)
    head_offsets = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    sequence_mask = sequence_offsets < sequence_length
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < HEAD_DIM_HALF
    activation_mask = (
        sequence_mask[:, None, None]
        & head_mask[None, :, None]
        & dim_mask[None, None, :]
    )

    qkv_base = (
        qkv_ptr
        + batch * qkv_stride_b
        + sequence_offsets[:, None, None] * qkv_stride_l
        + head_offsets[None, :, None] * qkv_stride_h
    )
    first_offsets = dim_offsets[None, None, :] * qkv_stride_d
    second_offsets = (HEAD_DIM_HALF + dim_offsets[None, None, :]) * qkv_stride_d

    query_first = tl.load(qkv_base + first_offsets, mask=activation_mask, other=0.0)
    query_second = tl.load(
        qkv_base + second_offsets,
        mask=activation_mask,
        other=0.0,
    )
    key_base = qkv_base + qkv_stride_p
    key_first = tl.load(key_base + first_offsets, mask=activation_mask, other=0.0)
    key_second = tl.load(
        key_base + second_offsets,
        mask=activation_mask,
        other=0.0,
    )

    query_square_sum = tl.sum(
        query_first.to(tl.float32) * query_first.to(tl.float32)
        + query_second.to(tl.float32) * query_second.to(tl.float32),
        axis=2,
    )
    key_square_sum = tl.sum(
        key_first.to(tl.float32) * key_first.to(tl.float32)
        + key_second.to(tl.float32) * key_second.to(tl.float32),
        axis=2,
    )
    query_scale = tl.rsqrt(query_square_sum / (2 * HEAD_DIM_HALF) + EPS)
    key_scale = tl.rsqrt(key_square_sum / (2 * HEAD_DIM_HALF) + EPS)

    query_weight_first = tl.load(
        query_weight_ptr + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )
    query_weight_second = tl.load(
        query_weight_ptr + HEAD_DIM_HALF + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )
    key_weight_first = tl.load(
        key_weight_ptr + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )
    key_weight_second = tl.load(
        key_weight_ptr + HEAD_DIM_HALF + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )

    query_first = (
        query_first * query_scale[:, :, None] * query_weight_first[None, None, :]
    ).to(query_first.dtype)
    query_second = (
        query_second * query_scale[:, :, None] * query_weight_second[None, None, :]
    ).to(query_second.dtype)
    key_first = (
        key_first * key_scale[:, :, None] * key_weight_first[None, None, :]
    ).to(key_first.dtype)
    key_second = (
        key_second * key_scale[:, :, None] * key_weight_second[None, None, :]
    ).to(key_second.dtype)

    frequency_offsets = (
        sequence_offsets[:, None] * rope_stride_l + dim_offsets[None, :] * rope_stride_d
    )
    frequency_mask = sequence_mask[:, None] & dim_mask[None, :]
    frequencies = tl.load(
        rope_freqs_ptr + frequency_offsets,
        mask=frequency_mask,
        other=0.0,
    ).to(tl.float32)
    cos_freqs = tl.cos(frequencies).to(query_first.dtype)[:, None, :]
    sin_freqs = tl.sin(frequencies).to(query_first.dtype)[:, None, :]

    query_rotated_first = query_first * cos_freqs - query_second * sin_freqs
    query_rotated_second = query_second * cos_freqs + query_first * sin_freqs
    key_rotated_first = key_first * cos_freqs - key_second * sin_freqs
    key_rotated_second = key_second * cos_freqs + key_first * sin_freqs

    query_base = (
        query_ptr
        + batch * query_stride_b
        + sequence_offsets[:, None, None] * query_stride_l
        + head_offsets[None, :, None] * query_stride_h
    )
    tl.store(
        query_base + dim_offsets[None, None, :] * query_stride_d,
        query_rotated_first,
        mask=activation_mask,
    )
    tl.store(
        query_base + (HEAD_DIM_HALF + dim_offsets[None, None, :]) * query_stride_d,
        query_rotated_second,
        mask=activation_mask,
    )

    cache_offsets = sequence_offsets - cache_read_start
    cache_mask = (
        activation_mask
        & (cache_offsets[:, None, None] >= 0)
        & (cache_offsets[:, None, None] < cache_write_length)
    )
    cache_sequence_offsets = cache_write_start + cache_offsets
    key_cache_base = (
        key_cache_ptr
        + batch * key_cache_stride_b
        + cache_sequence_offsets[:, None, None] * key_cache_stride_l
        + head_offsets[None, :, None] * key_cache_stride_h
    )
    value_cache_base = (
        value_cache_ptr
        + batch * value_cache_stride_b
        + cache_sequence_offsets[:, None, None] * value_cache_stride_l
        + head_offsets[None, :, None] * value_cache_stride_h
    )
    key_cache_fp8_base = (
        key_cache_fp8_ptr
        + batch * key_cache_stride_b
        + cache_sequence_offsets[:, None, None] * key_cache_stride_l
        + head_offsets[None, :, None] * key_cache_stride_h
    )
    value_cache_fp8_base = (
        value_cache_fp8_ptr
        + batch * value_cache_stride_b
        + cache_sequence_offsets[:, None, None] * value_cache_stride_l
        + head_offsets[None, :, None] * value_cache_stride_h
    )
    tl.store(
        key_cache_base + dim_offsets[None, None, :] * key_cache_stride_d,
        key_rotated_first,
        mask=cache_mask,
    )
    tl.store(
        key_cache_base
        + (HEAD_DIM_HALF + dim_offsets[None, None, :]) * key_cache_stride_d,
        key_rotated_second,
        mask=cache_mask,
    )
    tl.store(
        key_cache_fp8_base + dim_offsets[None, None, :] * key_cache_stride_d,
        key_rotated_first,
        mask=cache_mask,
    )
    tl.store(
        key_cache_fp8_base
        + (HEAD_DIM_HALF + dim_offsets[None, None, :]) * key_cache_stride_d,
        key_rotated_second,
        mask=cache_mask,
    )

    value_base = qkv_base + 2 * qkv_stride_p
    value_first = tl.load(
        value_base + first_offsets,
        mask=activation_mask,
        other=0.0,
    )
    value_second = tl.load(
        value_base + second_offsets,
        mask=activation_mask,
        other=0.0,
    )
    tl.store(
        value_cache_base + dim_offsets[None, None, :] * value_cache_stride_d,
        value_first,
        mask=cache_mask,
    )
    tl.store(
        value_cache_base
        + (HEAD_DIM_HALF + dim_offsets[None, None, :]) * value_cache_stride_d,
        value_second,
        mask=cache_mask,
    )
    tl.store(
        value_cache_fp8_base + dim_offsets[None, None, :] * value_cache_stride_d,
        value_first,
        mask=cache_mask,
    )
    tl.store(
        value_cache_fp8_base
        + (HEAD_DIM_HALF + dim_offsets[None, None, :]) * value_cache_stride_d,
        value_second,
        mask=cache_mask,
    )


class _AcceleratedBlockKVCache(BlockKVCache):
    """Block KV cache with an internal FP8 mirror for accelerated attention."""

    _k_fp8: Tensor | None
    """Cached keys quantized to FP8 E4M3 for the fused attention kernel."""

    _v_fp8: Tensor | None
    """Cached values quantized to FP8 E4M3 for the fused attention kernel."""

    def __post_init__(self) -> None:
        """Allocate the public cache and an eligible production FP8 mirror."""
        super().__post_init__()
        use_fp8 = (
            self.dtype == torch.bfloat16
            and self.k_shape[-1] == 128
            and torch.device(self.device).type == "cuda"
            and torch.cuda.get_device_capability(self.device)[0] >= 9
        )
        if use_fp8:
            self._k_fp8 = torch.empty(
                self.k_shape,
                device=self.device,
                dtype=torch.float8_e4m3fn,
            )
            self._v_fp8 = torch.empty(
                self.v_shape,
                device=self.device,
                dtype=torch.float8_e4m3fn,
            )
        else:
            self._k_fp8 = None
            self._v_fp8 = None

    def _roll_local_window_left(self) -> None:
        """Roll both public BF16 storage and the internal FP8 mirror."""
        super()._roll_local_window_left()
        if self._k_fp8 is None or self._v_fp8 is None:
            return

        tokens_to_keep = self.window_size - self.chunk_size
        if tokens_to_keep <= 0:
            return

        total_size = self._k_fp8.shape[self.seq_dim]
        src_start = self.sink_size + self.chunk_size
        dst_start = self.sink_size
        dst_end = self.sink_size + tokens_to_keep
        dst_slice = self._seq_slice(dst_start, dst_end)
        src_slice = self._seq_slice(src_start, total_size)
        self._k_fp8[dst_slice] = self._k_fp8[src_slice].clone()
        self._v_fp8[dst_slice] = self._v_fp8[src_slice].clone()

    def cached_k_fp8(self) -> Tensor:
        """Return the visible prefix of the internal FP8 key cache."""
        assert self._k_fp8 is not None
        return self._k_fp8[self._seq_slice(0, self._visible_end())]

    def cached_v_fp8(self) -> Tensor:
        """Return the visible prefix of the internal FP8 value cache."""
        assert self._v_fp8 is not None
        return self._v_fp8[self._seq_slice(0, self._visible_end())]


class AcceleratedSelfAttention(ReferenceSelfAttention):
    """Streaming self-attention with fused Triton preprocessing and attention."""

    _fused_qkv_weight: Tensor | None
    """Concatenated QKV projection weight quantized per output channel."""

    _fused_qkv_scale: Tensor | None
    """Per-output-channel dequantization scale for the fused QKV weight."""

    _output_weight_fp8: Tensor | None
    """Output projection weight quantized per output channel."""

    _output_weight_scale: Tensor | None
    """Per-output-channel dequantization scale for the output weight."""

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
    ) -> None:
        """Initialize projections, fused weights, and per-head normalization.

        Args:
            query_dim: Feature dimension of input tokens and projected output.
            n_heads: Number of attention heads.
            head_dim: Feature dimension of each attention head.
        """
        super().__init__(query_dim, n_heads, head_dim)
        self.register_buffer("_fused_qkv_weight", None, persistent=False)
        self.register_buffer("_fused_qkv_scale", None, persistent=False)
        self.register_buffer("_output_weight_fp8", None, persistent=False)
        self.register_buffer("_output_weight_scale", None, persistent=False)
        self._refresh_fused_qkv_weight()
        self.register_load_state_dict_post_hook(self._refresh_fused_qkv_weight)

    @staticmethod
    @torch.no_grad()
    def _quantize_linear_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize a linear weight to FP8 with one scale per output row."""
        weight_float = weight.detach().to(torch.float32)
        scale = (weight_float.abs().amax(dim=1) / 448.0).clamp_min(1e-12)
        weight_fp8 = (
            (weight_float / scale[:, None])
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        return weight_fp8, scale.reshape(1, -1).contiguous()

    @torch.no_grad()
    def _refresh_fused_qkv_weight(self, *args: object) -> None:
        """Rebuild FP8 projection weights from the authoritative parameters."""
        fused_qkv_weight = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight),
            dim=0,
        )
        self._fused_qkv_weight, self._fused_qkv_scale = self._quantize_linear_weight(
            fused_qkv_weight
        )
        self._output_weight_fp8, self._output_weight_scale = (
            self._quantize_linear_weight(self.output_proj.weight)
        )

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> "AcceleratedSelfAttention":
        """Apply a module conversion and rebuild derived FP8 weights."""
        module = super()._apply(fn, recurse=recurse)
        self._refresh_fused_qkv_weight()
        return module

    @staticmethod
    def _fp8_linear(
        x: Tensor,
        weight: Tensor,
        weight_scale: Tensor,
    ) -> Tensor:
        """Apply a unit-activation-scale, per-output-weight-scale FP8 GEMM."""
        input_shape = x.shape
        x_2d = x.reshape(-1, input_shape[-1])
        x_fp8 = (
            x_2d
            if x_2d.dtype == torch.float8_e4m3fn
            else x_2d.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        )
        input_scale = torch.ones(
            (x_2d.shape[0], 1),
            device=x.device,
            dtype=torch.float32,
        )
        output = torch._scaled_mm(
            x_fp8,
            weight.T,
            input_scale,
            weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=False,
        )
        return output.reshape(input_shape[:-1] + (weight.shape[0],))

    def initialize_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BlockKVCache:
        """Initialize a cache with an internal FP8 mirror when supported."""
        total_size = sink_size + window_size
        return _AcceleratedBlockKVCache(
            k_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            v_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            seq_dim=-3,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=dtype,
        )

    def _supports_fused_forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor,
    ) -> bool:
        """Return whether inputs satisfy the production FP8 fast-path contract."""
        return (
            not torch.is_grad_enabled()
            and x.is_cuda
            and isinstance(kv_cache, _AcceleratedBlockKVCache)
            and x.dtype == torch.bfloat16
            and self.head_dim == 128
            and x.shape[-1] == self.n_heads * self.head_dim
            and x.shape[-2] == kv_cache.chunk_size
            and kv_cache.seq_dim == 1
            and kv_cache._k.is_contiguous()
            and kv_cache._v.is_contiguous()
            and kv_cache._k.dtype == x.dtype
            and kv_cache._v.dtype == x.dtype
            and kv_cache._k.device == x.device
            and kv_cache._v.device == x.device
            and kv_cache._k_fp8 is not None
            and kv_cache._v_fp8 is not None
            and kv_cache._k_fp8.is_contiguous()
            and kv_cache._v_fp8.is_contiguous()
            and rope_freqs.ndim == 4
            and rope_freqs.shape[0] == x.shape[-2]
            and rope_freqs.shape[-1] == self.head_dim
            and rope_freqs.is_cuda
            and rope_freqs.device == x.device
            and self._fused_qkv_weight is not None
            and self._fused_qkv_weight.dtype == torch.float8_e4m3fn
            and self._fused_qkv_weight.device == x.device
            and self._fused_qkv_scale is not None
            and self._fused_qkv_scale.dtype == torch.float32
            and self._fused_qkv_scale.device == x.device
            and self._output_weight_fp8 is not None
            and self._output_weight_fp8.dtype == torch.float8_e4m3fn
            and self._output_weight_fp8.device == x.device
            and self._output_weight_scale is not None
            and self._output_weight_scale.dtype == torch.float32
            and self._output_weight_scale.device == x.device
        )

    @staticmethod
    def _cache_write_slice(kv_cache: BlockKVCache) -> tuple[int, int, int]:
        """Return the source offset, cache offset, and write length."""
        write_start, write_end = kv_cache._current_write_bounds()
        read_start = 0
        if (
            kv_cache.sink_size > 0
            and not kv_cache._current_chunk_overlaps_sink()
            and write_start < kv_cache.sink_size
        ):
            write_start = kv_cache.sink_size
            write_length = write_end - write_start
            read_start = kv_cache.chunk_size - write_length
        return read_start, write_start, write_end - write_start

    def _fused_qkv_and_cache(
        self,
        x: Tensor,
        kv_cache: _AcceleratedBlockKVCache,
        rope_freqs: Tensor,
    ) -> Tensor:
        """Project QKV and fuse normalization, RoPE, and cache writes.

        Args:
            x: Current token chunk of shape ``[..., L, D]``.
            kv_cache: Cache prepared for the current chunk.
            rope_freqs: Full-width RoPE angles of shape ``[L, 1, 1, head_dim]``.

        Returns:
            FP8 query heads in contiguous ``[B, L, H, D]`` layout.
        """
        batch_size = math.prod(x.shape[:-2])
        sequence_length = x.shape[-2]
        assert self._fused_qkv_weight is not None
        assert self._fused_qkv_scale is not None
        qkv = self._fp8_linear(
            x, self._fused_qkv_weight, self._fused_qkv_scale
        ).reshape(
            batch_size,
            sequence_length,
            3,
            self.n_heads,
            self.head_dim,
        )
        query = torch.empty(
            (batch_size, sequence_length, self.n_heads, self.head_dim),
            device=x.device,
            dtype=torch.float8_e4m3fn,
        )
        assert kv_cache._k_fp8 is not None and kv_cache._v_fp8 is not None

        cache_read_start, cache_write_start, cache_write_length = (
            self._cache_write_slice(kv_cache)
        )
        block_s = 4
        block_h = min(int(triton.next_power_of_2(self.n_heads)), 32)
        block_d = max(int(triton.next_power_of_2(self.head_dim // 2)), 16)
        grid = (
            batch_size,
            triton.cdiv(sequence_length, block_s),
            triton.cdiv(self.n_heads, block_h),
        )
        _qkv_postprocess_cache_kernel[grid](
            qkv,
            query,
            kv_cache._k,
            kv_cache._v,
            kv_cache._k_fp8,
            kv_cache._v_fp8,
            self.q_norm.weight,
            self.k_norm.weight,
            rope_freqs,
            *qkv.stride(),
            *query.stride(),
            *kv_cache._k.stride(),
            *kv_cache._v.stride(),
            rope_freqs.stride(0),
            rope_freqs.stride(3),
            sequence_length,
            self.n_heads,
            cache_read_start,
            cache_write_start,
            cache_write_length,
            EPS=self.q_norm.eps,
            HEAD_DIM_HALF=self.head_dim // 2,
            BLOCK_S=block_s,
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=1,
        )
        return query

    @nvtx.annotate("self_attention.forward")
    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor,
    ) -> Tensor:
        """Run the fused inference path when the production layout is supported.

        Args:
            x: Current token chunk of shape ``[..., L, D]``.
            kv_cache: Cache prepared for this chunk with
                :meth:`BlockKVCache.before_update`.
            rope_freqs: Full-width RoPE angles of shape ``[L, 1, 1, head_dim]``.

        Returns:
            Projected attention output with the same shape as ``x``.
        """
        if not self._supports_fused_forward(x, kv_cache, rope_freqs):
            return super().forward(x, kv_cache, rope_freqs)
        assert isinstance(kv_cache, _AcceleratedBlockKVCache)

        batch_shape = x.shape[:-2]
        sequence_length = x.shape[-2]
        inner_dim = self.n_heads * self.head_dim
        with nvtx.annotate("self_attention.qkv_projection_and_postprocess"):
            query = self._fused_qkv_and_cache(x, kv_cache, rope_freqs)

        with nvtx.annotate("self_attention.attention"):
            output = self._apply_attention_blhd(
                query,
                kv_cache.cached_k_fp8(),
                kv_cache.cached_v_fp8(),
            )

        with nvtx.annotate("self_attention.output_projection"):
            output = output.reshape(batch_shape + (sequence_length, inner_dim))
            assert self._output_weight_fp8 is not None
            assert self._output_weight_scale is not None
            return self._fp8_linear(
                output, self._output_weight_fp8, self._output_weight_scale
            )

    def _apply_attention_blhd(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """Apply TMA FlashAttention directly to contiguous BLHD tensors.

        Args:
            query: FP8 current query heads of shape ``[B, L, H, D]``.
            key: FP8 cached key heads of shape ``[B, S, H, D]``.
            value: FP8 cached value heads of shape ``[B, S, H, D]``.

        Returns:
            Contiguous FP8 attention output of shape ``[B, L, H, D]``.
        """
        batch_size, query_length, num_heads, head_dim = query.shape
        key_batch, key_length, key_heads, key_dim = key.shape
        assert (key_batch, key_heads, key_dim) == (
            batch_size,
            num_heads,
            head_dim,
        )
        assert value.shape == key.shape

        assert query.dtype == torch.float8_e4m3fn
        assert key.dtype == torch.float8_e4m3fn
        assert value.dtype == torch.float8_e4m3fn

        output = torch.empty(
            query.shape,
            device=query.device,
            dtype=torch.float8_e4m3fn,
        )
        if batch_size == 0 or num_heads == 0 or query_length == 0:
            return output
        if key_length == 0:
            return output.zero_()

        block_m = min(
            128,
            max(int(triton.next_power_of_2(query_length)), 16),
        )
        grid = (
            triton.cdiv(query_length, block_m),
            batch_size * num_heads,
        )
        query_bhld_strides = (
            query.stride(0),
            query.stride(2),
            query.stride(1),
            query.stride(3),
        )
        key_bhld_strides = (
            key.stride(0),
            key.stride(2),
            key.stride(1),
            key.stride(3),
        )
        value_bhld_strides = (
            value.stride(0),
            value.stride(2),
            value.stride(1),
            value.stride(3),
        )
        output_bhld_strides = (
            output.stride(0),
            output.stride(2),
            output.stride(1),
            output.stride(3),
        )
        _flash_attention_tma_kernel[grid](
            query,
            key,
            value,
            output,
            *query_bhld_strides,
            *key_bhld_strides,
            *value_bhld_strides,
            *output_bhld_strides,
            batch_size,
            num_heads,
            query_length,
            key_length,
            1.0 / math.sqrt(head_dim),
            HEAD_DIM=head_dim,
            BLOCK_M=block_m,
            BLOCK_N=64,
            WARP_SPECIALIZE=False,
            num_warps=4,
            num_stages=3,
        )
        return output

    def _apply_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply scaled dot-product attention with a fused Triton kernel.

        Args:
            query: Current query heads of shape ``[B, H, L, D]``.
            key: Cached key heads of shape ``[B, H, S, D]``.
            value: Cached value heads of shape ``[B, H, S, D]``.

        Returns:
            Attention output of shape ``[B, H, L, D]``.

        Raises:
            ValueError: Inputs are not compatible four-dimensional CUDA tensors,
                or their dtype or head dimension is unsupported.
        """
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("query, key, and value must have shape [B, H, S, D].")
        if not query.is_cuda or not key.is_cuda or not value.is_cuda:
            raise ValueError("AcceleratedSelfAttention requires CUDA tensors.")
        if query.device != key.device or query.device != value.device:
            raise ValueError("query, key, and value must be on the same device.")
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError("query, key, and value must have the same dtype.")
        if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "AcceleratedSelfAttention supports float16, bfloat16, and float32."
            )

        batch_size, num_heads, query_length, head_dim = query.shape
        key_batch, key_heads, key_length, key_dim = key.shape
        if (key_batch, key_heads, key_dim) != (batch_size, num_heads, head_dim):
            raise ValueError(
                "query and key batch, head, and feature dimensions differ."
            )
        if value.shape != key.shape:
            raise ValueError("key and value must have identical shapes.")
        if head_dim > 256:
            raise ValueError("Triton FlashAttention supports head_dim at most 256.")

        output = torch.empty_like(query, memory_format=torch.preserve_format)
        if batch_size == 0 or num_heads == 0 or query_length == 0 or head_dim == 0:
            return output
        if key_length == 0:
            return output.zero_()

        tma_supported = (
            query.dtype in (torch.float16, torch.bfloat16)
            and head_dim == 128
            and torch.cuda.get_device_capability(query.device)[0] >= 9
        )
        if tma_supported:
            block_m = min(
                128,
                max(int(triton.next_power_of_2(query_length)), 16),
            )
            block_n = 32
            grid = (
                triton.cdiv(query_length, block_m),
                batch_size * num_heads,
            )
            _flash_attention_tma_kernel[grid](
                query,
                key,
                value,
                output,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                *output.stride(),
                batch_size,
                num_heads,
                query_length,
                key_length,
                1.0 / math.sqrt(head_dim),
                HEAD_DIM=head_dim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                WARP_SPECIALIZE=False,
                num_warps=4,
                num_stages=2,
            )
            return output

        block_d = max(int(triton.next_power_of_2(head_dim)), 16)
        if query.dtype == torch.float32:
            max_block_m = 16 if block_d > 128 else 64
            block_n = 16 if block_d > 128 else 32
            num_stages = 1 if block_d > 128 else 2
        else:
            max_block_m = 64 if block_d > 128 else 128
            block_n = 32 if block_d > 128 else 64
            num_stages = 2 if block_d > 128 else 3
        block_m = min(
            max_block_m,
            max(int(triton.next_power_of_2(query_length)), 16),
        )
        num_warps = 4 if block_m * block_d <= 4096 else 8

        grid = (
            batch_size * num_heads,
            triton.cdiv(query_length, block_m),
        )
        _flash_attention_kernel[grid](
            query,
            key,
            value,
            output,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *output.stride(),
            num_heads,
            query_length,
            key_length,
            1.0 / math.sqrt(head_dim),
            HEAD_DIM=head_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output
