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

import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.core.attention.kvcache import BlockKVCache


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
    qk_scale = scale * 1.4426950408889634
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


class AcceleratedSelfAttention(ReferenceSelfAttention):
    """Streaming self-attention with fused Triton FlashAttention."""

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

        output = torch.empty_like(query, memory_format=torch.contiguous_format)
        if batch_size == 0 or num_heads == 0 or query_length == 0 or head_dim == 0:
            return output
        if key_length == 0:
            return output.zero_()

        block_d = max(int(triton.next_power_of_2(head_dim)), 16)
        if query.dtype == torch.float32:
            max_block_m = 16 if block_d > 128 else 64
            block_n = 16 if block_d > 128 else 32
            num_stages = 1 if block_d > 128 else 2
        else:
            max_block_m = 64 if block_d > 128 else 128
            block_n = 32 if block_d > 64 else 64
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
