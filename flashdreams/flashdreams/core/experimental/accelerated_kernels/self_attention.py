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

"""Configurable Triton-accelerated streaming self-attention."""

import math
from collections.abc import Callable

import nvtx
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor, nn
from torch.distributed import ProcessGroup

from flashdreams.core.attention.kvcache import BlockKVCache

_FP8_MAX = 448.0
"""Largest finite magnitude represented by ``torch.float8_e4m3fn``."""


def _allocate_triton_workspace(
    size: int,
    alignment: int,
    stream: int | None,
) -> Tensor:
    """Allocate tensor-descriptor workspace on the active CUDA device."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_allocate_triton_workspace)


def _supports_fp8_cache_shape(n_heads: int, head_dim: int) -> bool:
    """Return whether a head layout supports FP8 projection and attention."""
    return n_heads > 0 and 0 < head_dim <= 256 and n_heads * head_dim % 16 == 0


def _cache_write_slice(kv_cache: BlockKVCache) -> tuple[int, int, int]:
    """Return source offset, cache offset, and length for the current update."""
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


@torch.no_grad()
def _quantize_linear_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a linear weight to raw E4M3 bytes and FP32 row scales."""
    weight_float = weight.detach().to(torch.float32)
    scale = (weight_float.abs().amax(dim=1) / _FP8_MAX).clamp_min(1e-12)
    weight_fp8 = (
        (weight_float / scale[:, None])
        .clamp(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return weight_fp8.view(torch.uint8), scale.contiguous()


def _as_float8(weight: Tensor) -> Tensor:
    """View raw E4M3 bytes as an FP8 tensor without allocating a mirror."""
    if weight.dtype == torch.float8_e4m3fn:
        return weight
    if weight.dtype != torch.uint8:
        raise TypeError(f"FP8 weight storage must be uint8; got {weight.dtype}.")
    return weight.view(torch.float8_e4m3fn)


def _dequantize_linear_weight(weight: Tensor, weight_scale: Tensor) -> Tensor:
    """Dequantize raw E4M3 weight storage to FP32."""
    if weight_scale.ndim == 2 and weight_scale.shape[0] == 1:
        weight_scale = weight_scale.squeeze(0)
    if weight_scale.ndim != 1 or weight_scale.shape[0] != weight.shape[0]:
        raise ValueError("FP8 weight_scale must contain one value per output row.")
    return (
        _as_float8(weight).to(torch.float32) * weight_scale.to(torch.float32)[:, None]
    )


def _fp8_linear(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None,
    out_dtype: torch.dtype,
) -> Tensor:
    """Apply dynamically scaled activations to a row-scaled FP8 GEMM."""
    input_shape = x.shape
    x_2d = x.reshape(-1, input_shape[-1])
    if x_2d.dtype == torch.float8_e4m3fn:
        x_fp8 = x_2d
        input_scale = torch.ones(
            (x_2d.shape[0], 1), device=x.device, dtype=torch.float32
        )
    else:
        x_float = x_2d.to(torch.float32)
        input_scale = (x_float.abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(
            1e-12
        )
        x_fp8 = (
            (x_float / input_scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        )
    weight_fp8 = _as_float8(weight)
    scaled_mm_dtype = torch.bfloat16
    scaled_bias = bias.to(scaled_mm_dtype) if bias is not None else None
    output = torch._scaled_mm(
        x_fp8,
        weight_fp8.T,
        input_scale,
        weight_scale.reshape(1, -1),
        bias=scaled_bias,
        out_dtype=scaled_mm_dtype,
        use_fast_accum=False,
    )
    if out_dtype != scaled_mm_dtype:
        output = output.to(out_dtype)
    return output.reshape(input_shape[:-1] + (weight.shape[0],))


class _PackedFP8Linear(nn.Module):
    """Inference-only linear layer backed by one raw FP8 weight tensor."""

    in_features: int
    """Input feature width."""

    out_features: int
    """Output feature width."""

    weight: Tensor
    """Raw E4M3 bytes in ``[out_features, in_features]`` layout."""

    weight_scale: Tensor
    """FP32 dequantization scale for each output row."""

    bias: Tensor | None
    """Optional inference-only projection bias."""

    def __init__(self, weight: Tensor, bias: Tensor | None) -> None:
        """Quantize and retain only packed weight storage and row scales."""
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("Packed linear weight must be two-dimensional.")
        self.out_features, self.in_features = weight.shape
        packed_weight, weight_scale = _quantize_linear_weight(weight)
        self.register_buffer("weight", packed_weight, persistent=True)
        self.register_buffer("weight_scale", weight_scale, persistent=True)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> "_PackedFP8Linear":
        """Move row scales without allowing module dtype casts to change FP32."""
        weight_scale = self._buffers.pop("weight_scale")
        assert weight_scale is not None
        try:
            module = super()._apply(fn, recurse=recurse)
            scale_bytes = fn(weight_scale.contiguous().view(torch.uint8))
            self.register_buffer(
                "weight_scale", scale_bytes.view(torch.float32), persistent=True
            )
        except Exception:
            self.register_buffer("weight_scale", weight_scale, persistent=True)
            raise
        return module

    def forward(self, x: Tensor, out_dtype: torch.dtype) -> Tensor:
        """Apply the packed FP8 projection."""
        return _fp8_linear(
            x, self.weight, self.weight_scale, self.bias, out_dtype=out_dtype
        )

    def extra_repr(self) -> str:
        """Return feature and bias metadata for module summaries."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# TODO: Replace hard-coded heuristic with Triton auto tuning.
def _select_attention_config(
    query_length: int,
    head_dim: int,
    dtype: torch.dtype,
    *,
    use_tma: bool,
) -> tuple[int, int, int, int, int]:
    """Select deterministic attention launch metadata.

    Returns:
        ``(block_m, block_n, block_d, num_warps, num_stages)``.
    """
    if not 0 < head_dim <= 256:
        raise ValueError("Triton attention requires 0 < head_dim <= 256.")

    minimum_block_d = 32 if dtype == torch.float8_e4m3fn else 16
    block_d = max(int(triton.next_power_of_2(head_dim)), minimum_block_d)
    block_m = min(128, max(int(triton.next_power_of_2(query_length)), 16))
    if use_tma:
        if dtype == torch.float32:
            max_block_m = 64 if block_d <= 64 else 32
            block_m = min(block_m, max_block_m)
            block_n = 16 if block_d > 128 else 32
            num_stages = 1
        elif block_d > 128:
            block_m = min(block_m, 64)
            block_n = 32
            num_stages = 2
        else:
            block_n = 64
            num_stages = 3
        num_warps = 4 if block_m * block_d <= 4096 else 8
        return block_m, block_n, block_d, num_warps, num_stages

    if dtype == torch.float32:
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
    return block_m, block_n, block_d, num_warps, num_stages


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
    ``[BLOCK_M, BLOCK_D]`` output accumulator. ``BLOCK_D`` pads the physical
    head width to a dot-product-friendly power of two; masked lanes remain zero.
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

    # Initialize the online-softmax state for each query row. The accumulator
    # stays in FP32 while K/V blocks are incorporated one at a time.
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    denominator = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Base-2 exponentials are cheaper in Triton; fold log2(e) into the QK scale.
    qk_scale = scale.to(tl.float32) * 1.4426950408889634
    for key_start in range(0, key_length, BLOCK_N):
        # Stream one K/V tile from the cache and mask the final partial tile.
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

    # Normalize the accumulated weighted values and discard padded rows/features.
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
    """Apply non-causal FlashAttention with tensor-memory-accelerator loads.

    The launch grid is ``[ceil(L / BLOCK_M), B * H]``. Tensor descriptors move
    rectangular Q/K/V tiles while the kernel uses the same online-softmax
    recurrence as the pointer-based implementation.
    """
    # Select one query tile and one flattened batch/head pair.
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    # Describe the logical BHLD views once so TMA can issue tiled asynchronous
    # transfers without explicit pointer matrices in the loop below.
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

    # Load Q once and initialize one online-softmax state per query row.
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
        # Stream a K tile, form its score contribution, and mask descriptor
        # padding in the final key tile before the softmax reduction.
        key = key_desc.load([batch, head, key_start, 0]).reshape((BLOCK_N, HEAD_DIM))
        scores = tl.dot(query, tl.trans(key), input_precision="ieee")
        scores *= qk_scale
        key_mask = key_start + key_offsets < key_length
        scores = tl.where(key_mask[None, :], scores, -float("inf"))

        # Merge this tile into the running softmax without materializing the
        # complete query-by-key score matrix.
        block_max = tl.max(scores, axis=1)
        new_row_max = tl.maximum(row_max, block_max)
        correction = tl.exp2(row_max - new_row_max)
        probabilities = tl.exp2(scores - new_row_max[:, None])
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        # Delay the V transfer until its probabilities are available, then add
        # its weighted contribution to the FP32 output accumulator.
        value = value_desc.load([batch, head, key_start, 0]).reshape(
            (BLOCK_N, HEAD_DIM)
        )
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities.to(value.dtype),
            value,
            input_precision="ieee",
        )
        row_max = new_row_max

    # Normalize and let the descriptor store handle a partial final query tile.
    output = accumulator / denominator[:, None]
    output_desc.store(
        [batch, head, query_start, 0],
        output.reshape((1, 1, BLOCK_M, HEAD_DIM)),
    )


@triton.jit
def _qkv_postprocess_cache_kernel(
    query_input_ptr,
    key_input_ptr,
    value_input_ptr,
    query_output_ptr,
    key_cache_ptr,
    value_cache_ptr,
    query_weight_ptr,
    key_weight_ptr,
    rope_freqs_ptr,
    query_input_stride_b,
    query_input_stride_l,
    query_input_stride_h,
    query_input_stride_d,
    key_input_stride_b,
    key_input_stride_l,
    key_input_stride_h,
    key_input_stride_d,
    value_input_stride_b,
    value_input_stride_l,
    value_input_stride_h,
    value_input_stride_d,
    query_output_stride_b,
    query_output_stride_l,
    query_output_stride_h,
    query_output_stride_d,
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
    APPLY_NORM: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Normalize and rotate Q/K while writing K/V into the prepared cache.

    Each program handles one batch and a tile of sequence positions, heads, and
    paired feature dimensions. Compile-time flags remove normalization, RoPE,
    and cache conversion when those operations are disabled.
    """
    # Select a three-dimensional sequence/head/half-feature tile for one batch.
    batch = tl.program_id(0)
    sequence_block = tl.program_id(1)
    head_block = tl.program_id(2)

    # Rounded block sizes require independent masks along every tiled axis.
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

    # Express both RoPE layouts as pairs: adjacent even/odd features for the
    # interleaved layout, or corresponding features from the two head halves.
    if INTERLEAVED:
        first_feature_offsets = 2 * dim_offsets
        second_feature_offsets = 2 * dim_offsets + 1
    else:
        first_feature_offsets = dim_offsets
        second_feature_offsets = HEAD_DIM_HALF + dim_offsets

    # Build BLH bases and load both members of each Q/K feature pair.
    query_base = (
        query_input_ptr
        + batch * query_input_stride_b
        + sequence_offsets[:, None, None] * query_input_stride_l
        + head_offsets[None, :, None] * query_input_stride_h
    )
    key_base = (
        key_input_ptr
        + batch * key_input_stride_b
        + sequence_offsets[:, None, None] * key_input_stride_l
        + head_offsets[None, :, None] * key_input_stride_h
    )
    value_base = (
        value_input_ptr
        + batch * value_input_stride_b
        + sequence_offsets[:, None, None] * value_input_stride_l
        + head_offsets[None, :, None] * value_input_stride_h
    )
    query_first = tl.load(
        query_base + first_feature_offsets[None, None, :] * query_input_stride_d,
        mask=activation_mask,
        other=0.0,
    )
    query_second = tl.load(
        query_base + second_feature_offsets[None, None, :] * query_input_stride_d,
        mask=activation_mask,
        other=0.0,
    )
    key_first = tl.load(
        key_base + first_feature_offsets[None, None, :] * key_input_stride_d,
        mask=activation_mask,
        other=0.0,
    )
    key_second = tl.load(
        key_base + second_feature_offsets[None, None, :] * key_input_stride_d,
        mask=activation_mask,
        other=0.0,
    )

    if APPLY_NORM:
        # Compute one FP32 RMS statistic per sequence/head over both members of
        # every feature pair, then apply the learned per-feature weights.
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
            query_weight_ptr + first_feature_offsets,
            mask=dim_mask,
            other=0.0,
        )
        query_weight_second = tl.load(
            query_weight_ptr + second_feature_offsets,
            mask=dim_mask,
            other=0.0,
        )
        key_weight_first = tl.load(
            key_weight_ptr + first_feature_offsets,
            mask=dim_mask,
            other=0.0,
        )
        key_weight_second = tl.load(
            key_weight_ptr + second_feature_offsets,
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

    if APPLY_ROPE:
        # Frequencies are shared across heads. Selecting the first member's
        # offset addresses each pair's angle in either supported RoPE layout.
        frequency_offsets = (
            sequence_offsets[:, None] * rope_stride_l
            + first_feature_offsets[None, :] * rope_stride_d
        )
        frequency_mask = sequence_mask[:, None] & dim_mask[None, :]
        frequencies = tl.load(
            rope_freqs_ptr + frequency_offsets,
            mask=frequency_mask,
            other=0.0,
        ).to(tl.float32)
        cos_freqs = tl.cos(frequencies).to(query_first.dtype)[:, None, :]
        sin_freqs = tl.sin(frequencies).to(query_first.dtype)[:, None, :]
        # Apply the same two-dimensional rotation to Q and K for every pair.
        query_rotated_first = query_first * cos_freqs - query_second * sin_freqs
        query_rotated_second = query_second * cos_freqs + query_first * sin_freqs
        key_rotated_first = key_first * cos_freqs - key_second * sin_freqs
        key_rotated_second = key_second * cos_freqs + key_first * sin_freqs
    else:
        query_rotated_first = query_first
        query_rotated_second = query_second
        key_rotated_first = key_first
        key_rotated_second = key_second

    # Materialize processed Q for attention; Q is never retained in the cache.
    query_output_base = (
        query_output_ptr
        + batch * query_output_stride_b
        + sequence_offsets[:, None, None] * query_output_stride_l
        + head_offsets[None, :, None] * query_output_stride_h
    )
    tl.store(
        query_output_base
        + first_feature_offsets[None, None, :] * query_output_stride_d,
        query_rotated_first,
        mask=activation_mask,
    )
    tl.store(
        query_output_base
        + second_feature_offsets[None, None, :] * query_output_stride_d,
        query_rotated_second,
        mask=activation_mask,
    )

    # Map the writable source slice onto its physical cache destination. Sink
    # preservation can make this a suffix of the current input chunk.
    cache_offsets = sequence_offsets - cache_read_start
    cache_mask = (
        activation_mask
        & (cache_offsets[:, None, None] >= 0)
        & (cache_offsets[:, None, None] < cache_write_length)
    )
    cache_sequence_offsets = cache_write_start + cache_offsets
    # Build native cache addresses only for lanes selected by the cache mapping.
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
    # V bypasses RMSNorm and RoPE, so load it only when preparing cache stores.
    value_first = tl.load(
        value_base + first_feature_offsets[None, None, :] * value_input_stride_d,
        mask=activation_mask,
        other=0.0,
    )
    value_second = tl.load(
        value_base + second_feature_offsets[None, None, :] * value_input_stride_d,
        mask=activation_mask,
        other=0.0,
    )
    # Store K/V once; the cache tensor dtype performs any FP8 conversion.
    tl.store(
        key_cache_base + first_feature_offsets[None, None, :] * key_cache_stride_d,
        key_rotated_first,
        mask=cache_mask,
    )
    tl.store(
        key_cache_base + second_feature_offsets[None, None, :] * key_cache_stride_d,
        key_rotated_second,
        mask=cache_mask,
    )
    tl.store(
        value_cache_base + first_feature_offsets[None, None, :] * value_cache_stride_d,
        value_first,
        mask=cache_mask,
    )
    tl.store(
        value_cache_base + second_feature_offsets[None, None, :] * value_cache_stride_d,
        value_second,
        mask=cache_mask,
    )


class AcceleratedSelfAttention(nn.Module):
    """Streaming self-attention with independently configurable fast paths."""

    n_heads: int
    """Number of query, key, and value heads."""

    head_dim: int
    """Feature dimension of each attention head."""

    query_dim: int
    """Input and output feature width, independent of the attention inner width."""

    qkv_proj: _PackedFP8Linear
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    output_proj: nn.Linear | _PackedFP8Linear
    _fused_qkv_weight: Tensor | None
    _fused_qkv_bias: Tensor | None

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
        *,
        qkv_bias: bool = False,
        output_bias: bool = False,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        rope_interleaved: bool = False,
        use_tma: bool = True,
        fuse_qkv: bool = True,
        fuse_rope_kv_cache: bool = True,
        use_fp8: bool = True,
    ) -> None:
        """Initialize projections and independently configurable fast paths.

        Args:
            query_dim: Feature dimension of input tokens and projected output.
            n_heads: Number of query, key, and value heads.
            head_dim: Feature dimension of each attention head.
            qkv_bias: Add bias to the query, key, and value projections.
            output_bias: Add bias to the output projection.
            qk_norm: Apply per-head RMS normalization to projected Q/K.
            qk_norm_eps: Epsilon used by Q/K RMS normalization.
            rope_interleaved: Rotate adjacent feature pairs instead of half splits.
            use_tma: Prefer TMA attention when tensor descriptors support the layout.
            fuse_qkv: Combine Q/K/V projections into one matrix multiplication.
            fuse_rope_kv_cache: Fuse Q/K postprocessing and the cache write.
            use_fp8: Use FP8 projections and attention for eligible FP16/BF16 inputs.
        """
        super().__init__()
        if query_dim <= 0:
            raise ValueError(f"query_dim must be positive; got {query_dim}.")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive; got {n_heads}.")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive; got {head_dim}.")

        inner_dim = n_heads * head_dim
        self.query_dim = query_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_interleaved = rope_interleaved
        self.use_tma = use_tma
        self.fuse_qkv = fuse_qkv
        self.fuse_rope_kv_cache = fuse_rope_kv_cache
        self.use_fp8 = use_fp8

        if use_fp8:
            if query_dim % 16 != 0:
                raise ValueError(
                    "FP8 projection requires query_dim to be a multiple of 16."
                )
            if not _supports_fp8_cache_shape(n_heads, head_dim):
                raise ValueError(
                    "FP8 attention requires head_dim <= 256 and inner_dim to be "
                    "a multiple of 16."
                )

        q_proj = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        k_proj = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        v_proj = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        output_proj = nn.Linear(inner_dim, query_dim, bias=output_bias)
        if use_fp8:
            fused_weight = torch.cat(
                (q_proj.weight, k_proj.weight, v_proj.weight), dim=0
            )
            fused_bias = None
            if q_proj.bias is not None:
                assert k_proj.bias is not None and v_proj.bias is not None
                fused_bias = torch.cat((q_proj.bias, k_proj.bias, v_proj.bias), dim=0)
            self.qkv_proj = _PackedFP8Linear(fused_weight, fused_bias)
            self.output_proj = _PackedFP8Linear(output_proj.weight, output_proj.bias)
        else:
            self.q_proj = q_proj
            self.k_proj = k_proj
            self.v_proj = v_proj
            self.output_proj = output_proj
            self.register_buffer("_fused_qkv_weight", None, persistent=False)
            self.register_buffer("_fused_qkv_bias", None, persistent=False)

        self.q_norm: nn.Module = (
            nn.RMSNorm(head_dim, eps=qk_norm_eps) if qk_norm else nn.Identity()
        )
        self.k_norm: nn.Module = (
            nn.RMSNorm(head_dim, eps=qk_norm_eps) if qk_norm else nn.Identity()
        )
        if not use_fp8:
            self._refresh_derived_weights()
            self.register_load_state_dict_post_hook(self._refresh_derived_weights)

    @property
    def optimization_settings(self) -> dict[str, bool]:
        """Return the requested optimization settings."""
        return {
            "use_tma": self.use_tma,
            "fuse_qkv": self.fuse_qkv,
            "fuse_rope_kv_cache": self.fuse_rope_kv_cache,
            "use_fp8": self.use_fp8,
        }

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Reject context parallelism while accepting the single-rank no-op."""
        if cp_group is not None:
            raise NotImplementedError(
                "AcceleratedSelfAttention does not support context parallelism."
            )

    @staticmethod
    def _normalize_packed_weight(weight: Tensor) -> Tensor:
        """Normalize packed checkpoint storage to contiguous raw E4M3 bytes."""
        if weight.dtype == torch.float8_e4m3fn:
            return weight.contiguous().view(torch.uint8)
        if weight.dtype != torch.uint8:
            raise TypeError(f"Packed FP8 weights must be uint8; got {weight.dtype}.")
        return weight.contiguous()

    @staticmethod
    def _normalize_weight_scale(weight_scale: Tensor, rows: int) -> Tensor:
        """Normalize a packed checkpoint scale to one FP32 value per row."""
        scale = weight_scale.squeeze(0) if weight_scale.ndim == 2 else weight_scale
        if scale.ndim != 1 or scale.numel() != rows:
            raise ValueError(f"Packed FP8 scale must contain {rows} values.")
        return scale.to(torch.float32).contiguous()

    def _prepare_packed_state_dict(
        self, state_dict: dict[str, Tensor], prefix: str
    ) -> None:
        """Upgrade legacy split native projections to the packed FP8 schema."""
        qkv_key = prefix + "qkv_proj.weight"
        qkv_scale_key = prefix + "qkv_proj.weight_scale"
        legacy_weight_keys = [
            prefix + f"{name}_proj.weight" for name in ("q", "k", "v")
        ]
        legacy_present = [key in state_dict for key in legacy_weight_keys]
        if qkv_key in state_dict and any(legacy_present):
            raise RuntimeError("Checkpoint mixes packed and split Q/K/V weights.")
        if qkv_key not in state_dict and all(legacy_present):
            fused_weight = torch.cat(
                [state_dict.pop(key) for key in legacy_weight_keys], dim=0
            )
            packed_weight, weight_scale = _quantize_linear_weight(fused_weight)
            state_dict[qkv_key] = packed_weight
            state_dict[qkv_scale_key] = weight_scale
            legacy_bias_keys = [
                prefix + f"{name}_proj.bias" for name in ("q", "k", "v")
            ]
            if all(key in state_dict for key in legacy_bias_keys):
                state_dict[prefix + "qkv_proj.bias"] = torch.cat(
                    [state_dict.pop(key) for key in legacy_bias_keys], dim=0
                )
        elif qkv_key in state_dict:
            qkv_weight = state_dict[qkv_key]
            if qkv_weight.dtype not in (torch.uint8, torch.float8_e4m3fn):
                qkv_weight, qkv_scale = _quantize_linear_weight(qkv_weight)
                state_dict[qkv_key] = qkv_weight
                state_dict[qkv_scale_key] = qkv_scale
            elif qkv_scale_key not in state_dict:
                raise RuntimeError("Packed QKV checkpoint is missing weight_scale.")

        output_key = prefix + "output_proj.weight"
        output_scale_key = prefix + "output_proj.weight_scale"
        if output_key in state_dict:
            output_weight = state_dict[output_key]
            if output_weight.dtype not in (torch.uint8, torch.float8_e4m3fn):
                output_weight, output_scale = _quantize_linear_weight(output_weight)
                state_dict[output_key] = output_weight
                state_dict[output_scale_key] = output_scale
            elif output_scale_key not in state_dict:
                raise RuntimeError("Packed output checkpoint is missing weight_scale.")

        for weight_key, scale_key in (
            (qkv_key, qkv_scale_key),
            (output_key, output_scale_key),
        ):
            if weight_key not in state_dict or scale_key not in state_dict:
                continue
            state_dict[weight_key] = self._normalize_packed_weight(
                state_dict[weight_key]
            )
            state_dict[scale_key] = self._normalize_weight_scale(
                state_dict[scale_key], state_dict[weight_key].shape[0]
            )

    def _prepare_native_state_dict(
        self, state_dict: dict[str, Tensor], prefix: str
    ) -> None:
        """Downgrade the packed FP8 schema for a native attention instance."""
        qkv_key = prefix + "qkv_proj.weight"
        qkv_scale_key = prefix + "qkv_proj.weight_scale"
        if qkv_key in state_dict:
            if qkv_scale_key not in state_dict:
                raise RuntimeError("Packed QKV checkpoint is missing weight_scale.")
            fused_weight = _dequantize_linear_weight(
                state_dict.pop(qkv_key), state_dict.pop(qkv_scale_key)
            )
            q_weight, k_weight, v_weight = fused_weight.chunk(3, dim=0)
            for name, weight in zip(("q", "k", "v"), (q_weight, k_weight, v_weight)):
                state_dict[prefix + f"{name}_proj.weight"] = weight
            qkv_bias_key = prefix + "qkv_proj.bias"
            if qkv_bias_key in state_dict:
                q_bias, k_bias, v_bias = state_dict.pop(qkv_bias_key).chunk(3, dim=0)
                for name, bias in zip(("q", "k", "v"), (q_bias, k_bias, v_bias)):
                    state_dict[prefix + f"{name}_proj.bias"] = bias

        output_key = prefix + "output_proj.weight"
        output_scale_key = prefix + "output_proj.weight_scale"
        if output_key in state_dict and output_scale_key in state_dict:
            state_dict[output_key] = _dequantize_linear_weight(
                state_dict[output_key], state_dict.pop(output_scale_key)
            )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Load packed checkpoints and transparently upgrade legacy weights."""
        if self.use_fp8:
            self._prepare_packed_state_dict(state_dict, prefix)
        else:
            self._prepare_native_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.no_grad()
    def _refresh_derived_weights(self, *args: object) -> None:
        """Rebuild the native fused projection from authoritative parameters."""
        if self.use_fp8:
            return
        fused_weight = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0
        ).detach()
        self._fused_qkv_weight = fused_weight.contiguous()
        if self.q_proj.bias is None:
            self._fused_qkv_bias = None
        else:
            assert self.k_proj.bias is not None and self.v_proj.bias is not None
            self._fused_qkv_bias = torch.cat(
                (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias), dim=0
            ).detach()

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> "AcceleratedSelfAttention":
        """Apply a module conversion while preserving packed FP8 metadata."""
        module = super()._apply(fn, recurse=recurse)
        if not self.use_fp8:
            self._refresh_derived_weights()
        return module

    @staticmethod
    def _validate_fp8_device(device: torch.device) -> None:
        """Require a CUDA device with native FP8 tensor-core support."""
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FP8-only attention requires a CUDA device.")
        if torch.cuda.get_device_capability(device)[0] < 9:
            raise RuntimeError("FP8-only attention requires compute capability 9.0+.")

    def _validate_fp8_forward(self, x: Tensor, kv_cache: BlockKVCache) -> None:
        """Reject inputs that cannot execute the strict FP8-only path."""
        if not self.use_fp8:
            return
        if torch.is_grad_enabled():
            raise RuntimeError(
                "FP8-only attention is inference-only; disable gradients."
            )
        self._validate_fp8_device(x.device)
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("FP8-only attention requires FP16 or BF16 activations.")
        if (
            kv_cache._k.dtype != torch.float8_e4m3fn
            or kv_cache._v.dtype != torch.float8_e4m3fn
        ):
            raise TypeError("FP8-only attention requires an FP8 E4M3 KV cache.")
        if kv_cache._k.device != x.device or kv_cache._v.device != x.device:
            raise ValueError("FP8 KV cache and activations must be on the same device.")
        if not isinstance(self.qkv_proj, _PackedFP8Linear) or not isinstance(
            self.output_proj, _PackedFP8Linear
        ):
            raise RuntimeError("FP8-only attention is missing packed projections.")
        if (
            self.qkv_proj.weight.device != x.device
            or self.output_proj.weight.device != x.device
        ):
            raise ValueError(
                "Packed FP8 weights and activations must be on the same device."
            )

    def initialize_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BlockKVCache:
        """Initialize the fixed-size canonical KV cache."""
        cache_dtype = dtype
        if self.use_fp8:
            device = torch.device(device)
            self._validate_fp8_device(device)
            if dtype not in (torch.float16, torch.bfloat16):
                raise TypeError("FP8-only attention requires FP16 or BF16 activations.")
            cache_dtype = torch.float8_e4m3fn
        total_size = sink_size + window_size
        return BlockKVCache(
            k_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            v_shape=(batch_size, total_size, self.n_heads, self.head_dim),
            seq_dim=-3,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=cache_dtype,
        )

    def _supports_fused_postprocess(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None,
    ) -> bool:
        """Return whether Triton can fuse Q/K postprocessing and cache writes."""
        norm_supported = (
            isinstance(self.q_norm, nn.RMSNorm)
            and isinstance(self.k_norm, nn.RMSNorm)
            and self.q_norm.eps == self.k_norm.eps
        ) or (
            isinstance(self.q_norm, nn.Identity)
            and isinstance(self.k_norm, nn.Identity)
        )
        return (
            self.fuse_rope_kv_cache
            and not torch.is_grad_enabled()
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and self.head_dim % 2 == 0
            and self.head_dim <= 512
            and norm_supported
            and isinstance(kv_cache, BlockKVCache)
            and kv_cache.seq_dim == 1
            and kv_cache._k.is_contiguous()
            and kv_cache._v.is_contiguous()
            and kv_cache._k.device == x.device
            and kv_cache._v.device == x.device
            and (rope_freqs is None or rope_freqs.device == x.device)
        )

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project Q/K/V from either packed FP8 or native authoritative weights."""
        inner_dim = self.n_heads * self.head_dim
        use_fused_projection = self.fuse_qkv and not torch.is_grad_enabled()
        if self.use_fp8:
            if use_fused_projection:
                qkv = self.qkv_proj(x, out_dtype=x.dtype)
                qkv = qkv.reshape(-1, x.shape[-2], 3, self.n_heads, self.head_dim)
                query, key, value = qkv.unbind(dim=2)
                return query, key, value
            q_weight, k_weight, v_weight = self.qkv_proj.weight.split(inner_dim, dim=0)
            q_scale, k_scale, v_scale = self.qkv_proj.weight_scale.split(inner_dim)
            if self.qkv_proj.bias is None:
                q_bias = k_bias = v_bias = None
            else:
                q_bias, k_bias, v_bias = self.qkv_proj.bias.split(inner_dim)
            query = _fp8_linear(x, q_weight, q_scale, q_bias, x.dtype)
            key = _fp8_linear(x, k_weight, k_scale, k_bias, x.dtype)
            value = _fp8_linear(x, v_weight, v_scale, v_bias, x.dtype)
        elif use_fused_projection:
            assert self._fused_qkv_weight is not None
            qkv = F.linear(x, self._fused_qkv_weight, self._fused_qkv_bias)
            qkv = qkv.reshape(-1, x.shape[-2], 3, self.n_heads, self.head_dim)
            query, key, value = qkv.unbind(dim=2)
            return query, key, value
        else:
            query = self.q_proj(x)
            key = self.k_proj(x)
            value = self.v_proj(x)
        head_shape = (-1, x.shape[-2], self.n_heads, self.head_dim)
        return (
            query.reshape(head_shape),
            key.reshape(head_shape),
            value.reshape(head_shape),
        )

    def _apply_rope(self, x: Tensor, rope_freqs: Tensor | None) -> Tensor:
        """Apply optional rotary embeddings in the configured pair layout."""
        if rope_freqs is None:
            return x
        half_dim = x.shape[-1] // 2
        if self.rope_interleaved:
            paired = x.reshape(x.shape[:-1] + (half_dim, 2))
            first = paired[..., 0]
            second = paired[..., 1]
            freqs = rope_freqs[:, 0, 0, 0::2]
        else:
            first, second = x.chunk(2, dim=-1)
            freqs = rope_freqs[:, 0, 0, :half_dim]
        freqs = freqs.unsqueeze(0).unsqueeze(2)
        cos_freqs = torch.cos(freqs).to(dtype=x.dtype)
        sin_freqs = torch.sin(freqs).to(dtype=x.dtype)
        rotated_first = first * cos_freqs - second * sin_freqs
        rotated_second = second * cos_freqs + first * sin_freqs
        if self.rope_interleaved:
            return torch.stack((rotated_first, rotated_second), dim=-1).flatten(-2)
        return torch.cat((rotated_first, rotated_second), dim=-1)

    def _postprocess_and_update_cache(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None,
    ) -> Tensor:
        """Postprocess Q/K and write the current K/V chunk into the cache."""
        if not self._supports_fused_postprocess(query, kv_cache, rope_freqs):
            query = self._apply_rope(self.q_norm(query), rope_freqs)
            key = self._apply_rope(self.k_norm(key), rope_freqs)
            kv_cache.update(key, value)
            return query.to(torch.float8_e4m3fn) if self.use_fp8 else query

        query_output = torch.empty(
            query.shape,
            device=query.device,
            dtype=torch.float8_e4m3fn if self.use_fp8 else query.dtype,
        )
        cache_read_start, cache_write_start, cache_write_length = _cache_write_slice(
            kv_cache
        )
        block_s = 4 if query.shape[1] >= 512 else 1
        block_h = min(int(triton.next_power_of_2(self.n_heads)), 32)
        block_d = max(int(triton.next_power_of_2(self.head_dim // 2)), 16)
        grid = (
            query.shape[0],
            triton.cdiv(query.shape[1], block_s),
            triton.cdiv(self.n_heads, block_h),
        )
        apply_norm = isinstance(self.q_norm, nn.RMSNorm)
        query_weight = self.q_norm.weight if apply_norm else query.reshape(-1)
        key_weight = self.k_norm.weight if apply_norm else key.reshape(-1)
        rope_pointer = rope_freqs if rope_freqs is not None else query
        norm_eps = self.q_norm.eps if isinstance(self.q_norm, nn.RMSNorm) else 0.0
        _qkv_postprocess_cache_kernel[grid](
            query,
            key,
            value,
            query_output,
            kv_cache._k,
            kv_cache._v,
            query_weight,
            key_weight,
            rope_pointer,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *query_output.stride(),
            *kv_cache._k.stride(),
            *kv_cache._v.stride(),
            rope_pointer.stride(0),
            rope_pointer.stride(-1),
            query.shape[1],
            self.n_heads,
            cache_read_start,
            cache_write_start,
            cache_write_length,
            EPS=norm_eps,
            HEAD_DIM_HALF=self.head_dim // 2,
            APPLY_NORM=apply_norm,
            APPLY_ROPE=rope_freqs is not None,
            INTERLEAVED=self.rope_interleaved,
            BLOCK_S=block_s,
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=1,
        )
        return query_output

    @staticmethod
    def _validate_attention_inputs(
        query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[int, int, int, int, int]:
        """Validate BLHD attention tensors and return their dimensions."""
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("query, key, and value must have shape [B, L, H, D].")
        if query.device != key.device or query.device != value.device:
            raise ValueError("query, key, and value must be on the same device.")
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError("query, key, and value must have the same dtype.")
        batch_size, query_length, num_heads, head_dim = query.shape
        key_batch, key_length, key_heads, key_dim = key.shape
        if (key_batch, key_heads, key_dim) != (batch_size, num_heads, head_dim):
            raise ValueError(
                "query and key batch, head, and feature dimensions differ."
            )
        if value.shape != key.shape:
            raise ValueError("key and value must have identical shapes.")
        return batch_size, query_length, key_length, num_heads, head_dim

    def _supports_triton_attention(self, query: Tensor) -> bool:
        """Return whether the pointer-based Triton attention kernel is supported."""
        return (
            any(self.optimization_settings.values())
            and not torch.is_grad_enabled()
            and query.is_cuda
            and query.dtype
            in (
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
            )
            and 0 < query.shape[-1] <= 256
        )

    def _supports_tma_attention(self, query: Tensor) -> bool:
        """Return whether tensor descriptors support the current attention layout."""
        head_dim = query.shape[-1]
        return (
            self.use_tma
            and self._supports_triton_attention(query)
            and torch.cuda.get_device_capability(query.device)[0] >= 9
            and head_dim >= 16
            and head_dim & (head_dim - 1) == 0
            and query.stride(-1) == 1
        )

    @staticmethod
    def _apply_sdpa_blhd(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply PyTorch SDPA to BLHD tensors."""
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        return output.transpose(1, 2)

    def _apply_attention_blhd(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> Tensor:
        """Apply Triton attention or fall back to PyTorch SDPA."""
        batch_size, query_length, key_length, num_heads, head_dim = (
            self._validate_attention_inputs(query, key, value)
        )
        if not self._supports_triton_attention(query):
            if query.dtype == torch.float8_e4m3fn:
                native_dtype = torch.bfloat16
                query = query.to(native_dtype)
                key = key.to(native_dtype)
                value = value.to(native_dtype)
            return self._apply_sdpa_blhd(query, key, value)

        output = torch.empty_like(query)
        if batch_size == 0 or num_heads == 0 or query_length == 0:
            return output
        if key_length == 0:
            return output.zero_()

        use_tma = self._supports_tma_attention(query)
        block_m, block_n, block_d, num_warps, num_stages = _select_attention_config(
            query_length, head_dim, query.dtype, use_tma=use_tma
        )
        query_bhld_strides = (
            query.stride(0),
            query.stride(2),
            query.stride(1),
            query.stride(3),
        )
        key_bhld_strides = (key.stride(0), key.stride(2), key.stride(1), key.stride(3))
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
        if use_tma:
            grid = (triton.cdiv(query_length, block_m), batch_size * num_heads)
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
                BLOCK_N=block_n,
                WARP_SPECIALIZE=False,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            grid = (batch_size * num_heads, triton.cdiv(query_length, block_m))
            _flash_attention_kernel[grid](
                query,
                key,
                value,
                output,
                *query_bhld_strides,
                *key_bhld_strides,
                *value_bhld_strides,
                *output_bhld_strides,
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

    def _apply_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply attention to BHLD tensors for compatibility with existing callers."""
        return self._apply_attention_blhd(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)

    def _validate_forward_inputs(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None,
    ) -> None:
        """Validate module inputs before projection or cache mutation."""
        if x.ndim < 2:
            raise ValueError(f"x must have shape [..., L, D]; got {tuple(x.shape)}.")
        if x.shape[-1] != self.query_dim:
            raise ValueError(
                f"x feature width must equal query_dim={self.query_dim}; "
                f"got {x.shape[-1]}."
            )
        if x.shape[-2] != kv_cache.chunk_size:
            raise ValueError(
                f"x sequence length must equal cache chunk_size={kv_cache.chunk_size}; "
                f"got {x.shape[-2]}."
            )
        batch_size = math.prod(x.shape[:-2])
        expected_cache_shape = (batch_size, self.n_heads, self.head_dim)
        actual_cache_shape = (
            kv_cache._k.shape[0],
            kv_cache._k.shape[-2],
            kv_cache._k.shape[-1],
        )
        if actual_cache_shape != expected_cache_shape:
            raise ValueError(
                "cache batch, head, and feature dimensions must equal "
                f"{expected_cache_shape}; got {actual_cache_shape}."
            )
        if rope_freqs is None:
            return
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        expected_rope_shape = (x.shape[-2], 1, 1, self.head_dim)
        if tuple(rope_freqs.shape) != expected_rope_shape:
            raise ValueError(
                f"rope_freqs must have shape {expected_rope_shape}; "
                f"got {tuple(rope_freqs.shape)}."
            )
        if rope_freqs.device != x.device:
            raise ValueError("rope_freqs and x must be on the same device.")

    def _backend_metadata(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None,
    ) -> dict[str, str | bool]:
        """Resolve requested settings to auditable effective backend labels."""
        self._validate_fp8_forward(x, kv_cache)
        use_fp8 = self.use_fp8
        use_fused_projection = self.fuse_qkv and not torch.is_grad_enabled()
        use_fused_postprocess = self._supports_fused_postprocess(
            x, kv_cache, rope_freqs
        )
        attention_dtype = torch.float8_e4m3fn if use_fp8 else x.dtype
        attention_probe = torch.empty(
            (1, max(x.shape[-2], 1), self.n_heads, self.head_dim),
            device=x.device,
            dtype=attention_dtype,
        )
        if not self._supports_triton_attention(attention_probe):
            attention_backend = "sdpa"
        elif self._supports_tma_attention(attention_probe):
            attention_backend = "triton-tma"
        else:
            attention_backend = "triton-pointer"
        return {
            **self.optimization_settings,
            "fp8_effective": use_fp8,
            "qkv_backend": (
                f"{'fp8' if use_fp8 else 'native'}-"
                f"{'fused' if use_fused_projection else 'separate'}"
            ),
            "postprocess_backend": (
                "triton-fused" if use_fused_postprocess else "pytorch"
            ),
            "attention_backend": attention_backend,
            "output_backend": "fp8" if use_fp8 else "native",
            "projection_storage": "fp8-only" if use_fp8 else "native",
            "kv_cache_storage": "fp8-only" if use_fp8 else "native",
        }

    @nvtx.annotate("self_attention.forward")
    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """Update the KV cache and apply configurable streaming self-attention.

        Args:
            x: Current token chunk of shape ``[..., L, query_dim]``.
            kv_cache: Cache prepared with :meth:`BlockKVCache.before_update`.
            rope_freqs: Optional full-width RoPE angles with shape
                ``[L, 1, 1, head_dim]``; ``None`` disables RoPE.

        Returns:
            Projected attention output with shape ``[..., L, query_dim]``.
        """
        self._validate_forward_inputs(x, kv_cache, rope_freqs)
        batch_shape = x.shape[:-2]
        sequence_length = x.shape[-2]
        inner_dim = self.n_heads * self.head_dim
        self._validate_fp8_forward(x, kv_cache)

        with nvtx.annotate("self_attention.qkv_projection"):
            query, key, value = self._project_qkv(x)
        with nvtx.annotate("self_attention.rope_and_cache_update"):
            query = self._postprocess_and_update_cache(
                query,
                key,
                value,
                kv_cache,
                rope_freqs,
            )
        with nvtx.annotate("self_attention.attention"):
            cached_key = kv_cache.cached_k()
            cached_value = kv_cache.cached_v()
            output = self._apply_attention_blhd(query, cached_key, cached_value)
        with nvtx.annotate("self_attention.output_projection"):
            output = output.reshape(batch_shape + (sequence_length, inner_dim))
            if self.use_fp8:
                return self.output_proj(output, out_dtype=x.dtype)
            return self.output_proj(output)
