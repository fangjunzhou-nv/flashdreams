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

"""Fused Q/K RMS normalization, RoPE, and K/V-cache updates.

Two kernels reflect the normalization domains. ``QKNormScope.HEAD`` assigns a
program to each ``[D]`` head; ``QKNormScope.INNER`` assigns one to the complete
``[H * D]`` token width. Both optionally normalize and rotate Q/K, return
processed Q, cache processed K, and cache unnormalized, unrotated V. Shape
comments use ``B`` for batch, ``L`` for current tokens, ``S`` for cache
capacity, ``H`` for heads, and ``D`` for the head dimension.
"""

from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl
from flashdreams.accelerated.multi_head_attention import QKNormScope


@triton.jit
def _fused_head_rms_rope_kv_cache_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_output_ptr,
    key_cache_ptr,
    value_cache_ptr,
    query_weight_ptr,
    key_weight_ptr,
    rope_freqs_ptr,
    query_stride_b,
    query_stride_l,
    query_stride_h,
    query_stride_d,
    key_stride_b,
    key_stride_l,
    key_stride_h,
    key_stride_d,
    value_stride_b,
    value_stride_l,
    value_stride_h,
    value_stride_d,
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
    HEAD_DIM: tl.constexpr,
    APPLY_NORM: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Process one query/key head and write its current cache slice.

    Q/K/V and processed Q use logical shape ``[B, L, H, D]``; caches use
    ``[B, S, H, D]``. The one-dimensional grid has ``B * L * H`` programs,
    each reducing and rotating one ``[D]`` vector. Strides are in elements,
    and ``BLOCK_D`` is the power-of-two lane count covering ``D``.
    """
    # One program owns a (batch, token, head) vector: exactly the reduction
    # domain for head-scoped RMSNorm.
    program = tl.program_id(0)
    head = program % num_heads
    token_program = program // num_heads
    token = token_program % sequence_length
    batch = token_program // sequence_length

    # Mask padded lanes in ``[BLOCK_D]`` while loading one physical ``[D]``
    # head from token-major Q/K/V storage.
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    query_base = (
        query_ptr
        + batch * query_stride_b
        + token * query_stride_l
        + head * query_stride_h
    )
    key_base = (
        key_ptr + batch * key_stride_b + token * key_stride_l + head * key_stride_h
    )
    value_base = (
        value_ptr
        + batch * value_stride_b
        + token * value_stride_l
        + head * value_stride_h
    )
    query = tl.load(
        query_base + dim_offsets * query_stride_d,
        mask=dim_mask,
        other=0.0,
    )
    key = tl.load(
        key_base + dim_offsets * key_stride_d,
        mask=dim_mask,
        other=0.0,
    )

    if APPLY_NORM:
        # Reduce RMS statistics in FP32, then cast back after applying the
        # learned ``[D]`` weight so the result matches the projection dtype.
        query_scale = tl.rsqrt(
            tl.sum(query.to(tl.float32) * query.to(tl.float32), axis=0) / HEAD_DIM + EPS
        )
        key_scale = tl.rsqrt(
            tl.sum(key.to(tl.float32) * key.to(tl.float32), axis=0) / HEAD_DIM + EPS
        )
        query_weight = tl.load(
            query_weight_ptr + dim_offsets,
            mask=dim_mask,
            other=0.0,
        )
        key_weight = tl.load(
            key_weight_ptr + dim_offsets,
            mask=dim_mask,
            other=0.0,
        )
        query = (query * query_scale * query_weight).to(query.dtype)
        key = (key * key_scale * key_weight).to(key.dtype)

    if APPLY_ROPE:
        # Map every feature lane to its rotation partner within the same head:
        # adjacent pairs for interleaved RoPE, otherwise matching half splits.
        if INTERLEAVED:
            partner_offsets = tl.where(
                dim_offsets % 2 == 0,
                dim_offsets + 1,
                dim_offsets - 1,
            )
            rotation_sign = tl.where(dim_offsets % 2 == 0, -1.0, 1.0)
        else:
            partner_offsets = tl.where(
                dim_offsets < HEAD_DIM // 2,
                dim_offsets + HEAD_DIM // 2,
                dim_offsets - HEAD_DIM // 2,
            )
            rotation_sign = tl.where(dim_offsets < HEAD_DIM // 2, -1.0, 1.0)

        # Reload the unnormalized rotation partner and apply the same scalar
        # RMS scale, avoiding a materialized normalized Q/K intermediate.
        query_partner = tl.load(
            query_base + partner_offsets * query_stride_d,
            mask=dim_mask,
            other=0.0,
        )
        key_partner = tl.load(
            key_base + partner_offsets * key_stride_d,
            mask=dim_mask,
            other=0.0,
        )
        if APPLY_NORM:
            query_partner_weight = tl.load(
                query_weight_ptr + partner_offsets,
                mask=dim_mask,
                other=0.0,
            )
            key_partner_weight = tl.load(
                key_weight_ptr + partner_offsets,
                mask=dim_mask,
                other=0.0,
            )
            query_partner = (query_partner * query_scale * query_partner_weight).to(
                query_partner.dtype
            )
            key_partner = (key_partner * key_scale * key_partner_weight).to(
                key_partner.dtype
            )

        # RoPE angles ``[L, D]`` are shared across batch and head axes.
        frequencies = tl.load(
            rope_freqs_ptr + token * rope_stride_l + dim_offsets * rope_stride_d,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        cos_freqs = tl.cos(frequencies).to(query.dtype)
        sin_freqs = tl.sin(frequencies).to(query.dtype)
        query = query * cos_freqs + query_partner * sin_freqs * rotation_sign
        key = key * cos_freqs + key_partner * sin_freqs * rotation_sign

    # Store processed Q ``[D]`` for this ``(batch, token, head)`` program.
    query_output_base = (
        query_output_ptr
        + batch * query_output_stride_b
        + token * query_output_stride_l
        + head * query_output_stride_h
    )
    tl.store(
        query_output_base + dim_offsets * query_output_stride_d,
        query,
        mask=dim_mask,
    )

    # Map only the selected source slice onto its physical cache interval. A
    # nonzero read start occurs when an immutable sink clips a rolling write.
    cache_offset = token - cache_read_start
    cache_token = cache_write_start + cache_offset
    cache_mask = dim_mask & (cache_offset >= 0) & (cache_offset < cache_write_length)
    key_cache_base = (
        key_cache_ptr
        + batch * key_cache_stride_b
        + cache_token * key_cache_stride_l
        + head * key_cache_stride_h
    )
    value_cache_base = (
        value_cache_ptr
        + batch * value_cache_stride_b
        + cache_token * value_cache_stride_l
        + head * value_cache_stride_h
    )
    value = tl.load(
        value_base + dim_offsets * value_stride_d,
        mask=cache_mask,
        other=0.0,
    )
    tl.store(
        key_cache_base + dim_offsets * key_cache_stride_d,
        key,
        mask=cache_mask,
    )
    tl.store(
        value_cache_base + dim_offsets * value_cache_stride_d,
        value,
        mask=cache_mask,
    )


@triton.jit
def _fused_inner_rms_rope_kv_cache_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_output_ptr,
    key_cache_ptr,
    value_cache_ptr,
    query_weight_ptr,
    key_weight_ptr,
    rope_freqs_ptr,
    query_stride_b,
    query_stride_l,
    key_stride_b,
    key_stride_l,
    value_stride_b,
    value_stride_l,
    query_output_stride_b,
    query_output_stride_l,
    key_cache_stride_b,
    key_cache_stride_l,
    value_cache_stride_b,
    value_cache_stride_l,
    rope_stride_l,
    rope_stride_d,
    sequence_length,
    cache_read_start,
    cache_write_start,
    cache_write_length,
    EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INNER_DIM: tl.constexpr,
    APPLY_NORM: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    """Process one full-inner-width query/key token and update its cache.

    Q/K/V and processed Q use logical shape ``[B, L, H, D]`` but are traversed
    as contiguous ``[B, L, H * D]`` tensors. Caches use ``[B, S, H, D]``. The
    grid has ``B * L`` programs, each reducing one ``[H * D]`` vector;
    ``BLOCK_INNER`` is the power-of-two lane count covering that width.
    """
    # Inner-scoped RMSNorm couples all heads, so one program owns the complete
    # projected width for one (batch, token) pair.
    program = tl.program_id(0)
    token = program % sequence_length
    batch = program // sequence_length

    # Load flattened Q/K ``[H * D]``; padded ``BLOCK_INNER`` lanes are masked.
    inner_offsets = tl.arange(0, BLOCK_INNER)
    inner_mask = inner_offsets < INNER_DIM
    query_base = query_ptr + batch * query_stride_b + token * query_stride_l
    key_base = key_ptr + batch * key_stride_b + token * key_stride_l
    value_base = value_ptr + batch * value_stride_b + token * value_stride_l
    query = tl.load(query_base + inner_offsets, mask=inner_mask, other=0.0)
    key = tl.load(key_base + inner_offsets, mask=inner_mask, other=0.0)

    if APPLY_NORM:
        # One FP32 RMS reduction spans all heads, followed by learned
        # ``[H * D]`` weights for Q and K.
        query_scale = tl.rsqrt(
            tl.sum(query.to(tl.float32) * query.to(tl.float32), axis=0) / INNER_DIM
            + EPS
        )
        key_scale = tl.rsqrt(
            tl.sum(key.to(tl.float32) * key.to(tl.float32), axis=0) / INNER_DIM + EPS
        )
        query_weight = tl.load(
            query_weight_ptr + inner_offsets,
            mask=inner_mask,
            other=0.0,
        )
        key_weight = tl.load(
            key_weight_ptr + inner_offsets,
            mask=inner_mask,
            other=0.0,
        )
        query = (query * query_scale * query_weight).to(query.dtype)
        key = (key * key_scale * key_weight).to(key.dtype)

    if APPLY_ROPE:
        # RoPE still operates within each head. Reconstruct per-head partner
        # offsets from the flattened lane after reducing RMS over H * D.
        head_offsets = (inner_offsets // HEAD_DIM) * HEAD_DIM
        dim_offsets = inner_offsets % HEAD_DIM
        if INTERLEAVED:
            partner_dims = tl.where(
                dim_offsets % 2 == 0,
                dim_offsets + 1,
                dim_offsets - 1,
            )
            rotation_sign = tl.where(dim_offsets % 2 == 0, -1.0, 1.0)
        else:
            partner_dims = tl.where(
                dim_offsets < HEAD_DIM // 2,
                dim_offsets + HEAD_DIM // 2,
                dim_offsets - HEAD_DIM // 2,
            )
            rotation_sign = tl.where(dim_offsets < HEAD_DIM // 2, -1.0, 1.0)
        partner_offsets = head_offsets + partner_dims

        query_partner = tl.load(
            query_base + partner_offsets,
            mask=inner_mask,
            other=0.0,
        )
        key_partner = tl.load(
            key_base + partner_offsets,
            mask=inner_mask,
            other=0.0,
        )
        if APPLY_NORM:
            query_partner_weight = tl.load(
                query_weight_ptr + partner_offsets,
                mask=inner_mask,
                other=0.0,
            )
            key_partner_weight = tl.load(
                key_weight_ptr + partner_offsets,
                mask=inner_mask,
                other=0.0,
            )
            query_partner = (query_partner * query_scale * query_partner_weight).to(
                query_partner.dtype
            )
            key_partner = (key_partner * key_scale * key_partner_weight).to(
                key_partner.dtype
            )

        # Convert flattened lanes back to feature indices so ``[L, D]`` RoPE
        # angles broadcast identically across every head.
        frequencies = tl.load(
            rope_freqs_ptr + token * rope_stride_l + dim_offsets * rope_stride_d,
            mask=inner_mask,
            other=0.0,
        ).to(tl.float32)
        cos_freqs = tl.cos(frequencies).to(query.dtype)
        sin_freqs = tl.sin(frequencies).to(query.dtype)
        query = query * cos_freqs + query_partner * sin_freqs * rotation_sign
        key = key * cos_freqs + key_partner * sin_freqs * rotation_sign

    # Store the processed contiguous ``[H * D]`` query vector.
    query_output_base = (
        query_output_ptr + batch * query_output_stride_b + token * query_output_stride_l
    )
    tl.store(query_output_base + inner_offsets, query, mask=inner_mask)

    # Map the selected source interval ``[cache_read_start, ...]`` to the
    # physical cache interval beginning at ``cache_write_start``.
    cache_offset = token - cache_read_start
    cache_token = cache_write_start + cache_offset
    cache_mask = inner_mask & (cache_offset >= 0) & (cache_offset < cache_write_length)
    key_cache_base = (
        key_cache_ptr + batch * key_cache_stride_b + cache_token * key_cache_stride_l
    )
    value_cache_base = (
        value_cache_ptr
        + batch * value_cache_stride_b
        + cache_token * value_cache_stride_l
    )
    value = tl.load(value_base + inner_offsets, mask=cache_mask, other=0.0)
    tl.store(key_cache_base + inner_offsets, key, mask=cache_mask)
    tl.store(value_cache_base + inner_offsets, value, mask=cache_mask)


def fused_rms_rope_kv_cache_update(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    *,
    query_weight: Tensor | None,
    key_weight: Tensor | None,
    norm_eps: float,
    norm_scope: QKNormScope,
    rope_freqs: Tensor | None,
    rope_interleaved: bool,
    cache_read_start: int,
    cache_write_start: int,
    cache_write_length: int,
) -> Tensor:
    """Normalize and rotate Q/K while writing the current K/V cache slice.

    Normalization precedes RoPE. Processed K and unchanged V from source slice
    ``[cache_read_start:cache_read_start + cache_write_length]`` are cast to the
    cache dtype and written at ``cache_write_start``. Tokens outside that slice
    still produce processed queries but do not modify cache storage.

    Args:
        query: Projected queries with shape ``[B, L, H, D]``.
        key: Projected keys with shape ``[B, L, H, D]``.
        value: Projected values with shape ``[B, L, H, D]``.
        key_cache: Contiguous key storage with shape ``[B, S, H, D]``.
        value_cache: Contiguous value storage with shape ``[B, S, H, D]``.
        query_weight: RMSNorm query weight with shape ``[D]`` for head scope or
            ``[H * D]`` for inner scope; ``None`` skips normalization.
        key_weight: RMSNorm key weight with the same shape as ``query_weight``;
            ``None`` skips normalization.
        norm_eps: Epsilon used by Q/K RMS normalization.
        norm_scope: Normalize each head or the complete projected inner width.
        rope_freqs: Optional full-width RoPE angles with shape ``[L, 1, 1, D]``.
        rope_interleaved: Rotate adjacent pairs instead of half-split pairs.
        cache_read_start: First token on the ``L`` axis copied into the cache.
        cache_write_start: First token on the cache ``S`` axis written.
        cache_write_length: Number of consecutive source tokens written.

    Returns:
        Processed queries with shape ``[B, L, H, D]`` and the cache dtype.
        Processed K and unchanged V are written in place to their caches.

    Raises:
        ValueError: Tensor shapes, cache bounds, or normalization inputs differ.
        TypeError: ``norm_scope`` is not a :class:`QKNormScope`.
        RuntimeError: Tensors are not compatible CUDA inputs.
    """
    # Validate logical Q/K/V ``[B, L, H, D]`` and cache ``[B, S, H, D]``
    # geometry before checking storage properties used by the kernels.
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError(
            "query, key, and value must have identical [B, L, H, D] shapes"
        )
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError(
            "key_cache and value_cache must have identical [B, S, H, D] shapes"
        )
    batch_size, sequence_length, num_heads, head_dim = query.shape
    if key_cache.shape[0] != batch_size or key_cache.shape[2:] != (
        num_heads,
        head_dim,
    ):
        raise ValueError("cache batch, head, and feature dimensions differ from Q/K/V")
    tensors = (query, key, value, key_cache, value_cache)
    if not all(x.is_cuda for x in tensors):
        raise RuntimeError("fused RMS/RoPE/cache update requires CUDA tensors")
    if len({x.device for x in tensors}) != 1:
        raise RuntimeError("Q/K/V and cache tensors must share a device")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise RuntimeError("Q/K/V tensors must share a dtype")
    if key_cache.dtype != value_cache.dtype:
        raise RuntimeError("K/V cache tensors must share a dtype")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("fused RMS/RoPE/cache update requires FP16 or BF16")
    if key_cache.dtype not in (query.dtype, torch.float8_e4m3fn):
        raise RuntimeError(
            "K/V cache storage must match Q/K/V or use torch.float8_e4m3fn"
        )
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        raise RuntimeError("K/V cache storage must be contiguous")
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise RuntimeError("Q/K/V feature dimensions must be contiguous")
    if not isinstance(norm_scope, QKNormScope):
        raise TypeError(f"norm_scope must be a QKNormScope; got {norm_scope!r}")

    # Both normalization branches require a matched Q/K weight pair whose
    # width is determined by the selected reduction scope.
    apply_norm = query_weight is not None or key_weight is not None
    if (query_weight is None) != (key_weight is None):
        raise ValueError("query_weight and key_weight must both be present or absent")
    inner_dim = num_heads * head_dim
    if apply_norm:
        assert query_weight is not None and key_weight is not None
        expected_weight_size = head_dim if norm_scope is QKNormScope.HEAD else inner_dim
        if query_weight.shape != (expected_weight_size,) or key_weight.shape != (
            expected_weight_size,
        ):
            raise ValueError(
                f"RMSNorm weights must have shape ({expected_weight_size},)"
            )
        if (
            query_weight.device != query.device
            or key_weight.device != query.device
            or query_weight.dtype != query.dtype
            or key_weight.dtype != query.dtype
        ):
            raise RuntimeError("RMSNorm weights must match the Q/K device and dtype")

    # RoPE stores one full-width ``[D]`` angle vector per current token; batch
    # and head axes broadcast inside both kernels.
    if rope_freqs is not None:
        expected_rope_shape = (sequence_length, 1, 1, head_dim)
        if tuple(rope_freqs.shape) != expected_rope_shape:
            raise ValueError(
                f"rope_freqs must have shape {expected_rope_shape}; "
                f"got {tuple(rope_freqs.shape)}"
            )
        if rope_freqs.device != query.device:
            raise RuntimeError("rope_freqs must be on the Q/K device")
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")

    # The source ``L`` interval and destination ``S`` interval must each fit;
    # they may begin at different offsets when a rolling cache preserves sinks.
    cache_size = key_cache.shape[1]
    if not (0 <= cache_read_start <= sequence_length):
        raise ValueError("cache_read_start is outside the current sequence")
    if not (0 <= cache_write_start <= cache_size):
        raise ValueError("cache_write_start is outside the cache")
    if cache_write_length < 0:
        raise ValueError("cache_write_length must be non-negative")
    if cache_read_start + cache_write_length > sequence_length:
        raise ValueError("cache source slice exceeds the current sequence")
    if cache_write_start + cache_write_length > cache_size:
        raise ValueError("cache destination slice exceeds cache storage")

    # Match processed Q to cache storage so FP8 attention consumes it directly.
    query_output = torch.empty(
        query.shape,
        device=query.device,
        dtype=key_cache.dtype,
    )
    if batch_size == 0 or sequence_length == 0 or num_heads == 0:
        return query_output

    # Triton launch arguments cannot be None. Compile-time APPLY_NORM and
    # APPLY_ROPE eliminate every load from these aliases on disabled paths.
    query_weight_ptr = query if query_weight is None else query_weight
    key_weight_ptr = key if key_weight is None else key_weight
    rope_pointer = query if rope_freqs is None else rope_freqs
    common_meta = {
        "EPS": norm_eps,
        "HEAD_DIM": head_dim,
        "APPLY_NORM": apply_norm,
        "APPLY_ROPE": rope_freqs is not None,
        "INTERLEAVED": rope_interleaved,
    }
    if norm_scope is QKNormScope.HEAD:
        # The enum selects a distinct launch topology rather than a runtime
        # string branch inside one oversized kernel.
        # One program per ``(batch, token, head)`` and one power-of-two block
        # covering the ``D``-wide normalization/rotation domain.
        block_d = max(16, int(triton.next_power_of_2(head_dim)))
        grid = (batch_size * sequence_length * num_heads,)
        _fused_head_rms_rope_kv_cache_kernel[grid](
            query,
            key,
            value,
            query_output,
            key_cache,
            value_cache,
            query_weight_ptr,
            key_weight_ptr,
            rope_pointer,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *query_output.stride(),
            *key_cache.stride(),
            *value_cache.stride(),
            rope_pointer.stride(0),
            rope_pointer.stride(-1),
            sequence_length,
            num_heads,
            cache_read_start,
            cache_write_start,
            cache_write_length,
            BLOCK_D=block_d,
            **common_meta,
            num_warps=4,
            num_stages=1,
        )
    else:
        # Flattened inner-width loads require H and D to form one contiguous
        # physical span.
        if not all(
            x.stride(-2) == head_dim and x.stride(-1) == 1 for x in (query, key, value)
        ):
            raise RuntimeError(
                "inner-scope RMSNorm requires contiguous head and feature dimensions"
            )
        # One program per ``(batch, token)`` and one power-of-two block covering
        # the complete contiguous ``H * D`` normalization domain.
        block_inner = max(16, int(triton.next_power_of_2(inner_dim)))
        grid = (batch_size * sequence_length,)
        _fused_inner_rms_rope_kv_cache_kernel[grid](
            query,
            key,
            value,
            query_output,
            key_cache,
            value_cache,
            query_weight_ptr,
            key_weight_ptr,
            rope_pointer,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            query_output.stride(0),
            query_output.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            value_cache.stride(0),
            value_cache.stride(1),
            rope_pointer.stride(0),
            rope_pointer.stride(-1),
            sequence_length,
            cache_read_start,
            cache_write_start,
            cache_write_length,
            INNER_DIM=inner_dim,
            BLOCK_INNER=block_inner,
            **common_meta,
            num_warps=8 if inner_dim > 2048 else 4,
            num_stages=1,
        )
    return query_output


__all__ = ["fused_rms_rope_kv_cache_update"]
