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

"""TMA-backed Triton FlashAttention2 for projected attention tensors.

Each program owns one query tile in one batch/head plane. Q is loaded once,
K/V tiles stream through two-dimensional tensor descriptors, and the kernel
retains only the online-softmax state and output accumulator in SRAM. No score
matrix is materialized in global memory.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor


def _allocate_tma_workspace(
    size: int,
    alignment: int,
    stream: int | None,
) -> Tensor:
    """Allocate descriptor workspace on the active CUDA device."""
    del alignment, stream
    return torch.empty(size, device="cuda", dtype=torch.int8)


# In-kernel tensor descriptors need a small device allocation at launch time.
triton.set_allocator(_allocate_tma_workspace)


_TMA_ATTENTION_CONFIGS = [
    triton.Config(
        {"BLOCK_M": block_m, "BLOCK_N": block_n},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for block_m, block_n, num_warps, num_stages in (
        (16, 32, 4, 2),
        (32, 32, 4, 2),
        (64, 32, 4, 3),
        (64, 64, 4, 3),
        (64, 64, 8, 3),
        (128, 32, 4, 3),
        (128, 64, 8, 3),
    )
]


def _prune_tma_attention_configs(
    configs: list[triton.Config],
    named_args: dict[str, object],
    **meta: object,
) -> list[triton.Config]:
    """Drop tiles that waste work or exceed wide-head shared memory."""
    query_length = named_args["query_length"]
    key_length = named_args["key_length"]
    head_dim = meta["HEAD_DIM"]
    assert isinstance(query_length, int)
    assert isinstance(key_length, int)
    assert isinstance(head_dim, int)
    maximum_block_m = min(128, max(16, int(triton.next_power_of_2(query_length))))
    if head_dim > 128:
        maximum_block_m = min(maximum_block_m, 64)
    maximum_block_n = min(64, max(32, int(triton.next_power_of_2(key_length))))
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_M"] <= maximum_block_m
        and config.kwargs["BLOCK_N"] <= maximum_block_n
    ]


@triton.autotune(
    configs=_TMA_ATTENTION_CONFIGS,
    key=["num_heads", "query_length", "key_length", "element_size", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_tma_attention_configs},
    cache_results=True,
)
@triton.jit
def _flash_attention_2_tma_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    query_stride_b,
    query_stride_h,
    query_stride_l,
    query_stride_d: tl.constexpr,
    key_stride_b,
    key_stride_h,
    key_stride_s,
    key_stride_d: tl.constexpr,
    value_stride_b,
    value_stride_h,
    value_stride_s,
    value_stride_d: tl.constexpr,
    output_stride_b,
    output_stride_h,
    output_stride_l,
    output_stride_d: tl.constexpr,
    num_heads,
    query_length,
    key_length,
    element_size,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply non-causal FlashAttention2 with TMA loads and stores."""
    # The launch grid independently parallelizes query tiles and flattened
    # batch/head planes.
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    # Offset each descriptor to one batch/head plane. Two-dimensional
    # descriptors match the physical token/feature traversal issued by TMA.
    query_base = query_ptr + batch * query_stride_b + head * query_stride_h
    key_base = key_ptr + batch * key_stride_b + head * key_stride_h
    value_base = value_ptr + batch * value_stride_b + head * value_stride_h
    output_base = output_ptr + batch * output_stride_b + head * output_stride_h
    query_desc = tl.make_tensor_descriptor(
        query_base,
        shape=[query_length, HEAD_DIM],
        strides=[query_stride_l, query_stride_d],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    key_desc = tl.make_tensor_descriptor(
        key_base,
        shape=[key_length, HEAD_DIM],
        strides=[key_stride_s, key_stride_d],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    value_desc = tl.make_tensor_descriptor(
        value_base,
        shape=[key_length, HEAD_DIM],
        strides=[value_stride_s, value_stride_d],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    output_desc = tl.make_tensor_descriptor(
        output_base,
        shape=[query_length, HEAD_DIM],
        strides=[output_stride_l, output_stride_d],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    query_start = query_block * BLOCK_M
    query = query_desc.load([query_start, 0])

    # Keep only the FlashAttention2 online-softmax state and the output tile in
    # SRAM while K/V tiles stream through TMA.
    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    key_offsets = tl.arange(0, BLOCK_N)
    # exp2 is cheaper than exp. log2(e) preserves the requested softmax scale
    # while expressing the online recurrence in base two.
    qk_scale = scale.to(tl.float32) * 1.4426950408889634

    for key_start in tl.range(0, key_length, BLOCK_N):
        key = key_desc.load([key_start, 0])
        scores = tl.dot(query, tl.trans(key)) * qk_scale
        key_mask = key_start + key_offsets < key_length
        scores = tl.where(key_mask[None, :], scores, -float("inf"))

        # Rebase the previous numerator and denominator whenever a new row
        # maximum appears. FP32 state keeps long cache windows stable.
        tile_max = tl.max(scores, axis=1)
        next_row_max = tl.maximum(row_max, tile_max)
        correction = tl.exp2(row_max - next_row_max)
        probabilities = tl.exp2(scores - next_row_max[:, None])
        denominator = denominator * correction + tl.sum(probabilities, axis=1)

        value = value_desc.load([key_start, 0])
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities.to(value.dtype),
            value,
        )
        row_max = next_row_max

    output = accumulator / denominator[:, None]
    output_desc.store([query_start, 0], output)


def _descriptor_layout_supported(x: Tensor) -> bool:
    """Return whether ``x`` satisfies TMA tensor-descriptor stride rules."""
    element_size = x.element_size()
    # Public tensors are [B, L, H, D], but each descriptor traverses an [L, D]
    # plane selected by its B/H base pointer. TMA requires contiguous features
    # and 16-byte alignment for every outer byte stride.
    bhld_strides = (x.stride(0), x.stride(2), x.stride(1), x.stride(3))
    return bhld_strides[-1] == 1 and all(
        stride > 0 and stride * element_size % 16 == 0 for stride in bhld_strides[:-1]
    )


def is_tma_flash_attention_supported(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> bool:
    """Return whether projected tensors can use the TMA attention kernel."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        return False
    if query.device != key.device or query.device != value.device:
        return False
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        return False

    batch_size, _, num_heads, head_dim = query.shape
    if key.shape[0] != batch_size or key.shape[2:] != (num_heads, head_dim):
        return False
    if value.shape != key.shape:
        return False
    if not (16 <= head_dim <= 256 and head_dim & (head_dim - 1) == 0):
        return False
    if torch.cuda.get_device_capability(query.device)[0] < 9:
        return False
    return all(_descriptor_layout_supported(x) for x in (query, key, value))


def flash_attention_2_tma(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Apply non-causal TMA FlashAttention2 to token-major Q/K/V tensors.

    Args:
        query: Query tensor with shape ``[B, L, H, D]``.
        key: Key tensor with shape ``[B, S, H, D]``.
        value: Value tensor with shape ``[B, S, H, D]``.
        scale: QK scale; ``None`` uses ``1 / sqrt(D)``.

    Returns:
        Attention result with shape ``[B, L, H, D]``.

    Raises:
        ValueError: Q/K/V shapes are incompatible or contain an empty key axis.
        RuntimeError: The device, dtype, or layout does not support TMA.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B, L, H, D]")
    batch_size, query_length, num_heads, head_dim = query.shape
    if key.shape[0] != batch_size or key.shape[2:] != (num_heads, head_dim):
        raise ValueError("query and key batch, head, and feature dimensions differ")
    if value.shape != key.shape:
        raise ValueError("key and value must have identical shapes")
    key_length = key.shape[1]
    if key_length == 0:
        raise ValueError("key and value sequence length must be positive")
    if not is_tma_flash_attention_supported(query, key, value):
        raise RuntimeError(
            "TMA FlashAttention2 requires matching CUDA FP16/BF16/FP8 tensors, "
            "compute capability 9.0 or newer, a power-of-two head_dim in "
            "[16, 256], and tensor-descriptor-compatible strides"
        )

    output = torch.empty(
        query.shape,
        device=query.device,
        dtype=query.dtype,
    )
    if batch_size == 0 or num_heads == 0 or query_length == 0:
        return output

    # Reorder logical strides for the per-plane [B, H, L, D] descriptor view;
    # this is metadata only and does not transpose or copy a tensor.
    query_strides = (
        query.stride(0),
        query.stride(2),
        query.stride(1),
        query.stride(3),
    )
    key_strides = (key.stride(0), key.stride(2), key.stride(1), key.stride(3))
    value_strides = (
        value.stride(0),
        value.stride(2),
        value.stride(1),
        value.stride(3),
    )
    output_strides = (
        output.stride(0),
        output.stride(2),
        output.stride(1),
        output.stride(3),
    )

    # Autotuning selects the launch shape once per geometry and storage width,
    # then reuses the cached result for steady-state calls.
    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            triton.cdiv(query_length, meta["BLOCK_M"]),
            batch_size * num_heads,
        )

    _flash_attention_2_tma_kernel[grid](
        query,
        key,
        value,
        output,
        *query_strides,
        *key_strides,
        *value_strides,
        *output_strides,
        num_heads,
        query_length,
        key_length,
        query.element_size(),
        1.0 / math.sqrt(head_dim) if scale is None else scale,
        HEAD_DIM=head_dim,
    )
    return output


__all__ = ["flash_attention_2_tma", "is_tma_flash_attention_supported"]
