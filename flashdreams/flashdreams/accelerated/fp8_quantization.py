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

"""Row-scaled E4M3 FP8 quantization and linear projection helpers."""

from __future__ import annotations

import torch
from torch import Tensor

_FP8_MAX = 448.0
"""Largest finite magnitude represented by ``torch.float8_e4m3fn``."""


@torch.no_grad()
def quantize_fp8_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a linear weight with one FP32 scale per output row.

    Args:
        weight: Linear weight with shape ``[out_features, in_features]``.

    Returns:
        Contiguous E4M3 weight and its per-output-row FP32 scales.
    """
    weight_float = weight.detach().to(torch.float32)
    scale = (weight_float.abs().amax(dim=1) / _FP8_MAX).clamp_min(1e-12)
    weight_fp8 = (
        (weight_float / scale[:, None])
        .clamp(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return weight_fp8, scale.contiguous()


def fp8_linear(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None,
    out_dtype: torch.dtype,
) -> Tensor:
    """Apply a row-scaled FP8 GEMM and restore the requested activation dtype.

    Args:
        x: Native-precision or E4M3 input activations.
        weight: E4M3 weight produced by ``quantize_fp8_weight``.
        weight_scale: Per-output-row FP32 weight scales.
        bias: Optional native-precision output bias.
        out_dtype: Activation dtype returned to the caller.

    Returns:
        Projected activations with the leading shape of ``x``.
    """
    input_shape = x.shape
    x_2d = x.reshape(-1, input_shape[-1])
    if x_2d.dtype == torch.float8_e4m3fn:
        x_fp8 = x_2d
        input_scale = torch.ones(
            (x_2d.shape[0], 1),
            device=x.device,
            dtype=torch.float32,
        )
    else:
        x_float = x_2d.to(torch.float32)
        input_scale = (
            (x_float.abs().amax(dim=1, keepdim=True) / _FP8_MAX)
            .clamp_min(1e-12)
            .contiguous()
        )
        x_fp8 = (
            (x_float / input_scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        )
    scaled_bias = bias.to(torch.bfloat16) if bias is not None else None
    output = torch._scaled_mm(
        x_fp8,
        weight.T,
        input_scale,
        weight_scale.reshape(1, -1).contiguous(),
        bias=scaled_bias,
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )
    if out_dtype != torch.bfloat16:
        output = output.to(out_dtype)
    return output.reshape(input_shape[:-1] + (weight.shape[0],))


__all__ = ["fp8_linear", "quantize_fp8_weight"]
