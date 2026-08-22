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

"""Native cuDNN Frontend FP8 scaled-dot-product attention."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from functools import lru_cache

import torch
from torch import Tensor


@lru_cache(maxsize=256)
def _build_cudnn_fp8_sdpa(
    device: torch.device,
    stream: int,
    query_shape: tuple[int, ...],
    query_stride: tuple[int, ...],
    key_shape: tuple[int, ...],
    key_stride: tuple[int, ...],
    value_shape: tuple[int, ...],
    value_stride: tuple[int, ...],
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """Build one shape-, layout-, device-, and stream-specialized FP8 graph."""
    try:
        cudnn = importlib.import_module("cudnn")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "quantized cuDNN SDPA requires nvidia-cudnn-frontend"
        ) from error

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    def tensor(name: str, shape: tuple[int, ...], stride: tuple[int, ...]):
        return graph.tensor(
            name=name,
            dim=list(shape),
            stride=list(stride),
            data_type=cudnn.data_type.FP8_E4M3,
        )

    query_desc = tensor("query", query_shape, query_stride)
    key_desc = tensor("key", key_shape, key_stride)
    value_desc = tensor("value", value_shape, value_stride)
    scale_descriptors = tuple(
        graph.tensor(
            name=name,
            dim=[1, 1, 1, 1],
            stride=[1, 1, 1, 1],
            data_type=cudnn.data_type.FLOAT,
        )
        for name in (
            "descale_q",
            "descale_k",
            "descale_v",
            "descale_s",
            "scale_s",
            "scale_o",
        )
    )
    output_desc, _, amax_s_desc, amax_o_desc = graph.sdpa_fp8(
        q=query_desc,
        k=key_desc,
        v=value_desc,
        descale_q=scale_descriptors[0],
        descale_k=scale_descriptors[1],
        descale_v=scale_descriptors[2],
        descale_s=scale_descriptors[3],
        scale_s=scale_descriptors[4],
        scale_o=scale_descriptors[5],
        is_inference=True,
        attn_scale=1.0 / math.sqrt(query_shape[-1]),
        name="sdpa",
    )

    output = torch.empty_strided(
        query_shape, query_stride, dtype=torch.float8_e4m3fn, device=device
    )
    amax_s = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=device)
    amax_o = torch.empty_like(amax_s)
    output_desc.set_output(True).set_dim(list(output.shape)).set_stride(
        list(output.stride())
    )
    amax_s_desc.set_output(False).set_dim(list(amax_s.shape)).set_stride(
        list(amax_s.stride())
    )
    amax_o_desc.set_output(False).set_dim(list(amax_o.shape)).set_stride(
        list(amax_o.stride())
    )
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    scale = torch.ones_like(amax_s)
    workspace = torch.empty(
        graph.get_workspace_size(), dtype=torch.uint8, device=device
    )

    def execute(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        graph.execute(
            {
                query_desc: query,
                key_desc: key,
                value_desc: value,
                **{descriptor: scale for descriptor in scale_descriptors},
                output_desc: output,
                amax_s_desc: amax_s,
                amax_o_desc: amax_o,
            },
            workspace,
        )
        return output

    return execute


def native_cudnn_fp8_sdpa(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """Apply unscaled e4m3 attention with a cached cuDNN Frontend graph.

    Args:
        query: FP8 queries in ``[B, H, L, D]`` layout.
        key: FP8 keys in ``[B, H, S, D]`` layout.
        value: FP8 values in ``[B, H, S, D]`` layout.

    Returns:
        FP8 attention output shaped like ``query``.

    Raises:
        RuntimeError: The cuDNN Frontend package is unavailable.
    """
    execute = _build_cudnn_fp8_sdpa(
        query.device,
        torch.cuda.current_stream(query.device).cuda_stream,
        tuple(query.shape),
        query.stride(),
        tuple(key.shape),
        key.stride(),
        tuple(value.shape),
        value.stride(),
    )
    return execute(query, key, value)
