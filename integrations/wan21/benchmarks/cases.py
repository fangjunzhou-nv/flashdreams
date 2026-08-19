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

"""Shared Wan 2.1 attention benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from flashdreams.accelerated.multi_head_attention_triton import (
    QKVFusionOption,
    SDPABackend,
)
from flashdreams.recipes.wan.transformer.impl.modules import AttentionBackend


@dataclass(frozen=True)
class AttentionBenchmarkCase:
    """Configuration for one attention benchmark implementation."""

    implementation: str
    """Stable implementation name used by pytest and pipeline setup."""

    self_attention_backend: AttentionBackend
    """Self-attention implementation configured for this case."""

    cross_attention_backend: AttentionBackend
    """Cross-attention implementation configured for this case."""

    sdpa_backend: SDPABackend
    """SDPA backend configured for accelerated attention."""

    use_fp8: bool = True
    """Whether accelerated projections and supported cache storage use FP8."""

    self_attn_qkv_fusion_option: QKVFusionOption = QKVFusionOption.FULL
    """Projection fusion policy used by accelerated self-attention."""

    cross_attn_qkv_fusion_option: QKVFusionOption = QKVFusionOption.FUSE_KV
    """Projection fusion policy used by accelerated cross-attention."""

    minimum_compute_capability: tuple[int, int] | None = None
    """Minimum CUDA compute capability; ``None`` accepts any CUDA device."""

    @property
    def pytest_id(self) -> str:
        """Return the readable pytest parameter identifier."""
        return self.implementation.replace("_", "-")


BENCHMARK_CASES = [
    AttentionBenchmarkCase(
        implementation="wan_torch",
        self_attention_backend=AttentionBackend.WAN,
        cross_attention_backend=AttentionBackend.WAN,
        sdpa_backend=SDPABackend.CUDNN,
        use_fp8=False,
        self_attn_qkv_fusion_option=QKVFusionOption.NONE,
        cross_attn_qkv_fusion_option=QKVFusionOption.NONE,
    ),
    AttentionBenchmarkCase(
        implementation="triton_fa2_fp8_full",
        self_attention_backend=AttentionBackend.TRITON,
        cross_attention_backend=AttentionBackend.TRITON,
        sdpa_backend=SDPABackend.TRITON,
        minimum_compute_capability=(9, 0),
    ),
    AttentionBenchmarkCase(
        implementation="triton_cudnn_bf16_full",
        self_attention_backend=AttentionBackend.TRITON,
        cross_attention_backend=AttentionBackend.TRITON,
        sdpa_backend=SDPABackend.CUDNN,
        use_fp8=False,
    ),
    AttentionBenchmarkCase(
        implementation="triton_cudnn_bf16_full_wan_cross",
        self_attention_backend=AttentionBackend.TRITON,
        cross_attention_backend=AttentionBackend.WAN,
        sdpa_backend=SDPABackend.CUDNN,
        use_fp8=False,
        cross_attn_qkv_fusion_option=QKVFusionOption.NONE,
    ),
]
"""Attention cases exercised by the Wan 2.1 network and pipeline benchmarks."""


def skip_unsupported_device(
    case: AttentionBenchmarkCase,
    device: torch.device,
) -> None:
    """Skip a benchmark case when the device is older than its minimum."""
    minimum = case.minimum_compute_capability
    if minimum is None:
        return
    if torch.cuda.get_device_capability(device) < minimum:
        pytest.skip(
            f"{case.implementation} attention requires compute capability "
            f"{minimum[0]}.{minimum[1]}+"
        )
