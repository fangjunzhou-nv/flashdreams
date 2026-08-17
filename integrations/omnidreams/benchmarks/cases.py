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

"""Shared OmniDreams attention benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from omnidreams.transformer.impl.modules import AttentionBackend

from flashdreams.accelerated.multi_head_attention_triton import (
    QKVFusionOption,
    SDPABackend,
)


@dataclass(frozen=True)
class AttentionBenchmarkCase:
    """Configuration and metadata for one attention benchmark implementation."""

    implementation: str
    """Stable implementation name stored in benchmark metadata."""

    attention_backend: AttentionBackend
    """PyTorch network backend configured for this case."""

    sdpa_backend: SDPABackend
    """SDPA backend configured for accelerated self-attention."""

    self_attention_operator: str
    """Self-attention operator reported in benchmark metadata."""

    cross_attention_operator: str
    """Cross-attention operator reported in benchmark metadata."""

    use_fp8: bool = True
    """Whether accelerated projections and supported attention storage use FP8."""

    self_attn_qkv_fusion_option: QKVFusionOption = QKVFusionOption.FULL
    """Projection fusion policy used by accelerated self-attention."""

    cross_attn_qkv_fusion_option: QKVFusionOption = QKVFusionOption.FUSE_KV
    """Projection fusion policy used by accelerated cross-attention."""

    native_dit: bool = False
    """Whether the full-pipeline case bypasses the PyTorch network."""

    minimum_compute_capability: tuple[int, int] | None = None
    """Minimum CUDA compute capability; ``None`` accepts any CUDA device."""

    @property
    def pytest_id(self) -> str:
        """Return the readable pytest parameter identifier."""
        return self.implementation.replace("_", "-")


OMNIDREAMS_TORCH_CASE = AttentionBenchmarkCase(
    implementation="omnidreams_torch",
    attention_backend=AttentionBackend.OMNIDREAMS,
    sdpa_backend=SDPABackend.CUDNN,
    self_attention_operator="cudnn",
    cross_attention_operator="cudnn",
    use_fp8=False,
    self_attn_qkv_fusion_option=QKVFusionOption.NONE,
    cross_attn_qkv_fusion_option=QKVFusionOption.NONE,
)

TRITON_CUDNN_CASE = AttentionBenchmarkCase(
    implementation="triton_cudnn",
    attention_backend=AttentionBackend.TRITON,
    sdpa_backend=SDPABackend.CUDNN,
    self_attention_operator="torch_cudnn_sdpa",
    cross_attention_operator="triton_fa2",
    minimum_compute_capability=(9, 0),
)

TRITON_FA2_CASE = AttentionBenchmarkCase(
    implementation="triton_fa2",
    attention_backend=AttentionBackend.TRITON,
    sdpa_backend=SDPABackend.TRITON,
    self_attention_operator="triton_fa2",
    cross_attention_operator="triton_fa2",
    minimum_compute_capability=(9, 0),
)

NATIVE_CUDA_CASE = AttentionBenchmarkCase(
    implementation="cuda",
    attention_backend=AttentionBackend.OMNIDREAMS,
    sdpa_backend=SDPABackend.CUDNN,
    self_attention_operator="cudnn",
    cross_attention_operator="cudnn",
    use_fp8=False,
    self_attn_qkv_fusion_option=QKVFusionOption.NONE,
    cross_attn_qkv_fusion_option=QKVFusionOption.NONE,
    native_dit=True,
)

PYTORCH_ATTENTION_CASES = (
    OMNIDREAMS_TORCH_CASE,
    TRITON_CUDNN_CASE,
    TRITON_FA2_CASE,
)
"""Attention cases that execute the PyTorch DiT network."""

ATTENTION_POLICY_CASES = (
    OMNIDREAMS_TORCH_CASE,
    *(
        AttentionBenchmarkCase(
            implementation=(
                f"triton_{'fa2' if sdpa_backend is SDPABackend.TRITON else 'cudnn'}_"
                f"{'fp8' if use_fp8 else 'bf16'}_{qkv_fusion_option.value}"
            ),
            attention_backend=AttentionBackend.TRITON,
            sdpa_backend=sdpa_backend,
            self_attention_operator=(
                "triton_fa2"
                if sdpa_backend is SDPABackend.TRITON
                else "torch_cudnn_sdpa"
            ),
            cross_attention_operator=(
                "triton_fa2"
                if sdpa_backend is SDPABackend.TRITON
                else "torch_cudnn_sdpa"
            ),
            use_fp8=use_fp8,
            self_attn_qkv_fusion_option=qkv_fusion_option,
            cross_attn_qkv_fusion_option=(
                qkv_fusion_option
                if qkv_fusion_option is not QKVFusionOption.FULL
                else QKVFusionOption.FUSE_KV
            ),
            minimum_compute_capability=(
                (9, 0) if use_fp8 or sdpa_backend is SDPABackend.TRITON else None
            ),
        )
        for sdpa_backend in SDPABackend
        for use_fp8 in (False, True)
        for qkv_fusion_option in (
            QKVFusionOption.NONE,
            QKVFusionOption.FUSE_KV,
            QKVFusionOption.FULL,
        )
    ),
)
"""Omnidreams reference plus every PyTorch backend, precision, and fusion row.

Full QKV fusion applies to self-attention; the unequal-width production text
cross-attention branch retains K/V fusion for those block rows.
"""

MODULE_CROSS_ATTENTION_CASES = tuple(
    case
    for case in ATTENTION_POLICY_CASES
    if case.attention_backend is AttentionBackend.OMNIDREAMS
    or case.self_attn_qkv_fusion_option is not QKVFusionOption.FULL
)
"""Module cases valid for unequal-width production text cross-attention."""

CROSS_ATTENTION_CASES = (OMNIDREAMS_TORCH_CASE, TRITON_CUDNN_CASE)
"""One case per concrete cross-attention implementation."""

PIPELINE_CASES = (*ATTENTION_POLICY_CASES, NATIVE_CUDA_CASE)
"""Attention cases exercised by the full-pipeline benchmark."""

assert {case.attention_backend for case in PYTORCH_ATTENTION_CASES} == set(
    AttentionBackend
), "Every AttentionBackend must have a PyTorch benchmark case"
assert {
    case.sdpa_backend
    for case in PYTORCH_ATTENTION_CASES
    if case.attention_backend is AttentionBackend.TRITON
} == set(SDPABackend), "Every SDPABackend must have a Triton benchmark case"


def skip_unsupported_device(
    case: AttentionBenchmarkCase,
    device: torch.device,
) -> None:
    """Skip a benchmark case when device is older than its minimum capability."""
    minimum = case.minimum_compute_capability
    if minimum is None:
        return
    if torch.cuda.get_device_capability(device) < minimum:
        pytest.skip(
            f"{case.pytest_id} attention requires compute capability "
            f"{minimum[0]}.{minimum[1]}+"
        )
