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

"""CPU tests for multi-head attention and its PyTorch reference."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from flashdreams.accelerated.multi_head_attention import (
    MultiHeadAttention,
    QKNormScope,
)
from flashdreams.accelerated.multi_head_attention_torch import TorchMultiHeadAttention
from flashdreams.core.attention import RotaryPositionEmbedding3D

pytestmark = pytest.mark.ci_cpu


class _Attention(MultiHeadAttention[object]):
    """Minimal concrete attention used to test base initialization."""

    def initialize_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> object:
        """Return a placeholder cache for interface tests."""
        del batch_size, chunk_size, window_size, sink_size, device, dtype
        return object()

    def forward(
        self,
        x: Tensor,
        kv_cache: object,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """Return ``x`` unchanged."""
        del kv_cache, rope_freqs
        return x


def test_initialize_cache_is_abstract() -> None:
    """Require every accelerated attention implementation to create its cache."""
    assert "initialize_cache" in MultiHeadAttention.__abstractmethods__


def test_attention_geometry_and_policies_are_stored() -> None:
    """Preserve defaulted geometry and explicit policy overrides."""
    self_attention = _Attention(
        query_dim=48,
        n_heads=3,
        head_dim=16,
    )
    cross_attention = _Attention(
        query_dim=48,
        n_heads=3,
        head_dim=16,
        context_dim=32,
        qk_norm_scope=QKNormScope.INNER,
        rope_interleaved=True,
    )

    assert self_attention.context_dim == 48
    assert self_attention.inner_dim == 48
    assert self_attention.qk_norm_scope is QKNormScope.HEAD
    assert self_attention.rope_interleaved is False

    assert cross_attention.context_dim == 32
    assert cross_attention.qk_norm_scope is QKNormScope.INNER
    assert cross_attention.rope_interleaved is True


@pytest.mark.parametrize(
    ("dimension", "value"),
    [
        pytest.param("query_dim", 0, id="zero-query-dim"),
        pytest.param("context_dim", 0, id="zero-context-dim"),
        pytest.param("n_heads", 0, id="zero-heads"),
        pytest.param("head_dim", 0, id="zero-head-dim"),
    ],
)
def test_attention_dimensions_must_be_positive(dimension: str, value: int) -> None:
    """Reject zero values for every attention dimension."""
    with pytest.raises(ValueError, match=dimension):
        _Attention(
            query_dim=value if dimension == "query_dim" else 16,
            context_dim=value if dimension == "context_dim" else None,
            n_heads=value if dimension == "n_heads" else 2,
            head_dim=value if dimension == "head_dim" else 8,
        )


def test_torch_attention_runs_complete_streaming_forward_on_cpu() -> None:
    """Run the complete PyTorch reference forward path on CPU."""
    batch_size = 2
    chunk_size = 2
    n_heads = 2
    head_dim = 12
    query_dim = n_heads * head_dim
    x = torch.randn(batch_size, chunk_size, query_dim)

    attention = TorchMultiHeadAttention(
        query_dim=query_dim,
        n_heads=n_heads,
        head_dim=head_dim,
    )
    kv_cache = attention.initialize_cache(
        batch_size=batch_size,
        chunk_size=chunk_size,
        window_size=chunk_size,
        sink_size=0,
        device=torch.device("cpu"),
        dtype=x.dtype,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=head_dim,
        len_t=1,
        len_h=1,
        len_w=chunk_size,
        device=torch.device("cpu"),
    )

    kv_cache.before_update(0)
    output = attention(x, kv_cache, rope.shift_t(0))

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert kv_cache.cached_k().shape == (
        batch_size,
        chunk_size,
        n_heads,
        head_dim,
    )
    assert kv_cache.cached_v().shape == (
        batch_size,
        chunk_size,
        n_heads,
        head_dim,
    )
    kv_cache.after_update(0)
