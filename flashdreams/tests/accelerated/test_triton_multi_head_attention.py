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

"""GPU parity tests for Triton multi-head attention."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashdreams.accelerated.impl import TritonMultiHeadAttention
from flashdreams.accelerated.impl.triton import flash_attention_2_tma
from flashdreams.accelerated.multi_head_attention import QKNormScope
from flashdreams.accelerated.reference import TorchMultiHeadAttention
from flashdreams.core.attention import BlockKVCache, RotaryPositionEmbedding3D

pytestmark = pytest.mark.ci_gpu


@pytest.fixture(scope="module")
def tma_device() -> torch.device:
    """Return a CUDA device that supports tensor-memory acceleration."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 9:
        pytest.skip("TMA attention requires compute capability 9.0 or newer.")
    return device


def _make_cache(device: torch.device) -> BlockKVCache:
    """Build a two-chunk BF16 cache for one streaming parity path."""
    return BlockKVCache(
        k_shape=(1, 32, 2, 64),
        v_shape=(1, 32, 2, 64),
        seq_dim=1,
        chunk_size=16,
        window_size=32,
        device=device,
        dtype=torch.bfloat16,
    )


def test_tma_flash_attention_matches_sdpa(tma_device: torch.device) -> None:
    """Compare a partial-tile TMA attention launch with PyTorch SDPA."""
    generator = torch.Generator(device=tma_device).manual_seed(123)
    query = torch.randn(
        1,
        37,
        2,
        64,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        1,
        53,
        2,
        64,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        key.shape,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )

    actual = flash_attention_2_tma(query, key, value)
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    ("qk_norm_scope", "rope_interleaved", "projection_bias"),
    [
        pytest.param(QKNormScope.HEAD, False, False, id="cosmos"),
        pytest.param(QKNormScope.INNER, True, True, id="wan"),
    ],
)
def test_triton_attention_matches_reference_through_window_roll(
    tma_device: torch.device,
    qk_norm_scope: QKNormScope,
    rope_interleaved: bool,
    projection_bias: bool,
) -> None:
    """Compare full streaming attention across filling and rolling phases."""
    torch.manual_seed(7)
    reference = TorchMultiHeadAttention(
        query_dim=128,
        n_heads=2,
        head_dim=64,
        qkv_bias=projection_bias,
        output_bias=projection_bias,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
    ).to(device=tma_device, dtype=torch.bfloat16)
    triton_attention = TritonMultiHeadAttention(
        query_dim=128,
        n_heads=2,
        head_dim=64,
        qkv_bias=projection_bias,
        output_bias=projection_bias,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
    ).to(device=tma_device, dtype=torch.bfloat16)
    triton_attention.load_state_dict(reference.state_dict())
    reference.eval()
    triton_attention.eval()

    reference_cache = _make_cache(tma_device)
    triton_cache = _make_cache(tma_device)
    rope = RotaryPositionEmbedding3D(
        head_dim=64,
        len_t=1,
        len_h=1,
        len_w=16,
        interleaved=rope_interleaved,
        device=tma_device,
    )
    generator = torch.Generator(device=tma_device).manual_seed(11)

    with torch.inference_mode():
        for chunk_idx in range(3):
            x = torch.randn(
                1,
                16,
                128,
                generator=generator,
                device=tma_device,
                dtype=torch.bfloat16,
            )
            rope_freqs = rope.shift_t(chunk_idx)
            reference_cache.before_update(chunk_idx)
            triton_cache.before_update(chunk_idx)

            expected = reference(x, reference_cache, rope_freqs)
            actual = triton_attention(x, triton_cache, rope_freqs)

            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(
                triton_cache.cached_k(),
                reference_cache.cached_k(),
                atol=2e-2,
                rtol=2e-2,
            )
            torch.testing.assert_close(
                triton_cache.cached_v(),
                reference_cache.cached_v(),
                atol=0,
                rtol=0,
            )
            reference_cache.after_update(chunk_idx)
            triton_cache.after_update(chunk_idx)
