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

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

import flashdreams.accelerated.multi_head_attention_triton as triton_mha
from flashdreams.accelerated.multi_head_attention import QKNormScope
from flashdreams.accelerated.multi_head_attention_torch import TorchMultiHeadAttention
from flashdreams.accelerated.multi_head_attention_triton import (
    SDPABackend,
    TritonMultiHeadAttention,
)
from flashdreams.accelerated.triton import (
    flash_attention_2_tma,
    fp8_quantization,
    fused_rms_rope_kv_cache_update,
)
from flashdreams.core.attention import (
    BlockKVCache,
    RotaryPositionEmbedding3D,
)

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


@pytest.mark.parametrize(
    "sdpa_backend", tuple(SDPABackend), ids=lambda backend: backend.value
)
def test_attention_dispatches_configured_backend(
    monkeypatch: pytest.MonkeyPatch,
    tma_device: torch.device,
    sdpa_backend: SDPABackend,
) -> None:
    """Call exactly the selected cuDNN or Triton SDPA implementation."""
    attention = TritonMultiHeadAttention(
        128,
        n_heads=2,
        head_dim=64,
        sdpa_backend=sdpa_backend,
    )
    query = torch.empty((1, 2, 2, 64), device=tma_device, dtype=torch.bfloat16)
    calls: list[SDPABackend] = []

    def record_cudnn(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        del key, value
        calls.append(SDPABackend.CUDNN)
        return query

    def record_triton(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        del key, value
        calls.append(SDPABackend.TRITON)
        return query

    cudnn_kernel = MagicMock()
    monkeypatch.setattr(triton_mha.F, "scaled_dot_product_attention", record_cudnn)
    monkeypatch.setattr(
        triton_mha.torch.nn.attention,
        "sdpa_kernel",
        cudnn_kernel,
    )
    monkeypatch.setattr(triton_mha, "flash_attention_2_tma", record_triton)

    assert torch.equal(attention._attention(query, query, query), query)
    assert calls == [sdpa_backend]
    if sdpa_backend is SDPABackend.CUDNN:
        cudnn_kernel.assert_called_once_with(
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION
        )
    else:
        cudnn_kernel.assert_not_called()


def test_fused_fp8_row_quantization_matches_torch(
    tma_device: torch.device,
) -> None:
    """Match fused row quantization with the original PyTorch operations."""
    generator = torch.Generator(device=tma_device).manual_seed(5)
    x = torch.randn(
        (1536, 7),
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    ).T
    x[0].zero_()
    actual, actual_scales = fp8_quantization._quantize_fp8_rows(x)

    x_float = x.to(torch.float32)
    expected_scales = (
        (x_float.abs().amax(dim=1, keepdim=True) / fp8_quantization._FP8_MAX)
        .clamp_min(1e-12)
        .contiguous()
    )
    expected = (
        (x_float / expected_scales)
        .clamp(-fp8_quantization._FP8_MAX, fp8_quantization._FP8_MAX)
        .to(torch.float8_e4m3fn)
    )

    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
    assert torch.equal(actual_scales, expected_scales)
    assert torch.count_nonzero(actual[0]) == 0
    assert actual_scales[0, 0] == 1e-12

    nan_input = x.clone()
    nan_input[0, 0] = float("nan")
    nan_output, _ = fp8_quantization._quantize_fp8_rows(nan_input)
    assert torch.isnan(nan_output[0, 0].to(torch.float32))

    empty_output, empty_scales = fp8_quantization._quantize_fp8_rows(x[:0])
    assert empty_output.shape == x[:0].shape
    assert empty_scales.shape == (0, 1)


@pytest.mark.parametrize(
    ("rope_interleaved", "head_major_cache", "apply_rope"),
    [
        pytest.param(False, True, True, id="half-split-head-major"),
        pytest.param(True, True, True, id="interleaved-head-major"),
        pytest.param(False, False, True, id="half-split-token-major"),
        pytest.param(False, True, False, id="no-rope-head-major"),
    ],
)
def test_head_preprocessing_layouts_match_torch(
    tma_device: torch.device,
    rope_interleaved: bool,
    head_major_cache: bool,
    apply_rope: bool,
) -> None:
    """Cover tiled and fallback head preprocessing across dense cache layouts."""
    batch_size, sequence_length, num_heads, head_dim = 2, 3, 3, 16
    cache_size = 6
    generator = torch.Generator(device=tma_device).manual_seed(23)
    query = torch.randn(
        batch_size,
        sequence_length,
        num_heads,
        head_dim,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        query.shape,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    value = torch.randn(
        query.shape,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    sentinel = -123.0
    cache_shape = (
        (batch_size, num_heads, cache_size, head_dim)
        if head_major_cache
        else (batch_size, cache_size, num_heads, head_dim)
    )
    key_cache = torch.full(
        cache_shape,
        sentinel,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    if head_major_cache:
        key_cache = key_cache.transpose(1, 2)
    value_cache = torch.full_like(key_cache, sentinel)
    rope_freqs = (
        torch.arange(
            sequence_length * head_dim,
            device=tma_device,
            dtype=torch.float32,
        ).reshape(sequence_length, 1, 1, head_dim)
        / 37
    )

    actual_query = fused_rms_rope_kv_cache_update(
        query,
        key,
        value,
        key_cache,
        value_cache,
        query_weight=None,
        key_weight=None,
        norm_eps=0.0,
        norm_scope=QKNormScope.HEAD,
        rope_freqs=rope_freqs if apply_rope else None,
        rope_interleaved=rope_interleaved,
        cache_read_start=1,
        cache_write_start=2,
        cache_write_length=2,
    )

    reference = TorchMultiHeadAttention(
        query_dim=num_heads * head_dim,
        n_heads=num_heads,
        head_dim=head_dim,
        qk_norm=False,
        rope_interleaved=rope_interleaved,
    ).to(device=tma_device, dtype=torch.bfloat16)
    if apply_rope:
        expected_query = reference._apply_rope(query, rope_freqs)
        expected_key = reference._apply_rope(key, rope_freqs)
    else:
        expected_query = query
        expected_key = key
    torch.testing.assert_close(actual_query, expected_query, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        key_cache[:, 2:4], expected_key[:, 1:3], atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(value_cache[:, 2:4], value[:, 1:3])
    assert torch.all(key_cache[:, :2] == sentinel)
    assert torch.all(key_cache[:, 4:] == sentinel)
    assert torch.all(value_cache[:, :2] == sentinel)
    assert torch.all(value_cache[:, 4:] == sentinel)


def _make_cache(device: torch.device, n_heads: int, head_dim: int) -> BlockKVCache:
    """Build a two-chunk BF16 cache with an immutable sink prefix."""
    return BlockKVCache(
        k_shape=(1, 36, n_heads, head_dim),
        v_shape=(1, 36, n_heads, head_dim),
        seq_dim=1,
        chunk_size=16,
        window_size=32,
        sink_size=4,
        device=device,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    ("query_length", "key_length", "head_dim"),
    [
        pytest.param(37, 53, 64, id="partial-tiles"),
        pytest.param(129, 128, 128, id="production-head-divisible-key"),
    ],
)
def test_tma_flash_attention_matches_sdpa(
    tma_device: torch.device,
    query_length: int,
    key_length: int,
    head_dim: int,
) -> None:
    """Compare partial and production-head TMA launches with PyTorch SDPA."""
    generator = torch.Generator(device=tma_device).manual_seed(123)
    query = torch.randn(
        1,
        query_length,
        2,
        head_dim,
        generator=generator,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        1,
        key_length,
        2,
        head_dim,
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
    "sdpa_backend", tuple(SDPABackend), ids=lambda backend: backend.value
)
@pytest.mark.parametrize(
    (
        "qk_norm_scope",
        "rope_interleaved",
        "projection_bias",
        "n_heads",
        "head_dim",
    ),
    [
        pytest.param(QKNormScope.HEAD, False, False, 16, 128, id="cosmos"),
        pytest.param(QKNormScope.INNER, True, True, 12, 128, id="wan"),
    ],
)
def test_triton_attention_matches_reference_through_window_roll(
    tma_device: torch.device,
    qk_norm_scope: QKNormScope,
    rope_interleaved: bool,
    projection_bias: bool,
    n_heads: int,
    head_dim: int,
    sdpa_backend: SDPABackend,
) -> None:
    """Compare streaming attention across fill, roll, and overwrite phases."""
    torch.manual_seed(7)
    inner_dim = n_heads * head_dim
    reference = TorchMultiHeadAttention(
        query_dim=inner_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        qkv_bias=projection_bias,
        output_bias=projection_bias,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
    ).to(device=tma_device, dtype=torch.bfloat16)
    triton_attention = TritonMultiHeadAttention(
        query_dim=inner_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        qkv_bias=projection_bias,
        output_bias=projection_bias,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
        sdpa_backend=sdpa_backend,
    ).to(device=tma_device, dtype=torch.bfloat16)
    triton_attention.load_state_dict(reference.state_dict())
    reference.eval()
    triton_attention.eval()

    reference_cache = _make_cache(tma_device, n_heads, head_dim)
    triton_cache = triton_attention.initialize_cache(
        batch_size=1,
        chunk_size=16,
        window_size=32,
        sink_size=4,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    rope = RotaryPositionEmbedding3D(
        head_dim=head_dim,
        len_t=1,
        len_h=1,
        len_w=16,
        interleaved=rope_interleaved,
        device=tma_device,
    )
    generator = torch.Generator(device=tma_device).manual_seed(11)

    with torch.inference_mode():
        for chunk_idx in (0, 1, 2, 2):
            x = torch.randn(
                1,
                16,
                inner_dim,
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


@pytest.mark.parametrize(
    "sdpa_backend", tuple(SDPABackend), ids=lambda backend: backend.value
)
@pytest.mark.parametrize(
    (
        "qk_norm_scope",
        "rope_interleaved",
        "projection_bias",
        "n_heads",
        "head_dim",
    ),
    [
        pytest.param(QKNormScope.HEAD, False, False, 16, 128, id="cosmos"),
        pytest.param(QKNormScope.INNER, True, True, 12, 128, id="wan"),
    ],
)
def test_fp8_triton_attention_matches_bf16_reference_through_window_roll(
    tma_device: torch.device,
    qk_norm_scope: QKNormScope,
    rope_interleaved: bool,
    projection_bias: bool,
    n_heads: int,
    head_dim: int,
    sdpa_backend: SDPABackend,
) -> None:
    """Bound FP8 error across cache fill, roll, and overwrite phases."""
    torch.manual_seed(17)
    inner_dim = n_heads * head_dim
    reference = TorchMultiHeadAttention(
        query_dim=inner_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        qkv_bias=projection_bias,
        output_bias=projection_bias,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
    ).to(device=tma_device, dtype=torch.bfloat16)
    triton_attention = TritonMultiHeadAttention(
        query_dim=inner_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        qkv_bias=projection_bias,
        output_bias=projection_bias,
        qk_norm_scope=qk_norm_scope,
        rope_interleaved=rope_interleaved,
        use_fp8=True,
        sdpa_backend=sdpa_backend,
    ).to(device=tma_device, dtype=torch.bfloat16)
    triton_attention.load_state_dict(reference.state_dict())
    reference.eval()
    triton_attention.eval()

    reference_cache = _make_cache(tma_device, n_heads, head_dim)
    triton_cache = triton_attention.initialize_cache(
        batch_size=1,
        chunk_size=16,
        window_size=32,
        sink_size=4,
        device=tma_device,
        dtype=torch.bfloat16,
    )
    assert type(triton_cache) is BlockKVCache
    expected_cache_dtype = (
        torch.float8_e4m3fn if sdpa_backend is SDPABackend.TRITON else torch.bfloat16
    )
    assert triton_cache._k.dtype is expected_cache_dtype
    assert triton_cache._v.dtype is expected_cache_dtype
    assert triton_cache._k.shape == (1, 36, n_heads, head_dim)
    if qk_norm_scope is QKNormScope.HEAD:
        assert triton_cache._k.stride() == (
            36 * n_heads * head_dim,
            head_dim,
            36 * head_dim,
            1,
        )
        assert triton_cache._v.stride() == triton_cache._k.stride()
    else:
        assert triton_cache._k.is_contiguous()
        assert triton_cache._v.is_contiguous()
    rope = RotaryPositionEmbedding3D(
        head_dim=head_dim,
        len_t=1,
        len_h=1,
        len_w=16,
        interleaved=rope_interleaved,
        device=tma_device,
    )
    generator = torch.Generator(device=tma_device).manual_seed(19)

    with torch.inference_mode():
        for chunk_idx in (0, 1, 2, 2):
            x = torch.randn(
                1,
                16,
                inner_dim,
                generator=generator,
                device=tma_device,
                dtype=torch.bfloat16,
            )
            rope_freqs = rope.shift_t(chunk_idx)
            reference_cache.before_update(chunk_idx)
            triton_cache.before_update(chunk_idx)

            expected = reference(x, reference_cache, rope_freqs)
            actual = triton_attention(x, triton_cache, rope_freqs)

            torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
            torch.testing.assert_close(
                triton_cache.cached_k().to(torch.bfloat16),
                reference_cache.cached_k(),
                atol=1.5e-1,
                rtol=1.5e-1,
            )
            torch.testing.assert_close(
                triton_cache.cached_v().to(torch.bfloat16),
                reference_cache.cached_v(),
                atol=1.5e-1,
                rtol=1.5e-1,
            )
            reference_cache.after_update(chunk_idx)
            triton_cache.after_update(chunk_idx)
