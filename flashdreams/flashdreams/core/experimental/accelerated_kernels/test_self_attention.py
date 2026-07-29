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

"""Correctness tests and benchmarks for streaming self-attention.

Run correctness tests and manual benchmarks from the workspace root with::

    uv run --no-sync pytest \
      flashdreams/flashdreams/core/experimental/accelerated_kernels/test_self_attention.py \
      -m ci_gpu

    uv run --no-sync pytest \
      flashdreams/flashdreams/core/experimental/accelerated_kernels/test_self_attention.py \
      -m manual --runxfail --benchmark-only --benchmark-time-unit=ms
"""

from typing import Any

import pytest
import torch
from torch import Tensor

from flashdreams.core.experimental.accelerated_kernels.self_attention import (
    AcceleratedSelfAttention,
    FlashAttnSelfAttention,
    ReferenceSelfAttention,
    _AcceleratedBlockKVCache,
)


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    """Return a CUDA device capable of running PyTorch FlashAttention."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlashAttention.")
    if not torch.backends.cuda.is_flash_attention_available():
        pytest.skip("PyTorch FlashAttention is unavailable in this CUDA build.")

    device = torch.device("cuda")
    major, _ = torch.cuda.get_device_capability(device)
    if major < 8:
        pytest.skip("PyTorch FlashAttention requires an Ampere-or-newer GPU.")
    return device


def _make_rope_freqs(
    sequence_length: int,
    head_dim: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    """Create full-width non-interleaved RoPE angles for one input chunk."""
    half_freqs = torch.randn(
        sequence_length,
        head_dim // 2,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return torch.cat((half_freqs, half_freqs), dim=-1).reshape(
        sequence_length, 1, 1, head_dim
    )


def _make_matching_attention_modules(
    query_dim: int,
    n_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    attention_type: type[ReferenceSelfAttention] = FlashAttnSelfAttention,
) -> tuple[ReferenceSelfAttention, ReferenceSelfAttention]:
    """Create reference and candidate modules with identical parameters."""
    torch.manual_seed(0)
    reference = ReferenceSelfAttention(query_dim, n_heads, head_dim).to(
        device=device, dtype=dtype
    )
    candidate = attention_type(query_dim, n_heads, head_dim).to(
        device=device, dtype=dtype
    )
    candidate.load_state_dict(reference.state_dict())
    reference.eval()
    candidate.eval()
    return reference, candidate


def _assert_quantized_close(
    actual: Tensor,
    expected: Tensor,
    *,
    max_abs_error: float,
    max_rmse: float,
) -> None:
    """Check explicit maximum and aggregate error bounds for FP8 results."""
    error = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    observed_max = error.max().item()
    observed_rmse = error.square().mean().sqrt().item()
    assert observed_max <= max_abs_error, (
        f"maximum absolute error {observed_max} exceeds {max_abs_error}"
    )
    assert observed_rmse <= max_rmse, f"RMSE {observed_rmse} exceeds {max_rmse}"


_CORRECTNESS_CASES = (
    pytest.param(1, 7, 32, 1, 32, 32, id="b1-l7-d32-h1-hd32-q32"),
    pytest.param(2, 17, 128, 2, 64, 128, id="b2-l17-d128-h2-hd64-q128"),
    pytest.param(3, 31, 192, 3, 64, 192, id="b3-l31-d192-h3-hd64-q192"),
    pytest.param(4, 15, 256, 2, 128, 256, id="b4-l15-d256-h2-hd128-q256"),
)


@pytest.mark.ci_gpu
@pytest.mark.parametrize(
    "batch_size,sequence_length,hidden_dim,n_heads,head_dim,query_dim",
    _CORRECTNESS_CASES,
)
def test_flash_attention_matches_reference(
    cuda_device: torch.device,
    batch_size: int,
    sequence_length: int,
    hidden_dim: int,
    n_heads: int,
    head_dim: int,
    query_dim: int,
) -> None:
    """Match reference outputs and cache writes across attention dimensions."""
    assert hidden_dim == query_dim == n_heads * head_dim
    dtype = torch.float16
    generator = torch.Generator(device=cuda_device).manual_seed(1234)
    reference, flash = _make_matching_attention_modules(
        query_dim, n_heads, head_dim, cuda_device, dtype
    )
    x = torch.randn(
        batch_size,
        sequence_length,
        hidden_dim,
        device=cuda_device,
        dtype=dtype,
        generator=generator,
    )
    rope_freqs = _make_rope_freqs(sequence_length, head_dim, cuda_device, generator)
    reference_cache = reference.initialize_cache(
        batch_size,
        sequence_length,
        sequence_length,
        0,
        cuda_device,
        dtype,
    )
    flash_cache = flash.initialize_cache(
        batch_size,
        sequence_length,
        sequence_length,
        0,
        cuda_device,
        dtype,
    )

    reference_cache.before_update(0)
    flash_cache.before_update(0)
    with torch.inference_mode():
        expected = reference(x, reference_cache, rope_freqs)
        actual = flash(x, flash_cache, rope_freqs)

    assert actual.shape == (batch_size, sequence_length, query_dim)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(
        flash_cache.cached_k(), reference_cache.cached_k(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        flash_cache.cached_v(), reference_cache.cached_v(), atol=0, rtol=0
    )
    reference_cache.after_update(0)
    flash_cache.after_update(0)


@pytest.mark.ci_gpu
def test_fused_bfloat16_forward_matches_reference(
    cuda_device: torch.device,
) -> None:
    """Match the production fused QKV path against the reference module."""
    batch_size = 1
    sequence_length = 17
    n_heads = 16
    head_dim = 128
    query_dim = n_heads * head_dim
    dtype = torch.bfloat16
    generator = torch.Generator(device=cuda_device).manual_seed(2345)
    reference, accelerated = _make_matching_attention_modules(
        query_dim,
        n_heads,
        head_dim,
        cuda_device,
        dtype,
        AcceleratedSelfAttention,
    )
    assert isinstance(accelerated, AcceleratedSelfAttention)
    x = torch.randn(
        batch_size,
        sequence_length,
        query_dim,
        device=cuda_device,
        dtype=dtype,
        generator=generator,
    )
    rope_freqs = _make_rope_freqs(
        sequence_length,
        head_dim,
        cuda_device,
        generator,
    )
    reference_cache = reference.initialize_cache(
        batch_size,
        sequence_length,
        sequence_length,
        0,
        cuda_device,
        dtype,
    )
    accelerated_cache = accelerated.initialize_cache(
        batch_size,
        sequence_length,
        sequence_length,
        0,
        cuda_device,
        dtype,
    )
    reference_cache.before_update(0)
    accelerated_cache.before_update(0)

    with torch.inference_mode():
        assert accelerated._supports_fused_forward(
            x,
            accelerated_cache,
            rope_freqs,
        )
        expected = reference(x, reference_cache, rope_freqs)
        actual = accelerated(x, accelerated_cache, rope_freqs)

    _assert_quantized_close(
        actual,
        expected,
        max_abs_error=7e-2,
        max_rmse=2e-2,
    )
    _assert_quantized_close(
        accelerated_cache.cached_k(),
        reference_cache.cached_k(),
        max_abs_error=2.5e-1,
        max_rmse=5e-2,
    )
    _assert_quantized_close(
        accelerated_cache.cached_v(),
        reference_cache.cached_v(),
        max_abs_error=1.5e-1,
        max_rmse=3e-2,
    )
    reference_cache.after_update(0)
    accelerated_cache.after_update(0)


@pytest.mark.ci_gpu
@pytest.mark.parametrize(
    "batch_size,n_heads,query_length,key_length,head_dim,dtype,atol,rtol",
    (
        pytest.param(1, 2, 7, 11, 32, torch.float16, 5e-3, 5e-3, id="fp16-small"),
        pytest.param(2, 3, 13, 97, 64, torch.float16, 5e-3, 5e-3, id="fp16-key-tiles"),
        pytest.param(
            1, 2, 137, 97, 128, torch.float16, 5e-3, 5e-3, id="fp16-query-tiles"
        ),
        pytest.param(
            1, 2, 19, 73, 48, torch.float16, 5e-3, 5e-3, id="fp16-padded-head"
        ),
        pytest.param(1, 2, 19, 97, 64, torch.bfloat16, 2e-2, 2e-2, id="bf16"),
        pytest.param(1, 2, 19, 97, 64, torch.float32, 2e-3, 2e-3, id="fp32"),
        pytest.param(1, 1, 33, 73, 256, torch.float16, 5e-3, 5e-3, id="fp16-d256"),
        pytest.param(1, 1, 33, 73, 256, torch.bfloat16, 2e-2, 2e-2, id="bf16-d256"),
        pytest.param(1, 1, 33, 73, 65, torch.float32, 2e-3, 2e-3, id="fp32-d65"),
        pytest.param(1, 1, 7, 73, 129, torch.float32, 2e-3, 2e-3, id="fp32-d129"),
        pytest.param(1, 2, 7, 0, 32, torch.float16, 0.0, 0.0, id="empty-key"),
    ),
)
def test_triton_attention_matches_reference(
    cuda_device: torch.device,
    batch_size: int,
    n_heads: int,
    query_length: int,
    key_length: int,
    head_dim: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    """Match fused Triton attention across query, key, and feature tiles."""
    generator = torch.Generator(device=cuda_device).manual_seed(3456)
    query = torch.randn(
        batch_size,
        query_length,
        n_heads,
        head_dim,
        device=cuda_device,
        dtype=dtype,
        generator=generator,
    ).transpose(1, 2)
    key = torch.randn(
        batch_size,
        key_length,
        n_heads,
        head_dim,
        device=cuda_device,
        dtype=dtype,
        generator=generator,
    ).transpose(1, 2)
    value = torch.randn(
        batch_size,
        key_length,
        n_heads,
        head_dim,
        device=cuda_device,
        dtype=dtype,
        generator=generator,
    ).transpose(1, 2)
    reference = ReferenceSelfAttention(n_heads * head_dim, n_heads, head_dim).to(
        device=cuda_device, dtype=dtype
    )
    accelerated = AcceleratedSelfAttention(n_heads * head_dim, n_heads, head_dim).to(
        device=cuda_device, dtype=dtype
    )

    with torch.inference_mode():
        expected = reference._apply_attention(query, key, value)
        actual = accelerated._apply_attention(query, key, value)

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.ci_gpu
def test_triton_float32_attention_uses_ieee_dot(
    cuda_device: torch.device,
) -> None:
    """Retain FP32 score differences that TF32 would round away."""
    head_dim = 16
    query = torch.zeros((1, 1, 1, head_dim), device=cuda_device, dtype=torch.float32)
    key = torch.zeros((1, 1, 2, head_dim), device=cuda_device, dtype=torch.float32)
    value = torch.zeros_like(key)
    query[..., 0] = 4096.0
    key[:, :, 0, 0] = 1.0001
    key[:, :, 1, 0] = 1.0008
    value[:, :, 0, 0] = 1.0
    value[:, :, 1, 0] = -1.0
    reference = ReferenceSelfAttention(head_dim, 1, head_dim).to(cuda_device)
    accelerated = AcceleratedSelfAttention(head_dim, 1, head_dim).to(cuda_device)

    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.inference_mode():
            expected = reference._apply_attention(query, key, value)
            actual = accelerated._apply_attention(query, key, value)
    finally:
        torch.set_float32_matmul_precision(previous_precision)

    assert expected[0, 0, 0, 0].abs() > 0.3
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.ci_gpu
@pytest.mark.parametrize(
    "head_dim",
    (
        pytest.param(16, id="pointer"),
        pytest.param(128, id="tma"),
    ),
)
def test_triton_attention_compiles_with_inductor(
    cuda_device: torch.device,
    head_dim: int,
) -> None:
    """Compile the user Triton kernel through Inductor."""
    dtype = torch.float16
    generator = torch.Generator(device=cuda_device).manual_seed(7890)
    query = torch.randn(
        (1, 1, 1, head_dim), device=cuda_device, dtype=dtype, generator=generator
    )
    key = torch.randn(
        (1, 1, 2, head_dim), device=cuda_device, dtype=dtype, generator=generator
    )
    value = torch.randn(
        (1, 1, 2, head_dim), device=cuda_device, dtype=dtype, generator=generator
    )
    reference = ReferenceSelfAttention(head_dim, 1, head_dim).to(
        device=cuda_device, dtype=dtype
    )
    accelerated = AcceleratedSelfAttention(head_dim, 1, head_dim).to(
        device=cuda_device, dtype=dtype
    )
    compiled_attention = torch.compile(
        accelerated._apply_attention,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
    )

    with torch.inference_mode():
        expected = reference._apply_attention(query, key, value)
        actual = compiled_attention(query, key, value)
    torch.cuda.synchronize(cuda_device)

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.ci_gpu
@pytest.mark.parametrize(
    "attention_type,dtype,n_heads,head_dim,output_tolerance,cache_tolerance",
    (
        pytest.param(
            FlashAttnSelfAttention,
            torch.float16,
            2,
            32,
            5e-3,
            0.0,
            id="pytorch-flash",
        ),
        pytest.param(
            AcceleratedSelfAttention,
            torch.float16,
            2,
            32,
            5e-3,
            0.0,
            id="triton-flash",
        ),
        pytest.param(
            AcceleratedSelfAttention,
            torch.bfloat16,
            16,
            128,
            4e-2,
            2e-2,
            id="triton-fused-bf16",
        ),
    ),
)
def test_attention_updates_kv_cache(
    cuda_device: torch.device,
    attention_type: type[ReferenceSelfAttention],
    dtype: torch.dtype,
    n_heads: int,
    head_dim: int,
    output_tolerance: float,
    cache_tolerance: float,
) -> None:
    """Preserve sink tokens and roll cached K/V at the window boundary."""
    batch_size = 2
    chunk_size = 4
    window_size = 8
    sink_size = 4
    query_dim = n_heads * head_dim
    generator = torch.Generator(device=cuda_device).manual_seed(5678)
    reference, candidate = _make_matching_attention_modules(
        query_dim, n_heads, head_dim, cuda_device, dtype, attention_type
    )
    reference_cache = reference.initialize_cache(
        batch_size, chunk_size, window_size, sink_size, cuda_device, dtype
    )
    candidate_cache = candidate.initialize_cache(
        batch_size, chunk_size, window_size, sink_size, cuda_device, dtype
    )
    projected_keys: list[Tensor] = []
    projected_values: list[Tensor] = []
    sink_chunks = sink_size // chunk_size
    window_chunks = window_size // chunk_size

    with torch.inference_mode():
        for chunk_index in range(4):
            x = torch.randn(
                batch_size,
                chunk_size,
                query_dim,
                device=cuda_device,
                dtype=dtype,
                generator=generator,
            )
            rope_freqs = _make_rope_freqs(chunk_size, head_dim, cuda_device, generator)
            head_shape = (batch_size, chunk_size, n_heads, head_dim)
            projected_key = reference.k_norm(reference.k_proj(x).reshape(head_shape))
            projected_keys.append(reference._apply_rope(projected_key, rope_freqs))
            projected_values.append(reference.v_proj(x).reshape(head_shape))

            reference_cache.before_update(chunk_index)
            candidate_cache.before_update(chunk_index)
            expected_output = reference(x, reference_cache, rope_freqs)
            if dtype == torch.bfloat16:
                assert isinstance(candidate, AcceleratedSelfAttention)
                assert candidate._supports_fused_forward(x, candidate_cache, rope_freqs)
            actual_output = candidate(x, candidate_cache, rope_freqs)

            local_start = max(sink_chunks, chunk_index - window_chunks + 1)
            visible_chunks = list(range(min(chunk_index + 1, sink_chunks)))
            visible_chunks.extend(range(local_start, chunk_index + 1))
            expected_keys = torch.cat(
                [projected_keys[index] for index in visible_chunks], dim=1
            )
            expected_values = torch.cat(
                [projected_values[index] for index in visible_chunks], dim=1
            )

            uses_fp8 = (
                isinstance(candidate_cache, _AcceleratedBlockKVCache)
                and candidate_cache._k_fp8 is not None
            )
            if uses_fp8:
                _assert_quantized_close(
                    actual_output,
                    expected_output,
                    max_abs_error=7e-2,
                    max_rmse=2e-2,
                )
            else:
                torch.testing.assert_close(
                    actual_output,
                    expected_output,
                    atol=output_tolerance,
                    rtol=output_tolerance,
                )

            torch.testing.assert_close(
                reference_cache.cached_k(), expected_keys, atol=0, rtol=0
            )
            torch.testing.assert_close(
                reference_cache.cached_v(), expected_values, atol=0, rtol=0
            )
            if uses_fp8:
                _assert_quantized_close(
                    candidate_cache.cached_k(),
                    expected_keys,
                    max_abs_error=2.5e-1,
                    max_rmse=5e-2,
                )
                _assert_quantized_close(
                    candidate_cache.cached_v(),
                    expected_values,
                    max_abs_error=1.5e-1,
                    max_rmse=3e-2,
                )
            else:
                torch.testing.assert_close(
                    candidate_cache.cached_k(),
                    expected_keys,
                    atol=cache_tolerance,
                    rtol=cache_tolerance,
                )
                torch.testing.assert_close(
                    candidate_cache.cached_v(),
                    expected_values,
                    atol=cache_tolerance,
                    rtol=cache_tolerance,
                )

            if uses_fp8:
                cached_k_fp8 = candidate_cache.cached_k_fp8()
                cached_v_fp8 = candidate_cache.cached_v_fp8()
                assert cached_k_fp8.dtype == torch.float8_e4m3fn
                assert cached_v_fp8.dtype == torch.float8_e4m3fn
                torch.testing.assert_close(
                    cached_k_fp8,
                    candidate_cache.cached_k().to(torch.float8_e4m3fn),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    cached_v_fp8,
                    candidate_cache.cached_v().to(torch.float8_e4m3fn),
                    atol=0,
                    rtol=0,
                )

            reference_cache.after_update(chunk_index)
            candidate_cache.after_update(chunk_index)
            expected_size = min((chunk_index + 1) * chunk_size, sink_size + window_size)
            assert reference_cache.size == expected_size
            assert candidate_cache.size == expected_size


_BENCHMARK_CASES = (
    pytest.param(1, 40 * 73 * 2, 16, 128, 1, id="b1-l5840-k5840-h16-hd128"),
    pytest.param(1, 40 * 73 * 2, 16, 128, 3, id="b1-l5840-k17520-h16-hd128"),
    pytest.param(1, 80 * 44 * 2, 16, 128, 1, id="b1-l7040-k7040-h16-hd128"),
    pytest.param(1, 80 * 44 * 2, 16, 128, 3, id="b1-l7040-k21120-h16-hd128"),
)

_BENCHMARK_IMPLEMENTATIONS = (
    pytest.param(ReferenceSelfAttention, id="reference"),
    pytest.param(FlashAttnSelfAttention, id="flash"),
    pytest.param(AcceleratedSelfAttention, id="accelerated"),
)

_BENCHMARK_DTYPES = (
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
)


@pytest.mark.manual
@pytest.mark.parametrize("attention_type", _BENCHMARK_IMPLEMENTATIONS)
@pytest.mark.parametrize("dtype", _BENCHMARK_DTYPES)
@pytest.mark.parametrize(
    "batch_size,sequence_length,n_heads,head_dim,cache_chunks", _BENCHMARK_CASES
)
def test_self_attention_benchmark(
    benchmark: Any,
    cuda_device: torch.device,
    attention_type: type[ReferenceSelfAttention],
    dtype: torch.dtype,
    batch_size: int,
    sequence_length: int,
    n_heads: int,
    head_dim: int,
    cache_chunks: int,
) -> None:
    """Benchmark self-attention against a prefilled rolling KV cache."""
    if dtype == torch.float32 and attention_type is FlashAttnSelfAttention:
        pytest.skip("PyTorch FlashAttention does not support float32 inputs.")

    query_dim = n_heads * head_dim
    window_size = cache_chunks * sequence_length
    dtype_name = str(dtype).removeprefix("torch.")
    generator = torch.Generator(device=cuda_device).manual_seed(9012)
    torch.manual_seed(0)
    attention = attention_type(query_dim, n_heads, head_dim).to(
        device=cuda_device, dtype=dtype
    )
    attention.eval()
    x = torch.randn(
        batch_size,
        sequence_length,
        query_dim,
        device=cuda_device,
        dtype=dtype,
        generator=generator,
    )
    rope_freqs = _make_rope_freqs(sequence_length, head_dim, cuda_device, generator)
    kv_cache = attention.initialize_cache(
        batch_size,
        sequence_length,
        window_size,
        0,
        cuda_device,
        dtype,
    )

    with torch.inference_mode():
        for chunk_index in range(cache_chunks):
            kv_cache.before_update(chunk_index)
            attention(x, kv_cache, rope_freqs)
            kv_cache.after_update(chunk_index)
    torch.cuda.synchronize(cuda_device)
    assert kv_cache.size == window_size

    current_chunk_index = cache_chunks - 1

    @torch.inference_mode()
    def run_attention() -> Tensor:
        kv_cache.before_update(current_chunk_index)
        output = attention(x, kv_cache, rope_freqs)
        kv_cache.after_update(current_chunk_index)
        torch.cuda.synchronize(cuda_device)
        return output

    benchmark.group = (
        f"b{batch_size}-l{sequence_length}-k{window_size}-h{n_heads}-hd{head_dim}"
        f"-{dtype_name}"
    )
    benchmark.extra_info.update(
        {
            "implementation": attention_type.__name__,
            "device": torch.cuda.get_device_name(cuda_device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": str(dtype),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "cached_sequence_length": window_size,
            "cache_chunks": cache_chunks,
            "query_dim": query_dim,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "tokens_per_forward": batch_size * sequence_length,
        }
    )
    benchmark.pedantic(
        run_attention,
        rounds=50,
        warmup_rounds=10,
        iterations=1,
    )
