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
    FlashAttnSelfAttention,
    ReferenceSelfAttention,
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
) -> tuple[ReferenceSelfAttention, FlashAttnSelfAttention]:
    """Create reference and FlashAttention modules with identical parameters."""
    torch.manual_seed(0)
    reference = ReferenceSelfAttention(query_dim, n_heads, head_dim).to(
        device=device, dtype=dtype
    )
    flash = FlashAttnSelfAttention(query_dim, n_heads, head_dim).to(
        device=device, dtype=dtype
    )
    flash.load_state_dict(reference.state_dict())
    reference.eval()
    flash.eval()
    return reference, flash


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
    rope_freqs = _make_rope_freqs(
        sequence_length, head_dim, cuda_device, generator
    )
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
def test_flash_attention_updates_kv_cache(cuda_device: torch.device) -> None:
    """Preserve sink tokens and roll cached K/V at the window boundary."""
    batch_size = 2
    chunk_size = 4
    window_size = 8
    sink_size = 4
    n_heads = 2
    head_dim = 32
    query_dim = n_heads * head_dim
    dtype = torch.float16
    generator = torch.Generator(device=cuda_device).manual_seed(5678)
    reference, flash = _make_matching_attention_modules(
        query_dim, n_heads, head_dim, cuda_device, dtype
    )
    reference_cache = reference.initialize_cache(
        batch_size, chunk_size, window_size, sink_size, cuda_device, dtype
    )
    flash_cache = flash.initialize_cache(
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
            rope_freqs = _make_rope_freqs(
                chunk_size, head_dim, cuda_device, generator
            )
            head_shape = (batch_size, chunk_size, n_heads, head_dim)
            projected_key = reference.k_norm(
                reference.k_proj(x).reshape(head_shape)
            )
            projected_keys.append(reference._apply_rope(projected_key, rope_freqs))
            projected_values.append(reference.v_proj(x).reshape(head_shape))

            reference_cache.before_update(chunk_index)
            flash_cache.before_update(chunk_index)
            expected_output = reference(x, reference_cache, rope_freqs)
            actual_output = flash(x, flash_cache, rope_freqs)

            local_start = max(sink_chunks, chunk_index - window_chunks + 1)
            visible_chunks = list(range(min(chunk_index + 1, sink_chunks)))
            visible_chunks.extend(range(local_start, chunk_index + 1))
            expected_keys = torch.cat(
                [projected_keys[index] for index in visible_chunks], dim=1
            )
            expected_values = torch.cat(
                [projected_values[index] for index in visible_chunks], dim=1
            )

            torch.testing.assert_close(
                actual_output, expected_output, atol=5e-3, rtol=5e-3
            )
            for cache in (reference_cache, flash_cache):
                torch.testing.assert_close(
                    cache.cached_k(), expected_keys, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    cache.cached_v(), expected_values, atol=0, rtol=0
                )

            reference_cache.after_update(chunk_index)
            flash_cache.after_update(chunk_index)
            expected_size = min(
                (chunk_index + 1) * chunk_size, sink_size + window_size
            )
            assert reference_cache.size == expected_size
            assert flash_cache.size == expected_size


_BENCHMARK_CASES = (
    pytest.param(1, 80 * 44 * 2, 16, 128, 1, id="b1-l7040-k7040-h16-hd128"),
    pytest.param(1, 80 * 44 * 2, 16, 128, 3, id="b1-l7040-k21120-h16-hd128"),
)

_BENCHMARK_IMPLEMENTATIONS = (
    pytest.param(ReferenceSelfAttention, id="reference"),
    pytest.param(FlashAttnSelfAttention, id="flash"),
)


@pytest.mark.manual
@pytest.mark.parametrize("attention_type", _BENCHMARK_IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "batch_size,sequence_length,n_heads,head_dim,cache_chunks", _BENCHMARK_CASES
)
def test_self_attention_benchmark(
    benchmark: Any,
    cuda_device: torch.device,
    attention_type: type[ReferenceSelfAttention],
    batch_size: int,
    sequence_length: int,
    n_heads: int,
    head_dim: int,
    cache_chunks: int,
) -> None:
    """Benchmark self-attention against a prefilled rolling KV cache."""
    query_dim = n_heads * head_dim
    window_size = cache_chunks * sequence_length
    dtype = torch.float16
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
    rope_freqs = _make_rope_freqs(
        sequence_length, head_dim, cuda_device, generator
    )
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
    )
    benchmark.extra_info.update(
        {
            "implementation": attention_type.__name__,
            "device": torch.cuda.get_device_name(cuda_device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": str(dtype),
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
