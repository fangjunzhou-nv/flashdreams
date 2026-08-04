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

"""SDPA correctness tests and benchmarks for accelerated self-attention.

Run CUDA correctness and the manual benchmark from the workspace root with::

    uv run --package flashdreams pytest \
      flashdreams/flashdreams/core/experimental/accelerated_kernels/test_self_attention.py \
      -m ci_gpu -q

    FLASHDREAMS_ATTENTION_BENCHMARK_ROUNDS=50 \
    FLASHDREAMS_ATTENTION_BENCHMARK_WARMUP_ROUNDS=10 \
    uv run --package flashdreams pytest \
      flashdreams/flashdreams/core/experimental/accelerated_kernels/test_self_attention.py \
      -m manual --runxfail --benchmark-only --benchmark-time-unit=ms \
      --benchmark-save-data --benchmark-json=/tmp/self_attention_benchmark.json
"""

import itertools
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import triton
from torch import Tensor

from flashdreams.core.attention.kvcache import BlockKVCache
from flashdreams.core.experimental.accelerated_kernels.self_attention import (
    _POINTER_ATTENTION_CONFIGS,
    _TMA_ATTENTION_CONFIGS,
    AcceleratedSelfAttention,
    _prune_attention_configs,
    _prune_tma_attention_configs,
)


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    """Return a CUDA device supported by the Triton attention kernels."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for accelerated self-attention.")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 8:
        pytest.skip("Accelerated self-attention requires Ampere or newer.")
    return device


def _make_rope_freqs(
    sequence_length: int,
    head_dim: int,
    device: torch.device,
    generator: torch.Generator,
    *,
    interleaved: bool,
) -> Tensor:
    """Create full-width RoPE angles for one input chunk."""
    half_freqs = torch.randn(
        sequence_length,
        head_dim // 2,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    full_freqs = (
        torch.repeat_interleave(half_freqs, 2, dim=-1)
        if interleaved
        else torch.cat((half_freqs, half_freqs), dim=-1)
    )
    return full_freqs.reshape(sequence_length, 1, 1, head_dim)


def _initialize_reference_cache(
    attention: AcceleratedSelfAttention,
    batch_size: int,
    chunk_size: int,
    window_size: int,
    sink_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BlockKVCache:
    """Initialize a native-precision cache for the SDPA oracle."""
    total_size = sink_size + window_size
    return BlockKVCache(
        k_shape=(batch_size, total_size, attention.n_heads, attention.head_dim),
        v_shape=(batch_size, total_size, attention.n_heads, attention.head_dim),
        seq_dim=-3,
        chunk_size=chunk_size,
        window_size=window_size,
        sink_size=sink_size,
        device=device,
        dtype=dtype,
    )


def _make_native_reference(
    attention: AcceleratedSelfAttention,
    device: torch.device,
    dtype: torch.dtype,
) -> AcceleratedSelfAttention:
    """Build a native SDPA oracle from native or packed checkpoint state."""
    if attention.use_fp8:
        qkv_bias = attention.qkv_proj.bias is not None
        output_bias = attention.output_proj.bias is not None
    else:
        qkv_bias = attention.q_proj.bias is not None
        output_bias = attention.output_proj.bias is not None
    qk_norm = isinstance(attention.q_norm, torch.nn.RMSNorm)
    qk_norm_eps = float(attention.q_norm.eps or 1e-6) if qk_norm else 1e-6
    reference = AcceleratedSelfAttention(
        attention.query_dim,
        attention.n_heads,
        attention.head_dim,
        qkv_bias=qkv_bias,
        output_bias=output_bias,
        qk_norm=qk_norm,
        qk_norm_eps=qk_norm_eps,
        rope_interleaved=attention.rope_interleaved,
        use_tma=False,
        fuse_qkv=False,
        fuse_rope_kv_cache=False,
        use_cudnn=False,
        use_fp8=False,
    ).to(device=device, dtype=dtype)
    reference.load_state_dict(attention.state_dict(), strict=True)
    return reference.eval()


def _sdpa_reference_step(
    attention: AcceleratedSelfAttention,
    x: Tensor,
    kv_cache: BlockKVCache,
    rope_freqs: Tensor | None,
) -> Tensor:
    """Run one native projection/cache step with PyTorch SDPA."""
    batch_shape = x.shape[:-2]
    batch_size = math.prod(batch_shape)
    sequence_length = x.shape[-2]
    inner_dim = attention.n_heads * attention.head_dim
    head_shape = (
        batch_size,
        sequence_length,
        attention.n_heads,
        attention.head_dim,
    )
    query = attention.q_norm(attention.q_proj(x).reshape(head_shape))
    key = attention.k_norm(attention.k_proj(x).reshape(head_shape))
    value = attention.v_proj(x).reshape(head_shape)
    query = attention._apply_rope(query, rope_freqs)
    key = attention._apply_rope(key, rope_freqs)
    kv_cache.update(key, value)
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        kv_cache.cached_k().transpose(1, 2),
        kv_cache.cached_v().transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    )
    output = output.transpose(1, 2).reshape(batch_shape + (sequence_length, inner_dim))
    return attention.output_proj(output)


def _assert_error_bounds(
    actual: Tensor,
    expected: Tensor,
    *,
    max_abs_error: float,
    max_rmse: float,
) -> None:
    """Check explicit maximum and aggregate numerical error bounds."""
    error = actual.to(torch.float32) - expected.to(torch.float32)
    observed_max = error.abs().max().item()
    observed_rmse = error.square().mean().sqrt().item()
    assert observed_max <= max_abs_error, (
        f"maximum absolute error {observed_max} exceeds {max_abs_error}"
    )
    assert observed_rmse <= max_rmse, f"RMSE {observed_rmse} exceeds {max_rmse}"


def _native_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """Return absolute and relative tolerances for native-precision paths."""
    if dtype == torch.float32:
        return 2e-3, 2e-3
    if dtype == torch.float16:
        return 5e-3, 5e-3
    return 2e-2, 2e-2


def _assert_module_matches_sdpa(
    actual: Tensor,
    expected: Tensor,
    candidate_cache: BlockKVCache,
    reference_cache: BlockKVCache,
    *,
    fp8_effective: bool,
) -> None:
    """Compare module outputs and visible native cache contents."""
    if fp8_effective:
        _assert_error_bounds(
            actual,
            expected,
            max_abs_error=1.2e-1,
            max_rmse=2.5e-2,
        )
        _assert_error_bounds(
            candidate_cache.cached_k(),
            reference_cache.cached_k(),
            max_abs_error=2.5e-1,
            max_rmse=6e-2,
        )
        _assert_error_bounds(
            candidate_cache.cached_v(),
            reference_cache.cached_v(),
            max_abs_error=2e-1,
            max_rmse=5e-2,
        )
        return

    atol, rtol = _native_tolerance(actual.dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        candidate_cache.cached_k(),
        reference_cache.cached_k(),
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        candidate_cache.cached_v(),
        reference_cache.cached_v(),
        atol=atol,
        rtol=rtol,
    )


@dataclass(frozen=True)
class _CorrectnessCase:
    """One pairwise architecture and streaming-cache correctness case."""

    batch_shape: tuple[int, ...]
    sequence_length: int
    query_dim: int
    n_heads: int
    head_dim: int
    dtype: torch.dtype
    cache_chunks: int
    sink_chunks: int = 0
    qkv_bias: bool = False
    output_bias: bool = False
    qk_norm: bool = True
    use_rope: bool = True
    rope_interleaved: bool = False
    use_fp8: bool = False


_CORRECTNESS_CASES = (
    pytest.param(
        _CorrectnessCase((1,), 1, 80, 1, 32, torch.float32, 1, qk_norm=False),
        id="b1-l1-q80-h1-d32-fp32-no-norm",
    ),
    pytest.param(
        _CorrectnessCase((1,), 9, 37, 3, 7, torch.float16, 2, use_rope=False),
        id="b1-l9-q37-h3-d7-fp16-odd-no-rope",
    ),
    pytest.param(
        _CorrectnessCase((1,), 11, 48, 2, 8, torch.bfloat16, 2, use_fp8=True),
        id="b1-l11-q48-h2-d8-bf16-fp8-pointer",
    ),
    pytest.param(
        _CorrectnessCase((2,), 7, 112, 3, 32, torch.float16, 2, use_fp8=True),
        id="b2-l7-q112-h3-d32-fp16",
    ),
    pytest.param(
        _CorrectnessCase(
            (2, 2),
            17,
            320,
            4,
            64,
            torch.bfloat16,
            2,
            sink_chunks=1,
            qkv_bias=True,
            output_bias=True,
            qk_norm=False,
            rope_interleaved=True,
            use_fp8=True,
        ),
        id="leading-batch-l17-q320-h4-d64-bf16-interleaved",
    ),
    pytest.param(
        _CorrectnessCase((1,), 31, 224, 2, 96, torch.float16, 3, use_fp8=True),
        id="b1-l31-q224-h2-d96-fp16-pointer",
    ),
    pytest.param(
        _CorrectnessCase((1,), 17, 400, 2, 192, torch.bfloat16, 2, use_fp8=True),
        id="b1-l17-q400-h2-d192-bf16-pointer",
    ),
    pytest.param(
        _CorrectnessCase((1,), 7, 320, 1, 256, torch.float32, 1),
        id="b1-l7-q320-h1-d256-fp32-sdpa-fallback",
    ),
    pytest.param(
        _CorrectnessCase((1,), 5, 384, 1, 320, torch.float32, 2, use_rope=False),
        id="b1-l5-q384-h1-d320-fp32-sdpa-fallback",
    ),
)


@pytest.mark.ci_gpu
@pytest.mark.parametrize("case", _CORRECTNESS_CASES)
def test_accelerated_self_attention_matches_sdpa_across_architectures(
    cuda_device: torch.device,
    case: _CorrectnessCase,
) -> None:
    """Match SDPA across model widths, head layouts, dtypes, and cache phases."""
    torch.manual_seed(0)
    generator = torch.Generator(device=cuda_device).manual_seed(1234)
    attention = AcceleratedSelfAttention(
        case.query_dim,
        case.n_heads,
        case.head_dim,
        qkv_bias=case.qkv_bias,
        output_bias=case.output_bias,
        qk_norm=case.qk_norm,
        rope_interleaved=case.rope_interleaved,
        use_fp8=case.use_fp8,
    ).to(device=cuda_device, dtype=case.dtype)
    attention.eval()
    if case.use_fp8 and torch.cuda.get_device_capability(cuda_device)[0] < 9:
        pytest.skip("Strict FP8 attention requires compute capability 9.0+.")
    reference_attention = _make_native_reference(attention, cuda_device, case.dtype)

    batch_size = math.prod(case.batch_shape)
    window_size = case.cache_chunks * case.sequence_length
    sink_size = case.sink_chunks * case.sequence_length
    candidate_cache = attention.initialize_cache(
        batch_size,
        case.sequence_length,
        window_size,
        sink_size,
        cuda_device,
        case.dtype,
    )
    reference_cache = _initialize_reference_cache(
        reference_attention,
        batch_size,
        case.sequence_length,
        window_size,
        sink_size,
        cuda_device,
        case.dtype,
    )
    num_steps = case.cache_chunks + case.sink_chunks + 2

    with torch.inference_mode():
        for chunk_index in range(num_steps):
            x = torch.randn(
                case.batch_shape + (case.sequence_length, case.query_dim),
                device=cuda_device,
                dtype=case.dtype,
                generator=generator,
            )
            rope_freqs = (
                _make_rope_freqs(
                    case.sequence_length,
                    case.head_dim,
                    cuda_device,
                    generator,
                    interleaved=case.rope_interleaved,
                )
                if case.use_rope
                else None
            )
            candidate_cache.before_update(chunk_index)
            reference_cache.before_update(chunk_index)
            metadata = attention._backend_metadata(x, candidate_cache, rope_freqs)
            expected = _sdpa_reference_step(
                reference_attention,
                x,
                reference_cache,
                rope_freqs,
            )
            actual = attention(x, candidate_cache, rope_freqs)
            _assert_module_matches_sdpa(
                actual,
                expected,
                candidate_cache,
                reference_cache,
                fp8_effective=bool(metadata["fp8_effective"]),
            )
            candidate_cache.after_update(chunk_index)
            reference_cache.after_update(chunk_index)


_OPTIMIZATION_COMBINATIONS = tuple(
    pytest.param(
        *values,
        id=(
            f"tma-{int(values[0])}-qkv-{int(values[1])}-"
            f"rope-cache-{int(values[2])}-fp8-{int(values[3])}"
        ),
    )
    for values in itertools.product((False, True), repeat=4)
)


@pytest.mark.ci_gpu
@pytest.mark.parametrize(
    "use_tma,fuse_qkv,fuse_rope_kv_cache,use_fp8",
    _OPTIMIZATION_COMBINATIONS,
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_every_optimization_combination_matches_sdpa(
    cuda_device: torch.device,
    dtype: torch.dtype,
    use_tma: bool,
    fuse_qkv: bool,
    fuse_rope_kv_cache: bool,
    use_fp8: bool,
) -> None:
    """Match SDPA for the full factorial optimization configuration matrix."""
    torch.manual_seed(0)
    generator = torch.Generator(device=cuda_device).manual_seed(2345)
    attention = AcceleratedSelfAttention(
        320,
        4,
        64,
        use_tma=use_tma,
        fuse_qkv=fuse_qkv,
        fuse_rope_kv_cache=fuse_rope_kv_cache,
        use_fp8=use_fp8,
    ).to(device=cuda_device, dtype=dtype)
    attention.eval()
    if use_fp8 and torch.cuda.get_device_capability(cuda_device)[0] < 9:
        pytest.skip("Strict FP8 attention requires compute capability 9.0+.")
    reference_attention = _make_native_reference(attention, cuda_device, dtype)
    x = torch.randn(
        (1, 17, 320),
        device=cuda_device,
        dtype=dtype,
        generator=generator,
    )
    rope_freqs = _make_rope_freqs(
        17,
        64,
        cuda_device,
        generator,
        interleaved=False,
    )
    candidate_cache = attention.initialize_cache(1, 17, 34, 0, cuda_device, dtype)
    reference_cache = _initialize_reference_cache(
        reference_attention, 1, 17, 34, 0, cuda_device, dtype
    )
    candidate_cache.before_update(0)
    reference_cache.before_update(0)

    with torch.inference_mode():
        metadata = attention._backend_metadata(x, candidate_cache, rope_freqs)
        expected = _sdpa_reference_step(
            reference_attention, x, reference_cache, rope_freqs
        )
        actual = attention(x, candidate_cache, rope_freqs)
        _assert_module_matches_sdpa(
            actual,
            expected,
            candidate_cache,
            reference_cache,
            fp8_effective=bool(metadata["fp8_effective"]),
        )


_ATTENTION_CASES = (
    pytest.param(1, 1, 1, 1, 16, torch.bfloat16, True, id="d16-bf16-tma"),
    pytest.param(1, 2, 7, 11, 32, torch.float16, True, id="d32-fp16-tma"),
    pytest.param(2, 3, 13, 73, 48, torch.float16, True, id="d48-fp16-pointer"),
    pytest.param(1, 2, 17, 97, 64, torch.bfloat16, True, id="d64-bf16-tma"),
    pytest.param(1, 2, 19, 73, 96, torch.float32, True, id="d96-fp32-sdpa-fallback"),
    pytest.param(1, 1, 33, 97, 128, torch.float16, True, id="d128-fp16-tma"),
    pytest.param(
        1,
        2,
        129,
        257,
        128,
        torch.float16,
        True,
        id="d128-fp16-tma-large-tile",
    ),
    pytest.param(
        1,
        2,
        129,
        257,
        128,
        torch.float16,
        False,
        id="d128-fp16-pointer-large-tile",
    ),
    pytest.param(1, 1, 17, 73, 192, torch.bfloat16, False, id="d192-bf16-pointer"),
    pytest.param(1, 1, 7, 33, 256, torch.float16, True, id="d256-fp16-tma"),
    pytest.param(1, 1, 5, 19, 320, torch.float32, True, id="d320-sdpa-fallback"),
)


@pytest.mark.ci_gpu
@pytest.mark.parametrize(
    "batch_size,n_heads,query_length,key_length,head_dim,dtype,use_tma",
    _ATTENTION_CASES,
)
def test_accelerated_attention_kernel_matches_sdpa(
    cuda_device: torch.device,
    batch_size: int,
    n_heads: int,
    query_length: int,
    key_length: int,
    head_dim: int,
    dtype: torch.dtype,
    use_tma: bool,
) -> None:
    """Match SDPA across pointer, TMA, and structural fallback dimensions."""
    generator = torch.Generator(device=cuda_device).manual_seed(3456)
    query = torch.randn(
        (batch_size, n_heads, query_length, head_dim),
        device=cuda_device,
        dtype=dtype,
        generator=generator,
    )
    key = torch.randn(
        (batch_size, n_heads, key_length, head_dim),
        device=cuda_device,
        dtype=dtype,
        generator=generator,
    )
    value = torch.randn(
        (batch_size, n_heads, key_length, head_dim),
        device=cuda_device,
        dtype=dtype,
        generator=generator,
    )
    attention = AcceleratedSelfAttention(
        n_heads * head_dim,
        n_heads,
        head_dim,
        use_tma=use_tma,
        fuse_qkv=True,
        fuse_rope_kv_cache=False,
        use_fp8=False,
    ).to(device=cuda_device, dtype=dtype)
    attention.eval()

    with torch.inference_mode():
        expected = F.scaled_dot_product_attention(query, key, value)
        actual = attention._apply_attention(query, key, value)
    atol, rtol = _native_tolerance(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.ci_gpu
def test_tma_attention_torch_compile_analyzes_tensor_accesses(
    cuda_device: torch.device,
) -> None:
    """Compile TMA attention without conservatively marking every input mutated."""
    if torch.cuda.get_device_capability(cuda_device)[0] < 9:
        pytest.skip("TMA attention requires compute capability 9.0+.")

    generator = torch.Generator(device=cuda_device).manual_seed(4567)
    query = torch.randn(
        (1, 2, 17, 128),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )
    key = torch.randn(
        (1, 2, 33, 128),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )
    value = torch.randn(
        (1, 2, 33, 128),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )
    attention = AcceleratedSelfAttention(
        256,
        2,
        128,
        use_tma=True,
        fuse_rope_kv_cache=False,
        use_fp8=False,
    ).eval()

    with torch.inference_mode():
        assert attention._supports_tma_attention(query, key, value)
        expected = attention._apply_attention(query, key, value)

    captured: list[logging.LogRecord] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    logger = logging.getLogger("torch._dynamo")
    handler = _CaptureHandler(level=logging.WARNING)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    torch.compiler.reset()
    try:
        with torch.inference_mode():
            compiled_attention = torch.compile(
                attention._apply_attention,
                mode="default",
                fullgraph=True,
            )
            actual = compiled_attention(query, key, value)
            torch.cuda.synchronize(cuda_device)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    assert not any(
        "assuming every input is mutated" in record.getMessage() for record in captured
    )
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.ci_gpu
def test_tma_attention_falls_back_for_noncontiguous_kv_features(
    cuda_device: torch.device,
) -> None:
    """Use the pointer kernel when a K/V descriptor is not unit-stride."""
    if torch.cuda.get_device_capability(cuda_device)[0] < 9:
        pytest.skip("TMA attention requires compute capability 9.0+.")

    generator = torch.Generator(device=cuda_device).manual_seed(5678)
    query = torch.randn(
        (1, 2, 17, 128),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )
    key = torch.randn(
        (1, 2, 33, 256),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )[..., ::2]
    value = torch.randn(
        (1, 2, 33, 256),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )[..., ::2]
    attention = AcceleratedSelfAttention(
        256,
        2,
        128,
        use_tma=True,
        fuse_rope_kv_cache=False,
        use_fp8=False,
    ).eval()

    with torch.inference_mode():
        assert not attention._supports_tma_attention(query, key, value)
        expected = F.scaled_dot_product_attention(query, key, value)
        actual = attention._apply_attention(query, key, value)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.ci_cpu
def test_tma_tensor_layout_requires_descriptor_strides() -> None:
    """Reject final-contiguous views that violate TMA stride constraints."""
    contiguous = torch.empty((1, 2, 3, 128), dtype=torch.float16)
    padded = torch.empty((1, 2, 3, 129), dtype=torch.float16)[..., :128]
    broadcast = contiguous.expand(2, -1, -1, -1)

    assert AcceleratedSelfAttention._supports_tma_tensor_layout(contiguous)
    assert not AcceleratedSelfAttention._supports_tma_tensor_layout(padded)
    assert not AcceleratedSelfAttention._supports_tma_tensor_layout(broadcast)


@pytest.mark.ci_cpu
@pytest.mark.parametrize(
    "configs",
    (_POINTER_ATTENTION_CONFIGS, _TMA_ATTENTION_CONFIGS),
    ids=("pointer", "tma"),
)
@pytest.mark.parametrize(
    "query_length,head_dim,expected_max_block_m",
    (
        (1, 64, 16),
        (17, 64, 32),
        (129, 128, 128),
        (257, 192, 64),
    ),
)
def test_attention_autotune_prunes_tiles_and_retains_previous_default(
    configs: list[triton.Config],
    query_length: int,
    head_dim: int,
    expected_max_block_m: int,
) -> None:
    """Keep valid tiles and the former launch choice in each shape class."""
    pruned = _prune_attention_configs(
        configs,
        {"query_length": query_length},
        HEAD_DIM=head_dim,
    )
    assert pruned
    assert max(config.kwargs["BLOCK_M"] for config in pruned) == (expected_max_block_m)

    block_d = max(int(triton.next_power_of_2(head_dim)), 16)
    block_n = 32 if block_d > 128 else 64
    num_stages = 2 if block_d > 128 else 3
    num_warps = 4 if expected_max_block_m * block_d <= 4096 else 8
    assert any(
        config.kwargs == {"BLOCK_M": expected_max_block_m, "BLOCK_N": block_n}
        and config.num_warps == num_warps
        and config.num_stages == num_stages
        for config in pruned
    )


@pytest.mark.ci_cpu
def test_tma_attention_autotune_prunes_unsupported_hopper_warp_layouts() -> None:
    """Avoid Hopper 8-warp tiles excluded by Triton's attention tuner."""
    pruned = _prune_tma_attention_configs(
        _TMA_ATTENTION_CONFIGS,
        {"query_length": 129, "device_arch": 90},
        HEAD_DIM=128,
    )
    assert pruned
    assert all(
        not (
            config.kwargs["BLOCK_M"] * config.kwargs["BLOCK_N"] < 128 * 128
            and config.num_warps == 8
        )
        for config in pruned
    )


@pytest.mark.ci_cpu
def test_all_optimizations_disabled_matches_sdpa_on_cpu() -> None:
    """Use the complete PyTorch fallback when every optimization is disabled."""
    torch.manual_seed(0)
    attention = AcceleratedSelfAttention(
        80,
        3,
        16,
        use_tma=False,
        fuse_qkv=False,
        fuse_rope_kv_cache=False,
        use_fp8=False,
    ).eval()
    x = torch.randn((2, 7, 80))
    generator = torch.Generator().manual_seed(4567)
    rope_freqs = _make_rope_freqs(
        7,
        16,
        torch.device("cpu"),
        generator,
        interleaved=False,
    )
    candidate_cache = attention.initialize_cache(
        2, 7, 14, 0, torch.device("cpu"), torch.float32
    )
    reference_cache = _initialize_reference_cache(
        attention, 2, 7, 14, 0, torch.device("cpu"), torch.float32
    )
    candidate_cache.before_update(0)
    reference_cache.before_update(0)
    with torch.inference_mode():
        expected = _sdpa_reference_step(attention, x, reference_cache, rope_freqs)
        actual = attention(x, candidate_cache, rope_freqs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.ci_cpu
def test_fp8_mode_keeps_only_packed_projection_state() -> None:
    """Persist packed FP8 weights without native or derived weight mirrors."""
    torch.manual_seed(8)
    native = AcceleratedSelfAttention(
        32,
        2,
        16,
        qkv_bias=True,
        output_bias=True,
        use_fp8=False,
    )
    packed = AcceleratedSelfAttention(
        32,
        2,
        16,
        qkv_bias=True,
        output_bias=True,
        use_fp8=True,
    )
    packed.load_state_dict(native.state_dict(), strict=True)

    state = packed.state_dict()
    assert set(state) == {
        "qkv_proj.weight",
        "qkv_proj.weight_scale",
        "qkv_proj.bias",
        "output_proj.weight",
        "output_proj.weight_scale",
        "output_proj.bias",
        "q_norm.weight",
        "k_norm.weight",
    }
    assert state["qkv_proj.weight"].dtype == torch.uint8
    assert state["output_proj.weight"].dtype == torch.uint8
    assert state["qkv_proj.weight_scale"].dtype == torch.float32
    assert state["output_proj.weight_scale"].dtype == torch.float32
    assert not hasattr(packed, "q_proj")
    assert not hasattr(packed, "_fused_qkv_weight")

    packed_copy = AcceleratedSelfAttention(
        32,
        2,
        16,
        qkv_bias=True,
        output_bias=True,
        use_fp8=True,
    )
    packed_copy.load_state_dict(state, strict=True)
    native_copy = AcceleratedSelfAttention(
        32,
        2,
        16,
        qkv_bias=True,
        output_bias=True,
        use_fp8=False,
    )
    native_copy.load_state_dict(state, strict=True)
    packed_copy.to(dtype=torch.bfloat16)
    assert packed_copy.qkv_proj.weight.dtype == torch.uint8
    assert packed_copy.qkv_proj.weight_scale.dtype == torch.float32


@pytest.mark.ci_cpu
def test_fp8_cache_lifecycle_uses_one_canonical_allocation() -> None:
    """Preserve FP8 fill, same-chunk overwrite, roll, and reset semantics."""
    cache = BlockKVCache(
        k_shape=(1, 4, 1, 2),
        v_shape=(1, 4, 1, 2),
        seq_dim=1,
        chunk_size=2,
        window_size=4,
        device=torch.device("cpu"),
        dtype=torch.float8_e4m3fn,
    )
    k_ptr = cache._k.data_ptr()
    v_ptr = cache._v.data_ptr()

    def chunk(value: float) -> Tensor:
        return torch.full((1, 2, 1, 2), value, dtype=torch.float32)

    cache.before_update(0)
    cache.update(chunk(1.0), chunk(11.0))
    cache.after_update(0)
    cache.before_update(1)
    cache.update(chunk(2.0), chunk(12.0))
    cache.after_update(1)
    cache.before_update(1)
    cache.update(chunk(3.0), chunk(13.0))
    torch.testing.assert_close(cache.cached_k().float()[:, 2:], chunk(3.0))
    cache.after_update(1)
    cache.before_update(2)
    cache.update(chunk(4.0), chunk(14.0))
    torch.testing.assert_close(cache.cached_k().float()[:, :2], chunk(3.0))
    torch.testing.assert_close(cache.cached_k().float()[:, 2:], chunk(4.0))
    cache.after_update(2)

    cache.reset()
    assert cache.size == 0
    assert cache._k.data_ptr() == k_ptr
    assert cache._v.data_ptr() == v_ptr
    assert cache._k.dtype == torch.float8_e4m3fn
    assert cache._v.dtype == torch.float8_e4m3fn


@pytest.mark.ci_cpu
def test_strict_fp8_mode_rejects_unsupported_execution() -> None:
    """Reject invalid geometry and CPU cache initialization without fallback."""
    with pytest.raises(ValueError, match="multiple of 16"):
        AcceleratedSelfAttention(31, 1, 16, use_fp8=True)
    attention = AcceleratedSelfAttention(32, 2, 16, use_fp8=True)
    with pytest.raises(RuntimeError, match="CUDA device"):
        attention.initialize_cache(1, 2, 4, 0, torch.device("cpu"), torch.bfloat16)


@pytest.mark.ci_cpu
def test_speed_defaults_prefer_cudnn_with_native_storage() -> None:
    """Keep the speed-oriented backend and storage policy explicit."""
    attention = AcceleratedSelfAttention(32, 2, 16)

    assert attention.use_cudnn
    assert not attention.use_fp8
    assert attention.optimization_settings == {
        "use_tma": True,
        "fuse_qkv": True,
        "fuse_rope_kv_cache": True,
        "use_cudnn": True,
        "use_fp8": False,
    }


@pytest.mark.ci_gpu
def test_native_inference_selects_cudnn_attention(
    cuda_device: torch.device,
) -> None:
    """Select cuDNN for eligible native-precision inference inputs."""
    attention = AcceleratedSelfAttention(256, 2, 128).to(
        device=cuda_device,
        dtype=torch.bfloat16,
    )
    attention.eval()
    x = torch.zeros((1, 16, 256), device=cuda_device, dtype=torch.bfloat16)
    kv_cache = attention.initialize_cache(
        1,
        16,
        32,
        0,
        cuda_device,
        torch.bfloat16,
    )
    kv_cache.before_update(0)

    with torch.inference_mode():
        metadata = attention._backend_metadata(x, kv_cache, None)

    assert metadata["attention_backend"] == "cudnn"


@dataclass(frozen=True)
class _BenchmarkCase:
    """One full-forward benchmark shape and prefilled cache length."""

    name: str
    batch_size: int
    sequence_length: int
    query_dim: int
    n_heads: int
    head_dim: int
    cache_chunks: int


@dataclass(frozen=True)
class _BenchmarkVariant:
    """One requested optimization configuration for benchmark labeling."""

    name: str
    use_tma: bool
    fuse_qkv: bool
    fuse_rope_kv_cache: bool
    use_fp8: bool
    use_cudnn: bool = True

    def as_kwargs(self) -> dict[str, bool]:
        """Return constructor keyword arguments for this variant."""
        return {
            "use_tma": self.use_tma,
            "fuse_qkv": self.fuse_qkv,
            "fuse_rope_kv_cache": self.fuse_rope_kv_cache,
            "use_fp8": self.use_fp8,
            "use_cudnn": self.use_cudnn,
        }


_BENCHMARK_CASES = (
    _BenchmarkCase("small", 1, 256, 320, 4, 64, 1),
    _BenchmarkCase("batch2-history3", 2, 512, 384, 4, 64, 3),
    _BenchmarkCase("batch4-nonpower", 4, 257, 768, 8, 96, 1),
    _BenchmarkCase("omni-5840-history1", 1, 40 * 73 * 2, 2048, 16, 128, 1),
    _BenchmarkCase("omni-5840-history3", 1, 40 * 73 * 2, 2048, 16, 128, 3),
    _BenchmarkCase("omni-7040-history1", 1, 80 * 44 * 2, 2048, 16, 128, 1),
    _BenchmarkCase("omni-7040-history3", 1, 80 * 44 * 2, 2048, 16, 128, 3),
)

_DROP_ONE_VARIANTS = (
    _BenchmarkVariant("cudnn-reference", False, False, False, False),
    _BenchmarkVariant("default", True, True, True, False),
    _BenchmarkVariant("without-fused-qkv", True, False, True, False),
    _BenchmarkVariant("without-fused-rope-cache", True, True, False, False),
    _BenchmarkVariant("fp8", True, True, True, True),
    _BenchmarkVariant("triton-native", True, True, True, False, False),
    _BenchmarkVariant("triton-fp8", True, True, True, True, False),
)

_FACTORIAL_VARIANTS = tuple(
    _BenchmarkVariant(
        (
            f"tma-{int(values[0])}-qkv-{int(values[1])}-"
            f"rope-cache-{int(values[2])}-fp8-{int(values[3])}"
        ),
        values[0],
        values[1],
        values[2],
        values[3],
        False,
    )
    for values in itertools.product((False, True), repeat=4)
)

_BENCHMARK_DTYPES = (
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
)


def _benchmark_rounds() -> int:
    """Return the configured number of measured benchmark rounds."""
    return int(os.getenv("FLASHDREAMS_ATTENTION_BENCHMARK_ROUNDS", "50"))


def _benchmark_warmup_rounds() -> int:
    """Return the configured number of benchmark warmup rounds."""
    return int(os.getenv("FLASHDREAMS_ATTENTION_BENCHMARK_WARMUP_ROUNDS", "10"))


def _run_self_attention_benchmark(
    benchmark: Any,
    cuda_device: torch.device,
    case: _BenchmarkCase,
    variant: _BenchmarkVariant,
    dtype: torch.dtype,
    *,
    study: str,
) -> None:
    """Benchmark one prefilled steady-state full-forward configuration."""
    if dtype == torch.float32 and variant.use_fp8:
        pytest.skip("Strict FP8 variants apply only to FP16/BF16 inputs.")

    torch.manual_seed(0)
    generator = torch.Generator(device=cuda_device).manual_seed(9012)
    attention = AcceleratedSelfAttention(
        case.query_dim,
        case.n_heads,
        case.head_dim,
        **variant.as_kwargs(),
    ).to(device=cuda_device, dtype=dtype)
    attention.eval()
    x = torch.randn(
        (case.batch_size, case.sequence_length, case.query_dim),
        device=cuda_device,
        dtype=dtype,
        generator=generator,
    )
    rope_freqs = _make_rope_freqs(
        case.sequence_length,
        case.head_dim,
        cuda_device,
        generator,
        interleaved=False,
    )
    window_size = case.cache_chunks * case.sequence_length
    kv_cache = attention.initialize_cache(
        case.batch_size,
        case.sequence_length,
        window_size,
        0,
        cuda_device,
        dtype,
    )

    with torch.inference_mode():
        for chunk_index in range(case.cache_chunks):
            kv_cache.before_update(chunk_index)
            attention(x, kv_cache, rope_freqs)
            kv_cache.after_update(chunk_index)
        torch.cuda.synchronize(cuda_device)
        metadata = attention._backend_metadata(x, kv_cache, rope_freqs)
    current_chunk_index = case.cache_chunks - 1

    @torch.inference_mode()
    def run_attention() -> Tensor:
        kv_cache.before_update(current_chunk_index)
        output = attention(x, kv_cache, rope_freqs)
        kv_cache.after_update(current_chunk_index)
        torch.cuda.synchronize(cuda_device)
        return output

    dtype_name = str(dtype).removeprefix("torch.")
    benchmark.group = f"{case.name}-{dtype_name}"
    benchmark.extra_info.update(
        {
            "study": study,
            "variant": variant.name,
            "device": torch.cuda.get_device_name(cuda_device),
            "compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability(cuda_device)
            ),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "dtype": str(dtype),
            "batch_size": case.batch_size,
            "sequence_length": case.sequence_length,
            "cached_sequence_length": window_size,
            "cache_chunks": case.cache_chunks,
            "query_dim": case.query_dim,
            "n_heads": case.n_heads,
            "head_dim": case.head_dim,
            "tokens_per_forward": case.batch_size * case.sequence_length,
            "warmup_rounds": _benchmark_warmup_rounds(),
            "rounds": _benchmark_rounds(),
            "projection_weight_bytes": sum(
                tensor.numel() * tensor.element_size()
                for name, tensor in attention.state_dict().items()
                if name.endswith(("proj.weight", "proj.weight_scale"))
            ),
            "kv_cache_bytes": (
                kv_cache._k.numel() * kv_cache._k.element_size()
                + kv_cache._v.numel() * kv_cache._v.element_size()
            ),
            **metadata,
        }
    )
    benchmark.pedantic(
        run_attention,
        rounds=_benchmark_rounds(),
        warmup_rounds=_benchmark_warmup_rounds(),
        iterations=1,
    )


@pytest.mark.manual
@pytest.mark.parametrize("dtype", _BENCHMARK_DTYPES)
@pytest.mark.parametrize("variant", _DROP_ONE_VARIANTS, ids=lambda item: item.name)
@pytest.mark.parametrize("case", _BENCHMARK_CASES, ids=lambda item: item.name)
def test_self_attention_drop_one_benchmark(
    benchmark: Any,
    cuda_device: torch.device,
    case: _BenchmarkCase,
    variant: _BenchmarkVariant,
    dtype: torch.dtype,
) -> None:
    """Benchmark the default, reference, fallbacks, and drop-one variants."""
    _run_self_attention_benchmark(
        benchmark,
        cuda_device,
        case,
        variant,
        dtype,
        study="drop-one",
    )


@pytest.mark.manual
@pytest.mark.parametrize("dtype", _BENCHMARK_DTYPES)
@pytest.mark.parametrize("variant", _FACTORIAL_VARIANTS, ids=lambda item: item.name)
def test_self_attention_omnidreams_factorial_benchmark(
    benchmark: Any,
    cuda_device: torch.device,
    variant: _BenchmarkVariant,
    dtype: torch.dtype,
) -> None:
    """Benchmark all optimization combinations on an Omnidreams steady shape."""
    case = _BenchmarkCase(
        "omni-5840-history3-factorial",
        1,
        40 * 73 * 2,
        2048,
        16,
        128,
        3,
    )
    _run_self_attention_benchmark(
        benchmark,
        cuda_device,
        case,
        variant,
        dtype,
        study="factorial",
    )
