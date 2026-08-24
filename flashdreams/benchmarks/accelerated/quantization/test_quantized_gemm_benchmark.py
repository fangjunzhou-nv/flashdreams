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

"""Benchmarks for full-precision and quantized Torch matrix multiplication.

Quantized ``end-to-end`` rows include operand quantization and output conversion
in the timed region. ``gemm-only`` rows prepare operands before timing and omit
INT8 dequantization. FP8 scale application remains fused into ``torch._scaled_mm``.
CUDA does not support E5M2 by E5M2 GEMM, so E5M2 rows use E5M2 for the left
operand and E4M3 for the right operand.

Run the manual GPU benchmarks with::

    uv run --package flashdreams --group test pytest \
        flashdreams/benchmarks/accelerated/quantization/test_quantized_gemm_benchmark.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor

from flashdreams.accelerated.quantization.quantizer import (
    DTYPE_MAX,
    Granularity,
    dequantize,
    quantize,
)

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Quantized GEMM benchmarks require CUDA.",
    ),
]

_M = 4096
_K = 4096
_N = 4096
_SEED = 42
_WARMUP_ROUNDS = 5
"""Warmup calls used to absorb kernel initialization and autotuning."""

_BENCHMARK_ROUNDS = 50
"""Measured calls used for each GEMM comparison."""

_ORIGINAL_DTYPES = (
    pytest.param(torch.float16, "fp16", id="fp16"),
    pytest.param(torch.bfloat16, "bf16", id="bf16"),
    pytest.param(torch.float32, "fp32", id="fp32"),
)
"""Original matrix formats, each mapped to a separate benchmark group."""

_GEMM_CASES = (
    pytest.param(None, None, id="full-precision"),
    *(
        pytest.param(
            quantized_dtype,
            granularity,
            id=(
                f"{str(quantized_dtype).removeprefix('torch.')}"
                f"{'-x-float8_e4m3fn' if quantized_dtype is torch.float8_e5m2 else ''}"
                f"-{granularity.value}"
            ),
        )
        for quantized_dtype in DTYPE_MAX
        for granularity in Granularity
    ),
)
"""Full-precision baseline and every supported quantized GEMM configuration."""


def _quantized_gemm(
    left: Tensor,
    right: Tensor,
    quantized_dtype: torch.dtype,
    granularity: Granularity,
    output_dtype: torch.dtype,
    end_to_end: bool,
) -> tuple[Callable[[], Tensor], torch.dtype]:
    """Return the timed operation and the data type it produces."""
    if quantized_dtype is torch.int8:

        def quantize_operands() -> tuple[Tensor, Tensor, Tensor, Tensor]:
            left_quantized, left_scale = quantize(
                left, quantized_dtype, granularity, axis=1
            )
            right_quantized, right_scale = quantize(
                right, quantized_dtype, granularity, axis=0
            )
            return left_quantized, right_quantized, left_scale, right_scale

        prepared_operands = None if end_to_end else quantize_operands()

        def gemm() -> Tensor:
            operands = quantize_operands() if end_to_end else prepared_operands
            assert operands is not None
            left_quantized, right_quantized, left_scale, right_scale = operands
            output = torch._int_mm(left_quantized, right_quantized)
            if end_to_end:
                return dequantize(
                    output,
                    left_scale,
                    right_scale,
                    dtype=output_dtype,
                )
            return output

        return gemm, output_dtype if end_to_end else torch.int32

    right_dtype = (
        torch.float8_e4m3fn if quantized_dtype is torch.float8_e5m2 else quantized_dtype
    )
    scaled_output_dtype = (
        torch.bfloat16
        if granularity is Granularity.SLICE and output_dtype is torch.float32
        else output_dtype
    )

    def quantize_operands() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        left_quantized, left_scale = quantize(
            left, quantized_dtype, granularity, axis=1
        )
        right_quantized_transposed, right_scale_transposed = quantize(
            right.t().contiguous(), right_dtype, granularity, axis=1
        )
        return (
            left_quantized,
            right_quantized_transposed.t(),
            left_scale,
            right_scale_transposed.t(),
        )

    prepared_operands = None if end_to_end else quantize_operands()

    def gemm() -> Tensor:
        operands = quantize_operands() if end_to_end else prepared_operands
        assert operands is not None
        left_quantized, right_quantized, left_scale, right_scale = operands
        output = torch._scaled_mm(
            left_quantized,
            right_quantized,
            left_scale,
            right_scale,
            out_dtype=scaled_output_dtype,
        )
        return output.to(output_dtype) if end_to_end else output

    return gemm, output_dtype if end_to_end else scaled_output_dtype


def _benchmark_quantized_gemm(
    benchmark: BenchmarkFixture,
    original_dtype: torch.dtype,
    original_format: str,
    quantized_dtype: torch.dtype | None,
    granularity: Granularity | None,
    *,
    end_to_end: bool,
) -> None:
    """Benchmark one quantized GEMM configuration."""
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    left = torch.randn(
        (_M, _K), device="cuda", dtype=original_dtype, generator=generator
    )
    right = torch.randn(
        (_K, _N), device="cuda", dtype=original_dtype, generator=generator
    )

    if quantized_dtype is None:
        output_dtype = original_dtype

        def operation() -> Tensor:
            return left @ right

    else:
        assert granularity is not None
        operation, output_dtype = _quantized_gemm(
            left,
            right,
            quantized_dtype,
            granularity,
            original_dtype,
            end_to_end,
        )

    timing_scope = "end-to-end" if end_to_end else "gemm-only"
    benchmark.group = f"quantized-gemm-{original_format}-{timing_scope}"
    left_dtype = original_dtype if quantized_dtype is None else quantized_dtype
    right_dtype = (
        original_dtype
        if quantized_dtype is None
        else torch.float8_e4m3fn
        if quantized_dtype is torch.float8_e5m2
        else quantized_dtype
    )
    implementation = (
        "torch.matmul"
        if quantized_dtype is None
        else "torch._int_mm"
        if quantized_dtype is torch.int8
        else "torch._scaled_mm"
    )
    benchmark.extra_info.update(
        {
            "implementation": implementation,
            "timing_scope": timing_scope,
            "source_dtype": str(original_dtype),
            "left_dtype": str(left_dtype),
            "right_dtype": str(right_dtype),
            "output_dtype": str(output_dtype),
            "quantized_dtype": (
                None if quantized_dtype is None else str(quantized_dtype)
            ),
            "granularity": None if granularity is None else granularity.value,
            "operand_quantization_timed": end_to_end and quantized_dtype is not None,
            "output_conversion_timed": end_to_end and quantized_dtype is not None,
            "m": _M,
            "k": _K,
            "n": _N,
            "seed": _SEED,
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
        }
    )

    def synchronized_gemm() -> Tensor:
        output = operation()
        torch.cuda.synchronize()
        return output

    torch.cuda.synchronize()
    output = benchmark.pedantic(
        synchronized_gemm,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )

    assert output.shape == (_M, _N)
    assert output.dtype is output_dtype
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("quantized_dtype,granularity", _GEMM_CASES)
@pytest.mark.parametrize("original_dtype,original_format", _ORIGINAL_DTYPES)
def test_quantized_gemm_end_to_end_benchmark(
    benchmark: BenchmarkFixture,
    original_dtype: torch.dtype,
    original_format: str,
    quantized_dtype: torch.dtype | None,
    granularity: Granularity | None,
) -> None:
    """Benchmark quantization, GEMM, and output conversion together."""
    _benchmark_quantized_gemm(
        benchmark,
        original_dtype,
        original_format,
        quantized_dtype,
        granularity,
        end_to_end=True,
    )


@pytest.mark.parametrize("quantized_dtype,granularity", _GEMM_CASES)
@pytest.mark.parametrize("original_dtype,original_format", _ORIGINAL_DTYPES)
def test_quantized_gemm_only_benchmark(
    benchmark: BenchmarkFixture,
    original_dtype: torch.dtype,
    original_format: str,
    quantized_dtype: torch.dtype | None,
    granularity: Granularity | None,
) -> None:
    """Benchmark GEMM with operands quantized before timing."""
    _benchmark_quantized_gemm(
        benchmark,
        original_dtype,
        original_format,
        quantized_dtype,
        granularity,
        end_to_end=False,
    )
