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

"""Benchmarks for regular and quantized nonpersistent linear inference.

``full-precision-x`` rows include activation quantization in the timed region,
while ``prequantized-x`` rows prepare activations and scales before timing.
Weight quantization always happens during module construction.

Run the manual GPU benchmarks with::

    uv run --package flashdreams --group test pytest \
        flashdreams/benchmarks/accelerated/quantization/test_quantized_linear_benchmark.py \
        -p no:manual_marker -m manual --benchmark-only -v
"""

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor, nn

from flashdreams.accelerated.quantization.linear import (
    QuantizedNonPersistentLinear,
    WeightGranularity,
)
from flashdreams.accelerated.quantization.quantizer import (
    DTYPE_MAX,
    Granularity,
    quantize,
)

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Quantized linear benchmarks require CUDA.",
    ),
]

_BATCH_SIZE = 4096
_IN_FEATURES = 4096
_OUT_FEATURES = 4096
_SEED = 42
_WARMUP_ROUNDS = 5
"""Warmup calls used to absorb kernel initialization and autotuning."""

_BENCHMARK_ROUNDS = 50
"""Measured calls used for each linear comparison."""

_ORIGINAL_DTYPES = (
    pytest.param(torch.float16, "fp16", id="fp16"),
    pytest.param(torch.bfloat16, "bf16", id="bf16"),
    pytest.param(torch.float32, "fp32", id="fp32"),
)
"""Original activation and output formats, each in a separate benchmark group."""

_LINEAR_CASES = (
    pytest.param(None, None, None, None, id="nn-linear"),
    *(
        pytest.param(
            quantized_dtype,
            weight_granularity,
            input_granularity,
            prequantized,
            id=(
                f"{str(quantized_dtype).removeprefix('torch.')}"
                f"{'-x-float8_e4m3fn' if quantized_dtype is torch.float8_e5m2 else ''}"
                f"-weight-{weight_granularity.value}"
                f"-input-{input_granularity.value}"
                f"-{'prequantized-x' if prequantized else 'full-precision-x'}"
            ),
        )
        for quantized_dtype in DTYPE_MAX
        for weight_granularity in WeightGranularity
        for input_granularity in Granularity
        for prequantized in (False, True)
    ),
)
"""Regular linear baseline and every quantized linear inference configuration."""


@pytest.mark.parametrize(
    "quantized_dtype,weight_granularity,input_granularity,prequantized",
    _LINEAR_CASES,
)
@pytest.mark.parametrize("original_dtype,original_format", _ORIGINAL_DTYPES)
@torch.inference_mode()
def test_quantized_linear_benchmark(
    benchmark: BenchmarkFixture,
    original_dtype: torch.dtype,
    original_format: str,
    quantized_dtype: torch.dtype | None,
    weight_granularity: WeightGranularity | None,
    input_granularity: Granularity | None,
    prequantized: bool | None,
) -> None:
    """Benchmark regular and quantized linear inference."""
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    inputs = torch.randn(
        (_BATCH_SIZE, _IN_FEATURES),
        device="cuda",
        dtype=original_dtype,
        generator=generator,
    )
    weight = torch.randn(
        (_OUT_FEATURES, _IN_FEATURES),
        device="cuda",
        dtype=original_dtype,
        generator=generator,
    )

    operation: Callable[[], Tensor]
    if quantized_dtype is None:
        assert (
            weight_granularity is None
            and input_granularity is None
            and prequantized is None
        )
        linear = nn.Linear(
            _IN_FEATURES,
            _OUT_FEATURES,
            bias=False,
            device="cuda",
            dtype=original_dtype,
        ).requires_grad_(False)
        linear.weight.copy_(weight)

        def operation() -> Tensor:
            return linear(inputs)

    else:
        assert weight_granularity is not None
        assert input_granularity is not None
        assert prequantized is not None
        quantized_linear = QuantizedNonPersistentLinear(
            weight,
            None,
            weight_granularity,
            quantized_dtype,
        )
        if prequantized:
            quantized_inputs, input_scale = quantize(
                inputs,
                quantized_dtype,
                input_granularity,
                axis=-1,
            )

            def operation() -> Tensor:
                return quantized_linear(
                    quantized_inputs,
                    input_scale,
                    out_dtype=original_dtype,
                )

        else:

            def operation() -> Tensor:
                return quantized_linear(
                    inputs,
                    input_granularity,
                    out_dtype=original_dtype,
                )

    benchmark.group = f"quantized-linear-{original_format}"
    benchmark.extra_info.update(
        {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "batch_size": _BATCH_SIZE,
            "in_features": _IN_FEATURES,
            "out_features": _OUT_FEATURES,
            "warmup_rounds": _WARMUP_ROUNDS,
            "benchmark_rounds": _BENCHMARK_ROUNDS,
        }
    )

    def synchronized_linear() -> Tensor:
        output = operation()
        torch.cuda.synchronize()
        return output

    torch.cuda.synchronize()
    output = benchmark.pedantic(
        synchronized_linear,
        iterations=1,
        rounds=_BENCHMARK_ROUNDS,
        warmup_rounds=_WARMUP_ROUNDS,
    )

    assert output.shape == (_BATCH_SIZE, _OUT_FEATURES)
    assert output.dtype is original_dtype
    assert torch.isfinite(output).all()

    reference = F.linear(inputs, weight)
    relative_error = (
        output.float() - reference.float()
    ).norm() / reference.float().norm()
    tolerance = (
        0.0
        if quantized_dtype is None
        else (
            torch.finfo(quantized_dtype).eps
            if quantized_dtype.is_floating_point
            else 4 / DTYPE_MAX[quantized_dtype]
        )
    )
    assert relative_error.item() <= tolerance
