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

"""CPU tests for effective-precision quantization benchmark grouping."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import torch

from flashdreams.accelerated.quantization.linear import WeightGranularity
from flashdreams.accelerated.quantization.quantizer import Granularity
from scripts.benchmark.flashdreams.accelerated.quantization.plot_gemm import (
    _load_matrix as load_gemm_matrix,
    main as plot_gemm,
)
from scripts.benchmark.flashdreams.accelerated.quantization.plot_quantized_linear import (
    _load_matrix as load_linear_matrix,
    main as plot_linear,
)

pytestmark = pytest.mark.ci_cpu

_REPO_ROOT = Path(__file__).parents[5]


def _benchmark_globals(filename: str) -> dict[str, object]:
    return runpy.run_path(
        _REPO_ROOT
        / "flashdreams"
        / "benchmarks"
        / "accelerated"
        / "quantization"
        / filename
    )


def _record(group: str, param: str, median: float) -> dict[str, object]:
    return {"group": group, "param": param, "stats": {"median": median}}


def test_effective_gemm_dtype_classification() -> None:
    gemm = _benchmark_globals("test_quantized_gemm_benchmark.py")
    effective_gemm_dtype = gemm["_effective_gemm_dtype"]

    assert callable(effective_gemm_dtype)
    assert (
        effective_gemm_dtype(torch.float32, torch.float8_e4m3fn, Granularity.SLICE)
        is torch.bfloat16
    )
    assert (
        effective_gemm_dtype(torch.float32, torch.float8_e4m3fn, Granularity.TENSOR)
        is torch.float32
    )
    assert (
        effective_gemm_dtype(torch.float32, torch.int8, Granularity.SLICE)
        is torch.float32
    )


def test_effective_linear_dtype_classification() -> None:
    linear = _benchmark_globals("test_quantized_linear_benchmark.py")
    effective_gemm_dtype = linear["_effective_gemm_dtype"]

    assert callable(effective_gemm_dtype)
    assert (
        effective_gemm_dtype(
            torch.float32,
            torch.float8_e4m3fn,
            WeightGranularity.PER_OUT_CHANNEL,
            Granularity.TENSOR,
        )
        is torch.bfloat16
    )
    assert (
        effective_gemm_dtype(
            torch.float32,
            torch.float8_e4m3fn,
            WeightGranularity.TENSOR,
            Granularity.SLICE,
        )
        is torch.bfloat16
    )
    assert (
        effective_gemm_dtype(
            torch.float32,
            torch.float8_e4m3fn,
            WeightGranularity.TENSOR,
            Granularity.TENSOR,
        )
        is torch.float32
    )
    assert (
        effective_gemm_dtype(
            torch.float32,
            torch.int8,
            WeightGranularity.PER_OUT_CHANNEL,
            Granularity.SLICE,
        )
        is torch.float32
    )


def test_plot_rows_preserve_source_and_effective_dtypes(tmp_path: Path) -> None:
    gemm_configuration = "float8_e4m3fn-slice"
    linear_configuration = (
        "float8_e4m3fn-weight-per_out_channel-input-tensor-full-precision-x"
    )
    input_path = tmp_path / "benchmark.json"
    input_path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    _record(
                        "quantized-gemm-bf16-end-to-end",
                        "bf16-full-precision",
                        0.002,
                    ),
                    _record(
                        "quantized-gemm-bf16-end-to-end",
                        f"fp32-{gemm_configuration}",
                        0.001,
                    ),
                    _record("quantized-linear-bf16", "bf16-nn-linear", 0.003),
                    _record(
                        "quantized-linear-bf16",
                        f"fp32-{linear_configuration}",
                        0.0015,
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )

    gemm_rows, _, gemm_values, _ = load_gemm_matrix(input_path)
    linear_rows, _, linear_values, _ = load_linear_matrix(input_path)

    gemm_moved_row = ("end-to-end", "bf16", "fp32")
    linear_moved_row = ("bf16", "fp32")
    assert gemm_rows == [
        ("end-to-end", "bf16", "bf16"),
        gemm_moved_row,
    ]
    assert gemm_values[(gemm_moved_row, "full-precision")] == pytest.approx(2.0)
    assert gemm_values[(gemm_moved_row, gemm_configuration)] == pytest.approx(1.0)
    assert linear_rows == [("bf16", "bf16"), linear_moved_row]
    assert linear_values[(linear_moved_row, "nn-linear")] == pytest.approx(3.0)
    assert linear_values[(linear_moved_row, linear_configuration)] == pytest.approx(1.5)

    gemm_output_dir = tmp_path / "gemm"
    linear_output_dir = tmp_path / "linear"
    plot_gemm([str(input_path), "--output-dir", str(gemm_output_dir)])
    plot_linear([str(input_path), "--output-dir", str(linear_output_dir)])

    assert (gemm_output_dir / "end_to_end.png").is_file()
    assert (linear_output_dir / "quantized_linear_quantize_input.png").is_file()
