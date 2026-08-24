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

"""CPU tests for shared benchmark plotting helpers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from scripts.benchmark.common import (
    add_plot_io_arguments,
    annotate_relative_cell,
    benchmark_median_ms,
    benchmark_subtitle,
    draw_relative_heatmap,
    load_benchmark_json,
    percentage_latency_label,
    relative_matrix_with_fastest,
    save_figure,
)

pytestmark = pytest.mark.ci_cpu


def test_load_benchmark_json_and_standard_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "benchmark.json"
    input_path.write_text(
        json.dumps(
            {
                "benchmarks": [{"name": "demo", "stats": {"median": 0.012}}],
                "commit_info": {"id": "0123456789abcdef"},
                "datetime": "2026-08-24T12:00:00",
            }
        ),
        encoding="utf-8",
    )

    payload, records = load_benchmark_json(input_path)

    record = records[0]
    assert isinstance(record, dict)
    assert benchmark_median_ms(record) == pytest.approx(12.0)
    assert benchmark_subtitle(payload) == (
        "Median latency in ms (lower is faster) · commit 0123456789 · "
        "2026-08-24T12:00:00"
    )


@pytest.mark.parametrize("median", [True, 0, -1, math.inf, math.nan, "1"])
def test_benchmark_median_ms_rejects_invalid_values(median: object) -> None:
    with pytest.raises(SystemExit, match="has no positive median"):
        benchmark_median_ms({"name": "invalid", "stats": {"median": median}})


def test_load_benchmark_json_requires_benchmarks_list(tmp_path: Path) -> None:
    input_path = tmp_path / "benchmark.json"
    input_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="has no benchmarks list"):
        load_benchmark_json(input_path)


def test_add_plot_io_arguments_uses_defaults_and_overrides(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    default_input = tmp_path / "benchmark.json"
    default_output = tmp_path / "plots"
    add_plot_io_arguments(parser, default_input, default_output)

    defaults = parser.parse_args([])
    overrides = parser.parse_args(["custom.json", "--output-dir", "custom-plots"])

    assert defaults.input == default_input
    assert defaults.output_dir == default_output
    assert overrides.input == Path("custom.json")
    assert overrides.output_dir == Path("custom-plots")


def test_relative_matrix_tracks_fastest_candidate_and_missing_reference() -> None:
    rows = ("complete", "missing")
    columns = ("reference", "slow", "fast")
    values = {
        ("complete", "reference"): 10.0,
        ("complete", "slow"): 12.0,
        ("complete", "fast"): 8.0,
        ("missing", "fast"): 4.0,
    }

    matrix, fastest = relative_matrix_with_fastest(rows, columns, values, "reference")

    assert matrix[0] == pytest.approx([1.0, 1.2, 0.8, 0.8])
    assert all(math.isnan(value) for value in matrix[1])
    assert fastest == {"complete": "fast", "missing": "fast"}


def test_percentage_latency_label_covers_reference_and_candidate() -> None:
    assert percentage_latency_label(10.0, 10.0, is_reference=True) == (
        "10.00 ms\n1.00× reference"
    )
    assert percentage_latency_label(8.0, 10.0, is_reference=False) == (
        "8.00 ms\n20% faster"
    )
    assert percentage_latency_label(None, 10.0, is_reference=False) == "N/A"


def test_heatmap_annotation_and_save(tmp_path: Path) -> None:
    figure, axes = plt.subplots()
    image = draw_relative_heatmap(axes, [[1.0, 2.0]])

    reference = annotate_relative_cell(axes, image, 0, 0, "reference", 1.0)
    slower = annotate_relative_cell(axes, image, 0, 1, "slower", 2.0)
    output_path = tmp_path / "nested" / "heatmap.png"
    save_figure(figure, output_path)

    assert reference.get_color() in {"black", "white"}
    assert slower.get_color() in {"black", "white"}
    assert output_path.is_file()
