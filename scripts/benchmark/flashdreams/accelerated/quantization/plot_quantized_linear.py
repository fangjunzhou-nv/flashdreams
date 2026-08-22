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

"""Performance matrices for quantized linear benchmark results."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

_DEFAULT_OUTPUT_DIR = Path("artifacts/benchmark/flashdreams/accelerated/quantization")
_DEFAULT_INPUT = _DEFAULT_OUTPUT_DIR / "benchmark.json"
_FASTEST_CONFIG_COLUMN = "fastest-quantized-config"
_REFERENCE_CONFIG = "nn-linear"
"""Benchmark parameter ID used as each row's performance reference."""

_CONFIG_PATTERN = re.compile(
    r"(?P<format>float8_e4m3fn|float8_e5m2-x-float8_e4m3fn|int8)-"
    r"weight-(?P<weight>per_out_channel|tensor)-"
    r"input-(?P<input>slice|tensor)-"
    r"(?P<input_state>full-precision-x|prequantized-x)"
)
_QUANTIZED_FORMAT_LABELS = {
    "float8_e4m3fn": "FP8 E4M3",
    "float8_e5m2-x-float8_e4m3fn": "FP8 E5M2 × E4M3",
    "int8": "INT8",
}
_INPUT_STATE_LABELS = {
    "full-precision-x": "Quantize input",
    "prequantized-x": "Prequantized input",
}
_INPUT_STATE_OUTPUT_NAMES = {
    "full-precision-x": "quantize_input",
    "prequantized-x": "prequantized_input",
}
_SOURCE_FORMAT_LABELS = {"fp16": "FP16", "bf16": "BF16", "fp32": "FP32"}

CellKey = tuple[str, str]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark input and plot output paths.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot quantized linear median latency as PNG matrices."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"pytest-benchmark JSON path (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def _parse_configuration(configuration: str) -> re.Match[str]:
    """Parse a quantized linear configuration ID.

    Args:
        configuration: Pytest parameter suffix excluding the source dtype.

    Returns:
        Match containing quantized format, weight and input granularities, and
        input state.

    Raises:
        SystemExit: The configuration ID is unsupported.
    """
    match = _CONFIG_PATTERN.fullmatch(configuration)
    if match is None:
        raise SystemExit(
            f"Unsupported quantized linear configuration {configuration!r}"
        )
    return match


def _load_matrix(
    input_path: Path,
) -> tuple[list[str], list[str], dict[CellKey, float], str]:
    """Load quantized linear rows and median timings from benchmark JSON.

    Args:
        input_path: Pytest-benchmark JSON file to parse.

    Returns:
        Ordered source-format rows, configuration columns, median milliseconds
        by cell, and plot subtitle.

    Raises:
        SystemExit: The input cannot be read or does not contain compatible
            quantized linear records.
    """
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SystemExit(f"Cannot read benchmark JSON {input_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"Cannot parse benchmark JSON {input_path}: {error}"
        ) from error

    if not isinstance(payload, dict) or not isinstance(payload.get("benchmarks"), list):
        raise SystemExit(f"Benchmark JSON {input_path} has no benchmarks list")

    rows: list[str] = []
    columns: list[str] = []
    values_ms: dict[CellKey, float] = {}

    for record in payload["benchmarks"]:
        if not isinstance(record, dict):
            continue
        group = record.get("group")
        param = record.get("param")
        if not isinstance(group, str) or not isinstance(param, str):
            continue
        match = re.fullmatch(r"quantized-linear-(fp16|bf16|fp32)", group)
        if match is None:
            continue
        source_format = match.group(1)
        prefix = f"{source_format}-"
        if not param.startswith(prefix):
            raise SystemExit(f"Unsupported quantized linear parameter {param!r}")
        configuration = param.removeprefix(prefix)
        if configuration != _REFERENCE_CONFIG:
            _parse_configuration(configuration)

        stats = record.get("stats")
        median = stats.get("median") if isinstance(stats, dict) else None
        if (
            isinstance(median, bool)
            or not isinstance(median, (int, float))
            or not math.isfinite(median)
            or median <= 0
        ):
            raise SystemExit(
                f"Benchmark {record.get('name', '<unnamed>')} has no positive median"
            )

        cell = (source_format, configuration)
        if cell in values_ms:
            raise SystemExit(
                f"Duplicate benchmark cell for {source_format} and {configuration}"
            )
        if source_format not in rows:
            rows.append(source_format)
        if configuration not in columns:
            columns.append(configuration)
        values_ms[cell] = median * 1000.0

    if not values_ms:
        raise SystemExit(
            f"Benchmark JSON {input_path} contains no quantized linear results"
        )

    subtitle_parts = ["Median latency in ms (lower is faster)"]
    commit_info = payload.get("commit_info")
    commit_id = commit_info.get("id") if isinstance(commit_info, dict) else None
    if isinstance(commit_id, str) and commit_id:
        subtitle_parts.append(f"commit {commit_id[:10]}")
    timestamp = payload.get("datetime")
    if isinstance(timestamp, str) and timestamp:
        subtitle_parts.append(timestamp)
    return rows, columns, values_ms, " · ".join(subtitle_parts)


def _config_label(configuration: str) -> str:
    """Return a compact display label for a benchmark configuration."""
    if configuration == _REFERENCE_CONFIG:
        return "nn.Linear"
    if configuration == _FASTEST_CONFIG_COLUMN:
        return "Fastest quantized\nconfig"
    match = _parse_configuration(configuration)
    quantized_format = _QUANTIZED_FORMAT_LABELS[match.group("format")]
    weight = (
        "weight per output channel"
        if match.group("weight") == "per_out_channel"
        else "weight tensor"
    )
    return f"{quantized_format}\n{weight}\ninput {match.group('input')}"


def _cell_label(
    value: float | None, reference: float | None, *, is_reference: bool
) -> str:
    """Format a latency and its relationship to the row reference.

    Args:
        value: Cell latency in milliseconds; ``None`` marks a missing result.
        reference: ``nn.Linear`` latency in milliseconds; ``None`` marks a
            missing reference.
        is_reference: Whether the cell is the ``nn.Linear`` row reference.

    Returns:
        Two-line latency and relative-performance annotation.
    """
    if value is None:
        return "N/A"
    if is_reference:
        return f"{value:.2f} ms\n1.00× reference"
    if reference is None:
        return f"{value:.2f} ms\nreference unavailable"
    if value < reference:
        return f"{value:.2f} ms\n{(1 - value / reference) * 100:.0f}% faster"
    if value > reference:
        return f"{value:.2f} ms\n{(value / reference - 1) * 100:.0f}% slower"
    return f"{value:.2f} ms\nsame as reference"


def _write_png(
    output_path: Path,
    input_state: str,
    rows: list[str],
    columns: list[str],
    values_ms: dict[CellKey, float],
    subtitle: str,
) -> None:
    """Write one input state's median-latency heatmap as a PNG.

    Args:
        output_path: Destination PNG file.
        input_state: Whether input quantization occurs inside the timed region.
        rows: Ordered source-format rows.
        columns: Ordered linear configuration columns.
        values_ms: Median milliseconds keyed by row and configuration.
        subtitle: Benchmark environment summary.

    Raises:
        SystemExit: Results have no complete ``nn.Linear`` reference column.
    """
    if _REFERENCE_CONFIG not in columns or any(
        (row, _REFERENCE_CONFIG) not in values_ms for row in rows
    ):
        raise SystemExit("Benchmark results have no complete nn.Linear reference")

    display_columns = [*columns, _FASTEST_CONFIG_COLUMN]
    fastest_configurations: dict[str, str | None] = {}
    matrix = []
    for row in rows:
        reference = values_ms[(row, _REFERENCE_CONFIG)]
        fastest = min(
            (
                column
                for column in columns
                if column != _REFERENCE_CONFIG and (row, column) in values_ms
            ),
            key=lambda column: values_ms[(row, column)],
            default=None,
        )
        fastest_configurations[row] = fastest
        relative_values = [
            values_ms.get((row, column), math.nan) / reference for column in columns
        ]
        relative_values.append(
            values_ms[(row, fastest)] / reference if fastest is not None else math.nan
        )
        matrix.append(relative_values)

    figure, axes = plt.subplots()
    default_width, default_height = figure.get_size_inches()
    # ponytail: Linear sizing assumes current short labels; measure rendered
    # text extents if benchmark labels become substantially longer.
    figure.set_size_inches(
        max(default_width, len(display_columns) * 1.9),
        max(default_height, len(rows) * 2.0),
    )
    image = axes.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn_r",
        norm=CenteredNorm(vcenter=1.0),
    )
    axes.set_xticks(
        range(len(display_columns)),
        labels=[_config_label(column) for column in display_columns],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axes.axvline(len(columns) - 0.5, color="black", linewidth=1.5)
    axes.set_yticks(
        range(len(rows)),
        labels=[_SOURCE_FORMAT_LABELS.get(row, row) for row in rows],
    )
    axes.set_xlabel("Linear configuration")
    axes.set_ylabel("Source dtype")
    axes.set_title(f"Quantized linear · {_INPUT_STATE_LABELS[input_state]}\n{subtitle}")

    for row_index, row in enumerate(rows):
        reference = values_ms[(row, _REFERENCE_CONFIG)]
        for column_index, column in enumerate(display_columns):
            if column == _FASTEST_CONFIG_COLUMN:
                fastest = fastest_configurations[row]
                if fastest is None:
                    value = None
                    label = "N/A"
                else:
                    value = values_ms[(row, fastest)]
                    label = (
                        f"{_config_label(fastest)}\n"
                        f"{_cell_label(value, reference, is_reference=False)}"
                    )
            else:
                value = values_ms.get((row, column))
                label = _cell_label(
                    value,
                    reference,
                    is_reference=column == _REFERENCE_CONFIG,
                )

            text_color = "black"
            relative_value = matrix[row_index][column_index]
            if math.isfinite(relative_value):
                red, green, blue, _ = image.cmap(image.norm(relative_value))
                linear = tuple(
                    channel / 12.92
                    if channel <= 0.04045
                    else ((channel + 0.055) / 1.055) ** 2.4
                    for channel in (red, green, blue)
                )
                luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
                text_color = "white" if luminance < 0.179 else "black"
            axes.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=text_color,
            )

    colorbar = figure.colorbar(
        image, ax=axes, label="Runtime relative to nn.Linear (×)"
    )
    colorbar.ax.text(
        0.5, 1.02, "Slower ↑", ha="center", transform=colorbar.ax.transAxes
    )
    colorbar.ax.text(
        0.5,
        -0.04,
        "↓ Faster",
        ha="center",
        va="top",
        transform=colorbar.ax.transAxes,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    """Generate separate quantize-input and prequantized-input matrices."""
    args = _parse_args(argv)
    rows, columns, values_ms, subtitle = _load_matrix(args.input)
    input_states = list(
        dict.fromkeys(
            _parse_configuration(column).group("input_state")
            for column in columns
            if column != _REFERENCE_CONFIG
        )
    )
    for input_state in input_states:
        panel_columns = [
            column
            for column in columns
            if column == _REFERENCE_CONFIG
            or _parse_configuration(column).group("input_state") == input_state
        ]
        output_name = _INPUT_STATE_OUTPUT_NAMES[input_state]
        output = args.output_dir / f"quantized_linear_{output_name}.png"
        _write_png(
            output,
            input_state,
            rows,
            panel_columns,
            values_ms,
            subtitle,
        )
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
