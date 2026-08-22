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

"""Performance matrices for Torch and Triton quantizer benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm

_DEFAULT_OUTPUT_DIR = Path("artifacts/benchmark/flashdreams/accelerated/quantization")
_DEFAULT_INPUT = _DEFAULT_OUTPUT_DIR / "benchmark.json"

_OPERATIONS = ("quantize", "dequantize")
_FORMATS = ("float8_e4m3fn", "float8_e5m2", "int8")
_GRANULARITIES = ("slice", "tensor")
_IMPLEMENTATIONS = ("torch", "triton")
_COLUMNS = tuple(
    (granularity, implementation)
    for granularity in _GRANULARITIES
    for implementation in _IMPLEMENTATIONS
)
_GROUP_PATTERN = re.compile(
    r"(?P<operation>quantize|dequantize)-"
    r"(?P<format>float8_e4m3fn|float8_e5m2|int8)-"
    r"(?P<granularity>slice|tensor)"
)
_FORMAT_LABELS = {
    "float8_e4m3fn": "FP8 E4M3",
    "float8_e5m2": "FP8 E5M2",
    "int8": "INT8",
}

CellKey = tuple[str, str, str, str]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark input and plot output paths.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot Torch and Triton quantizer median latency as PNG matrices."
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


def _load_results(input_path: Path) -> tuple[dict[CellKey, float], str]:
    """Load quantizer median timings from benchmark JSON.

    Args:
        input_path: Pytest-benchmark JSON file to parse.

    Returns:
        Median milliseconds by operation, format, granularity, and
        implementation, plus the plot subtitle.

    Raises:
        SystemExit: The input cannot be read or lacks complete quantizer data.
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

    values_ms: dict[CellKey, float] = {}
    for record in payload["benchmarks"]:
        if not isinstance(record, dict):
            continue
        group = record.get("group")
        match = _GROUP_PATTERN.fullmatch(group) if isinstance(group, str) else None
        if match is None:
            continue

        extra_info = record.get("extra_info")
        implementation = (
            extra_info.get("implementation") if isinstance(extra_info, dict) else None
        )
        if implementation not in _IMPLEMENTATIONS:
            raise SystemExit(
                f"Benchmark {record.get('name', '<unnamed>')} has no supported "
                "implementation metadata"
            )

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

        cell = (
            match.group("operation"),
            match.group("format"),
            match.group("granularity"),
            implementation,
        )
        if cell in values_ms:
            raise SystemExit(f"Duplicate quantizer benchmark cell for {cell}")
        values_ms[cell] = median * 1000.0

    missing_operations = [
        operation
        for operation in _OPERATIONS
        if not any(cell[0] == operation for cell in values_ms)
    ]
    if missing_operations:
        raise SystemExit(
            f"Benchmark JSON {input_path} lacks quantizer operations: "
            f"{', '.join(missing_operations)}"
        )
    return values_ms, _subtitle(payload)


def _subtitle(payload: dict[str, object]) -> str:
    """Build a compact benchmark-environment subtitle."""
    parts = ["Median latency in ms (lower is faster)"]
    commit_info = payload.get("commit_info")
    commit_id = commit_info.get("id") if isinstance(commit_info, dict) else None
    if isinstance(commit_id, str) and commit_id:
        parts.append(f"commit {commit_id[:10]}")
    timestamp = payload.get("datetime")
    if isinstance(timestamp, str) and timestamp:
        parts.append(timestamp)
    return " · ".join(parts)


def _cell_label(
    value: float | None, reference: float | None, *, is_reference: bool
) -> str:
    """Format a latency and its relationship to the Torch reference.

    Args:
        value: Cell latency in milliseconds; ``None`` marks a missing result.
        reference: Torch latency for the same format and granularity.
        is_reference: Whether the cell is the Torch reference.

    Returns:
        Two-line latency and relative-performance annotation.
    """
    if value is None:
        return "N/A"
    if is_reference:
        return f"{value:.3f} ms\n1.00× reference"
    if reference is None:
        return f"{value:.3f} ms\nreference unavailable"
    if value < reference:
        return f"{value:.3f} ms\n{reference / value:.2f}× faster"
    if value > reference:
        return f"{value:.3f} ms\n{value / reference:.2f}× slower"
    return f"{value:.3f} ms\nsame as reference"


def _write_png(
    output_path: Path,
    operation: str,
    values_ms: dict[CellKey, float],
    subtitle: str,
) -> None:
    """Write one operation's Torch-versus-Triton latency heatmap.

    Args:
        output_path: Destination PNG file.
        operation: Quantizer operation represented by the matrix.
        values_ms: Median milliseconds keyed by benchmark configuration.
        subtitle: Benchmark environment summary.

    Raises:
        SystemExit: Any format and granularity pair lacks its Torch reference.
    """
    missing_references = [
        (format, granularity)
        for format in _FORMATS
        for granularity in _GRANULARITIES
        if (operation, format, granularity, "torch") not in values_ms
    ]
    if missing_references:
        raise SystemExit(
            f"{operation.capitalize()} results lack Torch references: "
            + ", ".join(
                f"{format}/{granularity}" for format, granularity in missing_references
            )
        )

    matrix: list[list[float]] = []
    for format in _FORMATS:
        row = []
        for granularity, implementation in _COLUMNS:
            reference = values_ms[(operation, format, granularity, "torch")]
            value = values_ms.get((operation, format, granularity, implementation))
            row.append(value / reference if value is not None else math.nan)
        matrix.append(row)

    figure, axes = plt.subplots()
    default_width, default_height = figure.get_size_inches()
    figure.set_size_inches(
        max(default_width, len(_COLUMNS) * 1.8),
        max(default_height, len(_FORMATS) * 1.2),
    )
    image = axes.imshow(
        np.asarray(matrix),
        aspect="auto",
        cmap="RdYlGn_r",
        norm=CenteredNorm(vcenter=1.0),
    )
    axes.set_xticks(
        range(len(_COLUMNS)),
        labels=[
            f"{granularity.capitalize()}\n{implementation.capitalize()}"
            for granularity, implementation in _COLUMNS
        ],
    )
    axes.set_yticks(
        range(len(_FORMATS)),
        labels=[_FORMAT_LABELS[format] for format in _FORMATS],
    )
    axes.set_xlabel("Scale granularity and implementation")
    axes.set_ylabel("Quantized dtype")
    axes.set_title(f"{operation.capitalize()} · Torch vs Triton\n{subtitle}")

    for row_index, format in enumerate(_FORMATS):
        for column_index, (granularity, implementation) in enumerate(_COLUMNS):
            reference = values_ms[(operation, format, granularity, "torch")]
            value = values_ms.get((operation, format, granularity, implementation))
            label = _cell_label(
                value,
                reference,
                is_reference=implementation == "torch",
            )

            text_color = "black"
            relative_value = matrix[row_index][column_index]
            if math.isfinite(relative_value):
                colors = image.cmap(image.norm(np.asarray([relative_value])))
                red, green, blue, _ = np.asarray(colors)[0]
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

    colorbar = figure.colorbar(image, ax=axes, label="Runtime relative to Torch (×)")
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
    """Generate separate quantize and dequantize performance matrices."""
    args = _parse_args(argv)
    values_ms, subtitle = _load_results(args.input)
    for operation in _OPERATIONS:
        output = args.output_dir / f"{operation}.png"
        _write_png(output, operation, values_ms, subtitle)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
