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

"""Performance matrices for accelerated attention benchmark results."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from matplotlib.patches import Rectangle

_DEFAULT_OUTPUT_DIR = Path(
    "artifacts/benchmark/flashdreams/accelerated/multi_head_attention"
)
_DEFAULT_INPUT = _DEFAULT_OUTPUT_DIR / "benchmark.json"
_FASTEST_IMPLEMENTATION_COLUMN = "fastest-implementation-config"
_HIGHLIGHTED_ROWS = {
    ("head", "before_kv_cache", False, False): "Cosmos",
    ("inner", "before_kv_cache", True, True): "Wan",
}
_REFERENCE_IMPLEMENTATION = "reference-torch"

RowKey = tuple[str, str, str, bool, bool]
CellKey = tuple[RowKey, str]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark input and plot output paths.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot self- and cross-attention median latency as PNG matrices."
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


def _load_matrix(
    input_path: Path,
) -> tuple[list[RowKey], list[str], dict[CellKey, float], str]:
    """Load accelerated attention rows and median timings from benchmark JSON.

    Args:
        input_path: Pytest-benchmark JSON file to parse.

    Returns:
        Ordered row keys, implementation columns, median milliseconds by cell,
        and plot subtitle.

    Raises:
        SystemExit: The input cannot be read or does not contain compatible
            accelerated attention records.
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

    rows: list[RowKey] = []
    columns: list[str] = []
    values_ms: dict[CellKey, float] = {}

    for record in payload["benchmarks"]:
        if not isinstance(record, dict):
            continue
        group = record.get("group")
        param = record.get("param")
        if not isinstance(group, str) or not isinstance(param, str):
            continue
        match = re.fullmatch(
            r"multi-head-attention-(self|cross)-norm-(.+)-rope-"
            r"(interleaved|split)(?:-scope-(before-kv-cache|after-kv-cache))?"
            r"-bias-(on|off)",
            group,
        )
        if match is None:
            continue
        attention, norm, rope, scope_state, bias_state = match.groups()
        attention_type = f"{attention}_attention"
        rope_scope = (scope_state or "before-kv-cache").replace("-", "_")
        rope_interleaved = rope == "interleaved"
        bias = bias_state == "on"
        shared_id = group.removeprefix(f"multi-head-attention-{attention}-")
        implementation_prefix = f"{shared_id}-"
        if not param.startswith(implementation_prefix):
            raise SystemExit(f"Unsupported attention parameter {param!r}")
        implementation = param.removeprefix(implementation_prefix)

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

        row = (attention_type, norm, rope_scope, rope_interleaved, bias)
        cell = (row, implementation)
        if cell in values_ms:
            raise SystemExit(f"Duplicate benchmark cell for {row} and {implementation}")
        if row not in rows:
            rows.append(row)
        if implementation not in columns:
            columns.append(implementation)
        values_ms[cell] = median * 1000.0

    if not values_ms:
        raise SystemExit(
            f"Benchmark JSON {input_path} contains no accelerated attention results"
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


def _row_label(row: RowKey) -> str:
    _, norm, rope_scope, rope_interleaved, bias = row
    rope = "interleaved" if rope_interleaved else "split"
    scope = rope_scope.removesuffix("_kv_cache").replace("_", " ")
    label = f"norm {norm} | rope {rope} {scope} cache | bias {'on' if bias else 'off'}"
    highlight = _HIGHLIGHTED_ROWS.get((norm, rope_scope, rope_interleaved, bias))
    return f"{label} | {highlight}" if highlight is not None else label


def _column_label(column: str) -> str:
    if column == _FASTEST_IMPLEMENTATION_COLUMN:
        return "fastest implementation config"
    return (
        " ".join(
            "fuse qkv" if token == "full" else token for token in column.split("-")
        )
        .replace(" projection ", "\nprojection ")
        .replace(" quantized sdpa", "\nquantized sdpa")
    )


def _cell_label(
    value: float | None, reference: float | None, *, is_reference: bool
) -> str:
    """Format a latency and its relationship to the row reference.

    Args:
        value: Cell latency in milliseconds; ``None`` marks a missing result.
        reference: Torch-reference latency in milliseconds; ``None`` marks a
            missing reference.
        is_reference: Whether the cell is the torch reference in its row.

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
    attention_type: str,
    rows: list[RowKey],
    columns: list[str],
    values_ms: dict[CellKey, float],
    subtitle: str,
) -> None:
    """Write one attention type's median-latency heatmap as a PNG.

    Args:
        output_path: Destination PNG file.
        attention_type: Attention family represented by every row.
        rows: Ordered attention policy rows.
        columns: Ordered implementation configuration columns.
        values_ms: Median milliseconds keyed by row and implementation.
        subtitle: Benchmark environment summary.
    """
    attention_label = attention_type.replace("_", " ").title()
    if _REFERENCE_IMPLEMENTATION not in columns:
        raise SystemExit("Benchmark results have no reference-torch implementation")
    display_columns = [*columns, _FASTEST_IMPLEMENTATION_COLUMN]
    fastest_implementations: dict[RowKey, str | None] = {}
    matrix = []
    for row in rows:
        reference = values_ms.get((row, _REFERENCE_IMPLEMENTATION))
        fastest = min(
            (
                column
                for column in columns
                if column != _REFERENCE_IMPLEMENTATION and (row, column) in values_ms
            ),
            key=lambda column: values_ms[(row, column)],
            default=None,
        )
        fastest_implementations[row] = fastest
        relative_values = [
            values_ms.get((row, column), math.nan) / reference
            if reference is not None
            else math.nan
            for column in columns
        ]
        relative_values.append(
            values_ms[(row, fastest)] / reference
            if reference is not None and fastest is not None
            else math.nan
        )
        matrix.append(relative_values)
    figure, axes = plt.subplots()
    default_width, default_height = figure.get_size_inches()
    # ponytail: Linear sizing assumes current short labels; measure rendered
    # text extents if benchmark labels become substantially longer.
    figure.set_size_inches(
        max(default_width, len(display_columns) * 2.0),
        max(default_height, len(rows) * 0.9),
    )
    image = axes.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn_r",
        norm=CenteredNorm(vcenter=1.0),
    )
    axes.set_xticks(
        range(len(display_columns)),
        labels=[_column_label(column) for column in display_columns],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axes.axvline(len(columns) - 0.5, color="black", linewidth=1.5)
    axes.set_yticks(range(len(rows)), labels=[_row_label(row) for row in rows])
    for row_index, (_, norm, rope_scope, rope_interleaved, bias) in enumerate(rows):
        if (norm, rope_scope, rope_interleaved, bias) in _HIGHLIGHTED_ROWS:
            axes.add_patch(
                Rectangle(
                    (-0.5, row_index - 0.5),
                    len(display_columns),
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.5,
                    clip_on=False,
                    zorder=3,
                )
            )
    axes.set_xlabel("Implementation configuration")
    axes.set_ylabel("Attention configuration")
    axes.set_title(f"{attention_label} performance\n{subtitle}")
    for row_index, row in enumerate(rows):
        reference = values_ms.get((row, _REFERENCE_IMPLEMENTATION))
        for column_index, column in enumerate(display_columns):
            if column == _FASTEST_IMPLEMENTATION_COLUMN:
                fastest = fastest_implementations[row]
                if fastest is None:
                    value = None
                    label = "N/A"
                else:
                    value = values_ms[(row, fastest)]
                    relative_label = _cell_label(
                        value, reference, is_reference=False
                    ).splitlines()[1]
                    implementation, backend, config = _column_label(fastest).split(
                        maxsplit=2
                    )
                    label = f"{implementation} {backend}\n{config}\n{relative_label}"
            else:
                value = values_ms.get((row, column))
                label = _cell_label(
                    value,
                    reference,
                    is_reference=column == _REFERENCE_IMPLEMENTATION,
                )
            text_color = "black"
            relative_value = matrix[row_index][column_index]
            if math.isfinite(relative_value):
                red, green, blue, _ = image.cmap(image.norm(relative_value))
                channels = (red, green, blue)
                linear = tuple(
                    channel / 12.92
                    if channel <= 0.04045
                    else ((channel + 0.055) / 1.055) ** 2.4
                    for channel in channels
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
        image, ax=axes, label="Runtime relative to torch reference (×)"
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
    """Generate separate self- and cross-attention performance matrices."""
    args = _parse_args(argv)
    rows, columns, values_ms, subtitle = _load_matrix(args.input)
    attention_types = list(dict.fromkeys(row[0] for row in rows))
    for attention_type in attention_types:
        panel_rows = [row for row in rows if row[0] == attention_type]
        panel_values = {
            cell: value
            for cell, value in values_ms.items()
            if cell[0][0] == attention_type
        }
        output = args.output_dir / f"{attention_type}.png"
        _write_png(
            output,
            attention_type,
            panel_rows,
            columns,
            panel_values,
            subtitle,
        )
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
