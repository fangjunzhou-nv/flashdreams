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

"""Performance plots for Omnidreams benchmark results."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm

_DEFAULT_INPUT = Path("artifacts/benchmark/omnidreams/benchmark.json")
_DEFAULT_OUTPUT_DIR = Path("artifacts/benchmark/omnidreams")

_ATTENTION_PANELS = (
    ("omnidreams-dit-self-attention", "Self-attention"),
    ("omnidreams-dit-cross-attention", "Cross-attention"),
)
_DIT_BLOCK_PANEL = ("omnidreams-dit-block", "DiT block")
_PIPELINE_GENERATE_PANEL = (
    "omnidreams-full-pipeline-generate",
    "Pipeline generate",
)
_END_TO_END_PANELS = (
    ("omnidreams-dit-network", "Network eval"),
    _PIPELINE_GENERATE_PANEL,
)
_PANELS = (*_ATTENTION_PANELS, _DIT_BLOCK_PANEL, *_END_TO_END_PANELS)

_NATIVE_LABELS = {
    "omnidreams-torch": "PyTorch Omnidreams BF16",
    "cuda": "CUDA cuDNN FP8",
    "cuda-sparge": "CUDA Sparge FP8",
    "cuda-sage3": "CUDA Sage3 BF16",
    "cuda-sage3-fp8": "CUDA Sage3 FP8",
}
"""Implementation labels for configurations without parseable optimized IDs."""

_FUSION_LABELS = {
    "none": "None",
    "fuse-kv": "Fuse KV",
    "full": "Fuse QKV",
}
"""Display labels by stable QKV fusion ID."""

BenchmarkValues = dict[str, dict[str, float]]
Panel = tuple[str, str]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark input and plot output paths.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot Omnidreams median benchmark latency as three figures."
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
    parser.add_argument(
        "--pipeline-generate-only",
        action="store_true",
        help="write only the pipeline generate benchmark figure",
    )
    return parser.parse_args(argv)


def _load_results(input_path: Path) -> tuple[BenchmarkValues, list[str], str]:
    """Load Omnidreams median timings and environment metadata.

    Args:
        input_path: Pytest-benchmark JSON file to parse.

    Returns:
        Median milliseconds by group and implementation, configuration order,
        and plot subtitle.

    Raises:
        SystemExit: The input cannot be read or lacks complete Omnidreams data.
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

    values: BenchmarkValues = {group: {} for group, _ in _PANELS}
    configurations: list[str] = []

    for record in payload["benchmarks"]:
        if not isinstance(record, dict):
            continue
        group = record.get("group")
        if not isinstance(group, str) or group not in values:
            continue
        param = record.get("param")
        if not isinstance(param, str) or not param:
            raise SystemExit(
                f"Benchmark {record.get('name', '<unnamed>')} has no parameter ID"
            )
        implementation = param
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
        if implementation in values[group]:
            raise SystemExit(f"Duplicate benchmark for {group} and {implementation}")

        values[group][implementation] = median * 1000.0
        if implementation not in configurations:
            configurations.append(implementation)

    missing = [group for group, results in values.items() if not results]
    if missing:
        raise SystemExit(
            f"Benchmark JSON {input_path} lacks groups: {', '.join(missing)}"
        )
    return values, configurations, _subtitle(payload)


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


def _backend_label(backend: str) -> str:
    return "cuDNN" if backend == "cudnn" else backend.upper()


def _attention_configuration_label(implementation: str) -> str:
    """Format one attention implementation as a compact axis label.

    Args:
        implementation: Stable benchmark implementation identifier.

    Returns:
        Backend, fusion, TMA, and projection-quantization label.
    """
    if implementation == "omnidreams":
        return "Omnidreams"
    match = re.fullmatch(
        r"(?:optimized|triton)-(cudnn|fa2)-(none|full|fuse-kv)-(no-tma|tma)"
        r"(?:-projection-(float8-e4m3fn))?",
        implementation,
    )
    if match is None:
        return implementation.replace("-", " ")
    backend, fusion, tma, projection = match.groups()
    tma_label = "no TMA" if tma == "no-tma" else "TMA"
    projection_label = "" if projection is None else "\nProjection: FP8 E4M3"
    return (
        f"Optimized {_backend_label(backend)}\n"
        f"{_FUSION_LABELS[fusion]} · {tma_label}{projection_label}"
    )


def _end_to_end_configuration_label(implementation: str) -> str:
    """Format one network or pipeline implementation as an axis label.

    Args:
        implementation: Stable benchmark implementation identifier.

    Returns:
        Implementation, self-attention, and cross-attention label.
    """
    native_label = _NATIVE_LABELS.get(implementation)
    if native_label is not None:
        return native_label

    selected = re.fullmatch(
        r"(?:optimized|triton)-(cudnn|fa2)-(bf16)-self-(none|full|fuse-kv)-"
        r"(no-tma|tma)-cross-(none|fuse-kv)-(no-tma|tma)",
        implementation,
    )
    if selected is not None:
        backend, precision, self_fusion, self_tma, cross_fusion, cross_tma = (
            selected.groups()
        )
        self_tma_label = "no TMA" if self_tma == "no-tma" else "TMA"
        cross_tma_label = "no TMA" if cross_tma == "no-tma" else "TMA"
        return (
            f"Optimized {_backend_label(backend)} {precision.upper()}\n"
            f"Self: {_FUSION_LABELS[self_fusion]} · {self_tma_label}\n"
            f"Cross: {_FUSION_LABELS[cross_fusion]} · {cross_tma_label}"
        )

    legacy = re.fullmatch(
        r"(?:optimized|triton)-(cudnn|fa2)-(bf16)-(none|full|fuse-kv)"
        r"(-omnidreams-cross)?",
        implementation,
    )
    if legacy is None:
        return implementation.replace("-", " ")
    backend, precision, self_fusion, omnidreams_cross = legacy.groups()
    cross_fusion = "fuse-kv" if self_fusion == "full" else self_fusion
    cross_label = (
        "Omnidreams · None"
        if omnidreams_cross is not None
        else f"{_FUSION_LABELS[cross_fusion]} · TMA"
    )
    return (
        f"Optimized {_backend_label(backend)} {precision.upper()}\n"
        f"Self: {_FUSION_LABELS[self_fusion]} · TMA\n"
        f"Cross: {cross_label}"
    )


def _bar_label(
    latency_ms: float,
    reference_ms: float,
    *,
    is_reference: bool,
) -> str:
    """Format latency and relative performance against a reference.

    Args:
        latency_ms: Bar latency in milliseconds.
        reference_ms: Reference latency in milliseconds.
        is_reference: Whether the bar or cell is the reference.

    Returns:
        Two-line latency and relative-performance annotation.
    """
    if latency_ms < 1:
        latency_label = f"{latency_ms:.3f} ms"
    elif latency_ms < 10:
        latency_label = f"{latency_ms:.2f} ms"
    else:
        latency_label = f"{latency_ms:.1f} ms"
    if is_reference:
        return f"{latency_label}\nreference"
    if latency_ms < reference_ms:
        return f"{latency_label}\n{(1 - latency_ms / reference_ms) * 100:.0f}% faster"
    if latency_ms > reference_ms:
        return f"{latency_label}\n{(latency_ms / reference_ms - 1) * 100:.0f}% slower"
    return f"{latency_label}\nsame as reference"


def _panel_configurations(
    panels: tuple[Panel, ...],
    values: BenchmarkValues,
    configurations: list[str],
) -> list[str]:
    return [
        configuration
        for configuration in configurations
        if any(configuration in values[group] for group, _ in panels)
    ]


def _write_bar_figure(
    output_path: Path,
    title: str,
    panels: tuple[Panel, ...],
    values: BenchmarkValues,
    configurations: list[str],
    subtitle: str,
) -> None:
    """Write aligned median-latency bar charts as one PNG.

    Args:
        output_path: Destination PNG file.
        title: Figure title.
        panels: Benchmark group and display-title pairs.
        values: Median milliseconds by group and implementation.
        configurations: Stable implementation order.
        subtitle: Benchmark environment summary.
    """
    end_to_end = all(panel in _END_TO_END_PANELS for panel in panels)
    reference_configuration = "omnidreams-torch" if end_to_end else "omnidreams"
    configuration_label = (
        _end_to_end_configuration_label
        if end_to_end
        else _attention_configuration_label
    )
    panel_configurations = _panel_configurations(panels, values, configurations)
    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(
            max(16.0, len(panel_configurations) * 1.3),
            len(panels) * 5.0,
        ),
        layout="constrained",
    )
    axes_list = [axes] if len(panels) == 1 else list(axes)
    for axes_item, (group, panel_title) in zip(axes_list, panels, strict=True):
        reference = values[group].get(reference_configuration)
        if reference is None:
            raise SystemExit(
                f"Benchmark results for {group} have no {reference_configuration} reference"
            )
        ordered_configurations = sorted(
            panel_configurations,
            key=lambda configuration: values[group].get(configuration, math.inf),
        )
        latencies = [
            values[group].get(configuration, math.nan)
            for configuration in ordered_configurations
        ]
        colors = [
            "C2"
            if configuration == reference_configuration
            else "C1"
            if configuration.startswith("cuda")
            else "C0"
            for configuration in ordered_configurations
        ]
        bars = axes_item.bar(
            range(len(ordered_configurations)),
            latencies,
            color=colors,
        )
        finite_latencies = [value for value in latencies if math.isfinite(value)]
        axes_item.set_ylim(0, max(finite_latencies) * 1.2)
        axes_item.set_title(panel_title)
        axes_item.set_ylabel("Median latency (ms)")

        for index, (bar, latency, configuration) in enumerate(
            zip(bars, latencies, ordered_configurations, strict=True)
        ):
            if not math.isfinite(latency):
                bar.set_visible(False)
                axes_item.text(
                    index,
                    0.02,
                    "N/A",
                    transform=axes_item.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                )
                continue
            axes_item.annotate(
                _bar_label(
                    latency,
                    reference,
                    is_reference=configuration == reference_configuration,
                ),
                xy=(bar.get_x() + bar.get_width() / 2, latency),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
        axes_item.set_xticks(
            range(len(ordered_configurations)),
            labels=[
                configuration_label(configuration)
                for configuration in ordered_configurations
            ],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

    figure.supxlabel(
        "Configuration (self-attention + cross-attention)"
        if end_to_end
        else "Configuration (SDPA backend / QKV fusion / TMA / projection dtype)"
    )
    figure.suptitle(f"{title}\n{subtitle}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def _split_block_implementation(
    implementation: str,
) -> tuple[str, str]:
    """Split one DiT block ID into self- and cross-attention implementations.

    Args:
        implementation: Stable DiT block benchmark identifier.

    Returns:
        Self- and cross-attention implementation identifiers.

    Raises:
        SystemExit: The identifier does not encode both implementations.
    """
    prefix = "self-"
    if not implementation.startswith(prefix):
        raise SystemExit(f"Unsupported DiT block parameter {implementation!r}")
    self_implementation, separator, cross_implementation = implementation[
        len(prefix) :
    ].partition("-cross-")
    if not separator or not self_implementation or not cross_implementation:
        raise SystemExit(f"Unsupported DiT block parameter {implementation!r}")
    return self_implementation, cross_implementation


def _write_block_figure(
    output_path: Path,
    values: BenchmarkValues,
    subtitle: str,
) -> None:
    """Write the DiT block self-by-cross implementation heatmap.

    Args:
        output_path: Destination PNG file.
        values: Median milliseconds by group and implementation.
        subtitle: Benchmark environment summary.
    """
    block_values = values[_DIT_BLOCK_PANEL[0]]
    self_configurations: list[str] = []
    cross_configurations: list[str] = []
    latencies: dict[tuple[str, str], float] = {}
    for implementation, latency in block_values.items():
        self_implementation, cross_implementation = _split_block_implementation(
            implementation
        )
        cell = (self_implementation, cross_implementation)
        if cell in latencies:
            raise SystemExit(
                "Duplicate DiT block benchmark for "
                f"{self_implementation} and {cross_implementation}"
            )
        latencies[cell] = latency
        if self_implementation not in self_configurations:
            self_configurations.append(self_implementation)
        if cross_implementation not in cross_configurations:
            cross_configurations.append(cross_implementation)

    reference_key = ("omnidreams", "omnidreams")
    reference = latencies.get(reference_key)
    if reference is None:
        raise SystemExit("DiT block results have no all-Omnidreams reference")

    matrix = [
        [
            latencies.get((self_implementation, cross_implementation), math.nan)
            / reference
            for cross_implementation in cross_configurations
            # ponytail: Linear sizing assumes current short labels; measure rendered
            # text extents if benchmark labels become substantially longer.
        ]
        for self_implementation in self_configurations
    ]
    figure, axes = plt.subplots(
        figsize=(
            max(12.0, len(cross_configurations) * 2.0),
            max(8.0, len(self_configurations) * 0.9),
        ),
        layout="constrained",
    )
    image = axes.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn_r",
        norm=CenteredNorm(vcenter=1.0),
    )
    axes.set_xticks(
        range(len(cross_configurations)),
        labels=[
            _attention_configuration_label(configuration).removeprefix("Optimized ")
            for configuration in cross_configurations
        ],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axes.set_yticks(
        range(len(self_configurations)),
        labels=[
            _attention_configuration_label(configuration).removeprefix("Optimized ")
            for configuration in self_configurations
        ],
    )
    axes.set_xlabel("Cross-attention implementation")
    axes.set_ylabel("Self-attention implementation")
    axes.set_title(f"Omnidreams DiT block performance\n{subtitle}")

    for self_index, self_implementation in enumerate(self_configurations):
        for cross_index, cross_implementation in enumerate(cross_configurations):
            latency = latencies.get((self_implementation, cross_implementation))
            if latency is None:
                label = "N/A"
            else:
                label = _bar_label(
                    latency,
                    reference,
                    is_reference=(
                        self_implementation,
                        cross_implementation,
                    )
                    == reference_key,
                )
            relative_value = matrix[self_index][cross_index]
            text_color = "black"
            if math.isfinite(relative_value):
                red, green, blue, _ = image.cmap(image.norm(np.asarray(relative_value)))
                linear = tuple(
                    channel / 12.92
                    if channel <= 0.04045
                    else ((channel + 0.055) / 1.055) ** 2.4
                    for channel in (red, green, blue)
                )
                luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
                text_color = "white" if luminance < 0.179 else "black"
            axes.text(
                cross_index,
                self_index,
                label,
                ha="center",
                va="center",
                color=text_color,
            )

    colorbar = figure.colorbar(
        image,
        ax=axes,
        label="Runtime relative to all-Omnidreams reference (×)",
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    """Generate the three Omnidreams benchmark figures."""
    args = _parse_args(argv)
    values, configurations, subtitle = _load_results(args.input)
    if args.pipeline_generate_only:
        output_path = args.output_dir / "pipeline_generate.png"
        _write_bar_figure(
            output_path,
            "Omnidreams pipeline generate benchmark",
            (_PIPELINE_GENERATE_PANEL,),
            values,
            configurations,
            subtitle,
        )
        print(f"Wrote {output_path}")
        return

    attention_output = args.output_dir / "modules.png"
    _write_bar_figure(
        attention_output,
        "Omnidreams attention module benchmarks",
        _ATTENTION_PANELS,
        values,
        configurations,
        subtitle,
    )
    print(f"Wrote {attention_output}")

    block_output = args.output_dir / "dit_block.png"
    _write_block_figure(block_output, values, subtitle)
    print(f"Wrote {block_output}")

    end_to_end_output = args.output_dir / "network_pipeline.png"
    _write_bar_figure(
        end_to_end_output,
        "Omnidreams network and pipeline benchmarks",
        _END_TO_END_PANELS,
        values,
        configurations,
        subtitle,
    )
    print(f"Wrote {end_to_end_output}")


if __name__ == "__main__":
    main()
