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

"""Combined pipeline-generate benchmark plots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

_DEFAULT_LINGBOT_INPUT = Path("artifacts/benchmark/lingbot/benchmark.json")
_DEFAULT_OMNIDREAMS_INPUT = Path("artifacts/benchmark/omnidreams/benchmark.json")
_DEFAULT_WAN21_INPUT = Path("artifacts/benchmark/wan21/benchmark.json")
_DEFAULT_OUTPUT = Path("artifacts/benchmark/pipeline_generate.png")

_PIPELINE_GROUPS = (
    ("LingBot", "lingbot-full-pipeline-generate", "wan_torch"),
    ("Omnidreams", "omnidreams-full-pipeline-generate", "omnidreams_torch"),
    ("Wan 2.1", "wan21-full-pipeline-generate", "wan_torch"),
)
"""Model labels, pytest-benchmark groups, and reference implementations."""

_NATIVE_LABELS = {
    "wan_torch": "PyTorch Wan BF16",
    "omnidreams_torch": "PyTorch Omnidreams BF16",
    "cuda": "CUDA cuDNN FP8",
    "cuda_sparge": "CUDA Sparge FP8",
    "cuda_sage3": "CUDA Sage3 BF16",
    "cuda_sage3_fp8": "CUDA Sage3 FP8",
}
"""Implementation labels for configurations without parseable Triton IDs."""

_FUSION_LABELS = {
    "none": ("None", "None"),
    "fuse_kv": ("Fuse KV", "Fuse KV"),
    "full": ("Fuse QKV", "Fuse KV"),
    "full_wan_cross": ("Fuse QKV", "None"),
    "full_omnidreams_cross": ("Fuse QKV", "None"),
}
"""Self- and cross-attention labels by stable fusion ID."""

BenchmarkValues = dict[str, float]
PipelineResults = tuple[str, BenchmarkValues, str, str]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark input and plot output paths.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot pipeline-generate latency for all model benchmarks."
    )
    parser.add_argument(
        "--lingbot-input",
        type=Path,
        default=_DEFAULT_LINGBOT_INPUT,
        help=f"LingBot benchmark JSON path (default: {_DEFAULT_LINGBOT_INPUT})",
    )
    parser.add_argument(
        "--omnidreams-input",
        type=Path,
        default=_DEFAULT_OMNIDREAMS_INPUT,
        help=(f"Omnidreams benchmark JSON path (default: {_DEFAULT_OMNIDREAMS_INPUT})"),
    )
    parser.add_argument(
        "--wan21-input",
        type=Path,
        default=_DEFAULT_WAN21_INPUT,
        help=f"Wan 2.1 benchmark JSON path (default: {_DEFAULT_WAN21_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"output PNG path (default: {_DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def _load_results(input_path: Path, group: str) -> tuple[BenchmarkValues, str]:
    """Load positive median timings for one pipeline benchmark group.

    Args:
        input_path: Pytest-benchmark JSON file to parse.
        group: Pipeline-generate benchmark group to select.

    Returns:
        Median seconds by implementation and a compact benchmark context.

    Raises:
        SystemExit: The input cannot be read or has no valid selected results.
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

    values: BenchmarkValues = {}
    for record in payload["benchmarks"]:
        if not isinstance(record, dict) or record.get("group") != group:
            continue
        param = record.get("param")
        if not isinstance(param, str) or not param:
            raise SystemExit(
                f"Benchmark {record.get('name', '<unnamed>')} has no parameter ID"
            )
        implementation = param.replace("-", "_")
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
        if implementation in values:
            raise SystemExit(f"Duplicate benchmark for {group} and {implementation}")
        values[implementation] = median

    if not values:
        raise SystemExit(f"Benchmark JSON {input_path} lacks group: {group}")
    return values, _benchmark_context(payload)


def _benchmark_context(payload: dict[str, object]) -> str:
    """Build a compact commit and timestamp label."""
    parts: list[str] = []
    commit_info = payload.get("commit_info")
    commit_id = commit_info.get("id") if isinstance(commit_info, dict) else None
    if isinstance(commit_id, str) and commit_id:
        parts.append(f"commit {commit_id[:10]}")
    timestamp = payload.get("datetime")
    if isinstance(timestamp, str) and timestamp:
        parts.append(timestamp)
    return " · ".join(parts)


def _configuration_label(implementation: str) -> str:
    """Format a stable implementation name as a compact axis label.

    Args:
        implementation: Stable benchmark implementation identifier.

    Returns:
        Three-line implementation, self-attention, and cross-attention label.
    """
    implementation_label = _NATIVE_LABELS.get(implementation)
    fusion = "none"
    if implementation.startswith("triton_"):
        parts = implementation.removeprefix("triton_").split("_", maxsplit=2)
        if len(parts) == 3:
            backend, precision, fusion = parts
            backend_label = "cuDNN" if backend == "cudnn" else backend.upper()
            implementation_label = f"Triton {backend_label} {precision.upper()}"
    self_fusion, cross_fusion = _FUSION_LABELS.get(fusion, ("None", "None"))
    implementation_label = implementation_label or implementation.replace("_", " ")
    return f"{implementation_label}\nSelf Attn {self_fusion}\nCross Attn {cross_fusion}"


def _bar_label(latency: float, reference: float, *, is_reference: bool) -> str:
    """Format latency and relative performance against the native pipeline."""
    if latency < 1:
        latency_label = f"{latency:.3f} s"
    elif latency < 10:
        latency_label = f"{latency:.2f} s"
    else:
        latency_label = f"{latency:.1f} s"
    if is_reference:
        return f"{latency_label}\nreference"
    if latency < reference:
        return f"{latency_label}\n{(1 - latency / reference) * 100:.0f}% faster"
    if latency > reference:
        return f"{latency_label}\n{(latency / reference - 1) * 100:.0f}% slower"
    return f"{latency_label}\nsame as reference"


def _write_figure(output_path: Path, pipelines: tuple[PipelineResults, ...]) -> None:
    """Write all pipeline-generate median latencies as one PNG.

    Args:
        output_path: Destination PNG file.
        pipelines: Model labels, timings, references, and benchmark contexts.
    """
    width = max(16.0, max(len(values) for _, values, _, _ in pipelines) * 1.3)
    figure, axes = plt.subplots(
        len(pipelines),
        1,
        figsize=(width, len(pipelines) * 5.0),
        layout="constrained",
    )
    axes_list = [axes] if len(pipelines) == 1 else list(axes)
    for axes_item, (model, values, reference_name, context) in zip(
        axes_list, pipelines, strict=True
    ):
        if reference_name not in values:
            raise SystemExit(f"{model} results lack reference: {reference_name}")
        reference = values[reference_name]
        configurations = sorted(values, key=lambda configuration: values[configuration])
        latencies = [values[configuration] for configuration in configurations]
        colors = [
            "C2"
            if configuration == reference_name
            else "C1"
            if configuration.startswith("cuda")
            else "C0"
            for configuration in configurations
        ]
        bars = axes_item.bar(range(len(configurations)), latencies, color=colors)
        axes_item.set_ylim(0, max(latencies) * 1.2)
        axes_item.set_title(f"{model} · {context}" if context else model)
        axes_item.set_ylabel("Median latency (s)")
        axes_item.bar_label(
            bars,
            labels=[
                _bar_label(
                    latency,
                    reference,
                    is_reference=configuration == reference_name,
                )
                for latency, configuration in zip(
                    latencies, configurations, strict=True
                )
            ],
            padding=3,
        )
        axes_item.set_xticks(
            range(len(configurations)),
            labels=[
                _configuration_label(configuration) for configuration in configurations
            ],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

    figure.supxlabel("Configuration (self-attention + cross-attention)")
    figure.suptitle(
        "Pipeline generate benchmarks\nMedian latency in seconds (lower is faster)"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    """Generate the combined pipeline-generate benchmark figure."""
    args = _parse_args(argv)
    input_paths = (
        args.lingbot_input,
        args.omnidreams_input,
        args.wan21_input,
    )
    loaded = (
        (model, *_load_results(input_path, group), reference)
        for input_path, (model, group, reference) in zip(
            input_paths, _PIPELINE_GROUPS, strict=True
        )
    )
    pipelines = tuple(
        (model, values, reference, context)
        for model, values, context, reference in loaded
    )
    _write_figure(args.output, pipelines)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
