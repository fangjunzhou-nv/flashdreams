#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run and plot the representative module benchmark:
#   ./scripts/benchmark/omnidreams/run_module.sh
# Replot its saved results:
#   uv run python -m scripts.benchmark.omnidreams.plot_module
#
# Run and plot all 925 full-block self/cross pairs:
#   OMNIDREAMS_DIT_BLOCK_FULL_SEARCH=1 \
#     ./scripts/benchmark/omnidreams/run_module.sh
# Replot their saved results:
#   uv run python -m scripts.benchmark.omnidreams.plot_module \
#     artifacts/benchmark/omnidreams/full/module.json \
#     --output-dir artifacts/benchmark/omnidreams/full

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
artifact_dir=artifacts/benchmark/omnidreams
if [[ "${OMNIDREAMS_DIT_BLOCK_FULL_SEARCH:-0}" == "1" ]]; then
    artifact_dir="$artifact_dir/full"
fi
mkdir -p "$artifact_dir"

uv run --project integrations/omnidreams --group test pytest \
    integrations/omnidreams/benchmarks/test_modules.py \
    -p no:manual_marker -m manual --benchmark-only -v "$@" \
    --benchmark-json="$artifact_dir/module.json"

uv run python -m scripts.benchmark.omnidreams.plot_module \
    "$artifact_dir/module.json" --output-dir "$artifact_dir"
