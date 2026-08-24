#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
artifact_dir=artifacts/benchmark/omnidreams
mkdir -p "$artifact_dir"

uv run --project integrations/omnidreams --group test pytest \
    integrations/omnidreams/benchmarks/test_modules.py \
    -p no:manual_marker -m manual --benchmark-only -v "$@" \
    --benchmark-json="$artifact_dir/module.json"

uv run python -m scripts.benchmark.omnidreams.plot_module \
    "$artifact_dir/module.json"
