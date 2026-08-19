#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
shopt -s globstar nullglob

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$(cd "$script_dir/../.." && pwd)"

status=0
for plot_script in "$script_dir"/**/plot.py; do
    uv run python "$plot_script" || status=$?
done

exit "$status"
