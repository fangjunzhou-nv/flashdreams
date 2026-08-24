#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
shopt -s globstar nullglob

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
cd "$repo_root"

status=0
for plot_script in "$script_dir"/**/plot*.py; do
    plot_module=${plot_script#"$repo_root/"}
    plot_module=${plot_module%.py}
    uv run python -m "${plot_module//\//.}" || status=$?
done

exit "$status"
