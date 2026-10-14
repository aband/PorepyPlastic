#!/usr/bin/env bash
# Remove this project's generated results, independent of the working directory.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
results_dir="$project_dir/results"

if [[ -e "$results_dir" || -L "$results_dir" ]]; then
    rm -rf -- "$results_dir"
    printf 'Removed %s\n' "$results_dir"
else
    printf 'No results to remove: %s\n' "$results_dir"
fi
