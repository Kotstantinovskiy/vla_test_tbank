#!/usr/bin/env bash
set -euo pipefail

experiment_root="$(cd "$(dirname "$0")/.." && pwd)"
repo_root="$(git -C "$experiment_root" rev-parse --show-toplevel)"
prediction_path="$(realpath --relative-to="$repo_root" "$experiment_root/reports/PREDICTIONS.md")"

git -C "$repo_root" diff --exit-code -- "$prediction_path"
git -C "$repo_root" diff --cached --exit-code -- "$prediction_path"

commit="$(git -C "$repo_root" log --follow -1 --format=%H -- "$prediction_path")"
if [[ -z "$commit" ]]; then
  echo "$prediction_path has not been committed" >&2
  exit 1
fi

echo "$commit"
