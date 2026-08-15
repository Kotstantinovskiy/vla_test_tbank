#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

git diff --exit-code -- reports/PREDICTIONS.md
git diff --cached --exit-code -- reports/PREDICTIONS.md

commit="$(git log -1 --format=%H -- reports/PREDICTIONS.md)"
if [[ -z "$commit" ]]; then
  echo "reports/PREDICTIONS.md has not been committed" >&2
  exit 1
fi

echo "$commit"
