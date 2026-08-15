#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_root="${VLA_ENV_ROOT:-$repo_root/.venv}"
if [[ ! -x "$env_root/bin/trackio" ]]; then
  echo "Trackio is missing. Run 'uv sync --frozen' in $repo_root." >&2
  exit 1
fi

dashboard_root="${TRACKIO_DASHBOARD_DIR:-$repo_root/.trackio-dashboard}"
mkdir -p "$dashboard_root/media"

found=0
shopt -s nullglob
for source_db in "$repo_root"/experiments/*/artifacts/trackio/*.db; do
  project="$(basename "$source_db" .db)"
  source_dir="$(dirname "$source_db")"

  ln -sfn "$source_db" "$dashboard_root/$project.db"
  if [[ -e "$source_dir/$project.lock" ]]; then
    ln -sfn "$source_dir/$project.lock" "$dashboard_root/$project.lock"
  fi
  if [[ -d "$source_dir/media/$project" ]]; then
    ln -sfn "$source_dir/media/$project" "$dashboard_root/media/$project"
  fi
  found=$((found + 1))
done

if [[ "$found" -eq 0 ]]; then
  echo "No experiment-local Trackio databases found." >&2
  exit 1
fi

export PATH="$env_root/bin:$PATH"
export TRACKIO_DIR="$dashboard_root"
trackio list projects --json
