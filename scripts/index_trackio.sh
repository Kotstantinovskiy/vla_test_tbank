#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_root="${VLA_ENV_ROOT:-$repo_root/.venv}"
if [[ ! -x "$env_root/bin/trackio" ]]; then
  echo "Trackio is missing. Run 'uv sync --frozen' in $repo_root." >&2
  exit 1
fi

dashboard_root="${TRACKIO_DASHBOARD_DIR:-$repo_root/.trackio-dashboard}"

export PATH="$env_root/bin:$PATH"
export TRACKIO_DIR="$dashboard_root"
"$env_root/bin/python" "$repo_root/scripts/index_trackio.py"
