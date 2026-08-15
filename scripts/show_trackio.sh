#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_root="${VLA_ENV_ROOT:-$repo_root/.venv}"
dashboard_root="${TRACKIO_DASHBOARD_DIR:-$repo_root/.trackio-dashboard}"

"$repo_root/scripts/index_trackio.sh"

export PATH="$env_root/bin:$PATH"
export TRACKIO_DIR="$dashboard_root"
exec trackio show --host "${TRACKIO_HOST:-0.0.0.0}" "$@"
