#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$REPOSITORY_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing root uv environment: $PYTHON" >&2
  echo "Run: uv sync --frozen" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-/var/tmp/hf_home_vla}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

exec "$PYTHON" "$REPOSITORY_ROOT/scripts/download_official_libero.py" "$@"
