#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_ROOT/../.." && pwd)"
PYTHONPATH="$EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$REPO_ROOT/.venv/bin/python" -m analyst_curve_baseline.build "$@"
