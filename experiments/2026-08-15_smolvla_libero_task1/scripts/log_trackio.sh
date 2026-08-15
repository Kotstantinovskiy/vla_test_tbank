#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

if [[ -n "${TRACKIO_SPACE_ID:-}" ]]; then
  # Model and dataset execution stays offline, while an explicitly requested
  # Trackio Space publication needs Hub access in this reporting subprocess.
  export HF_HUB_OFFLINE=0
fi

"$VLA_PYTHON" -m vla_cost_curve.trackio_report "$@"
