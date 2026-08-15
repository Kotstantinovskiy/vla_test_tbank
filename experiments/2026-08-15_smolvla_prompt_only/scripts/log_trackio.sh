#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

if [[ -n "${TRACKIO_SPACE_ID:-}" ]]; then
  export HF_HUB_OFFLINE=0
fi

"$VLA_PYTHON" -m smolvla_prompt_only.trackio_report "$@"
