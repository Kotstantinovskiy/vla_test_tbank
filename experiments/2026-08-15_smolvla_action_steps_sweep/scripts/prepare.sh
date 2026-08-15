#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

"$VLA_PYTHON" -m smolvla_action_steps.prepare \
  --external-checkpoints /var/tmp/vla_outputs \
  --checkpoint-link artifacts/checkpoints \
  --libero-config-dir artifacts/libero_config \
  --manifest artifacts/checkpoint_manifest.json
