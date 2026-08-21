#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
"$VLA_PYTHON" -m pretrain_smolvla_prompt_only_2.prepare \
  --config-dir artifacts/libero_config \
  --manifest artifacts/checkpoint_manifest.json
