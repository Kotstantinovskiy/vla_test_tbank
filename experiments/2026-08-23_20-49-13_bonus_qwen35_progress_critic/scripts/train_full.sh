#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$VLA_PYTHON" -m bonus_qwen35_progress_critic.train \
  --mode train \
  --max-steps 2000 \
  --output "$VLA_EXPERIMENT_ROOT/results/training" \
  "$@"
