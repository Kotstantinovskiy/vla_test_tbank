#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

condition="${1:?usage: eval_prompt.sh CONDITION [GPU]}"
gpu="${2:-0}"
videos=20
# 2026-08-18: all episodes of all conditions are recorded to disk; Trackio still
# receives only episode 0 per task (see trackio_report.py).

CUDA_VISIBLE_DEVICES="$gpu" "$VLA_PYTHON" -m pretrain_smolvla_prompt_only_2.evaluate \
  --model "$VLA_CHECKPOINT_PATH" \
  --manifest artifacts/checkpoint_manifest.json \
  --condition "$condition" \
  --n-episodes 20 \
  --batch-size 4 \
  --seed 1000 \
  --videos "$videos" \
  --output "results/raw/$condition.json"
