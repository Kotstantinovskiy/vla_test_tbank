#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

condition="${1:?usage: eval_prompt.sh CONDITION [GPU]}"
gpu="${2:-0}"
videos=0
if [[ "$condition" == "true" ]]; then videos=1; fi

CUDA_VISIBLE_DEVICES="$gpu" "$VLA_PYTHON" -m smolvla_prompt_only.evaluate \
  --model crislmfroes/smolvla-libero-90 \
  --revision "$VLA_CHECKPOINT_REVISION" \
  --condition "$condition" \
  --n-episodes 20 \
  --batch-size 4 \
  --seed 1000 \
  --videos "$videos" \
  --output "results/raw/$condition.json"
