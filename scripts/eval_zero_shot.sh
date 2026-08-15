#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_ROOT"

condition="${1:?usage: eval_zero_shot.sh CONDITION [GPU]}"
gpu="${2:-0}"
videos=0
if [[ "$condition" == "true" ]]; then videos=1; fi

CUDA_VISIBLE_DEVICES="$gpu" "$VLA_PYTHON" -m vla_cost_curve.evaluate \
  --model crislmfroes/smolvla-libero-90 \
  --revision "$VLA_SEEN_REVISION" \
  --condition "$condition" \
  --n-episodes 20 \
  --batch-size 4 \
  --seed 1000 \
  --videos "$videos" \
  --output "results/zero_shot/$condition.json"

