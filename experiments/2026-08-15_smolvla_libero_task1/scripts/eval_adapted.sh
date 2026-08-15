#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

task_id="${1:?usage: eval_adapted.sh TASK_ID K [GPU]}"
k="${2:?usage: eval_adapted.sh TASK_ID K [GPU]}"
gpu="${3:-0}"
run_dir="artifacts/checkpoints/naive/task_${task_id}/k_${k}"
model="$(find "$run_dir/checkpoints" -type d -name pretrained_model | sort | tail -1)"
if [[ -z "$model" ]]; then
  echo "No final checkpoint found below $run_dir" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$gpu" "$VLA_PYTHON" -m vla_cost_curve.evaluate \
  --model "$model" \
  --revision none \
  --condition true \
  --task-ids "$task_id" \
  --n-episodes 20 \
  --batch-size 4 \
  --seed 1000 \
  --videos 1 \
  --output "results/raw/adapted/task_${task_id}/k_${k}.json"
