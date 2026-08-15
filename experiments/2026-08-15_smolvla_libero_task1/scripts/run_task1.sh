#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

scripts/lock_predictions.sh >/dev/null
scripts/prepare.sh

scripts/eval_zero_shot.sh true 0 & p0=$!
scripts/eval_zero_shot.sh wrong 1 & p1=$!
scripts/eval_zero_shot.sh nonsense 2 & p2=$!
wait "$p0" "$p1" "$p2"

for task_id in 0 1 2; do
  (
    for k in 5 10 25; do
      scripts/train_naive_ft.sh "$task_id" "$k" "$task_id"
      scripts/eval_adapted.sh "$task_id" "$k" "$task_id"
    done
  ) &
done
wait

"$VLA_PYTHON" -m vla_cost_curve.aggregate
