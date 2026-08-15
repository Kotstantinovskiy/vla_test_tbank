#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

gpu0="${1:-0}"
gpu1="${2:-1}"
gpu2="${3:-2}"
gpu3="${4:-3}"
mkdir -p results/logs
scripts/prepare.sh > results/logs/prepare.log 2>&1

worker_0() {
  scripts/eval_job.sh zero_shot 0 0 "$gpu0" > results/logs/task_0_k_0.log 2>&1
  scripts/eval_job.sh adapted 0 5 "$gpu0" > results/logs/task_0_k_5.log 2>&1
  scripts/eval_job.sh adapted 0 10 "$gpu0" > results/logs/task_0_k_10.log 2>&1
}
worker_1() {
  scripts/eval_job.sh zero_shot 1 0 "$gpu1" > results/logs/task_1_k_0.log 2>&1
  scripts/eval_job.sh adapted 1 5 "$gpu1" > results/logs/task_1_k_5.log 2>&1
  scripts/eval_job.sh adapted 1 10 "$gpu1" > results/logs/task_1_k_10.log 2>&1
}
worker_2() {
  scripts/eval_job.sh zero_shot 2 0 "$gpu2" > results/logs/task_2_k_0.log 2>&1
  scripts/eval_job.sh adapted 2 5 "$gpu2" > results/logs/task_2_k_5.log 2>&1
  scripts/eval_job.sh adapted 2 10 "$gpu2" > results/logs/task_2_k_10.log 2>&1
}
worker_3() {
  scripts/eval_job.sh adapted 0 25 "$gpu3" > results/logs/task_0_k_25.log 2>&1
  scripts/eval_job.sh adapted 1 25 "$gpu3" > results/logs/task_1_k_25.log 2>&1
  scripts/eval_job.sh adapted 2 25 "$gpu3" > results/logs/task_2_k_25.log 2>&1
}

worker_0 & pid0=$!
worker_1 & pid1=$!
worker_2 & pid2=$!
worker_3 & pid3=$!
status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one primary worker failed; inspect results/logs/*.log" >&2
  exit "$status"
fi

"$VLA_PYTHON" -m smolvla_action_steps.aggregate > results/logs/aggregate_primary.log
gate=$("$VLA_PYTHON" -m smolvla_action_steps.control_gate)
if [[ "$gate" == "run" ]]; then
  control_worker() {
    local task_id="$1"
    local gpu="$2"
    scripts/eval_job.sh zero_shot "$task_id" 0 "$gpu" wrong \
      > "results/logs/task_${task_id}_k_0_wrong.log" 2>&1
    scripts/eval_job.sh zero_shot "$task_id" 0 "$gpu" nonsense \
      > "results/logs/task_${task_id}_k_0_nonsense.log" 2>&1
  }
  control_worker 0 "$gpu0" & pid0=$!
  control_worker 1 "$gpu1" & pid1=$!
  control_worker 2 "$gpu2" & pid2=$!
  status=0
  for pid in "$pid0" "$pid1" "$pid2"; do
    wait "$pid" || status=1
  done
  if [[ "$status" -ne 0 ]]; then
    echo "At least one language-control worker failed" >&2
    exit "$status"
  fi
fi

"$VLA_PYTHON" -m smolvla_action_steps.aggregate > results/logs/aggregate_final.log
scripts/log_trackio.sh > results/logs/trackio.log 2>&1
