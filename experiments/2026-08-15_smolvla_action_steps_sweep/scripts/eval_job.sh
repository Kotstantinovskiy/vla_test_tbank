#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

kind="${1:?usage: eval_job.sh KIND TASK_ID DEMO_BUDGET GPU [CONDITION]}"
task_id="${2:?usage: eval_job.sh KIND TASK_ID DEMO_BUDGET GPU [CONDITION]}"
demo_budget="${3:?usage: eval_job.sh KIND TASK_ID DEMO_BUDGET GPU [CONDITION]}"
gpu="${4:?usage: eval_job.sh KIND TASK_ID DEMO_BUDGET GPU [CONDITION]}"
condition="${5:-true}"

case "$kind" in
  zero_shot)
    model="crislmfroes/smolvla-libero-90"
    revision="$VLA_CHECKPOINT_REVISION"
    if [[ "$condition" == "true" ]]; then
      output="results/raw/zero_shot/task_${task_id}.json"
    else
      output="results/raw/zero_shot_controls/${condition}/task_${task_id}.json"
    fi
    ;;
  adapted)
    if [[ "$condition" != "true" ]]; then
      echo "Adapted checkpoints are evaluated only with the true prompt" >&2
      exit 2
    fi
    model="artifacts/checkpoints/naive/task_${task_id}/k_${demo_budget}/checkpoints/last/pretrained_model"
    revision="none"
    output="results/raw/adapted/task_${task_id}/k_${demo_budget}.json"
    ;;
  *)
    echo "Unknown checkpoint kind: $kind" >&2
    exit 2
    ;;
esac

read -r -a action_steps <<< "${VLA_ACTION_STEPS:-1 5 10 25 50}"
CUDA_VISIBLE_DEVICES="$gpu" "$VLA_PYTHON" -m smolvla_action_steps.evaluate \
  --model "$model" \
  --revision "$revision" \
  --task-id "$task_id" \
  --demo-budget "$demo_budget" \
  --condition "$condition" \
  --action-steps "${action_steps[@]}" \
  --video-action-steps "${action_steps[@]}" \
  --n-episodes "${VLA_N_EPISODES:-20}" \
  --batch-size "${VLA_BATCH_SIZE:-4}" \
  --seed 1000 \
  --videos "${VLA_VIDEOS:-1}" \
  --output "$output"
