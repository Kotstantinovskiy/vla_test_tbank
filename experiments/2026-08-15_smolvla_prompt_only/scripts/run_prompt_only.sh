#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

gpu_true="${1:-0}"
gpu_wrong="${2:-1}"
gpu_nonsense="${3:-2}"

mkdir -p results/logs
scripts/prepare.sh > results/logs/prepare.log 2>&1

scripts/eval_prompt.sh true "$gpu_true" > results/logs/eval_true.log 2>&1 &
pid_true=$!
scripts/eval_prompt.sh wrong "$gpu_wrong" > results/logs/eval_wrong.log 2>&1 &
pid_wrong=$!
scripts/eval_prompt.sh nonsense "$gpu_nonsense" > results/logs/eval_nonsense.log 2>&1 &
pid_nonsense=$!

wait "$pid_true"
wait "$pid_wrong"
wait "$pid_nonsense"

"$VLA_PYTHON" -m smolvla_prompt_only.aggregate
scripts/log_trackio.sh
