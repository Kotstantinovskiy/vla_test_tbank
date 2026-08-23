#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

if (( $# == 0 )); then
  echo "Usage: $0 PID [PID ...]" >&2
  exit 2
fi

for inflight_pid in "$@"; do
  if [[ ! "$inflight_pid" =~ ^[0-9]+$ ]]; then
    echo "Invalid in-flight PID: $inflight_pid" >&2
    exit 2
  fi
  echo "Waiting for retained assignment trainer PID $inflight_pid"
  if kill -0 "$inflight_pid" 2>/dev/null; then
    tail --pid="$inflight_pid" -f /dev/null
  fi
done

cd "$VLA_EXPERIMENT_ROOT"
"$VLA_PYTHON" -c '
from pretrain_smolvla_naive_deterministic_repro.training import training_complete

missing = [(task_id, 5) for task_id in (1, 2) if not training_complete(task_id, 5)]
if missing:
    raise SystemExit(f"Retained assignment training did not complete: {missing}")
'

echo "Retained task-1/task-2 k=5 checkpoints complete; starting scoped orchestrator"
exec env VLA_GPU_IDS="${VLA_GPU_IDS:-1,2,3}" scripts/run_all.sh
