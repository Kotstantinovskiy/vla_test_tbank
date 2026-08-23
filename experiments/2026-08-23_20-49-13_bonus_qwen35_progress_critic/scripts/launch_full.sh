#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

pid_path="$VLA_EXPERIMENT_ROOT/results/train_full.pid"
log_path="$VLA_EXPERIMENT_ROOT/results/logs/train_full.log"
status_path="$VLA_EXPERIMENT_ROOT/results/status.json"

if [[ -f "$pid_path" ]]; then
  existing_pid="$(<"$pid_path")"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "full training is already running with PID $existing_pid" >&2
    exit 1
  fi
fi

if [[ -f "$status_path" ]] && "$VLA_PYTHON" - "$status_path" <<'PY'
import json
import sys

state = json.load(open(sys.argv[1])).get("state")
raise SystemExit(0 if state in {"starting", "running"} else 1)
PY
then
  echo "status reports an active full-training run; refusing to launch a duplicate" >&2
  exit 1
fi

if [[ -e "$VLA_EXPERIMENT_ROOT/results/training/metrics.jsonl" ]]; then
  echo "training metrics already exist; refusing to overwrite them" >&2
  exit 1
fi

mkdir -p "$VLA_EXPERIMENT_ROOT/results/logs"
if [[ -e "$log_path" ]]; then
  echo "log already exists: $log_path" >&2
  exit 1
fi

nohup setsid "$VLA_EXPERIMENT_ROOT/scripts/train_full.sh" >"$log_path" 2>&1 </dev/null &
training_pid=$!
tmp_pid_path="$(mktemp "$VLA_EXPERIMENT_ROOT/results/.train_full.pid.XXXXXX")"
printf '%s\n' "$training_pid" >"$tmp_pid_path"
mv "$tmp_pid_path" "$pid_path"
echo "launched full training: PID=$training_pid log=$log_path"
