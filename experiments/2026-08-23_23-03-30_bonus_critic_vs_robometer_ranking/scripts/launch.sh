#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$EXPERIMENT_ROOT"
mkdir -p results/logs

if [[ -f results/run.pid ]]; then
  existing_pid="$(tr -d '[:space:]' < results/run.pid)"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "run already active: PID $existing_pid"
    exit 0
  fi
fi

setsid bash scripts/run_all.sh </dev/null > results/logs/run.log 2>&1 &
pid=$!
printf '%s\n' "$pid" > results/run.pid
echo "launched PID $pid; log: results/logs/run.log"
