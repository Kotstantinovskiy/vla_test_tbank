#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

naive_root="$VLA_REPO_ROOT/experiments/2026-08-21_20-37-56_pretrain_smolvla_naive_deterministic_repro"
naive_pid_file="$naive_root/results/orchestrator.pid"
naive_status="$naive_root/results/status.json"

if [[ ! -f "$naive_pid_file" ]]; then
  echo "Naive orchestrator PID file is missing: $naive_pid_file" >&2
  exit 1
fi
naive_pid="$(<"$naive_pid_file")"
if [[ ! "$naive_pid" =~ ^[0-9]+$ ]]; then
  echo "Invalid naive orchestrator PID: $naive_pid" >&2
  exit 1
fi

echo "Waiting for naive orchestrator PID $naive_pid"
if kill -0 "$naive_pid" 2>/dev/null; then
  tail --pid="$naive_pid" -f /dev/null
fi

"$VLA_PYTHON" -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Naive status is missing: {path}")
payload = json.loads(path.read_text())
if payload.get("state") != "completed":
    raise SystemExit("Naive orchestration did not complete: " + str(payload.get("state")))
' "$naive_status"

echo "Naive completed; starting low-k on GPUs ${VLA_GPU_IDS:-1,2,3}"
cd "$VLA_EXPERIMENT_ROOT"
exec env VLA_GPU_IDS="${VLA_GPU_IDS:-1,2,3}" scripts/run_all.sh
