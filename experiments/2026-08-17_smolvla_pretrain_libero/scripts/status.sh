#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
if [[ -f results/status.json ]]; then
  "$VLA_PYTHON" - <<'PY'
import json
from pathlib import Path

payload = {"trainer": json.loads(Path("results/status.json").read_text())}
exact = Path("results/live_trackio_status.json")
if exact.is_file():
    payload["exact_progress"] = json.loads(exact.read_text())
print(json.dumps(payload, indent=2))
PY
else
  echo "results/status.json does not exist yet"
fi
ps -eo pid,etimes,cmd | grep -E '[t]orchrun.*smolvla|[l]erobot-train' || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader || true
