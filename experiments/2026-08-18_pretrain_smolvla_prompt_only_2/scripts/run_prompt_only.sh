#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

gpu_true="${1:-0}"
gpu_wrong="${2:-1}"
gpu_nonsense="${3:-2}"

mkdir -p results/logs
scripts/prepare.sh > results/logs/prepare.log 2>&1
scripts/smoke_env.sh > results/logs/smoke_env.log 2>&1

# Stage 1: run the production "true" condition first and verify it fully
# before fanning out the remaining prompt controls.
scripts/eval_prompt.sh true "$gpu_true" > results/logs/eval_true.log 2>&1
"$VLA_PYTHON" - <<'PY'
import json
from pathlib import Path

result = json.loads(Path("results/raw/true.json").read_text())
assert len(result["tasks"]) == 10, len(result["tasks"])
for task_id, task in result["tasks"].items():
    assert len(task["per_episode"]) == 20, (task_id, len(task["per_episode"]))
    for path in task.get("video_paths", []):
        assert Path(path).is_file(), path
print("true.json verified")
PY

# Stage 2: prompt controls in parallel.
scripts/eval_prompt.sh wrong "$gpu_wrong" > results/logs/eval_wrong.log 2>&1 &
pid_wrong=$!
scripts/eval_prompt.sh nonsense "$gpu_nonsense" > results/logs/eval_nonsense.log 2>&1 &
pid_nonsense=$!

wait "$pid_wrong"
wait "$pid_nonsense"

"$VLA_PYTHON" -m pretrain_smolvla_prompt_only_2.aggregate
scripts/log_trackio.sh
