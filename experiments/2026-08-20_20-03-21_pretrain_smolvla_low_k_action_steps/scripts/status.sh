#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
if [[ -f results/status.json ]]; then
  "$VLA_PYTHON" -c "
import json
s = json.load(open('results/status.json'))
print(
    s['state'],
    f\"training {s['completed_training_jobs']}/{s['total_training_jobs']},\",
    f\"reused checkpoints {s.get('reused_checkpoint_count', 0)},\",
    f\"evaluation {s['completed_evaluation_points']}/{s['total_evaluation_points']},\",
    f\"{s['failed_jobs']} failed\",
)
for key, job in s['jobs'].items():
    if job['state'] not in ('completed',):
        print(' ', key, job['state'], job.get('gpu', ''))
"
else
  echo "results/status.json does not exist yet"
fi
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
