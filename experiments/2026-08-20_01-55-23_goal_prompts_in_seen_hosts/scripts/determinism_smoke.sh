#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
gpu="${1:-0}"
left="$VLA_EXPERIMENT_ROOT/results/determinism_check/a"
right="$VLA_EXPERIMENT_ROOT/results/determinism_check/b"
CUDA_VISIBLE_DEVICES="$gpu" "$VLA_PYTHON" -m goal_prompts_in_seen_hosts.evaluate \
  --labels goal__goal_0 --out-dir "$left" --force
CUDA_VISIBLE_DEVICES="$gpu" "$VLA_PYTHON" -m goal_prompts_in_seen_hosts.evaluate \
  --labels seen__goal_0 goal__goal_0 --out-dir "$right" --force
"$VLA_PYTHON" -m goal_prompts_in_seen_hosts.determinism \
  --label goal__goal_0 --left "$left" --right "$right"
