#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
gpu="${1:?usage: eval_point.sh GPU LABEL...}"
shift
CUDA_VISIBLE_DEVICES="$gpu" "$VLA_PYTHON" -m goal_prompts_in_seen_hosts.evaluate --labels "$@"
