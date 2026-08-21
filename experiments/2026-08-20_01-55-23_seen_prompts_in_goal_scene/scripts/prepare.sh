#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
"$VLA_PYTHON" -m seen_prompts_in_goal_scene.prepare "$@"
