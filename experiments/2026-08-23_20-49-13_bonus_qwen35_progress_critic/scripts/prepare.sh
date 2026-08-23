#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
exec "$VLA_PYTHON" -m bonus_qwen35_progress_critic.prepare "$@"
