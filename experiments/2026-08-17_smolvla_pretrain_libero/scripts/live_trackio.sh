#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
exec "$VLA_PYTHON" -m smolvla_pretrain_libero.live_trackio
