#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
mkdir -p artifacts results/logs results/raw results/summary results/media
"$VLA_PYTHON" -m smolvla_pretrain_libero.prepare
