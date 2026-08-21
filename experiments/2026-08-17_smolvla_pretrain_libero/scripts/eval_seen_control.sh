#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
gpu="${1:-0}"
shift || true
CUDA_VISIBLE_DEVICES="$gpu" "$VLA_PYTHON" -m smolvla_pretrain_libero.eval_seen_control "$@"
