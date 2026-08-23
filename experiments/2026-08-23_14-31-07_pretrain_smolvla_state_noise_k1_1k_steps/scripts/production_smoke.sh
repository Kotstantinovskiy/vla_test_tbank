#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
gpu="${1:-0}"
CUDA_VISIBLE_DEVICES="$gpu" "$VLA_PYTHON" -m pretrain_smolvla_state_noise_k1_1k.training 0 0.0
CUDA_VISIBLE_DEVICES="$gpu" "$VLA_PYTHON" -m pretrain_smolvla_state_noise_k1_1k.determinism
