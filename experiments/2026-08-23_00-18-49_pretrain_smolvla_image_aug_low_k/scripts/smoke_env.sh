#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
"$VLA_PYTHON" -m pretrain_smolvla_image_aug_low_k.smoke "$@"
