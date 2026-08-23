#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
"$VLA_PYTHON" -m pretrain_smolvla_low_k_deterministic_repro.trackio_report "$@"
