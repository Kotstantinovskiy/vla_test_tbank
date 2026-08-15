#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
"$VLA_PYTHON" -m vla_cost_curve.prepare \
  --dataset-root "$VLA_DATA_ROOT" \
  --manifest artifacts/episode_manifest.json \
  --seen-output artifacts/seen_image_schema
