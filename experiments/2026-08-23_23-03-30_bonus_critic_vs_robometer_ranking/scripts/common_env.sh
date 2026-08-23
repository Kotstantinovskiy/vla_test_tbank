#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_ROOT}/../.." && pwd)"
export EXPERIMENT_ROOT REPO_ROOT
export HF_HOME=/var/tmp/vla_hf
export HF_HUB_CACHE=/var/tmp/vla_hf/hub
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRACKIO_DIR="${TRACKIO_DIR:-$EXPERIMENT_ROOT/artifacts/trackio}"
export TRACKIO_PROJECT="${TRACKIO_PROJECT:-bonus-critic-vs-robometer-ranking}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
