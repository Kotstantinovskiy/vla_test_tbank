#!/usr/bin/env bash
set -euo pipefail

export VLA_EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_REPO_ROOT="$(cd "$VLA_EXPERIMENT_ROOT/../.." && pwd)"
export VLA_ENV_ROOT="${VLA_ENV_ROOT:-$VLA_REPO_ROOT/.venv}"
export PATH="$VLA_ENV_ROOT/bin:$PATH"
export PYTHONPATH="$VLA_EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/var/tmp/vla_hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/var/tmp/vla_lerobot}"
export TRACKIO_DIR="${TRACKIO_DIR:-$VLA_EXPERIMENT_ROOT/artifacts/trackio}"
export TRACKIO_PROJECT="${TRACKIO_PROJECT:-bonus-qwen35-progress-critic}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export VLA_PYTHON="${VLA_PYTHON:-$VLA_ENV_ROOT/bin/python}"
