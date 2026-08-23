#!/usr/bin/env bash
set -euo pipefail

export VLA_EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_REPO_ROOT="$(cd "$VLA_EXPERIMENT_ROOT/../.." && pwd)"
export VLA_ENV_ROOT="${VLA_ENV_ROOT:-$VLA_REPO_ROOT/.venv}"
export PATH="$VLA_ENV_ROOT/bin:$PATH"
export PYTHONPATH="$VLA_EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/var/tmp/vla_hf}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/var/tmp/vla_lerobot}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$VLA_EXPERIMENT_ROOT/artifacts/libero_config}"
export TRACKIO_DIR="${TRACKIO_DIR:-$VLA_EXPERIMENT_ROOT/artifacts/trackio}"
export TRACKIO_PROJECT="${TRACKIO_PROJECT:-pretrain-smolvla-bundle-all-k}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export PYTHONUNBUFFERED=1
export VLA_PYTHON="${VLA_PYTHON:-$VLA_ENV_ROOT/bin/python}"
