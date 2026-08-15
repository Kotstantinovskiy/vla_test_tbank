#!/usr/bin/env bash
set -euo pipefail

export VLA_EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_REPO_ROOT="$(cd "$VLA_EXPERIMENT_ROOT/../.." && pwd)"
if [[ -z "${VLA_ENV_ROOT:-}" ]]; then
  if [[ -x "$VLA_REPO_ROOT/.venv/bin/python" ]]; then
    export VLA_ENV_ROOT="$VLA_REPO_ROOT/.venv"
  else
    export VLA_ENV_ROOT="/var/tmp/vla_env"
  fi
fi
export PATH="$VLA_ENV_ROOT/bin:$PATH"
export PYTHONPATH="$VLA_EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/var/tmp/vla_hf}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/var/tmp/vla_lerobot}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$VLA_EXPERIMENT_ROOT/artifacts/libero_config}"
export TRACKIO_DIR="${TRACKIO_DIR:-$VLA_EXPERIMENT_ROOT/artifacts/trackio}"
export TRACKIO_PROJECT="${TRACKIO_PROJECT:-smolvla-prompt-only}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export VLA_PYTHON="${VLA_PYTHON:-$VLA_ENV_ROOT/bin/python}"
export VLA_CHECKPOINT_REVISION="418f9d0e5b48585bcee1e1a7d47e302629af78da"
