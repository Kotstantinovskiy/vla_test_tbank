#!/usr/bin/env bash
set -euo pipefail

export VLA_EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_ENV_ROOT="${VLA_ENV_ROOT:-/var/tmp/vla_env}"
export PATH="$VLA_ENV_ROOT/bin:$PATH"
export PYTHONPATH="$VLA_EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/var/tmp/vla_hf}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/var/tmp/vla_lerobot}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$VLA_EXPERIMENT_ROOT/artifacts/libero_config}"
export TRACKIO_DIR="${TRACKIO_DIR:-$VLA_EXPERIMENT_ROOT/artifacts/trackio}"
export TRACKIO_PROJECT="${TRACKIO_PROJECT:-smolvla-libero-task1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# All model and tokenizer files are pinned and cached during preparation.  Keep
# experiment runs offline so an optional Hub HEAD request cannot abort a rollout.
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"

export VLA_PYTHON="${VLA_PYTHON:-$VLA_ENV_ROOT/bin/python}"
export VLA_DATA_ROOT="${VLA_DATA_ROOT:-/var/tmp/vla_target_9176}"
export VLA_DATA_REVISION="9176d427966503c81ac9f8f96502e50861a15ee7"
export VLA_SEEN_REVISION="418f9d0e5b48585bcee1e1a7d47e302629af78da"
