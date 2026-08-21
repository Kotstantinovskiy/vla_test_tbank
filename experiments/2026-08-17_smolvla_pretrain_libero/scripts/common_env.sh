#!/usr/bin/env bash
set -euo pipefail

export VLA_EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_REPO_ROOT="$(cd "$VLA_EXPERIMENT_ROOT/../.." && pwd)"
export VLA_ENV_ROOT="${VLA_ENV_ROOT:-$VLA_REPO_ROOT/.venv}"
export PATH="$VLA_ENV_ROOT/bin:$PATH"
export PYTHONPATH="$VLA_EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HF_HOME:-/var/tmp/vla_hf}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/var/tmp/vla_lerobot}"
export VLA_OFFICIAL_OUTPUT_ROOT="${VLA_OFFICIAL_OUTPUT_ROOT:-/var/tmp/vla_outputs/seen_libero90_official_20260817}"
export TRACKIO_DIR="${TRACKIO_DIR:-$VLA_EXPERIMENT_ROOT/artifacts/trackio}"
export TRACKIO_PROJECT="${TRACKIO_PROJECT:-smolvla-pretrain-libero}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export VLA_PYTHON="$VLA_ENV_ROOT/bin/python"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$VLA_EXPERIMENT_ROOT/artifacts/libero_config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export VLA_OFFICIAL_SEEN_DATA_ROOT="${VLA_OFFICIAL_SEEN_DATA_ROOT:-/var/tmp/vla_libero_official_rot180/libero_90}"
export VLA_OFFICIAL_TARGET_DATA_ROOT="${VLA_OFFICIAL_TARGET_DATA_ROOT:-/var/tmp/vla_libero_official_rot180/libero_goal}"
