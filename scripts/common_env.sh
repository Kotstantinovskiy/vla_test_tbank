#!/usr/bin/env bash
set -euo pipefail

export PATH="/var/tmp/vla_env/bin:$PATH"
export PYTHONPATH="/home/nbagent174/vla_test/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="/var/tmp/vla_hf"
export HF_LEROBOT_HOME="/var/tmp/vla_lerobot"
export LIBERO_CONFIG_PATH="/home/nbagent174/vla_test/.libero"
export MUJOCO_GL="egl"
export TOKENIZERS_PARALLELISM="false"
# All model and tokenizer files are pinned and cached during preparation.  Keep
# experiment runs offline so an optional Hub HEAD request cannot abort a rollout.
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"

export VLA_PYTHON="/var/tmp/vla_env/bin/python"
export VLA_ROOT="/home/nbagent174/vla_test"
export VLA_DATA_ROOT="/var/tmp/vla_target_9176"
export VLA_DATA_REVISION="9176d427966503c81ac9f8f96502e50861a15ee7"
export VLA_SEEN_REVISION="418f9d0e5b48585bcee1e1a7d47e302629af78da"
