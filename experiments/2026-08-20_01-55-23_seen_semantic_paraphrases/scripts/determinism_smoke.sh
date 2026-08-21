#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
gpu="${1:-0}"
left="$VLA_EXPERIMENT_ROOT/results/determinism_check/a"
right="$VLA_EXPERIMENT_ROOT/results/determinism_check/b"
CUDA_VISIBLE_DEVICES="$gpu" "$VLA_PYTHON" -m seen_semantic_paraphrases.evaluate \
  --labels paraphrase__task_0 --out-dir "$left" --force
CUDA_VISIBLE_DEVICES="$gpu" "$VLA_PYTHON" -m seen_semantic_paraphrases.evaluate \
  --labels exact__task_0 paraphrase__task_0 --out-dir "$right" --force
"$VLA_PYTHON" -m seen_semantic_paraphrases.determinism \
  --label paraphrase__task_0 --left "$left" --right "$right"
