#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"
# Train the gate job (task 0 / k=1) on the first GPU argument, then run the
# three determinism variants concurrently, one per GPU, and combine.
gpu_train="${1:-1}"
gpu_a="${2:-1}"; gpu_b="${3:-2}"; gpu_c="${4:-3}"
CUDA_VISIBLE_DEVICES="$gpu_train" "$VLA_PYTHON" -m pretrain_smolvla_bundle_all_k.training 0 1
pids=()
CUDA_VISIBLE_DEVICES="$gpu_a" "$VLA_PYTHON" -m pretrain_smolvla_bundle_all_k.determinism --variant 50 > results/logs/determinism_n50.log 2>&1 & pids+=($!)
CUDA_VISIBLE_DEVICES="$gpu_b" "$VLA_PYTHON" -m pretrain_smolvla_bundle_all_k.determinism --variant 35 > results/logs/determinism_n35.log 2>&1 & pids+=($!)
CUDA_VISIBLE_DEVICES="$gpu_c" "$VLA_PYTHON" -m pretrain_smolvla_bundle_all_k.determinism --variant 25 > results/logs/determinism_n25.log 2>&1 & pids+=($!)
code=0
for pid in "${pids[@]}"; do wait "$pid" || code=1; done
if [[ "$code" != 0 ]]; then echo "determinism variant failed; see results/logs/determinism_n*.log" >&2; exit 1; fi
"$VLA_PYTHON" -m pretrain_smolvla_bundle_all_k.determinism --combine
