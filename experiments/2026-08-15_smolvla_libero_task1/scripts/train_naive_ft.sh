#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
cd "$VLA_EXPERIMENT_ROOT"

task_id="${1:?usage: train_naive_ft.sh TASK_ID K [GPU]}"
k="${2:?usage: train_naive_ft.sh TASK_ID K [GPU]}"
gpu="${3:-0}"
case "$task_id" in 0|1|2) ;; *) echo "task_id must be 0, 1, or 2" >&2; exit 2;; esac
case "$k" in 5|10|25) ;; *) echo "k must be 5, 10, or 25" >&2; exit 2;; esac

episodes="$("$VLA_PYTHON" -m vla_cost_curve.selection \
  --manifest artifacts/episode_manifest.json --task-id "$task_id" --k "$k" --print-episodes)"
output="artifacts/checkpoints/naive/task_${task_id}/k_${k}"
log_file="results/logs/train/task_${task_id}_k_${k}.log"
mkdir -p "$(dirname "$log_file")"

CUDA_VISIBLE_DEVICES="$gpu" lerobot-train \
  --policy.path=artifacts/seen_image_schema \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.revision="$VLA_DATA_REVISION" \
  --dataset.root="$VLA_DATA_ROOT" \
  --dataset.episodes="$episodes" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=false \
  --output_dir="$output" \
  --job_name="naive_task_${task_id}_k_${k}" \
  --steps=2000 \
  --batch_size=32 \
  --num_workers=8 \
  --dataloader_multiprocessing_context=spawn \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --save_checkpoint=true \
  --save_freq=0 \
  --log_freq=25 \
  --seed=1000 \
  --wandb.enable=false \
  2>&1 | tee "$log_file"
