from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .constants import (
    BASE_CHECKPOINT,
    DEMO_BUDGETS,
    MASTER_SEED,
    OUTPUT_ROOT,
    TARGET_DATASET_REPO,
    TARGET_DATASET_ROOT,
    TARGET_INSTRUCTIONS,
    TRAIN_BATCH_SIZE,
    TRAIN_LOG_FREQ,
    TRAIN_STEPS,
    TRAIN_WORKERS,
    experiment_root,
)
from .selection import episodes_from_manifest


def job_output(task_id: int, budget: int) -> Path:
    return OUTPUT_ROOT / f"task_{task_id}" / f"k_{budget}"


def final_model(task_id: int, budget: int) -> Path:
    return job_output(task_id, budget) / f"checkpoints/{TRAIN_STEPS:06d}/pretrained_model"


def training_complete(task_id: int, budget: int) -> bool:
    model = final_model(task_id, budget)
    state = (
        job_output(task_id, budget)
        / f"checkpoints/{TRAIN_STEPS:06d}/training_state/training_step.json"
    )
    if not model.joinpath("model.safetensors").is_file() or not state.is_file():
        return False
    return json.loads(state.read_text()).get("step") == TRAIN_STEPS


def build_command(root: Path, task_id: int, budget: int) -> list[str]:
    if task_id not in TARGET_INSTRUCTIONS:
        raise ValueError(f"Unknown task_id {task_id}")
    if budget not in DEMO_BUDGETS:
        raise ValueError(f"Unsupported budget {budget}; zero-shot is intentionally absent")
    episodes = episodes_from_manifest(
        root / "artifacts/episode_manifest.json", task_id, budget
    )
    trainer = (
        Path(os.environ.get("VLA_ENV_ROOT", root.parents[1] / ".venv"))
        / "bin/lerobot-train"
    )
    # The base checkpoint and the target dataset share one schema (top /
    # wrist_image cameras at 128x128, 8-dim state) and one frame convention
    # (rot180 == eval), so no adapter or rename map is needed.  LeRobot still
    # replaces the checkpoint's normalization statistics with target-dataset
    # statistics at fine-tune time; both now come from the same conversion
    # pipeline, and the swap is disclosed in configs/protocol.yaml.
    return [
        str(trainer),
        f"--policy.path={BASE_CHECKPOINT}",
        "--policy.freeze_vision_encoder=true",
        "--policy.train_expert_only=true",
        "--policy.train_state_proj=true",
        "--policy.use_amp=false",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={TARGET_DATASET_REPO}",
        f"--dataset.root={TARGET_DATASET_ROOT}",
        f"--dataset.episodes={json.dumps(episodes, separators=(',', ':'))}",
        "--dataset.video_backend=pyav",
        "--dataset.image_transforms.enable=false",
        f"--output_dir={job_output(task_id, budget)}",
        f"--job_name=few_shot_task_{task_id}_k_{budget}",
        f"--steps={TRAIN_STEPS}",
        f"--batch_size={TRAIN_BATCH_SIZE}",
        f"--num_workers={TRAIN_WORKERS}",
        "--dataloader_multiprocessing_context=spawn",
        "--persistent_workers=true",
        "--prefetch_factor=4",
        "--env_eval_freq=0",
        "--eval_steps=0",
        "--save_checkpoint=true",
        "--save_freq=0",
        f"--log_freq={TRAIN_LOG_FREQ}",
        "--use_policy_training_preset=true",
        f"--seed={MASTER_SEED}",
        "--wandb.enable=false",
    ]


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Train one expert-only task/budget adaptation")
    parser.add_argument("task_id", type=int)
    parser.add_argument("budget", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = build_command(root, args.task_id, args.budget)
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return
    if training_complete(args.task_id, args.budget):
        print(json.dumps({"state": "already_complete", "model": str(final_model(args.task_id, args.budget))}))
        return
    output = job_output(args.task_id, args.budget)
    if output.exists():
        raise FileExistsError(
            f"Incomplete output already exists; inspect before retrying: {output}"
        )
    subprocess.run(command, cwd=root, check=True)
    if not training_complete(args.task_id, args.budget):
        raise RuntimeError("Trainer exited successfully but final checkpoint is incomplete")
    print(json.dumps({"state": "completed", "model": str(final_model(args.task_id, args.budget))}))


if __name__ == "__main__":
    main()
