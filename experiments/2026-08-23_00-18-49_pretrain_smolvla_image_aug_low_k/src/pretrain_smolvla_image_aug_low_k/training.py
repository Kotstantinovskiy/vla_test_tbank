from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from .constants import (
    BASE_CHECKPOINT,
    DEMO_BUDGETS,
    EXPERIMENT_NAME,
    FREEZE_VISION_ENCODER,
    MASTER_SEED,
    OUTPUT_ROOT,
    TARGET_DATASET_REPO,
    TARGET_DATASET_ROOT,
    TARGET_INSTRUCTIONS,
    TRAIN_BATCH_SIZE,
    TRAIN_LOG_FREQ,
    TRAIN_EXPERT_ONLY,
    TRAIN_STATE_PROJ,
    TRAIN_STEPS_BY_BUDGET,
    TRAIN_WORKERS,
    TRAINED_ACTION_STEPS,
    TRAINED_CHUNK_SIZE,
    VLM_BACKBONE,
    experiment_root,
)
from .selection import episodes_from_manifest


def job_output(task_id: int, budget: int) -> Path:
    return OUTPUT_ROOT / f"task_{task_id}" / f"k_{budget}"


def train_steps(budget: int) -> int:
    return TRAIN_STEPS_BY_BUDGET[budget]


def final_model(task_id: int, budget: int) -> Path:
    return (
        job_output(task_id, budget)
        / f"checkpoints/{train_steps(budget):06d}/pretrained_model"
    )


def training_complete(task_id: int, budget: int) -> bool:
    model = final_model(task_id, budget)
    state = (
        job_output(task_id, budget)
        / f"checkpoints/{train_steps(budget):06d}/training_state/training_step.json"
    )
    if not model.joinpath("model.safetensors").is_file() or not state.is_file():
        return False
    return json.loads(state.read_text()).get("step") == train_steps(budget)


def adapted_manifest_path(root: Path, task_id: int, budget: int) -> Path:
    return (
        root
        / "artifacts/adapted_checkpoints"
        / f"task_{task_id}"
        / f"k_{budget}.json"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_adapted_manifest(root: Path, task_id: int, budget: int) -> Path:
    if not training_complete(task_id, budget):
        raise RuntimeError(f"Training is incomplete for task={task_id}, k={budget}")
    model = final_model(task_id, budget)
    config = json.loads((model / "config.json").read_text())
    if config.get("chunk_size") != TRAINED_CHUNK_SIZE:
        raise ValueError(
            f"Unexpected chunk_size in {model}: {config.get('chunk_size')}"
        )
    if config.get("n_action_steps") != TRAINED_ACTION_STEPS:
        raise ValueError(
            f"Unexpected trained n_action_steps in {model}: "
            f"{config.get('n_action_steps')}"
        )
    weights = model / "model.safetensors"
    output = adapted_manifest_path(root, task_id, budget)
    weights_sha256 = _sha256(weights)
    if output.is_file():
        existing = json.loads(output.read_text())
        if (
            existing.get("model_safetensors_bytes") == weights.stat().st_size
            and existing.get("training_step") == train_steps(budget)
            and existing.get("model_safetensors_sha256") == weights_sha256
        ):
            return output
    payload = {
        "task_id": task_id,
        "demo_budget": budget,
        "model": str(model),
        "training_step": train_steps(budget),
        "chunk_size": config["chunk_size"],
        "trained_n_action_steps": config["n_action_steps"],
        "model_safetensors_bytes": weights.stat().st_size,
        "model_safetensors_sha256": weights_sha256,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output


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
    runtime_base = root / "artifacts/runtime_base_checkpoint"
    if not (runtime_base / "model.safetensors").is_file():
        raise FileNotFoundError(f"Offline runtime checkpoint missing: {runtime_base}")
    # The base checkpoint and the target dataset share one schema (top /
    # wrist_image cameras at 128x128, 8-dim state) and one frame convention
    # (rot180 == eval), so no adapter or rename map is needed.  LeRobot still
    # replaces the checkpoint's normalization statistics with target-dataset
    # statistics at fine-tune time; both now come from the same conversion
    # pipeline, and the swap is disclosed in configs/protocol.yaml.
    return [
        str(trainer),
        f"--policy.path={runtime_base}",
        f"--policy.vlm_model_name={VLM_BACKBONE}",
        f"--policy.freeze_vision_encoder={str(FREEZE_VISION_ENCODER).lower()}",
        f"--policy.train_expert_only={str(TRAIN_EXPERT_ONLY).lower()}",
        f"--policy.train_state_proj={str(TRAIN_STATE_PROJ).lower()}",
        "--policy.use_amp=false",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={TARGET_DATASET_REPO}",
        f"--dataset.root={TARGET_DATASET_ROOT}",
        f"--dataset.episodes={json.dumps(episodes, separators=(',', ':'))}",
        "--dataset.video_backend=pyav",
        "--dataset.image_transforms.enable=true",
        f"--output_dir={job_output(task_id, budget)}",
        f"--job_name={EXPERIMENT_NAME}_task_{task_id}_k_{budget}",
        f"--steps={train_steps(budget)}",
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
    parser = argparse.ArgumentParser(
        description="Train one image-augmented full-fine-tune task/budget adaptation"
    )
    parser.add_argument("task_id", type=int)
    parser.add_argument("budget", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = build_command(root, args.task_id, args.budget)
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return
    if training_complete(args.task_id, args.budget):
        manifest = ensure_adapted_manifest(root, args.task_id, args.budget)
        print(
            json.dumps(
                {
                    "state": "already_complete",
                    "model": str(final_model(args.task_id, args.budget)),
                    "manifest": str(manifest),
                }
            )
        )
        return
    output = job_output(args.task_id, args.budget)
    if output.exists():
        raise FileExistsError(
            f"Incomplete output already exists; inspect before retrying: {output}"
        )
    subprocess.run(command, cwd=root, check=True)
    if not training_complete(args.task_id, args.budget):
        raise RuntimeError("Trainer exited successfully but final checkpoint is incomplete")
    manifest = ensure_adapted_manifest(root, args.task_id, args.budget)
    print(
        json.dumps(
            {
                "state": "completed",
                "model": str(final_model(args.task_id, args.budget)),
                "manifest": str(manifest),
            }
        )
    )


if __name__ == "__main__":
    main()
