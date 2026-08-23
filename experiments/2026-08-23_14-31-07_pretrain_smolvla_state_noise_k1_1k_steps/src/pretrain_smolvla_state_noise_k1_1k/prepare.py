from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .constants import (
    ALPHAS,
    BASE_CHECKPOINT,
    BASE_MODEL_SHA256,
    BASE_PROVENANCE,
    DEMO_BUDGET,
    EVAL_ACTION_STEPS,
    EVAL_EPISODES,
    EVAL_HORIZON,
    EXPERIMENT_NAME,
    MASTER_SEED,
    LIBERO_ASSETS_REVISION,
    LIBERO_ASSETS_ROOT,
    OFFICIAL_SOURCE_REVISION,
    OUTPUT_ROOT,
    TARGET_DATASET_REPO,
    TARGET_DATASET_ROOT,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
    TRAINED_ACTION_STEPS,
    TRAINED_CHUNK_SIZE,
    VLM_BACKBONE,
    VLM_BACKBONE_REVISION,
    VLM_MODEL_BYTES,
    VLM_MODEL_SHA256,
    alpha_tag,
    experiment_root,
    noise_seed,
)
from .libero_setup import ensure_libero_config
from .selection import build_manifest

REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "train_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink artifact: {link}")
    link.symlink_to(target, target_is_directory=True)


def write_runtime_base_checkpoint(root: Path) -> dict[str, object]:
    """Create a lightweight offline view without modifying canonical files."""

    runtime = root / "artifacts/runtime_base_checkpoint"
    runtime.mkdir(parents=True, exist_ok=True)
    linked_files = (
        "model.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "train_config.json",
    )
    for name in linked_files:
        source = BASE_CHECKPOINT / name
        destination = runtime / name
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise RuntimeError(f"Runtime checkpoint link points elsewhere: {destination}")
        elif destination.exists():
            raise FileExistsError(f"Refusing to replace runtime artifact: {destination}")
        else:
            destination.symlink_to(source)

    config = json.loads((BASE_CHECKPOINT / "config.json").read_text())
    config["pretrained_path"] = str(runtime)
    config["pretrained_revision"] = None
    config["vlm_model_name"] = str(VLM_BACKBONE)
    (runtime / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    preprocessor = json.loads(
        (BASE_CHECKPOINT / "policy_preprocessor.json").read_text()
    )
    tokenizer_steps = [
        step
        for step in preprocessor["steps"]
        if step.get("registry_name") == "tokenizer_processor"
    ]
    if len(tokenizer_steps) != 1:
        raise ValueError("Expected exactly one tokenizer processor in base checkpoint")
    tokenizer_steps[0]["config"]["tokenizer_name"] = str(VLM_BACKBONE)
    (runtime / "policy_preprocessor.json").write_text(
        json.dumps(preprocessor, indent=2) + "\n"
    )
    manifest = {
        "runtime_checkpoint": str(runtime),
        "canonical_checkpoint": str(BASE_CHECKPOINT),
        "model_safetensors_sha256": BASE_MODEL_SHA256,
        "model_safetensors_is_symlink": (runtime / "model.safetensors").is_symlink(),
        "patched_fields": {
            "config.pretrained_path": str(runtime),
            "config.vlm_model_name": str(VLM_BACKBONE),
            "policy_preprocessor.tokenizer_name": str(VLM_BACKBONE),
        },
    }
    (root / "artifacts/runtime_base_checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def base_checkpoint_manifest() -> dict[str, object]:
    missing = [
        name for name in REQUIRED_CHECKPOINT_FILES if not (BASE_CHECKPOINT / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete base checkpoint, missing: {missing}")
    config = json.loads((BASE_CHECKPOINT / "config.json").read_text())
    if config.get("chunk_size") != TRAINED_CHUNK_SIZE:
        raise ValueError(f"Base chunk_size changed: {config.get('chunk_size')}")
    if config.get("n_action_steps") != TRAINED_ACTION_STEPS:
        raise ValueError(f"Base n_action_steps changed: {config.get('n_action_steps')}")
    weights = BASE_CHECKPOINT / "model.safetensors"
    weights_sha256 = sha256(weights)
    if weights_sha256 != BASE_MODEL_SHA256:
        raise ValueError(
            f"Base checkpoint SHA-256 changed: {weights_sha256} != {BASE_MODEL_SHA256}"
        )
    required_backbone = (
        "config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
    )
    missing_backbone = [
        name for name in required_backbone if not (VLM_BACKBONE / name).is_file()
    ]
    if missing_backbone:
        raise FileNotFoundError(f"Incomplete pinned VLM backbone: {missing_backbone}")
    backbone_weights = VLM_BACKBONE / "model.safetensors"
    backbone_sha256 = sha256(backbone_weights)
    if backbone_weights.stat().st_size != VLM_MODEL_BYTES:
        raise ValueError("Pinned VLM backbone size changed")
    if backbone_sha256 != VLM_MODEL_SHA256:
        raise ValueError("Pinned VLM backbone SHA-256 changed")
    return {
        "checkpoint_path": str(BASE_CHECKPOINT),
        "model_safetensors_sha256": weights_sha256,
        "model_safetensors_bytes": weights.stat().st_size,
        "chunk_size": config["chunk_size"],
        "n_action_steps": config["n_action_steps"],
        "provenance": BASE_PROVENANCE,
        "vlm_backbone_path": str(VLM_BACKBONE),
        "vlm_backbone_revision": VLM_BACKBONE_REVISION,
        "vlm_model_safetensors_sha256": backbone_sha256,
        "vlm_model_safetensors_bytes": backbone_weights.stat().st_size,
        "libero_assets_path": str(LIBERO_ASSETS_ROOT),
        "libero_assets_revision": LIBERO_ASSETS_REVISION,
    }


def validate_task_mapping() -> list[dict[str, object]]:
    from libero.libero.benchmark import get_benchmark

    benchmark = get_benchmark(TARGET_SUITE)()
    total = getattr(benchmark, "n_tasks", None) or benchmark.get_num_tasks()
    if total != 10:
        raise ValueError(f"Expected the LIBERO goal suite to expose 10 tasks, got {total}")
    mapping = []
    for logical_task_id, expected in TARGET_INSTRUCTIONS.items():
        env_task_id = TARGET_ENV_TASK_IDS[logical_task_id]
        task = benchmark.get_task(env_task_id)
        if task.language != expected:
            raise ValueError(
                f"Task mapping mismatch for logical task {logical_task_id}: "
                f"{task.language!r} != {expected!r}"
            )
        mapping.append(
            {
                "logical_task_id": logical_task_id,
                "env_task_id": env_task_id,
                "name": task.name,
                "instruction": task.language,
            }
        )
    return mapping


def evaluation_plan(task_mapping: list[dict[str, object]]) -> dict[str, object]:
    training_jobs = [
        {
            "label": f"task_{task_id}__alpha_{alpha_tag(alpha)}",
            "logical_task_id": task_id,
            "env_task_id": TARGET_ENV_TASK_IDS[task_id],
            "instruction": TARGET_INSTRUCTIONS[task_id],
            "demo_budget": DEMO_BUDGET,
            "state_noise_alpha": alpha,
            "chunk_size": TRAINED_CHUNK_SIZE,
            "trained_n_action_steps": TRAINED_ACTION_STEPS,
        }
        for alpha in ALPHAS
        for task_id in TARGET_INSTRUCTIONS
    ]
    points = [
        {
            **job,
            "label": f"{job['label']}__n_{action_steps}",
            "n_action_steps": action_steps,
        }
        for job in training_jobs
        for action_steps in EVAL_ACTION_STEPS
    ]
    return {
        "experiment": EXPERIMENT_NAME,
        "training_seed": MASTER_SEED,
        "training_jobs": len(training_jobs),
        "evaluation_points": len(points),
        "eval_action_steps": list(EVAL_ACTION_STEPS),
        "state_noise_alphas": list(ALPHAS),
        "state_noise_location": (
            "normalized observation.state (MEAN_STD) inside policy.forward, "
            "training only; alpha=0.00 is a strict no-op"
        ),
        "episodes_per_point": EVAL_EPISODES,
        "episode_horizon": EVAL_HORIZON,
        "main_rollout_videos": len(points) * EVAL_EPISODES,
        "maximum_policy_invocations": (
            sum(
                math.ceil(EVAL_HORIZON / action_steps)
                for action_steps in EVAL_ACTION_STEPS
            )
            * len(training_jobs)
            * EVAL_EPISODES
        ),
        "eval_batch_size": 1,
        "env_seeds": [MASTER_SEED + index for index in range(EVAL_EPISODES)],
        "noise_seeds": [noise_seed(index) for index in range(EVAL_EPISODES)],
        "noise_protocol": "torch/CUDA reseed before each single-episode eval_policy call",
        "vlm_backbone_revision": VLM_BACKBONE_REVISION,
        "libero_assets_revision": LIBERO_ASSETS_REVISION,
        "task_mapping": task_mapping,
        "points": points,
    }


def main() -> None:
    root = experiment_root()
    predictions = root / "reports/PRIOR_EXPECTATION.md"
    if not predictions.is_file():
        raise FileNotFoundError("Predictions must be recorded before preparation")

    config_path = ensure_libero_config()
    base_manifest = base_checkpoint_manifest()
    (root / "artifacts/base_checkpoint_manifest.json").write_text(
        json.dumps(base_manifest, indent=2) + "\n"
    )
    write_runtime_base_checkpoint(root)

    conversion_path = TARGET_DATASET_ROOT / "conversion_manifest.json"
    if not conversion_path.is_file():
        raise FileNotFoundError(f"Target conversion manifest missing: {conversion_path}")
    conversion = json.loads(conversion_path.read_text())
    if conversion.get("source_revision") != OFFICIAL_SOURCE_REVISION:
        raise ValueError("Target dataset source revision changed")
    episode_manifest = build_manifest(conversion)
    (root / "artifacts/episode_manifest.json").write_text(
        json.dumps(episode_manifest, indent=2) + "\n"
    )

    task_mapping = validate_task_mapping()
    plan = evaluation_plan(task_mapping)
    plan["predictions_sha256"] = sha256(predictions)
    (root / "artifacts/evaluation_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )

    replace_symlink(root / "artifacts/base_checkpoint", BASE_CHECKPOINT)
    replace_symlink(root / "artifacts/vlm_backbone", VLM_BACKBONE)
    replace_symlink(root / "artifacts/libero_assets", LIBERO_ASSETS_ROOT)
    replace_symlink(root / "artifacts/dataset_target", TARGET_DATASET_ROOT)
    replace_symlink(root / "artifacts/checkpoints", OUTPUT_ROOT)
    print(
        json.dumps(
            {
                "libero_config": str(config_path),
                "base_sha256": base_manifest["model_safetensors_sha256"],
                "dataset": TARGET_DATASET_REPO,
                "training_jobs": plan["training_jobs"],
                "evaluation_points": plan["evaluation_points"],
                "main_rollout_videos": plan["main_rollout_videos"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
