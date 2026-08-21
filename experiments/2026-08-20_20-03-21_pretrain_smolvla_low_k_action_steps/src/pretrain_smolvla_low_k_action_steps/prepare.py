from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .constants import (
    ACTION_STEPS,
    BASE_CHECKPOINT,
    BASE_PROVENANCE,
    DEMO_BUDGETS,
    EVAL_EPISODES,
    EVAL_HORIZON,
    EXPERIMENT_NAME,
    MASTER_SEED,
    TARGET_DATASET_REPO,
    TARGET_DATASET_ROOT,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
    TRAINED_ACTION_STEPS,
    TRAINED_CHUNK_SIZE,
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
        raise ValueError(
            f"Base n_action_steps changed: {config.get('n_action_steps')}"
        )
    weights = BASE_CHECKPOINT / "model.safetensors"
    return {
        "checkpoint_path": str(BASE_CHECKPOINT),
        "model_safetensors_sha256": sha256(weights),
        "model_safetensors_bytes": weights.stat().st_size,
        "chunk_size": config["chunk_size"],
        "n_action_steps": config["n_action_steps"],
        "provenance": BASE_PROVENANCE,
    }


def validate_task_mapping() -> list[dict[str, object]]:
    # Import only after ensure_libero_config() has written config.yaml.
    from libero.libero.benchmark import get_benchmark

    benchmark = get_benchmark(TARGET_SUITE)()
    total = getattr(benchmark, "n_tasks", None) or benchmark.get_num_tasks()
    if total != len(TARGET_INSTRUCTIONS):
        raise ValueError(f"Expected 10 target tasks, got {total}")
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
    points = [
        {
            "label": f"task_{task_id}__k_{budget}__n_{action_steps}",
            "logical_task_id": task_id,
            "env_task_id": TARGET_ENV_TASK_IDS[task_id],
            "instruction": TARGET_INSTRUCTIONS[task_id],
            "demo_budget": budget,
            "n_action_steps": action_steps,
            "chunk_size": TRAINED_CHUNK_SIZE,
        }
        for budget in DEMO_BUDGETS
        for task_id in TARGET_INSTRUCTIONS
        for action_steps in ACTION_STEPS
    ]
    return {
        "experiment": EXPERIMENT_NAME,
        "training_jobs": len(TARGET_INSTRUCTIONS) * len(DEMO_BUDGETS),
        "evaluation_points": len(points),
        "episodes_per_point": EVAL_EPISODES,
        "episode_horizon": EVAL_HORIZON,
        "main_rollout_videos": len(points) * EVAL_EPISODES,
        "maximum_policy_invocations": sum(
            math.ceil(EVAL_HORIZON / action_steps)
            * EVAL_EPISODES
            * len(TARGET_INSTRUCTIONS)
            * len(DEMO_BUDGETS)
            for action_steps in ACTION_STEPS
        ),
        "master_seed": MASTER_SEED,
        "env_seeds": [MASTER_SEED + index for index in range(EVAL_EPISODES)],
        "noise_seeds": [noise_seed(index) for index in range(EVAL_EPISODES)],
        "task_mapping": task_mapping,
        "points": points,
    }


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(
        description="Freeze LIBERO, checkpoint, demo-selection and eval manifests"
    )
    parser.parse_args()

    config_path = ensure_libero_config()
    base_manifest = base_checkpoint_manifest()
    base_path = root / "artifacts/base_checkpoint_manifest.json"
    base_path.write_text(json.dumps(base_manifest, indent=2) + "\n")

    conversion_manifest_path = TARGET_DATASET_ROOT / "conversion_manifest.json"
    if not conversion_manifest_path.is_file():
        raise FileNotFoundError(
            f"Target conversion manifest missing: {conversion_manifest_path}"
        )
    conversion = json.loads(conversion_manifest_path.read_text())
    episode_manifest = build_manifest(conversion)
    episode_path = root / "artifacts/episode_manifest.json"
    episode_path.write_text(json.dumps(episode_manifest, indent=2) + "\n")

    task_mapping = validate_task_mapping()
    plan = evaluation_plan(task_mapping)
    prior_path = root / "artifacts/prior_action_steps_evidence.json"
    prior = json.loads(prior_path.read_text())
    canonical = prior["canonical_official_data_lineage"]
    if canonical["adapted_checkpoint_configs_checked"] != 30:
        raise ValueError("Prior action-step evidence did not inspect all 30 checkpoints")
    if canonical["distinct_n_action_steps"] != [TRAINED_ACTION_STEPS]:
        raise ValueError("Canonical prior action-step evidence is not n=50-only")
    plan_path = root / "artifacts/evaluation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    print(
        json.dumps(
            {
                "libero_config": str(config_path),
                "base_sha256": base_manifest["model_safetensors_sha256"],
                "dataset": TARGET_DATASET_REPO,
                "episode_manifest": str(episode_path),
                "evaluation_plan": str(plan_path),
                "prior_action_steps_evidence": str(prior_path),
                "training_jobs": plan["training_jobs"],
                "evaluation_points": plan["evaluation_points"],
                "main_rollout_videos": plan["main_rollout_videos"],
                "maximum_policy_invocations": plan["maximum_policy_invocations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
