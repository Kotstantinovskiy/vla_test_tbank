from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import model_info

from .constants import ARTIFACTS_DIR, CONFIG_PATH, RESULTS_DIR, TARGET_GOAL_INSTRUCTIONS
from .data import load_episode_records, split_records_by_task, task_counts
from .utils import atomic_json, load_config, now_iso


def prepare(config_path: str | Path = CONFIG_PATH) -> dict:
    config = load_config(config_path)
    data_config = config["data"]
    model_config = config["model"]
    root = Path(data_config["root"])
    conversion = json.loads((root / "conversion_manifest.json").read_text())
    expected = {
        "suite": data_config["suite"],
        "fps": data_config["fps"],
        "image_size": data_config["image_size"],
        "orientation": data_config["expected_orientation"],
        "source_revision": data_config["expected_source_revision"],
    }
    observed = {key: conversion.get(key) for key in expected}
    if observed != expected:
        raise AssertionError(f"dataset contract mismatch: expected={expected}, observed={observed}")

    records = load_episode_records(root, data_config["camera"])
    train, validation, validation_tasks = split_records_by_task(
        records,
        float(data_config["validation_task_fraction"]),
        int(config["experiment"]["seed"]),
    )
    all_tasks = {row.task for row in records}
    leaked_targets = sorted(all_tasks & TARGET_GOAL_INSTRUCTIONS)
    if leaked_targets:
        raise AssertionError(f"target Goal instructions leaked into LIBERO-90: {leaked_targets}")

    split_manifest = {
        "created_at": now_iso(),
        "seed": config["experiment"]["seed"],
        "split_unit": "natural-language task instruction",
        "validation_task_fraction": data_config["validation_task_fraction"],
        "dataset_root": str(root),
        "camera": data_config["camera"],
        "total_episodes": len(records),
        "total_tasks": len(all_tasks),
        "train_episode_indices": [row.episode_index for row in train],
        "validation_episode_indices": [row.episode_index for row in validation],
        "train_task_counts": task_counts(train),
        "validation_task_counts": task_counts(validation),
        "validation_tasks": validation_tasks,
        "target_goal_instructions_present": leaked_targets,
    }
    atomic_json(ARTIFACTS_DIR / "dataset_split.json", split_manifest)

    hub_cache = Path(os.environ.get("HF_HUB_CACHE", "/var/tmp/vla_hf/hub"))
    cache_name = f"models--{model_config['repo_id'].replace('/', '--')}"
    snapshot_path = hub_cache / cache_name / "snapshots" / model_config["revision"]
    if snapshot_path.is_dir():
        resolved_revision = model_config["revision"]
    else:
        info = model_info(model_config["repo_id"], revision=model_config["revision"])
        resolved_revision = info.sha
    if resolved_revision != model_config["revision"]:
        raise AssertionError(
            f"model revision resolved to {resolved_revision}, expected {model_config['revision']}"
        )
    model_manifest = {
        "created_at": now_iso(),
        "repo_id": model_config["repo_id"],
        "revision": resolved_revision,
        "local_snapshot": str(snapshot_path) if snapshot_path.is_dir() else None,
        "num_progress_bins": model_config["num_progress_bins"],
        "max_frames": model_config["max_frames"],
        "gradient_checkpointing": model_config["gradient_checkpointing"],
        "training_heads": ["progress"],
        "omitted_heads": ["preference", "success"],
    }
    atomic_json(ARTIFACTS_DIR / "model_manifest.json", model_manifest)
    if snapshot_path.is_dir():
        model_link = ARTIFACTS_DIR / "qwen35_4b_base"
        if model_link.is_symlink() and model_link.resolve() != snapshot_path.resolve():
            raise AssertionError(f"existing model link points to {model_link.resolve()}")
        if not model_link.exists():
            os.symlink(snapshot_path, model_link, target_is_directory=True)
    atomic_json(
        ARTIFACTS_DIR / "protocol_manifest.json",
        {"created_at": now_iso(), "config": config, "full_training_launched": False},
    )
    root_status = RESULTS_DIR / "status.json"
    if not root_status.exists():
        atomic_json(
            root_status,
            {
                "state": "prepared",
                "updated_at": now_iso(),
                "full_training_launched": False,
            },
        )
    dataset_link = ARTIFACTS_DIR / "libero_90"
    if dataset_link.is_symlink() and dataset_link.resolve() != root.resolve():
        raise AssertionError(f"existing dataset link points to {dataset_link.resolve()}")
    if not dataset_link.exists():
        os.symlink(root, dataset_link, target_is_directory=True)
    return split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), indent=2))


if __name__ == "__main__":
    main()
