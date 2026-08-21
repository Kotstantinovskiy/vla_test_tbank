from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from .constants import (
    BASE_MODEL_REPO,
    BASE_MODEL_REVISION,
    EFFECTIVE_BATCH_SIZE,
    EXPECTED_SEEN_TASKS,
    EXPECTED_TARGET_TASKS,
    FPS,
    GPU_IDS,
    INPUT_ORIENTATION,
    LEARNING_RATE,
    OFFICIAL_REPO,
    OFFICIAL_REVISION,
    OFFICIAL_ROOT,
    PER_RANK_BATCH_SIZE,
    SEED,
    TRAIN_STEPS,
    WORLD_SIZE,
    experiment_root,
    seen_dataset_root,
    target_dataset_root,
)
from .schema_adapter import adapt_base_snapshot


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink artifact: {link}")
    link.symlink_to(target, target_is_directory=True)


def converted_dataset_status(root: Path, expected_tasks: int) -> dict[str, Any]:
    manifest_path = root / "conversion_manifest.json"
    if not manifest_path.is_file():
        return {"root": str(root), "state": "pending", "reason": "not converted yet"}
    manifest = json.loads(manifest_path.read_text())
    info = json.loads((root / "meta/info.json").read_text())
    problems = []
    if manifest["total_tasks"] != expected_tasks:
        problems.append(f"tasks {manifest['total_tasks']} != {expected_tasks}")
    if info.get("fps") != FPS:
        problems.append(f"fps {info.get('fps')} != {FPS}")
    if info.get("total_episodes") != manifest["total_episodes"]:
        problems.append("info/manifest episode mismatch")
    if problems:
        raise ValueError(f"Converted dataset {root} invalid: {problems}")
    return {
        "root": str(root),
        "state": "converted",
        "repo_id": manifest["repo_id"],
        "total_tasks": manifest["total_tasks"],
        "total_episodes": manifest["total_episodes"],
        "total_frames": manifest["total_frames"],
        "fps": info["fps"],
        "orientation": manifest["orientation"],
    }


def main() -> None:
    root = experiment_root()
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    hf_home = Path(os.environ.get("HF_HOME", "/var/tmp/vla_hf"))
    output_root = Path(
        os.environ.get(
            "VLA_OFFICIAL_OUTPUT_ROOT",
            "/var/tmp/vla_outputs/seen_libero90_official_20260817",
        )
    ).resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)

    if not OFFICIAL_ROOT.is_dir():
        raise FileNotFoundError(
            f"Official download root missing: {OFFICIAL_ROOT} (run the pinned "
            "root downloader scripts/download_official_libero.py first)"
        )

    seen = converted_dataset_status(seen_dataset_root(), EXPECTED_SEEN_TASKS)
    target = converted_dataset_status(target_dataset_root(), EXPECTED_TARGET_TASKS)

    base_snapshot = Path(
        snapshot_download(
            BASE_MODEL_REPO,
            revision=BASE_MODEL_REVISION,
            cache_dir=hf_home / "hub",
        )
    ).resolve()

    adapted_snapshot = (
        output_root.parent / "smolvla_base_libero_official_schema_c83c3163"
    )
    adapter_manifest = adapt_base_snapshot(base_snapshot, adapted_snapshot)
    replace_symlink(artifacts / "base_model_source", base_snapshot)
    replace_symlink(artifacts / "base_model", adapted_snapshot)
    replace_symlink(artifacts / "official_source", OFFICIAL_ROOT)
    if seen["state"] == "converted":
        replace_symlink(artifacts / "dataset_seen", seen_dataset_root())
    if target["state"] == "converted":
        replace_symlink(artifacts / "dataset_target", target_dataset_root())
    # LeRobot deliberately refuses a fresh run when output_dir already exists.
    # A dangling symlink is fine here and becomes live as soon as training starts.
    replace_symlink(artifacts / "checkpoints", output_root)

    manifest = {
        "base_model": {"repo_id": BASE_MODEL_REPO, "revision": BASE_MODEL_REVISION},
        "schema_adapter": adapter_manifest,
        "official_source": {
            "repo_id": OFFICIAL_REPO,
            "revision": OFFICIAL_REVISION,
            "root": str(OFFICIAL_ROOT),
        },
        "input_orientation": INPUT_ORIENTATION,
        "datasets": {"seen": seen, "target": target},
        "training": {
            "strategy": "ddp",
            "gpu_ids": list(GPU_IDS),
            "world_size": WORLD_SIZE,
            "per_rank_batch_size": PER_RANK_BATCH_SIZE,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "steps": TRAIN_STEPS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "output_root": str(output_root),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    if manifest["runtime"]["cuda_device_count"] < WORLD_SIZE:
        raise RuntimeError(
            f"Need {WORLD_SIZE} GPUs, found {manifest['runtime']['cuda_device_count']}"
        )
    (artifacts / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
