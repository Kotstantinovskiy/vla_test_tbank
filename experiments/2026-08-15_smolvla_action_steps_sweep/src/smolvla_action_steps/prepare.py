from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from .constants import ADAPTED_BUDGETS, TARGET_INSTRUCTIONS


def libero_paths() -> dict[str, str]:
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("LIBERO is not installed in the active environment")
    package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    benchmark_root = package_root / "libero"
    return {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(package_root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_entry(checkpoints_root: Path, task_id: int, budget: int) -> dict[str, Any]:
    checkpoint = (
        checkpoints_root
        / "naive"
        / f"task_{task_id}"
        / f"k_{budget}"
        / "checkpoints"
        / "last"
        / "pretrained_model"
    )
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Incomplete checkpoint: {checkpoint}")
    config = json.loads(config_path.read_text())
    if config.get("chunk_size") != 50 or config.get("n_action_steps") != 50:
        raise ValueError(f"Unexpected action horizon in {config_path}")
    return {
        "task_id": task_id,
        "demo_budget": budget,
        "path": str(checkpoint),
        "resolved_path": str(checkpoint.resolve()),
        "training_step": 2000,
        "chunk_size": config["chunk_size"],
        "original_n_action_steps": config["n_action_steps"],
        "model_size_bytes": weights_path.stat().st_size,
        "config_sha256": sha256(config_path),
        "_weights_path": weights_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare checkpoint and LIBERO provenance")
    parser.add_argument("--external-checkpoints", type=Path, default=Path("/var/tmp/vla_outputs"))
    parser.add_argument("--checkpoint-link", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--libero-config-dir", type=Path, default=Path("artifacts/libero_config"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/checkpoint_manifest.json"))
    args = parser.parse_args()

    external = args.external_checkpoints.resolve()
    if not external.is_dir():
        raise FileNotFoundError(external)
    args.checkpoint_link.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_link.is_symlink():
        if args.checkpoint_link.resolve() != external:
            raise RuntimeError(f"Checkpoint link points elsewhere: {args.checkpoint_link}")
    elif args.checkpoint_link.exists():
        raise RuntimeError(f"Checkpoint link target already exists: {args.checkpoint_link}")
    else:
        args.checkpoint_link.symlink_to(external, target_is_directory=True)

    paths = libero_paths()
    required = ("benchmark_root", "bddl_files", "init_states")
    missing = [name for name in required if not Path(paths[name]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing LIBERO resources: {missing}")
    args.libero_config_dir.mkdir(parents=True, exist_ok=True)
    (args.libero_config_dir / "config.yaml").write_text(
        yaml.safe_dump(paths, sort_keys=False)
    )

    entries = [
        checkpoint_entry(args.checkpoint_link, task_id, budget)
        for task_id in TARGET_INSTRUCTIONS
        for budget in ADAPTED_BUDGETS
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = list(pool.map(lambda item: sha256(item["_weights_path"]), entries))
    for entry, weights_hash in zip(entries, hashes):
        entry.pop("_weights_path")
        entry["model_sha256"] = weights_hash

    manifest = {
        "weights_modified": False,
        "external_checkpoints_root": str(external),
        "experiment_checkpoint_link": str(args.checkpoint_link),
        "checkpoints": entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
