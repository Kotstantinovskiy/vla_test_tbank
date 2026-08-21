from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import yaml

from .constants import CHECKPOINT_PATH, CHECKPOINT_PROVENANCE

CHECKPOINT_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "train_config.json",
)


def libero_paths() -> dict[str, str]:
    """Resolve LIBERO resources from this experiment's active Python environment."""

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


def write_libero_config(config_dir: Path) -> Path:
    paths = libero_paths()
    # The PyPI build omits optional asset and demonstration directories.  They
    # still belong in LIBERO's five-key config, while prompt-only simulation
    # requires benchmark definitions, BDDL files, and initial states.
    required = ("benchmark_root", "bddl_files", "init_states")
    missing = [name for name in required if not Path(paths[name]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing LIBERO resource directories: {missing}")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(paths, sort_keys=False))
    return config_path


def checkpoint_manifest(checkpoint: Path) -> dict[str, object]:
    """Validate checkpoint completeness and fingerprint the weights."""

    missing = [
        name for name in CHECKPOINT_REQUIRED_FILES if not (checkpoint / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint {checkpoint} is incomplete, missing: {missing}"
        )
    digest = hashlib.sha256()
    with (checkpoint / "model.safetensors").open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return {
        "checkpoint_path": str(checkpoint),
        "files": sorted(
            path.name for path in checkpoint.iterdir() if path.is_file()
        ),
        "model_safetensors_sha256": digest.hexdigest(),
        "model_safetensors_bytes": (checkpoint / "model.safetensors").stat().st_size,
        "provenance": CHECKPOINT_PROVENANCE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create the experiment-local LIBERO configuration and the frozen "
            "checkpoint manifest"
        )
    )
    parser.add_argument(
        "--config-dir", type=Path, default=Path("artifacts/libero_config")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/checkpoint_manifest.json")
    )
    args = parser.parse_args()

    config_path = write_libero_config(args.config_dir)
    manifest = checkpoint_manifest(CHECKPOINT_PATH)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"config": str(config_path), "manifest": str(args.manifest), **manifest},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
