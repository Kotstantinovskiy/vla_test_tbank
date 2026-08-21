from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .constants import (
    BASE_CHECKPOINT,
    BASE_PROVENANCE,
    TARGET_DATASET_REPO,
    TARGET_DATASET_ROOT,
    experiment_root,
)
from .evaluate import ensure_libero_config
from .selection import build_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(
        description="LIBERO config, base-checkpoint manifest, and frozen demo selection"
    )
    args = parser.parse_args()

    config_path = ensure_libero_config()

    if not (BASE_CHECKPOINT / "model.safetensors").is_file():
        raise FileNotFoundError(f"Base checkpoint missing: {BASE_CHECKPOINT}")
    base_manifest = {
        "checkpoint_path": str(BASE_CHECKPOINT),
        "model_safetensors_sha256": sha256(BASE_CHECKPOINT / "model.safetensors"),
        "provenance": BASE_PROVENANCE,
    }
    (root / "artifacts/base_checkpoint_manifest.json").write_text(
        json.dumps(base_manifest, indent=2) + "\n"
    )

    conversion_manifest_path = TARGET_DATASET_ROOT / "conversion_manifest.json"
    if not conversion_manifest_path.is_file():
        raise FileNotFoundError(
            f"Target conversion manifest missing: {conversion_manifest_path}"
        )
    conversion = json.loads(conversion_manifest_path.read_text())
    episode_manifest = build_manifest(conversion)
    (root / "artifacts/episode_manifest.json").write_text(
        json.dumps(episode_manifest, indent=2) + "\n"
    )

    print(
        json.dumps(
            {
                "libero_config": str(config_path),
                "base_sha256": base_manifest["model_safetensors_sha256"],
                "dataset": TARGET_DATASET_REPO,
                "episode_manifest": "artifacts/episode_manifest.json",
                "tasks": len(episode_manifest["tasks"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
