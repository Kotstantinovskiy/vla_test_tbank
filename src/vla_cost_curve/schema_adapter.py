from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from .constants import CAMERA_KEY_RENAMES, SEEN_REPO, SEEN_REVISION


def _rename_feature_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            CAMERA_KEY_RENAMES.get(key, key): _rename_feature_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_feature_keys(item) for item in value]
    return value


def _rename_safetensor_keys(path: Path) -> None:
    with safe_open(path, framework="pt", device="cpu") as source:
        tensors = {
            next(
                (
                    new + key[len(old) :]
                    for old, new in CAMERA_KEY_RENAMES.items()
                    if key.startswith(old)
                ),
                key,
            ): source.get_tensor(key)
            for key in source.keys()
        }
        metadata = source.metadata()
    save_file(tensors, path, metadata=metadata)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapt_checkpoint(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "SCHEMA_ADAPTER.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source_revision") != SEEN_REVISION:
            raise RuntimeError(f"Existing adapter has unexpected provenance: {manifest_path}")
        return manifest
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")

    source = Path(
        snapshot_download(
            SEEN_REPO,
            revision=SEEN_REVISION,
            local_dir=output_dir,
        )
    )
    if source != output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, output_dir, dirs_exist_ok=True)

    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        path = output_dir / name
        data = json.loads(path.read_text())
        path.write_text(json.dumps(_rename_feature_keys(data), indent=2) + "\n")

    for path in output_dir.glob("*processor*.safetensors"):
        _rename_safetensor_keys(path)

    manifest = {
        "source_repo": SEEN_REPO,
        "source_revision": SEEN_REVISION,
        "model_sha256": _sha256(output_dir / "model.safetensors"),
        "weight_change": "none; only feature names and processor-stat keys were renamed",
        "renames": CAMERA_KEY_RENAMES,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/seen_image_schema"))
    args = parser.parse_args()
    print(json.dumps(adapt_checkpoint(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()

