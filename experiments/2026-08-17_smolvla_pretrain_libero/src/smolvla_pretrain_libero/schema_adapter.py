from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .constants import BASE_MODEL_REPO, BASE_MODEL_REVISION, INPUT_ORIENTATION

# Orientation and resolution contract: the converted dataset stores
# rot180(official) frames — already the exact eval convention that
# lerobot/processor/env_processor.py produces at rollout time — so no runtime
# transform is needed.  Frames stay at the native official 128x128; SmolVLA
# resizes every camera to resize_imgs_with_padding=(512,512) internally, and
# evaluation renders at the same 128x128 (observation size is derived from
# these feature shapes), so train and eval share one resize path.
LIBERO_INPUT_FEATURES = {
    "observation.images.top": {"type": "VISUAL", "shape": [3, 128, 128]},
    "observation.images.wrist_image": {
        "type": "VISUAL",
        "shape": [3, 128, 128],
    },
    "observation.state": {"type": "STATE", "shape": [8]},
}
LIBERO_OUTPUT_FEATURES = {
    "action": {"type": "ACTION", "shape": [7]},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_snapshot(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        elif item.name == "model.safetensors":
            os.link(item.resolve(), target)
        else:
            shutil.copy2(item, target, follow_symlinks=True)


def _adapt_processor_configs(destination: Path) -> None:
    pre_path = destination / "policy_preprocessor.json"
    pre = json.loads(pre_path.read_text())
    for step in pre["steps"]:
        if step["registry_name"] == "tokenizer_processor":
            step["config"]["padding"] = "longest"
        elif step["registry_name"] == "normalizer_processor":
            step["config"]["features"] = {
                **LIBERO_INPUT_FEATURES,
                **LIBERO_OUTPUT_FEATURES,
            }
    pre_path.write_text(json.dumps(pre, indent=2) + "\n")

    post_path = destination / "policy_postprocessor.json"
    post = json.loads(post_path.read_text())
    for step in post["steps"]:
        if step["registry_name"] == "unnormalizer_processor":
            step["config"]["features"] = LIBERO_OUTPUT_FEATURES
    post_path.write_text(json.dumps(post, indent=2) + "\n")


def adapt_base_snapshot(source: Path, destination: Path) -> dict[str, Any]:
    manifest_path = destination / "SCHEMA_ADAPTER.json"
    source_weights = source / "model.safetensors"
    source_hash = sha256(source_weights)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("source_revision") != BASE_MODEL_REVISION
            or manifest.get("model_sha256") != source_hash
        ):
            raise RuntimeError(f"Existing schema adapter has wrong provenance: {destination}")
        if sha256(destination / "model.safetensors") != source_hash:
            raise RuntimeError(f"Adapted model weights changed: {destination}")
        return manifest
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty {destination}")
        destination.rmdir()

    _copy_snapshot(source, destination)
    config_path = destination / "config.json"
    config = json.loads(config_path.read_text())
    config["input_features"] = LIBERO_INPUT_FEATURES
    config["output_features"] = LIBERO_OUTPUT_FEATURES
    config["prefix_length"] = -1
    config["pad_language_to"] = "longest"
    config["num_expert_layers"] = -1
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    _adapt_processor_configs(destination)

    if sha256(destination / "model.safetensors") != source_hash:
        raise RuntimeError("Schema adaptation unexpectedly changed model weights")
    manifest = {
        "source_repo": BASE_MODEL_REPO,
        "source_revision": BASE_MODEL_REVISION,
        "model_sha256": source_hash,
        "weight_change": "none; model.safetensors is a hard link to the pinned source",
        "input_orientation": INPUT_ORIENTATION,
        "input_features": LIBERO_INPUT_FEATURES,
        "output_features": LIBERO_OUTPUT_FEATURES,
        "config_overrides": {
            "prefix_length": -1,
            "pad_language_to": "longest",
            "num_expert_layers": -1,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
