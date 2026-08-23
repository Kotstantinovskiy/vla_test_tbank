from __future__ import annotations

"""LoRA trainable-parameter audit.

Wraps the pinned base policy exactly the way training does
(lerobot ``wrap_with_peft`` with this experiment's overrides) and asserts:

- every trainable parameter is either a LoRA adapter tensor on an intended
  VLM target module or part of a ``modules_to_save`` full-training copy of
  the expert / projections;
- every intended target group actually received adapters (text layers 0..14,
  vision encoder, connector) and none landed on the guard text layer 15;
- the base VLM, vision encoder, and original expert weights stay frozen.
"""

import argparse
import json
import re
from pathlib import Path

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from .constants import (
    FREEZE_VISION_ENCODER,
    FULL_TRAINING_MODULES,
    LORA_ALPHA,
    LORA_RANK,
    LORA_TARGET_REGEX,
    NUM_VLM_LAYERS,
    TRAIN_EXPERT_ONLY,
    TRAIN_STATE_PROJ,
    VLM_BACKBONE,
    experiment_root,
)

LORA_MARKERS = (".lora_A.", ".lora_B.")
SAVE_MARKER = ".modules_to_save."


def strip_peft_prefix(name: str) -> str:
    return name.removeprefix("base_model.model.")


def target_module_of_lora(name: str) -> str:
    """`<module>.lora_A.default.weight` -> `<module>` (peft prefix stripped)."""

    stripped = strip_peft_prefix(name)
    for marker in LORA_MARKERS:
        if marker in stripped:
            return stripped.split(marker)[0].removesuffix(".base_layer")
    raise ValueError(f"Not a LoRA parameter: {name}")


def classify_trainable(names: list[str]) -> dict[str, list[str]]:
    lora = [name for name in names if any(marker in name for marker in LORA_MARKERS)]
    saved = [name for name in names if SAVE_MARKER in strip_peft_prefix(name)]
    lora_targets = sorted({target_module_of_lora(name) for name in lora})
    return {
        "lora_parameters": lora,
        "full_training_parameters": saved,
        "lora_target_modules": lora_targets,
        "lora_text_targets": [t for t in lora_targets if ".text_model." in t],
        "lora_vision_targets": [t for t in lora_targets if ".vision_model." in t],
        "lora_connector_targets": [t for t in lora_targets if ".connector." in t],
    }


def audit(base: Path, output: Path, device: str) -> dict[str, object]:
    cfg = SmolVLAConfig.from_pretrained(base)
    cfg.pretrained_path = str(base)
    cfg.pretrained_revision = None
    cfg.vlm_model_name = str(VLM_BACKBONE)
    cfg.device = device
    cfg.freeze_vision_encoder = FREEZE_VISION_ENCODER
    cfg.train_expert_only = TRAIN_EXPERT_ONLY
    cfg.train_state_proj = TRAIN_STATE_PROJ
    if cfg.num_vlm_layers != NUM_VLM_LAYERS:
        raise RuntimeError(
            f"Base checkpoint num_vlm_layers changed: {cfg.num_vlm_layers} != {NUM_VLM_LAYERS}"
        )
    policy = SmolVLAPolicy.from_pretrained(base, config=cfg, local_files_only=True)
    peft_model = policy.wrap_with_peft(
        peft_cli_overrides={
            "method_type": "LORA",
            "r": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "target_modules": LORA_TARGET_REGEX,
            "modules_to_save": list(FULL_TRAINING_MODULES),
        }
    )
    trainable = [
        (name, parameter)
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    ]
    frozen = [
        (name, parameter)
        for name, parameter in peft_model.named_parameters()
        if not parameter.requires_grad
    ]
    names = [name for name, _ in trainable]
    groups = classify_trainable(names)

    unexpected = [
        name
        for name in names
        if not any(marker in name for marker in LORA_MARKERS)
        and SAVE_MARKER not in strip_peft_prefix(name)
    ]
    stray_targets = [
        target
        for target in groups["lora_target_modules"]
        if not re.fullmatch(LORA_TARGET_REGEX, target)
    ]
    guard_layer = f".text_model.layers.{NUM_VLM_LAYERS - 1}."
    guard_hits = [t for t in groups["lora_target_modules"] if guard_layer in t]
    saved_outside_protocol = [
        name
        for name in groups["full_training_parameters"]
        if not any(
            strip_peft_prefix(name).startswith(prefix + ".")
            for prefix in FULL_TRAINING_MODULES
        )
    ]
    missing = [
        key
        for key in ("lora_text_targets", "lora_vision_targets", "lora_connector_targets", "full_training_parameters")
        if not groups[key]
    ]
    expected_text = 15 * 7
    expected_vision = 12 * 6
    counts_ok = (
        len(groups["lora_text_targets"]) == expected_text
        and len(groups["lora_vision_targets"]) == expected_vision
        and len(groups["lora_connector_targets"]) == 1
    )
    if unexpected:
        raise RuntimeError(f"Non-adapter parameters are trainable: {unexpected[:10]}")
    if stray_targets or guard_hits:
        raise RuntimeError(
            f"LoRA adapters landed outside the frozen target set: {stray_targets[:5]} {guard_hits[:5]}"
        )
    if saved_outside_protocol:
        raise RuntimeError(
            f"Full-training copies outside the protocol: {saved_outside_protocol[:5]}"
        )
    if missing:
        raise RuntimeError(f"Expected trainable groups are missing: {missing}")
    if not counts_ok:
        raise RuntimeError(
            "LoRA target counts changed: "
            f"text={len(groups['lora_text_targets'])}/{expected_text}, "
            f"vision={len(groups['lora_vision_targets'])}/{expected_vision}, "
            f"connector={len(groups['lora_connector_targets'])}/1"
        )

    result: dict[str, object] = {
        "base": str(base.resolve()),
        "peft": {
            "method": "LORA",
            "r": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "target_modules": LORA_TARGET_REGEX,
            "modules_to_save": list(FULL_TRAINING_MODULES),
        },
        "flags": {
            "freeze_vision_encoder": FREEZE_VISION_ENCODER,
            "train_expert_only": TRAIN_EXPERT_ONLY,
            "train_state_proj": TRAIN_STATE_PROJ,
        },
        "total_parameters": sum(parameter.numel() for _, parameter in trainable + frozen),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "frozen_parameters": sum(parameter.numel() for _, parameter in frozen),
        "trainable_tensor_count": len(trainable),
        "frozen_tensor_count": len(frozen),
        "lora_parameter_count": sum(
            parameter.numel()
            for name, parameter in trainable
            if any(marker in name for marker in LORA_MARKERS)
        ),
        "full_training_parameter_count": sum(
            parameter.numel()
            for name, parameter in trainable
            if SAVE_MARKER in strip_peft_prefix(name)
        ),
        "lora_target_counts": {
            "text": len(groups["lora_text_targets"]),
            "vision": len(groups["lora_vision_targets"]),
            "connector": len(groups["lora_connector_targets"]),
        },
        "trainable_groups": {
            key: {"tensor_count": len(values), "examples": values[:6]}
            for key, values in groups.items()
        },
        "forbidden_trainable": unexpected + stray_targets + guard_hits + saved_outside_protocol,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Audit LoRA trainable parameters")
    parser.add_argument("--base", type=Path, default=root / "artifacts/base_checkpoint")
    parser.add_argument("--output", type=Path, default=root / "artifacts/trainable_parameters.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(audit(args.base, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
