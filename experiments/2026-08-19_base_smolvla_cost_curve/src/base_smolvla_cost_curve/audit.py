from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from .constants import experiment_root


def classify_trainable(names: list[str]) -> dict[str, list[str]]:
    groups = {
        "action_expert": [name for name in names if ".lm_expert." in name],
        "state_projection": [name for name in names if ".state_proj." in name],
        "action_input_projection": [name for name in names if ".action_in_proj." in name],
        "action_output_projection": [name for name in names if ".action_out_proj." in name],
        "action_time_mlp": [name for name in names if ".action_time_mlp_" in name],
    }
    return groups


def audit(base: Path, output: Path, device: str) -> dict[str, object]:
    cfg = SmolVLAConfig.from_pretrained(base)
    cfg.pretrained_path = str(base)
    cfg.pretrained_revision = None
    cfg.device = device
    cfg.freeze_vision_encoder = True
    cfg.train_expert_only = True
    cfg.train_state_proj = True
    policy = SmolVLAPolicy.from_pretrained(base, config=cfg, local_files_only=True)
    trainable = [(name, param) for name, param in policy.named_parameters() if param.requires_grad]
    frozen = [(name, param) for name, param in policy.named_parameters() if not param.requires_grad]
    names = [name for name, _ in trainable]
    groups = classify_trainable(names)
    forbidden = [name for name in names if ".vlm_with_expert.vlm." in name or "vision_model" in name]
    missing = [group for group, members in groups.items() if not members]
    if forbidden:
        raise RuntimeError(f"Frozen VLM/vision parameters are unexpectedly trainable: {forbidden[:10]}")
    if missing:
        raise RuntimeError(f"Expected trainable parameter groups are missing: {missing}")
    result: dict[str, object] = {
        "base": str(base.resolve()),
        "flags": {
            "freeze_vision_encoder": True,
            "train_expert_only": True,
            "train_state_proj": True,
        },
        "total_parameters": sum(param.numel() for _, param in trainable + frozen),
        "trainable_parameters": sum(param.numel() for _, param in trainable),
        "frozen_parameters": sum(param.numel() for _, param in frozen),
        "trainable_tensor_count": len(trainable),
        "frozen_tensor_count": len(frozen),
        "trainable_groups": {
            group: {
                "tensor_count": len(members),
                "examples": members[:8],
            }
            for group, members in groups.items()
        },
        "forbidden_trainable": forbidden,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Audit expert-only trainable parameters")
    parser.add_argument("--base", type=Path, default=root / "artifacts/base_checkpoint")
    parser.add_argument("--output", type=Path, default=root / "artifacts/trainable_parameters.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(audit(args.base, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
