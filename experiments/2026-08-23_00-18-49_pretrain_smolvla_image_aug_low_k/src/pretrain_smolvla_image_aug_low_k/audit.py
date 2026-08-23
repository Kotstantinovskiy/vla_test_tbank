from __future__ import annotations

"""Full-fine-tune trainable-parameter audit.

With ``train_expert_only=False`` and ``freeze_vision_encoder=False`` LeRobot's
SmolVLA trains the whole policy except four unused-by-design tensor groups it
keeps frozen as a distributed-training guard (their outputs never reach the
action head): the VLM ``lm_head``, the final ``text_model.norm``, the last
retained VLM text layer, and the expert's ``lm_head``.  The audit asserts that
exactly this frozen set remains and that every intended group (vision encoder,
connector, VLM text layers/embeddings, action expert, projections) is
trainable.
"""

import argparse
import json
from pathlib import Path

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from .constants import (
    FREEZE_VISION_ENCODER,
    NUM_VLM_LAYERS,
    TRAIN_EXPERT_ONLY,
    TRAIN_STATE_PROJ,
    VLM_BACKBONE,
    experiment_root,
)

# Substring patterns of the tensors LeRobot intentionally leaves frozen in a
# full fine-tune (see SmolVLMWithExpertModel.set_requires_grad).
INTENDED_FROZEN_PATTERNS = (
    ".vlm.lm_head.",
    ".vlm.model.text_model.norm.",
    f".vlm.model.text_model.layers.{NUM_VLM_LAYERS - 1}.",
    ".lm_expert.lm_head.",
)


def intended_frozen(name: str) -> bool:
    return any(pattern in name for pattern in INTENDED_FROZEN_PATTERNS)


def classify_trainable(names: list[str]) -> dict[str, list[str]]:
    return {
        "vlm_vision_encoder": [n for n in names if ".vlm.model.vision_model." in n],
        "vlm_connector": [n for n in names if ".vlm.model.connector." in n],
        "vlm_text_layers": [n for n in names if ".vlm.model.text_model.layers." in n],
        "vlm_text_embeddings": [n for n in names if ".vlm.model.text_model.embed_tokens." in n],
        "action_expert": [n for n in names if ".lm_expert." in n],
        "state_projection": [n for n in names if ".state_proj." in n],
        "action_input_projection": [n for n in names if ".action_in_proj." in n],
        "action_output_projection": [n for n in names if ".action_out_proj." in n],
        "action_time_mlp": [n for n in names if ".action_time_mlp_" in n],
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
    trainable = [(name, param) for name, param in policy.named_parameters() if param.requires_grad]
    frozen = [(name, param) for name, param in policy.named_parameters() if not param.requires_grad]
    names = [name for name, _ in trainable]
    groups = classify_trainable(names)
    # Trainable tensors that must be frozen (the intentional guard set).
    forbidden = [name for name in names if intended_frozen(name)]
    # Frozen tensors outside the intentional guard set: in a full fine-tune
    # nothing else may stay frozen.
    unexpectedly_frozen = [name for name, _ in frozen if not intended_frozen(name)]
    missing = [group for group, members in groups.items() if not members]
    if forbidden:
        raise RuntimeError(
            f"Intentionally-frozen guard tensors are unexpectedly trainable: {forbidden[:10]}"
        )
    if unexpectedly_frozen:
        raise RuntimeError(
            f"Full fine-tune left unexpected parameters frozen: {unexpectedly_frozen[:10]}"
        )
    if missing:
        raise RuntimeError(f"Expected trainable parameter groups are missing: {missing}")
    result: dict[str, object] = {
        "base": str(base.resolve()),
        "flags": {
            "freeze_vision_encoder": FREEZE_VISION_ENCODER,
            "train_expert_only": TRAIN_EXPERT_ONLY,
            "train_state_proj": TRAIN_STATE_PROJ,
        },
        "intended_frozen_patterns": list(INTENDED_FROZEN_PATTERNS),
        "total_parameters": sum(param.numel() for _, param in trainable + frozen),
        "trainable_parameters": sum(param.numel() for _, param in trainable),
        "frozen_parameters": sum(param.numel() for _, param in frozen),
        "trainable_tensor_count": len(trainable),
        "frozen_tensor_count": len(frozen),
        "frozen_tensors": [name for name, _ in frozen],
        "trainable_groups": {
            group: {
                "tensor_count": len(members),
                "examples": members[:8],
            }
            for group, members in groups.items()
        },
        "forbidden_trainable": forbidden,
        "unexpectedly_frozen": unexpectedly_frozen,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Audit full-fine-tune trainable parameters")
    parser.add_argument("--base", type=Path, default=root / "artifacts/base_checkpoint")
    parser.add_argument("--output", type=Path, default=root / "artifacts/trainable_parameters.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(audit(args.base, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
