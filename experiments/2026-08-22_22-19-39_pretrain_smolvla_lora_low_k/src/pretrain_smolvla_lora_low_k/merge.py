from __future__ import annotations

"""Merge a trained LoRA adapter into a plain SmolVLA checkpoint.

The evaluation pipeline (paired n_action_steps=50/25 rollouts, determinism
gate, SHA-256 manifests) consumes standard SmolVLA checkpoints.  This module
loads the pinned base policy, applies the trained PEFT adapter
(LoRA weights + the fully-trained modules_to_save copies of the expert and
projections), merges it, and saves ``pretrained_model_merged`` next to the
adapter checkpoint.  Processor pipelines (which carry the target-dataset
normalization statistics captured at training time) are copied verbatim from
the adapter checkpoint directory.
"""

import argparse
import json
import re
import shutil

import torch

from .constants import (
    FREEZE_VISION_ENCODER,
    FULL_TRAINING_MODULES,
    LORA_ALPHA,
    LORA_RANK,
    LORA_TARGET_REGEX,
    TRAIN_EXPERT_ONLY,
    TRAIN_STATE_PROJ,
    VLM_BACKBONE,
    experiment_root,
)

PROCESSOR_FILE_PATTERNS = (
    "policy_preprocessor*",
    "policy_postprocessor*",
    "train_config.json",
)


def load_base_policy(device: str = "cpu"):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    runtime_base = experiment_root() / "artifacts/runtime_base_checkpoint"
    cfg = SmolVLAConfig.from_pretrained(runtime_base)
    cfg.pretrained_path = str(runtime_base)
    cfg.pretrained_revision = None
    cfg.vlm_model_name = str(VLM_BACKBONE)
    cfg.device = device
    cfg.freeze_vision_encoder = FREEZE_VISION_ENCODER
    cfg.train_expert_only = TRAIN_EXPERT_ONLY
    cfg.train_state_proj = TRAIN_STATE_PROJ
    return SmolVLAPolicy.from_pretrained(runtime_base, config=cfg, local_files_only=True)


def validate_adapter_config(adapter_dir) -> dict:
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA":
        raise ValueError(f"Unexpected peft_type: {config.get('peft_type')}")
    if config.get("r") != LORA_RANK or config.get("lora_alpha") != LORA_ALPHA:
        raise ValueError(
            f"Adapter rank/alpha changed: r={config.get('r')}, alpha={config.get('lora_alpha')}"
        )
    if config.get("target_modules") != LORA_TARGET_REGEX:
        raise ValueError("Adapter target_modules regex differs from the frozen protocol")
    if sorted(config.get("modules_to_save") or []) != sorted(FULL_TRAINING_MODULES):
        raise ValueError("Adapter modules_to_save differ from the frozen protocol")
    return config


def merge_adapter(task_id: int, budget: int, device: str = "cpu") -> None:
    from peft import PeftModel

    from .training import adapter_model, final_model

    adapter_dir = adapter_model(task_id, budget)
    merged_dir = final_model(task_id, budget)
    validate_adapter_config(adapter_dir)

    policy = load_base_policy(device)
    reference = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if re.fullmatch(LORA_TARGET_REGEX, name.rsplit(".", 1)[0])
        or any(name.startswith(prefix + ".") for prefix in FULL_TRAINING_MODULES)
    }
    peft_model = PeftModel.from_pretrained(policy, adapter_dir, is_trainable=False)
    merged = peft_model.merge_and_unload()

    changed = 0
    with torch.no_grad():
        for name, parameter in merged.named_parameters():
            baseline = reference.get(name)
            if baseline is not None and not torch.equal(baseline, parameter):
                changed += 1
    if changed == 0:
        raise RuntimeError(
            "Merged weights are identical to the base checkpoint; the adapter "
            "was not applied"
        )

    merged.config.pretrained_path = None
    if hasattr(merged.config, "use_peft"):
        merged.config.use_peft = False
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged.save_pretrained(merged_dir)
    for pattern in PROCESSOR_FILE_PATTERNS:
        for source in adapter_dir.glob(pattern):
            shutil.copy2(source, merged_dir / source.name)
    print(
        json.dumps(
            {
                "merged": str(merged_dir),
                "adapter": str(adapter_dir),
                "changed_target_tensors": changed,
                "reference_target_tensors": len(reference),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", type=int)
    parser.add_argument("budget", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    merge_adapter(args.task_id, args.budget, args.device)


if __name__ == "__main__":
    main()
