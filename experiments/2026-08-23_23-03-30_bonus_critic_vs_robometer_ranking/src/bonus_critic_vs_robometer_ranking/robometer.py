from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from accelerate.utils import load_checkpoint_in_model
from lerobot.rewards.robometer.configuration_robometer import RobometerConfig
from lerobot.rewards.robometer.modeling_robometer import (
    RobometerPredictionHead,
    RobometerRewardModel,
    convert_bins_to_continuous,
)
from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep
from peft import LoraConfig, inject_adapter_in_model

from .utils import sha256_file


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_robometer(config: dict[str, Any]) -> tuple[RobometerRewardModel, RobometerEncoderProcessorStep, dict]:
    cfg = config["robometer"]
    checkpoint = Path(cfg["local_snapshot"])
    backbone_config = json.loads((checkpoint / "config.json").read_text())
    reward_config = RobometerConfig(
        pretrained_path=str(checkpoint),
        device="cuda",
        base_model_id=str(cfg["base_processor_snapshot"]),
        max_frames=4,
        reward_output="progress",
        torch_dtype="bfloat16",
        use_multi_image=True,
        use_per_frame_progress_token=True,
        frame_pooling="mean",
        progress_loss_type="discrete",
        progress_discrete_bins=int(cfg["progress_discrete_bins"]),
        vlm_config=backbone_config,
    )
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=LORA_TARGETS,
        inference_mode=True,
    )
    with init_empty_weights():
        model = RobometerRewardModel(reward_config)
        model.model = inject_adapter_in_model(
            lora_config, model.model, adapter_name="default", low_cpu_mem_usage=True
        )
        hidden_size = int(model.model.config.text_config.hidden_size)
        model.similarity_head = RobometerPredictionHead(
            hidden_size, 1, dropout=0.1, with_sigmoid=False
        ).to(dtype=torch.bfloat16)
        model.model.tie_weights()

    checkpoint_keys = set(json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"])
    model_keys = set(model.state_dict())
    unexpected = sorted(checkpoint_keys - model_keys)
    allowed_missing = {"model.lm_head.weight"}
    missing = sorted(model_keys - checkpoint_keys - allowed_missing)
    if unexpected or missing:
        raise RuntimeError(
            f"Robometer architecture mismatch: unexpected={unexpected[:20]}, missing={missing[:20]}"
        )
    load_checkpoint_in_model(
        model,
        checkpoint=str(checkpoint),
        device_map={"": "cuda:0"},
        dtype=torch.bfloat16,
        strict=False,
    )
    # Safetensors correctly stores tied embeddings once, so loading replaces
    # the embedding parameter but leaves its omitted lm_head alias on meta.
    # Re-establish the declared tie after shard loading and before inference.
    model.model.tie_weights()
    model.to("cuda")
    model.eval()
    processor = RobometerEncoderProcessorStep(
        base_model_id=str(cfg["base_processor_snapshot"]),
        max_frames=4,
        use_multi_image=True,
        use_per_frame_progress_token=True,
        max_length=1024,
    )
    audit = {
        "repo": cfg["repo"],
        "revision": cfg["revision"],
        "checkpoint_keys": len(checkpoint_keys),
        "model_keys": len(model_keys),
        "allowed_missing_keys": sorted(allowed_missing.intersection(model_keys - checkpoint_keys)),
        "unexpected_keys": unexpected,
        "progress_bins": reward_config.progress_discrete_bins,
        "lora_rank": 32,
        "lora_alpha": 64,
        "weight_shards": [
            {
                "name": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": sha256_file(shard),
            }
            for shard in sorted(checkpoint.glob("model-*.safetensors"))
        ],
    }
    return model, processor, audit


def score_robometer_batch(
    model: RobometerRewardModel,
    processor: RobometerEncoderProcessorStep,
    rows: list[dict],
    frames: list[Any],
) -> tuple[list[float], list[list[float]]]:
    samples = [(sample_frames, row["instruction"]) for row, sample_frames in zip(rows, frames, strict=True)]
    encoded = processor.encode_samples(samples)
    inputs = {
        key: value.to("cuda") if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    with torch.inference_mode():
        progress_logits, _ = model._compute_rbm_logits(dict(inputs))
    progress = convert_bins_to_continuous(progress_logits.detach().float().cpu())
    sequences = progress.tolist()
    return [float(sequence[-1]) for sequence in sequences], sequences
