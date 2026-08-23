from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


class QwenProgressCritic(nn.Module):
    """Frozen inference copy of the model used by the training experiment."""

    def __init__(self, backbone: nn.Module, hidden_size: int, num_bins: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_bins = num_bins
        self.progress_head = nn.Linear(hidden_size, num_bins, bias=True).to(
            dtype=next(backbone.parameters()).dtype
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.backbone.get_base_model().model(
            **inputs, use_cache=False, return_dict=True
        )
        hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
        last_indices = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]
        logits = self.progress_head(pooled)
        probabilities = torch.softmax(logits.float(), dim=-1)
        centers = torch.linspace(0.0, 1.0, self.num_bins, device=logits.device)
        return {"logits": logits, "progress": (probabilities * centers).sum(dim=-1)}


def load_own_critic(config: dict[str, Any], cache_dir: str | Path) -> tuple[QwenProgressCritic, Any]:
    model_cfg = config["own_critic"]
    checkpoint = Path(model_cfg["checkpoint"])
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_cfg["base_repo"],
        revision=model_cfg["base_revision"],
        cache_dir=str(cache_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    backbone = PeftModel.from_pretrained(base, checkpoint / "adapter", is_trainable=False)
    hidden_size = int(backbone.get_base_model().config.text_config.hidden_size)
    model = QwenProgressCritic(backbone, hidden_size, int(model_cfg["num_progress_bins"]))
    state = load_file(str(checkpoint / "progress_head.safetensors"), device="cpu")
    model.progress_head.load_state_dict(state, strict=True)
    model.to("cuda").eval()
    processor = AutoProcessor.from_pretrained(
        model_cfg["base_repo"],
        revision=model_cfg["base_revision"],
        cache_dir=str(cache_dir),
        local_files_only=True,
    )
    return model, processor


def encode_own_batch(processor: Any, rows: list[dict], frames: list[Any], config: dict) -> dict[str, torch.Tensor]:
    model_cfg = config["own_critic"]
    conversations = []
    for row, sample_frames in zip(rows, frames, strict=True):
        content: list[dict[str, Any]] = [
            {"type": "text", "text": model_cfg["prompt_prefix"].format(task=row["instruction"])}
        ]
        for frame_number, frame in enumerate(sample_frames, start=1):
            content.append({"type": "text", "text": f"\nFrame {frame_number}:"})
            content.append({"type": "image", "image": Image.fromarray(frame)})
        content.append({"type": "text", "text": f"\n{model_cfg['prompt_suffix']}"})
        conversations.append([{"role": "user", "content": content}])
    encoded = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
    )
    return {key: value.to("cuda") for key, value in dict(encoded).items()}
