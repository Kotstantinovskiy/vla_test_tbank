from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch import nn
from transformers import Qwen3_5ForConditionalGeneration


@dataclass(frozen=True)
class TrainableAudit:
    total_parameters: int
    trainable_parameters: int
    trainable_fraction: float
    trainable_vision_parameters: int
    gradient_checkpointing: bool
    trainable_names: list[str]


class QwenProgressCritic(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_bins: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_bins = num_bins
        backbone_dtype = next(backbone.parameters()).dtype
        self.progress_head = nn.Linear(hidden_size, num_bins, bias=True).to(dtype=backbone_dtype)

    def _core_model(self) -> nn.Module:
        base = self.backbone.get_base_model()
        return base.model

    def forward(self, inputs: dict[str, torch.Tensor], labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        core = self._core_model()
        outputs = core(**inputs, use_cache=False, return_dict=True)
        hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
        last_indices = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]
        logits = self.progress_head(pooled)
        probabilities = torch.softmax(logits.float(), dim=-1)
        centers = torch.linspace(0.0, 1.0, self.num_bins, device=logits.device)
        progress = (probabilities * centers).sum(dim=-1)
        result = {"logits": logits, "progress": progress}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits.float(), labels)
        return result

    def save_adapter_checkpoint(self, output_dir: str | Path, metadata: dict[str, Any]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output_dir / "adapter")
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in self.progress_head.state_dict().items()},
            output_dir / "progress_head.safetensors",
            metadata={key: str(value) for key, value in metadata.items()},
        )


def build_model(config: dict[str, Any], cache_dir: str | Path) -> QwenProgressCritic:
    model_config = config["model"]
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        cache_dir=str(cache_dir),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_disable()
    if model_config["gradient_checkpointing"]:
        raise ValueError("protocol requires gradient_checkpointing=false")
    lora = model_config["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias="none",
        target_modules=list(lora["target_modules"]),
    )
    backbone = get_peft_model(backbone, peft_config)
    # Module suffixes such as q_proj also occur in the vision tower.  Keep the
    # protocol invariant explicit even if a future PEFT version matches them.
    for name, parameter in backbone.named_parameters():
        if "visual" in name or "vision" in name:
            parameter.requires_grad_(False)
    return QwenProgressCritic(
        backbone=backbone,
        hidden_size=int(backbone.get_base_model().config.text_config.hidden_size),
        num_bins=int(model_config["num_progress_bins"]),
    )


def audit_trainable_parameters(model: QwenProgressCritic) -> TrainableAudit:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable_vision = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("visual" in name or "vision" in name)
    )
    checkpointing = bool(model.backbone.get_base_model().is_gradient_checkpointing)
    audit = TrainableAudit(
        total_parameters=total,
        trainable_parameters=trainable,
        trainable_fraction=trainable / total,
        trainable_vision_parameters=trainable_vision,
        gradient_checkpointing=checkpointing,
        trainable_names=trainable_names,
    )
    if trainable == 0:
        raise AssertionError("model has no trainable parameters")
    if trainable_vision != 0:
        raise AssertionError("vision encoder unexpectedly has trainable parameters")
    if checkpointing:
        raise AssertionError("gradient checkpointing is unexpectedly enabled")
    if not any(name.startswith("progress_head") for name in trainable_names):
        raise AssertionError("progress head is frozen")
    if not any("lora_" in name for name in trainable_names):
        raise AssertionError("no LoRA parameters are trainable")
    return audit


def audit_to_dict(audit: TrainableAudit) -> dict[str, Any]:
    return asdict(audit)
