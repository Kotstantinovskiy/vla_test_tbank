from __future__ import annotations

from pathlib import Path

EXPERIMENT_NAME = "pretrain_smolvla_lora_low_k"
TRACKIO_PROJECT = "pretrain-smolvla-lora-low-k"
CREATED_AT = "2026-08-22T22:19:39+03:00"
OFFICIAL_SOURCE_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"

# Base: the official-data pretrain (seen control 20/20, zero-shot 0.005 on the
# ten goal tasks, replicated).  Consumed read-only; every task/budget
# adaptation starts from this checkpoint independently.
BASE_CHECKPOINT = Path(
    "/var/tmp/vla_outputs/seen_libero90_official_20260817/checkpoints/030000/pretrained_model"
)
BASE_MODEL_SHA256 = "0a410cc6887e71c0dd2d21dd8a5deb7aba66e291dba9acba8656320dc076a3cc"
VLM_BACKBONE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
VLM_MODEL_SHA256 = "b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3"
VLM_MODEL_BYTES = 2_029_990_624
VLM_BACKBONE = Path(
    "/var/tmp/vla_backbones/"
    f"SmolVLM2-500M-Video-Instruct_{VLM_BACKBONE_REVISION}"
)
LIBERO_ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
LIBERO_ASSETS_ROOT = Path(f"/var/tmp/vla_libero_assets_{LIBERO_ASSETS_REVISION}")
BASE_PROVENANCE = {
    "pretraining_experiment": "2026-08-17_smolvla_pretrain_libero",
    "base_model_repo": "lerobot/smolvla_base",
    "base_model_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
    "seen_positive_control": "libero_90 task 0: 20/20",
    "zero_shot_reference": (
        "2026-08-19_pretrain_smolvla_prompt_only_3: true 1/200; "
        "per-episode deterministic noise"
    ),
}

# Target demos: the in-repo conversion of official libero_goal HDF5.
# Episodes preserve official order, so "first k" == official demo_0..demo_{k-1}.
TARGET_DATASET_REPO = "official/libero_goal_rot180_128"
TARGET_DATASET_ROOT = Path("/var/tmp/vla_libero_official_rot180/libero_goal")
TARGET_SUITE = "libero_goal"

TARGET_INSTRUCTIONS: dict[int, str] = {
    0: "open the middle drawer of the cabinet",
    1: "put the bowl on the stove",
    2: "put the wine bottle on top of the cabinet",
}
TARGET_ENV_TASK_IDS = {task_id: task_id for task_id in TARGET_INSTRUCTIONS}

DEMO_BUDGETS = (1, 2, 3)
MASTER_SEED = 1_000
TRAIN_STEPS = 2_000
TRAIN_BATCH_SIZE = 32
# Raised 8 -> 32 mid-run on 2026-08-22 (see reports/EXECUTION_NOTES.md): the
# dataloader was the bottleneck (data_s ~1.2s vs updt_s ~0.23s on 256 idle
# CPUs).  Worker count does not affect batch order or numerics; jobs already
# in flight kept 8 and each job's train_config.json records its own value.
TRAIN_WORKERS = 32
TRAIN_LOG_FREQ = 25
TRAINED_CHUNK_SIZE = 50
TRAINED_ACTION_STEPS = 50
# Every adapted checkpoint is evaluated at both inference-time action-step
# settings: the trained default (50 = execute the whole chunk) and 25
# (re-predict twice per chunk).  Training always uses n_action_steps=50; the
# checkpoint config keeps 50 and the override is applied at load time only.
EVAL_ACTION_STEPS = (50, 25)
EVAL_EPISODES = 20
# One eval_policy call per episode is required to make flow-sampling noise a
# deterministic function of episode index rather than process history.
EVAL_BATCH_SIZE = 1
EVAL_HORIZON = 300
GPU_IDS = (0, 1, 2, 3)

# The single protocol change against the expert-only low-k reproduction
# (2026-08-21_20-37-56_pretrain_smolvla_low_k_deterministic_repro): the VLM is
# additionally adapted through LoRA adapters, while the action expert and the
# projections train fully (via PEFT modules_to_save), exactly matching the
# expert-only baseline's trainable set.  The base policy is loaded with the
# baseline freeze flags; lerobot's wrap_with_peft then freezes every base
# parameter and leaves only the adapters and full-training copies trainable.
FREEZE_VISION_ENCODER = True
TRAIN_EXPERT_ONLY = True
TRAIN_STATE_PROJ = True
# Retained VLM text layers in the base checkpoint (config num_vlm_layers);
# the frozen-by-design last layer index is NUM_VLM_LAYERS - 1 and gets no
# adapters (its outputs never reach the action head).
NUM_VLM_LAYERS = 16

# LoRA on every linear layer of the VLM that the forward pass uses: text
# layers 0..14 (attention + MLP projections), all vision-encoder layers
# (attention + MLP), and the connector projection.  Embeddings, norms, the
# guard tensors, and the expert receive no adapters.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_REGEX = (
    r"model\.vlm_with_expert\.vlm\.model\."
    r"(?:text_model\.layers\.(?:[0-9]|1[0-4])\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
    r"|vision_model\.encoder\.layers\.\d+\."
    r"(?:self_attn\.(?:q|k|v|out)_proj|mlp\.fc[12])"
    r"|connector\.modality_projection\.proj)"
)
# Fully trained (PEFT modules_to_save): the expert-only baseline's trainable
# set.  lm_expert is a bare LlamaModel (layers, norm, rotary_emb; no lm_head).
FULL_TRAINING_MODULES = (
    "model.vlm_with_expert.lm_expert",
    "model.state_proj",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.action_time_mlp_in",
    "model.action_time_mlp_out",
)

OUTPUT_ROOT = Path(
    "/var/tmp/vla_outputs/lora_low_k_20260822_221939"
)
PRODUCTION_SMOKE_POINT = {"task_id": 0, "budget": 1}


def noise_seed(episode_index: int) -> int:
    return MASTER_SEED + episode_index


def result_path(
    results_root: Path, task_id: int, budget: int, action_steps: int
) -> Path:
    return (
        results_root
        / f"task_{task_id}"
        / f"k_{budget}"
        / f"n_{action_steps}.json"
    )


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]
