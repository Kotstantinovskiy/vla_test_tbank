from __future__ import annotations

from pathlib import Path

EXPERIMENT_NAME = "pretrain_smolvla_image_aug_low_k"
TRACKIO_PROJECT = "pretrain-smolvla-image-aug-low-k"
CREATED_AT = "2026-08-23T00:18:49+03:00"
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
# Budget-dependent optimization length (protocol choice of this experiment):
# fewer demonstrations get fewer steps to curb single-episode overfitting.
# Note: lerobot auto-scales the preset LR schedule (warmup 1000 / decay
# 30000) proportionally to --steps, so the k=1/k=2 arms also run compressed
# schedules.  The sibling experiments all trained 2000 steps at every k, so
# for k=1/k=2 the comparison against them differs in BOTH augmentation and
# steps (disclosed in configs/protocol.yaml).
TRAIN_STEPS_BY_BUDGET = {1: 1_000, 2: 1_500, 3: 2_000}
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

# The single protocol change against the full-FT experiment
# (2026-08-22_18-10-40_pretrain_smolvla_full_ft_low_k): training-time image
# augmentation is enabled via lerobot's native dataset image_transforms
# (photometric jitter + small random affine; see IMAGE_TRANSFORMS_EXPECTED).
# The trainable set stays the full fine-tune: the whole policy trains, and
# LeRobot's SmolVLA keeps only its unused-by-design guard tensors frozen
# (vlm.lm_head, the final text_model.norm, the last retained VLM text layer,
# and the expert's lm_head).
FREEZE_VISION_ENCODER = False
TRAIN_EXPERT_ONLY = False
TRAIN_STATE_PROJ = True
# Retained VLM text layers in the base checkpoint (config num_vlm_layers);
# the frozen-by-design last layer index is NUM_VLM_LAYERS - 1.
NUM_VLM_LAYERS = 16

OUTPUT_ROOT = Path(
    "/var/tmp/vla_outputs/image_aug_low_k_20260823_001849"
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


# Frozen view of lerobot 0.6.1's default ImageTransformsConfig, which this
# experiment enables verbatim (--dataset.image_transforms.enable=true, no
# other overrides).  Up to MAX_NUM_TRANSFORMS of these are sampled per frame
# (RandomSubsetApply, equal weights, torchvision order).  The protocol test
# asserts the installed lerobot still carries exactly these defaults.
IMAGE_TRANSFORMS_MAX_NUM = 3
IMAGE_TRANSFORMS_EXPECTED = {
    "brightness": ("ColorJitter", {"brightness": (0.8, 1.2)}),
    "contrast": ("ColorJitter", {"contrast": (0.8, 1.2)}),
    "saturation": ("ColorJitter", {"saturation": (0.5, 1.5)}),
    "hue": ("ColorJitter", {"hue": (-0.05, 0.05)}),
    "sharpness": ("SharpnessJitter", {"sharpness": (0.5, 1.5)}),
    "affine": ("RandomAffine", {"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)}),
}
