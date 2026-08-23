from __future__ import annotations

from pathlib import Path

EXPERIMENT_NAME = "pretrain_smolvla_state_noise_k1_1k"
TRACKIO_PROJECT = "pretrain-smolvla-state-noise-k1"
CREATED_AT = "2026-08-22T22:43:55+03:00"
OFFICIAL_SOURCE_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"

# Base: the official-data pretrain (seen control 20/20, zero-shot 0.005 on the
# ten goal tasks, replicated).  Consumed read-only; every task/alpha
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
TARGET_DATASET_REPO = "official/libero_goal_rot180_128"
TARGET_DATASET_ROOT = Path("/var/tmp/vla_libero_official_rot180/libero_goal")
TARGET_SUITE = "libero_goal"

TARGET_INSTRUCTIONS: dict[int, str] = {
    0: "open the middle drawer of the cabinet",
    1: "put the bowl on the stove",
    2: "put the wine bottle on top of the cabinet",
}
TARGET_ENV_TASK_IDS = {task_id: task_id for task_id in TARGET_INSTRUCTIONS}

# This experiment probes proprioception-noise augmentation at the hardest
# budget only: k=1 (official demo_0 of each task).
DEMO_BUDGET = 1
DEMO_BUDGETS = (DEMO_BUDGET,)

# High-alpha continuation of 2026-08-22_22-43-55_pretrain_smolvla_state_noise_k1
# (there the mean was still rising at the grid edge: 0.617/0.583/0.667/0.683
# at alpha 0.00..0.05, 2000 steps), probed at HALF the training budget:
# 1000 optimizer steps.  Additive zero-mean Gaussian noise on the NORMALIZED
# proprioceptive state inside policy.forward (STATE is MEAN_STD-normalized,
# so alpha == sigma_i = alpha * Std(s_i) in raw units); actions and images
# untouched; no noise at evaluation.  alpha=0.0 is the in-experiment control
# at 1000 steps -- REQUIRED because halving the steps also compresses the
# auto-scaled LR schedule, so 2000-step arms are not a valid baseline for the
# noise effect at 1000 steps.
ALPHAS = (0.0, 0.08, 0.1, 0.2)
# Per-job dedicated RNG stream for the augmentation noise (torch.Generator
# seeded independently of the training seed so the main RNG stream is
# untouched relative to the full-FT experiment).
NOISE_STREAM_SEED = 91_000

MASTER_SEED = 1_000
TRAIN_STEPS = 1_000
TRAIN_BATCH_SIZE = 32
TRAIN_WORKERS = 32
TRAIN_LOG_FREQ = 25
TRAINED_CHUNK_SIZE = 50
TRAINED_ACTION_STEPS = 50
# Inference at the trained default only (n_action_steps=50).
EVAL_ACTION_STEPS = (50,)
EVAL_EPISODES = 20
EVAL_BATCH_SIZE = 1
EVAL_HORIZON = 300
GPU_IDS = (0, 1, 2, 3)

# Training recipe: full fine-tune, identical to
# 2026-08-22_18-10-40_pretrain_smolvla_full_ft_low_k (the alpha=0.0 arm of
# this sweep is that experiment's k=1 recipe).
FREEZE_VISION_ENCODER = False
TRAIN_EXPERT_ONLY = False
TRAIN_STATE_PROJ = True
NUM_VLM_LAYERS = 16

OUTPUT_ROOT = Path(
    "/var/tmp/vla_outputs/state_noise_k1_1k_20260823_143107"
)
PRODUCTION_SMOKE_POINT = {"task_id": 0, "alpha": 0.0}


def alpha_tag(alpha: float) -> str:
    if alpha not in ALPHAS:
        raise ValueError(f"Unsupported alpha {alpha}")
    return format(alpha, ".2f")


def noise_seed(episode_index: int) -> int:
    return MASTER_SEED + episode_index


def result_path(
    results_root: Path, task_id: int, alpha: float, action_steps: int
) -> Path:
    return (
        results_root
        / f"task_{task_id}"
        / f"alpha_{alpha_tag(alpha)}"
        / f"n_{action_steps}.json"
    )


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]
