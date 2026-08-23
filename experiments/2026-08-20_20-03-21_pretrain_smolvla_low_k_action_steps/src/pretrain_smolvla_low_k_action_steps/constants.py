from __future__ import annotations

from pathlib import Path

EXPERIMENT_NAME = "pretrain_smolvla_low_k_action_steps"
TRACKIO_PROJECT = "pretrain-low-k-action-steps"
CREATED_AT = "2026-08-20T20:03:21+03:00"
OFFICIAL_SOURCE_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"

# Frozen official-data pretrain. Every task/budget adaptation starts here;
# action-step settings share the same adapted weights and differ only at eval.
BASE_CHECKPOINT = Path(
    "/var/tmp/vla_outputs/seen_libero90_official_20260817/"
    "checkpoints/030000/pretrained_model"
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
    "pretraining_dataset": "official/libero_90_rot180_128 (in-repo conversion)",
    "official_source_repo": "yifengzhu-hf/LIBERO-datasets",
    "official_source_revision": OFFICIAL_SOURCE_REVISION,
    "pretraining_steps": 30_000,
    "seen_positive_control": "libero_90 task 0: 20/20",
}

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
ACTION_STEPS = (1, 10, 25)
TRAINED_CHUNK_SIZE = 50
TRAINED_ACTION_STEPS = 50

MASTER_SEED = 1_000
TRAIN_STEPS = 2_000
TRAIN_BATCH_SIZE = 32
TRAIN_WORKERS = 8
TRAIN_LOG_FREQ = 25
EVAL_EPISODES = 20
EVAL_HORIZON = 300
# One episode per eval_policy call is required to make policy sampling noise a
# deterministic function of episode index rather than process/job history.
EVAL_BATCH_SIZE = 1
GPU_IDS = (0, 1, 2, 3)

REUSED_CHECKPOINT_EXPERIMENT = (
    "2026-08-21_20-37-56_pretrain_smolvla_low_k_deterministic_repro"
)
OUTPUT_ROOT = Path(
    "/var/tmp/vla_outputs/low_k_deterministic_repro_20260821_203756"
)

PRODUCTION_SMOKE_POINT = {"task_id": 0, "budget": 1, "action_steps": 10}


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
