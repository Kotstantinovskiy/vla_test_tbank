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
    3: "open the top drawer and put the bowl inside",
    4: "put the bowl on top of the cabinet",
    5: "push the plate to the front of the stove",
    6: "put the cream cheese in the bowl",
    7: "turn on the stove",
    8: "put the bowl on the plate",
    9: "put the wine bottle on the rack",
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

OUTPUT_ROOT = Path(
    "/var/tmp/vla_outputs/pretrain_low_k_action_steps_20260820_200321"
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
