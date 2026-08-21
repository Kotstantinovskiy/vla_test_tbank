from __future__ import annotations

from pathlib import Path

EXPERIMENT_NAME = "pretrain_smolvla_few_shot_tune_low_k"
TRACKIO_PROJECT = "pretrain-few-shot-low-k"
DATE = "2026-08-18"

# Base: the official-data pretrain (seen control 20/20, zero-shot 0.005 on the
# ten goal tasks, replicated).  Consumed read-only; every task/budget
# adaptation starts from this checkpoint independently.
BASE_CHECKPOINT = Path(
    "/var/tmp/vla_outputs/seen_libero90_official_20260817/checkpoints/030000/pretrained_model"
)
BASE_PROVENANCE = {
    "pretraining_experiment": "2026-08-17_smolvla_pretrain_libero",
    "base_model_repo": "lerobot/smolvla_base",
    "base_model_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
    "seen_positive_control": "libero_90 task 0: 20/20",
    "zero_shot_reference": "2026-08-18_pretrain_smolvla_prompt_only: true 1/200",
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
MASTER_SEED = 1_000
TRAIN_STEPS = 2_000
TRAIN_BATCH_SIZE = 32
TRAIN_WORKERS = 8
TRAIN_LOG_FREQ = 25
EVAL_EPISODES = 20
EVAL_BATCH_SIZE = 4
GPU_IDS = (0, 1, 2, 3)

OUTPUT_ROOT = Path("/var/tmp/vla_outputs/pretrain_few_shot_low_k_20260818")


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]
