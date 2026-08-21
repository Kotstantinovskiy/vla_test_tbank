from __future__ import annotations

from pathlib import Path

EXPERIMENT_NAME = "pretrain_smolvla_prompt_only_2"
TRACKIO_PROJECT = "pretrain-smolvla-prompt-only-2"
DATE = "2026-08-18"

# Immutable evaluation input: the final checkpoint of the official-data
# pretrain (experiments/2026-08-17_smolvla_pretrain_libero).  Its seen-task
# positive control scored 20/20, so the evaluation pipeline is validated
# end-to-end for this checkpoint's conventions (rot180 frames, native
# 128x128 rendering, official state recipe).
CHECKPOINT_PATH = Path(
    "/var/tmp/vla_outputs/seen_libero90_official_20260817/checkpoints/030000/pretrained_model"
)
CHECKPOINT_PROVENANCE = {
    "pretraining_experiment": "2026-08-17_smolvla_pretrain_libero",
    "base_model_repo": "lerobot/smolvla_base",
    "base_model_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
    "pretraining_dataset": "official/libero_90_rot180_128 (in-repo conversion)",
    "official_source_repo": "yifengzhu-hf/LIBERO-datasets",
    "official_source_revision": "f13aa24a3da8c43c7225569f28c562979fa0e35a",
    "pretraining_steps": 30_000,
    "seen_positive_control": "libero_90 task 0: 20/20",
}

TARGET_SUITE = "libero_goal"

# All ten libero_goal tasks; logical IDs equal environment IDs and every
# instruction is asserted against the live environment before rollouts.
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
TARGET_ENV_TASK_IDS: dict[int, int] = {task_id: task_id for task_id in TARGET_INSTRUCTIONS}

PROMPT_CONDITIONS = ("true", "wrong", "nonsense")
NONSENSE_PROMPT = "perform the dax florp twice"
MASTER_SEED = 1000
N_EVAL_EPISODES = 20
EVAL_BATCH_SIZE = 4


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]
