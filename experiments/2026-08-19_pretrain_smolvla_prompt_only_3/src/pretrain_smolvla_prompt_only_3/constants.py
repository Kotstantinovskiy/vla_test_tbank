from __future__ import annotations

from pathlib import Path

EXPERIMENT_NAME = "pretrain_smolvla_prompt_only_3"
TRACKIO_PROJECT = "pretrain-smolvla-prompt-only-3"
DATE = "2026-08-19"

# Redo of 2026-08-18_pretrain_smolvla_prompt_only_2 with ONE protocol change:
# the policy's flow-sampling noise is reseeded from the episode seed at the
# start of every episode, so rollouts are reproducible independent of process
# layout (the _2 harness drew action noise from the process-global torch RNG
# stream; see 2026-08-19_goal_scene_seen_prompts/reports/HARNESS_NOTE.md).
# To make per-episode noise well-defined, episodes run with batch size 1
# (batched draws would interleave one stream across sub-envs).  Everything
# else — checkpoint, suite, conditions, seeds, init states, all-videos
# policy — matches _2.

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
# Batch size is 1 BY PROTOCOL in _3 (see note above); _2 used 4.
EVAL_BATCH_SIZE = 1


def noise_seed(episode_index: int) -> int:
    """Deterministic per-episode seed for the policy's sampling noise.

    Equal to the episode's env seed (MASTER_SEED + episode); the env RNG
    (numpy/mujoco) and the policy RNG (torch) are separate systems, so
    sharing the integer is harmless and keeps the manifest simple.
    """

    return MASTER_SEED + episode_index


def prompt_for(condition: str, task_id: int) -> str:
    if condition == "true":
        return TARGET_INSTRUCTIONS[task_id]
    if condition == "wrong":
        return TARGET_INSTRUCTIONS[(task_id + 1) % len(TARGET_INSTRUCTIONS)]
    if condition == "nonsense":
        return NONSENSE_PROMPT
    raise ValueError(f"Unknown condition: {condition}")


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]
