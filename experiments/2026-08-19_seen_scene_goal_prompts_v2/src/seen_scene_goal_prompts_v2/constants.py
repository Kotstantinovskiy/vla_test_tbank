from __future__ import annotations

from pathlib import Path

EXPERIMENT_NAME = "seen_scene_goal_prompts_v2"
TRACKIO_PROJECT = "seen-scene-goal-prompts-v2"
DATE = "2026-08-19"

# Frozen official-data pretrain (seen positive control 20/20) — same as v1.
CHECKPOINT_PATH = Path(
    "/var/tmp/vla_outputs/seen_libero90_official_20260817/checkpoints/030000/pretrained_model"
)

SUITE = "libero_90"
GOAL_SUITE = "libero_goal"
MASTER_SEED = 1000
N_EVAL_EPISODES = 20
EVAL_BATCH_SIZE = 4

# v2 changes vs v1 (2026-08-18_seen_scene_goal_prompts):
#   1. Primary success = the PROMPTED task's goal predicate, evaluated per step
#      on the running scene (episodes also terminate on prompted success).
#      The env task's own predicate is kept as a secondary, v1-comparable
#      metric.
#   2. New block `goal`: every libero_goal instruction whose goal predicate is
#      evaluable in some libero_90 scene gets one seen-scene point; the rest
#      are recorded as skipped (they would be absent-object probes).
#   3. The absent block is dropped (user decision).

# Block A: goal-style paraphrases that are TRUTHFUL for a seen task (verbatim
# libero_goal instructions; goal_ref is the libero_goal task id).
PARAPHRASE_PAIRS = (
    {
        "seen": "put the black bowl on top of the cabinet",
        "prompt": "put the bowl on top of the cabinet",
        "goal_ref": 4,
    },
    {
        "seen": "put the black bowl on the plate",
        "prompt": "put the bowl on the plate",
        "goal_ref": 8,
    },
    {
        "seen": "put the wine bottle on the wine rack",
        "prompt": "put the wine bottle on the rack",
        "goal_ref": 9,
    },
    {
        "seen": "open the top drawer of the cabinet and put the bowl in it",
        "prompt": "open the top drawer and put the bowl inside",
        "goal_ref": 3,
    },
)

# Block B: language causality inside one multi-task scene (2x2 around anchor).
CROSS_ANCHOR_INSTRUCTION = "turn on the stove"

# Block D: nonsense control (no prompted predicate; env metric only).
NONSENSE_PROMPT = "perform the dax florp twice"


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]
