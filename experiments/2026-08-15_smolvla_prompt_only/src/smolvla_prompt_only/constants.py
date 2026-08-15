from __future__ import annotations

CHECKPOINT_REPO = "crislmfroes/smolvla-libero-90"
CHECKPOINT_REVISION = "418f9d0e5b48585bcee1e1a7d47e302629af78da"
TARGET_SUITE = "libero_goal"

TARGET_INSTRUCTIONS: dict[int, str] = {
    0: "open the middle drawer of the cabinet",
    1: "put the wine bottle on the rack",
    2: "open the top drawer and put the bowl inside",
}

PROMPT_CONDITIONS = ("true", "wrong", "nonsense")
NONSENSE_PROMPT = "perform the dax florp twice"
MASTER_SEED = 1000
N_EVAL_EPISODES = 20
EVAL_BATCH_SIZE = 4
