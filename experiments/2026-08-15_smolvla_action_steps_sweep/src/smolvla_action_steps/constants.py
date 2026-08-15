from __future__ import annotations

CHECKPOINT_REPO = "crislmfroes/smolvla-libero-90"
CHECKPOINT_REVISION = "418f9d0e5b48585bcee1e1a7d47e302629af78da"
CHECKPOINT_MODEL_SHA256 = "2e5b0a69e1ad03520f8d6a8abb940f22f361250db9cbe4a47feacfcbe35eda6e"

TARGET_SUITE = "libero_goal"
TARGET_INSTRUCTIONS: dict[int, str] = {
    0: "open the middle drawer of the cabinet",
    1: "put the wine bottle on the rack",
    2: "open the top drawer and put the bowl inside",
}

ACTION_STEPS = (1, 5, 10, 25, 50)
DEMO_BUDGETS = (0, 5, 10, 25)
ADAPTED_BUDGETS = (5, 10, 25)
PROMPT_CONDITIONS = ("true", "wrong", "nonsense")
NONSENSE_PROMPT = "perform the dax florp twice"
MASTER_SEED = 1000
N_EVAL_EPISODES = 20
EVAL_BATCH_SIZE = 4
