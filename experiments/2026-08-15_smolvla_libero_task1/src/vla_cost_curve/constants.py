from __future__ import annotations

SEEN_REPO = "crislmfroes/smolvla-libero-90"
SEEN_REVISION = "418f9d0e5b48585bcee1e1a7d47e302629af78da"

TARGET_DATASET_REPO = "HuggingFaceVLA/libero"
TARGET_DATASET_REVISION = "9176d427966503c81ac9f8f96502e50861a15ee7"
TARGET_SUITE = "libero_goal"

# Logical task IDs are stable experiment labels. They are deliberately separate
# from LIBERO's suite-local environment IDs: the assignment selects three
# instructions, not the first three tasks in the suite's native ordering.
TARGET_INSTRUCTIONS: dict[int, str] = {
    0: "open the middle drawer of the cabinet",
    1: "put the wine bottle on the rack",
    2: "open the top drawer and put the bowl inside",
}
TARGET_ENV_TASK_IDS: dict[int, int] = {
    0: 0,
    1: 9,
    2: 3,
}

DEMO_BUDGETS = (5, 10, 25)
MASTER_SEED = 1000
N_EVAL_EPISODES = 20
EVAL_BATCH_SIZE = 4
NONSENSE_PROMPT = "perform the dax florp twice"

CAMERA_KEY_RENAMES = {
    "observation.images.top": "observation.images.image",
    "observation.images.wrist_image": "observation.images.image2",
}
