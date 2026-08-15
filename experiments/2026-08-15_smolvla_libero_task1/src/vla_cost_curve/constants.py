from __future__ import annotations

SEEN_REPO = "crislmfroes/smolvla-libero-90"
SEEN_REVISION = "418f9d0e5b48585bcee1e1a7d47e302629af78da"

TARGET_DATASET_REPO = "HuggingFaceVLA/libero"
TARGET_DATASET_REVISION = "9176d427966503c81ac9f8f96502e50861a15ee7"
TARGET_SUITE = "libero_goal"

# These strings come from hf-libero 0.1.4's benchmark order. Selection in the
# combined Hub dataset is deliberately done by instruction, because its global
# task_index values (19, 11, 12) are not the suite-local environment IDs.
TARGET_INSTRUCTIONS: dict[int, str] = {
    0: "open the middle drawer of the cabinet",
    1: "put the wine bottle on the rack",
    2: "open the top drawer and put the bowl inside",
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
