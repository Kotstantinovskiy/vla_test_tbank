from __future__ import annotations

from pathlib import Path

BASE_MODEL_REPO = "lerobot/smolvla_base"
BASE_MODEL_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"

# Official LIBERO HDF5 source, downloaded by the repository's revision-pinned
# downloader (scripts/download_official_libero.py) for the provenance audit.
OFFICIAL_REPO = "yifengzhu-hf/LIBERO-datasets"
OFFICIAL_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
OFFICIAL_ROOT = Path("/var/tmp/libero_official_f13aa24")

# Converted LeRobot v3 datasets (this experiment's own conversion).
CONVERTED_ROOT = Path("/var/tmp/vla_libero_official_rot180")
SEEN_SUITE = "libero_90"
TARGET_SUITE = "libero_goal"
SEEN_REPO_ID = "official/libero_90_rot180_128"
TARGET_REPO_ID = "official/libero_goal_rot180_128"

# Conversion contract.  Frames are stored as rot180(official rendering) —
# exactly the orientation LeRobot's LIBERO evaluation feeds the policy — at
# the native official resolution 128x128 (no fabricated upscale).  Actions are
# the official float32 values bit-for-bit; the 8-dim state is the official
# ee_pos(3) + ee_ori(3, axis-angle) + gripper_states(2), matching the layout
# LiberoProcessorStep assembles at rollout time (eef_pos, eef_axisangle,
# gripper_qpos).  Episodes keep the official order: files sorted by name,
# demos sorted by numeric demo index, so "first k episodes" means the official
# demo_0..demo_{k-1} of each task.
INPUT_ORIENTATION = "rot180_of_official_equals_eval_convention"
IMAGE_SIZE = 128
FPS = 20
VIDEO_CRF = 18  # default crf=30 visibly degrades 128x128 frames; 18 keeps them near-lossless

CAMERA_KEYS = {
    "observation.images.top": "agentview_rgb",
    "observation.images.wrist_image": "eye_in_hand_rgb",
}

EXPECTED_SEEN_TASKS = 90
EXPECTED_TARGET_TASKS = 10
DEMOS_PER_TASK = 50

WORLD_SIZE = 4
GPU_IDS = (0, 1, 2, 3)
PER_RANK_BATCH_SIZE = 8
EFFECTIVE_BATCH_SIZE = WORLD_SIZE * PER_RANK_BATCH_SIZE
TRAIN_STEPS = 30_000
SMOKE_STEPS = 4
SMOKE_EPISODES = tuple(range(16))
SEED = 1_000
LEARNING_RATE = 1e-4
SAVE_FREQ = 5_000
LOG_FREQ = 50
NUM_WORKERS_PER_RANK = 16

SEEN_CONTROL_SUITE = "libero_90"
SEEN_CONTROL_ENV_TASK_ID = 0
SEEN_CONTROL_EPISODES = 20
SEEN_CONTROL_BATCH_SIZE = 4

TRACKIO_PROJECT = "smolvla-pretrain-libero"
TRACKIO_GROUP = "seen-pretrain-official"
FULL_RUN_NAME = "2026-08-17-pretrain-libero-ddp4"
SMOKE_RUN_NAME = "2026-08-17-pretrain-libero-ddp4-smoke"
RESUMED_LIVE_RUN_NAME = "2026-08-17-pretrain-libero-ddp4-resumed"


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]


def seen_dataset_root() -> Path:
    return CONVERTED_ROOT / SEEN_SUITE


def target_dataset_root() -> Path:
    return CONVERTED_ROOT / TARGET_SUITE


def default_output_root() -> Path:
    return Path("/var/tmp/vla_outputs/seen_libero90_official_20260817")
