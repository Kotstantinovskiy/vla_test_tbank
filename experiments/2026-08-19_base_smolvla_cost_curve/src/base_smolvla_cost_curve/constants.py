from __future__ import annotations

from pathlib import Path

EXPERIMENT_NAME = "base_smolvla_cost_curve"
TRACKIO_PROJECT = "base-smolvla-cost-curve"
DATE = "2026-08-19"

# Base: PLAIN lerobot/smolvla_base (community SO-100 pretraining, has never
# seen LIBERO or a Franka) via the schema-adapted local snapshot built by
# 2026-08-17_smolvla_pretrain_libero (LIBERO feature schema in the processor
# configs; weights untouched).  This ablation removes the libero_90 pretrain
# from the pipeline; everything downstream (recipe, demos, eval) is
# byte-identical to the pretrained cost curve.
BASE_CHECKPOINT = Path(
    "/var/tmp/vla_outputs/smolvla_base_libero_official_schema_c83c3163"
)
BASE_PROVENANCE = {
    "ablation": "NO libero_90 pretrain — plain smolvla_base",
    "base_model_repo": "lerobot/smolvla_base",
    "base_model_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
    "schema_adapter": "2026-08-17_smolvla_pretrain_libero SCHEMA_ADAPTER.json",
    "embodiment_gap": "base trained on real SO-100; LIBERO is simulated Franka",
    "no_zero_shot_point": "k=0 undefined: state/action projections are "
    "re-initialized for the 8-dim LIBERO schema only at fine-tune time, so "
    "an untrained evaluation would sample through random projections",
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

DEMO_BUDGETS = (1, 2, 3, 5, 10, 25)
MASTER_SEED = 1_000
TRAIN_STEPS = 2_000
TRAIN_BATCH_SIZE = 32
TRAIN_WORKERS = 8
TRAIN_LOG_FREQ = 25
EVAL_EPISODES = 20
EVAL_BATCH_SIZE = 4
GPU_IDS = (0, 1, 2, 3)
# Co-tenancy: training jobs are dataloader-bound (GPU busy ~10% of the time),
# so the orchestrator runs several jobs per GPU. Declared in protocol.yaml;
# per-job commands are byte-identical to the reference recipe.
JOBS_PER_GPU = 3

OUTPUT_ROOT = Path("/var/tmp/vla_outputs/base_smolvla_cost_curve_20260819")


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]
