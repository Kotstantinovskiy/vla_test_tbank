from __future__ import annotations

from pathlib import Path

EXPERIMENT_NAME = "seen_semantic_paraphrases"
TRACKIO_PROJECT = "seen-semantic-paraphrases"
DATE = "2026-08-20"
CHECKPOINT_PATH = Path(
    "/var/tmp/vla_outputs/seen_libero90_official_20260817/checkpoints/030000/pretrained_model"
)
CHECKPOINT_PROVENANCE = {
    "pretraining_experiment": "2026-08-17_smolvla_pretrain_libero",
    "base_model_repo": "lerobot/smolvla_base",
    "base_model_revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
    "pretraining_dataset": "official/libero_90_rot180_128 (in-repo conversion)",
    "official_source_revision": "f13aa24a3da8c43c7225569f28c562979fa0e35a",
    "pretraining_steps": 30_000,
    "seen_positive_control": "libero_90 task 0: 20/20",
}
SUITE = "libero_90"
EXPECTED_SUITE_TASKS = 90
MASTER_SEED = 1000
N_EVAL_EPISODES = 20
EVAL_BATCH_SIZE = 1
DETERMINISM_LABEL = "paraphrase__task_0"
DETERMINISM_PREFIX_LABEL = "exact__task_0"


def noise_seed(episode_index: int) -> int:
    return MASTER_SEED + episode_index


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]
