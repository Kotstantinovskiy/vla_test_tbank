import json
from pathlib import Path

from bonus_critic_vs_robometer_ranking.aggregate import summarize
from bonus_critic_vs_robometer_ranking.manifest import FORBIDDEN_KEYS, build_manifest
from bonus_critic_vs_robometer_ranking.utils import load_config
from bonus_critic_vs_robometer_ranking.video import uniform_frame_indices


def test_uniform_frame_contract():
    assert uniform_frame_indices(101, 4) == [0, 33, 67, 100]
    assert uniform_frame_indices(4, 4) == [0, 1, 2, 3]


def test_protocol_is_n_action_steps_50_only():
    config = load_config()
    assert config["scope"]["n_action_steps"] == 50
    assert config["scope"]["expected_videos"] == 3 * 7 * 20


def test_blind_manifest_has_no_labels_and_all_media():
    rows = build_manifest(load_config())
    assert len(rows) == 420
    assert {row["n_action_steps"] for row in rows} == {50}
    assert not any(FORBIDDEN_KEYS.intersection(row) for row in rows)
    assert all(Path(row["video_path"]).is_file() for row in rows)


def test_own_checkpoint_was_fixed_at_saved_step_2000():
    config = load_config()
    checkpoint = Path(config["own_critic"]["checkpoint"])
    assert checkpoint.name == "002000"
    assert (checkpoint / "adapter" / "adapter_model.safetensors").is_file()
    assert (checkpoint / "progress_head.safetensors").is_file()


def test_robometer_revision_and_bins_are_fixed():
    config = load_config()
    assert config["robometer"]["revision"] == "637fa8ecb7fb872cb5783c19d0825a08dc20fc8c"
    assert config["robometer"]["progress_discrete_bins"] == 32


def test_ranking_aggregation_is_per_task():
    config = load_config()
    rows = []
    candidates = config["scope"]["candidates"]
    for task_id in config["scope"]["task_ids"]:
        for candidate_ix, candidate in enumerate(candidates):
            for episode_ix in range(20):
                success = episode_ix < candidate_ix * 3
                rows.append(
                    {
                        "task_id": task_id,
                        "candidate": candidate,
                        "success": success,
                        "own_score": candidate_ix / 10 + episode_ix / 1000,
                        "robometer_score": -candidate_ix / 10 + episode_ix / 1000,
                    }
                )
    summary, metrics = summarize(rows, config)
    assert len(summary) == 21
    assert metrics["macro"]["own"]["spearman_rho"] == 1.0
    assert metrics["macro"]["robometer"]["spearman_rho"] == -1.0
