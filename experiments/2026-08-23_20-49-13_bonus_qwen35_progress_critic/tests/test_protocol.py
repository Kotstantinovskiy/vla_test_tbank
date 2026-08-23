from bonus_qwen35_progress_critic.constants import CONFIG_PATH, TARGET_GOAL_INSTRUCTIONS
from bonus_qwen35_progress_critic.data import (
    EpisodeRecord,
    build_validation_specs,
    endpoint_and_bin,
    frame_indices_for_endpoint,
    split_records_by_task,
)
from bonus_qwen35_progress_critic.utils import load_config
from bonus_qwen35_progress_critic.utils import estimate_full_runtime


def record(index: int, task: str, length: int = 101) -> EpisodeRecord:
    return EpisodeRecord(index, task, length, "/tmp/video.mp4", 0.0, length / 20)


def test_protocol_is_progress_only_and_no_gradient_checkpointing():
    config = load_config(CONFIG_PATH)
    assert config["model"]["num_progress_bins"] == 32
    assert config["model"]["max_frames"] == 4
    assert config["model"]["gradient_checkpointing"] is False
    assert config["experiment"]["seed"] == 1000
    assert "ranking" in config["experiment"]["scope"]


def test_task_split_is_deterministic_and_disjoint():
    records = [record(index, f"task-{index // 2}") for index in range(20)]
    train_a, validation_a, tasks_a = split_records_by_task(records, 0.2, 1000)
    train_b, validation_b, tasks_b = split_records_by_task(records, 0.2, 1000)
    assert tasks_a == tasks_b
    assert [row.episode_index for row in train_a] == [row.episode_index for row in train_b]
    assert [row.episode_index for row in validation_a] == [row.episode_index for row in validation_b]
    assert {row.task for row in train_a}.isdisjoint({row.task for row in validation_a})


def test_absolute_progress_targets_and_four_frame_prefix():
    endpoint, target_bin = endpoint_and_bin(length=101, requested_bin=31, num_bins=32)
    assert endpoint == 100
    assert target_bin == 31
    indices = frame_indices_for_endpoint(endpoint, max_frames=4)
    assert indices == [0, 33, 66, 100]
    endpoint, target_bin = endpoint_and_bin(length=101, requested_bin=0, num_bins=32)
    assert endpoint == 0
    assert target_bin == 0
    assert frame_indices_for_endpoint(endpoint, 4) == [0, 0, 0, 0]


def test_validation_specs_are_fixed_and_bounded():
    records = [record(index, f"task-{index}") for index in range(5)]
    specs_a = build_validation_specs(records, [0, 10, 20, 31], max_samples=7, seed=1000)
    specs_b = build_validation_specs(records, [0, 10, 20, 31], max_samples=7, seed=1000)
    assert specs_a == specs_b
    assert len(specs_a) == 7
    assert all(spec.target_bin in {0, 10, 20, 31} for spec in specs_a)


def test_target_goal_instructions_are_explicitly_guarded():
    assert TARGET_GOAL_INSTRUCTIONS == {
        "open the middle drawer of the cabinet",
        "put the bowl on the stove",
        "put the wine bottle on top of the cabinet",
    }


def test_runtime_estimate_scales_validation_sample_count():
    estimate = estimate_full_runtime(
        median_step_seconds=2.0,
        full_steps=100,
        validation_seconds_total=3.0,
        observed_validation_runs=3,
        observed_validation_samples=10,
        full_validation_runs=5,
        full_validation_samples=40,
    )
    assert estimate["mean_validation_seconds_per_sample"] == 0.1
    assert estimate["estimated_full_validation_seconds"] == 20.0
    assert estimate["estimated_full_training_seconds"] == 220.0
