from __future__ import annotations

import numpy as np
import pytest

from smolvla_pretrain_libero.constants import (
    EFFECTIVE_BATCH_SIZE,
    FPS,
    IMAGE_SIZE,
    PER_RANK_BATCH_SIZE,
    TRAIN_STEPS,
    WORLD_SIZE,
)
from smolvla_pretrain_libero.convert import (
    dataset_features,
    natural_demo_index,
    rot180,
    task_from_file,
)
from smolvla_pretrain_libero.runner import base_command, parse_training_metrics
from smolvla_pretrain_libero.schema_adapter import (
    LIBERO_INPUT_FEATURES,
    LIBERO_OUTPUT_FEATURES,
)

from pathlib import Path


def test_protocol_matches_reference_global_batch() -> None:
    assert WORLD_SIZE == 4
    assert PER_RANK_BATCH_SIZE == 8
    assert EFFECTIVE_BATCH_SIZE == 32
    assert TRAIN_STEPS == 30_000
    assert FPS == 20
    assert IMAGE_SIZE == 128


def test_schema_uses_native_official_resolution() -> None:
    assert LIBERO_INPUT_FEATURES["observation.images.top"]["shape"] == [3, 128, 128]
    assert LIBERO_INPUT_FEATURES["observation.images.wrist_image"]["shape"] == [3, 128, 128]
    assert LIBERO_INPUT_FEATURES["observation.state"]["shape"] == [8]
    assert LIBERO_OUTPUT_FEATURES["action"]["shape"] == [7]


def test_dataset_features_match_schema() -> None:
    features = dataset_features()
    assert features["observation.images.top"]["shape"] == (128, 128, 3)
    assert features["observation.images.top"]["dtype"] == "video"
    assert features["observation.state"]["shape"] == (8,)
    assert features["action"]["shape"] == (7,)


def test_task_from_file_strips_scene_prefix_only_for_libero_90() -> None:
    assert (
        task_from_file(Path("KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5"), "libero_90")
        == "close the top drawer of the cabinet"
    )
    assert (
        task_from_file(Path("open_the_middle_drawer_of_the_cabinet_demo.hdf5"), "libero_goal")
        == "open the middle drawer of the cabinet"
    )


def test_natural_demo_index_orders_numerically() -> None:
    names = ["demo_10", "demo_2", "demo_0", "demo_30"]
    assert sorted(names, key=natural_demo_index) == ["demo_0", "demo_2", "demo_10", "demo_30"]
    with pytest.raises(ValueError):
        natural_demo_index("demo_x")


def test_rot180_flips_both_axes() -> None:
    frames = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    rotated = rot180(frames)
    assert np.array_equal(rotated, frames[:, ::-1, ::-1])
    assert np.array_equal(rot180(rotated), frames)
    assert rotated.flags["C_CONTIGUOUS"]


def test_commands_use_stock_trainer_and_local_dataset() -> None:
    for mode in ("smoke", "full"):
        command = base_command(mode)
        assert any(item.endswith("bin/lerobot-train") for item in command)
        assert "--dataset.repo_id=official/libero_90_rot180_128" in command
        assert not any(item.startswith("--dataset.revision=") for item in command)
        assert any(
            item == "--dataset.root=/var/tmp/vla_libero_official_rot180/libero_90"
            for item in command
        )


def test_full_command_is_ddp_and_does_not_subset_episodes() -> None:
    command = base_command("full")
    assert "--nproc-per-node=4" in command
    assert "--batch_size=8" in command
    assert "--steps=30000" in command
    assert not any(item.startswith("--dataset.episodes=") for item in command)


def test_training_log_parser() -> None:
    parsed = parse_training_metrics(
        "INFO step:200 smpl:6K ep:41 epch:0.29 loss:0.321 grdn:2.5 "
        "lr:1.0e-04 updt_s:0.812 data_s:0.123 smp/s:39 mem_gb:15.7"
    )
    assert parsed is not None
    step, metrics = parsed
    assert step == 200
    assert metrics["train/loss"] == 0.321
