from __future__ import annotations

import json
from pathlib import Path

import pytest

from pretrain_few_shot_low_k.aggregate import wilson_interval
from pretrain_few_shot_low_k.constants import (
    DEMO_BUDGETS,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TRAIN_STEPS,
)
from pretrain_few_shot_low_k.selection import build_manifest
from pretrain_few_shot_low_k.training import build_command
from pretrain_few_shot_low_k.trackio_report import first_outcome_gifs


def _conversion_manifest() -> dict:
    episodes = []
    index = 0
    for instruction in TARGET_INSTRUCTIONS.values():
        for demo in range(50):
            episodes.append(
                {
                    "episode_index": index,
                    "official_file": instruction.replace(" ", "_") + "_demo.hdf5",
                    "official_demo": f"demo_{demo}",
                    "official_demo_index": demo,
                    "task": instruction,
                    "length": 100,
                }
            )
            index += 1
    return {
        "repo_id": "official/libero_goal_rot180_128",
        "source_repo": "yifengzhu-hf/LIBERO-datasets",
        "source_revision": "f13aa24a3da8c43c7225569f28c562979fa0e35a",
        "episodes": episodes,
    }


def test_selection_is_official_first_k():
    manifest = build_manifest(_conversion_manifest())
    assert len(manifest["tasks"]) == 10
    task0 = manifest["tasks"]["0"]
    assert task0["episodes"]["1"] == [0]
    assert task0["episodes"]["3"] == [0, 1, 2]
    assert task0["official_demos"]["2"] == ["demo_0", "demo_1"]
    task1 = manifest["tasks"]["1"]
    assert task1["episodes"]["2"] == [50, 51]


def test_selection_rejects_broken_official_order():
    manifest = _conversion_manifest()
    manifest["episodes"][0], manifest["episodes"][1] = (
        manifest["episodes"][1],
        manifest["episodes"][0],
    )
    with pytest.raises(ValueError, match="order broken"):
        build_manifest(manifest)


def test_training_command_freezes_vlm_and_uses_official_demos(tmp_path: Path):
    manifest = build_manifest(_conversion_manifest())
    root = tmp_path
    (root / "artifacts").mkdir()
    (root / "artifacts/episode_manifest.json").write_text(json.dumps(manifest))
    command = build_command(root, 1, 2)
    assert "--policy.train_expert_only=true" in command
    assert "--policy.freeze_vision_encoder=true" in command
    assert "--policy.use_amp=false" in command
    assert f"--steps={TRAIN_STEPS}" in command
    assert "--dataset.episodes=[50,51]" in command
    assert not any(item.startswith("--dataset.revision") for item in command)
    with pytest.raises(ValueError, match="zero-shot"):
        build_command(root, 0, 0)


def test_env_task_ids_are_identity():
    assert TARGET_ENV_TASK_IDS == {i: i for i in range(10)}
    assert sorted(DEMO_BUDGETS) == [1, 2, 3]


def test_wilson_interval_bounds():
    low, high = wilson_interval(20, 20)
    assert high == 1.0 and low == pytest.approx(0.8389, abs=1e-3)


def test_first_outcome_gifs_picks_first_success_and_failure(tmp_path: Path):
    results = tmp_path / "raw"
    gifs = tmp_path / "gifs"
    video = tmp_path / "video.mp4"
    # minimal valid mp4 via imageio
    import numpy as np
    import imageio.v2 as imageio

    imageio.mimwrite(video, [np.zeros((16, 16, 3), dtype=np.uint8)] * 4, fps=4)
    for task_id, outcomes in {0: ["failure"] * 20, 3: ["success"] + ["failure"] * 19}.items():
        for budget in DEMO_BUDGETS:
            payload = {
                "per_episode": [
                    {"outcome": outcome, "video_path": str(video)}
                    for outcome in outcomes
                ]
            }
            path = results / f"task_{task_id}" / f"k_{budget}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
    items = first_outcome_gifs(results, gifs)
    by_key = {(item["budget"], item["outcome"]): item for item in items}
    assert by_key[(1, "failure")]["task_id"] == 0
    assert by_key[(1, "failure")]["episode_index"] == 0
    assert by_key[(1, "success")]["task_id"] == 3
    assert len(items) == 6
    assert all(Path(item["gif"]).is_file() for item in items)
