from __future__ import annotations

import json
from pathlib import Path

import pytest

from pretrain_smolvla_naive_deterministic_repro.aggregate import wilson_interval
from pretrain_smolvla_naive_deterministic_repro.constants import (
    DEMO_BUDGETS,
    EVAL_BATCH_SIZE,
    EVAL_EPISODES,
    MASTER_SEED,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TRAIN_STEPS,
    noise_seed,
)
from pretrain_smolvla_naive_deterministic_repro.dataset_smoke import (
    verify_loaded_episode_indices,
)
from pretrain_smolvla_naive_deterministic_repro.determinism import compare
from pretrain_smolvla_naive_deterministic_repro.selection import build_manifest
from pretrain_smolvla_naive_deterministic_repro.training import build_command
from pretrain_smolvla_naive_deterministic_repro.trackio_report import first_outcome_gifs


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
    assert len(manifest["tasks"]) == 3
    task0 = manifest["tasks"]["0"]
    assert task0["episodes"]["5"] == [0, 1, 2, 3, 4]
    assert task0["official_demos"]["5"] == [f"demo_{i}" for i in range(5)]
    task1 = manifest["tasks"]["1"]
    assert task1["episodes"]["10"] == list(range(50, 60))


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
    runtime = root / "artifacts/runtime_base_checkpoint"
    runtime.mkdir()
    (runtime / "model.safetensors").write_bytes(b"test")
    command = build_command(root, 1, 5)
    assert f"--policy.path={runtime}" in command
    assert "--policy.train_expert_only=true" in command
    assert "--policy.freeze_vision_encoder=true" in command
    assert "--policy.use_amp=false" in command
    assert f"--steps={TRAIN_STEPS}" in command
    assert any(item.startswith("--policy.vlm_model_name=/var/tmp/") for item in command)
    assert "--dataset.episodes=[50,51,52,53,54]" in command
    assert not any(item.startswith("--dataset.revision") for item in command)
    with pytest.raises(ValueError, match="zero-shot"):
        build_command(root, 0, 0)


def test_env_task_ids_are_identity():
    assert TARGET_ENV_TASK_IDS == {0: 0, 1: 1, 2: 2}
    assert sorted(DEMO_BUDGETS) == [5, 10, 25]


def test_evaluation_seed_bank_is_per_episode():
    assert EVAL_BATCH_SIZE == 1
    assert [noise_seed(index) for index in range(EVAL_EPISODES)] == list(
        range(MASTER_SEED, MASTER_SEED + EVAL_EPISODES)
    )


def test_dataset_smoke_rejects_wrong_loaded_episode():
    verify_loaded_episode_indices([5, 6, 7], [5, 5, 6, 7])
    with pytest.raises(RuntimeError, match="expected exactly"):
        verify_loaded_episode_indices([5, 6, 7], [0, 1, 2])


def test_determinism_comparison_is_episode_index_stable():
    episodes = [
        {
            "episode_ix": index,
            "env_seed": MASTER_SEED + index,
            "noise_seed": MASTER_SEED + index,
            "init_state_id": index,
            "seed": MASTER_SEED + index,
            "success": index % 2 == 0,
            "sum_reward": float(index % 2 == 0),
            "max_reward": float(index % 2 == 0),
        }
        for index in range(EVAL_EPISODES)
    ]
    payload = {
        "logical_task_id": 0,
        "demo_budget": 5,
        "model_safetensors_sha256": "abc",
        "successes": 10,
        "per_episode": episodes,
    }
    assert compare(payload, json.loads(json.dumps(payload)))["passed"]
    changed = json.loads(json.dumps(payload))
    changed["per_episode"][3]["sum_reward"] = 1.0
    result = compare(payload, changed)
    assert not result["passed"]
    assert result["mismatched_episode_indices"] == [3]
    changed_state = json.loads(json.dumps(payload))
    changed_state["per_episode"][4]["init_state_id"] = 17
    assert compare(payload, changed_state)["mismatched_episode_indices"] == [4]


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
    for task_id, outcomes in {0: ["failure"] * 20, 2: ["success"] + ["failure"] * 19}.items():
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
    assert by_key[(5, "failure")]["task_id"] == 0
    assert by_key[(5, "failure")]["episode_index"] == 0
    assert by_key[(5, "success")]["task_id"] == 2
    assert len(items) == 6
    assert all(Path(item["gif"]).is_file() for item in items)
