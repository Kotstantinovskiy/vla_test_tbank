from __future__ import annotations

import json
from pathlib import Path

import pytest

from pretrain_smolvla_low_k_action_steps.aggregate import aggregate, wilson_interval
from pretrain_smolvla_low_k_action_steps.constants import (
    ACTION_STEPS,
    DEMO_BUDGETS,
    EVAL_BATCH_SIZE,
    EVAL_EPISODES,
    MASTER_SEED,
    OFFICIAL_SOURCE_REVISION,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TRAIN_STEPS,
    noise_seed,
    result_path,
)
from pretrain_smolvla_low_k_action_steps.determinism import compare
from pretrain_smolvla_low_k_action_steps.dataset_smoke import verify_loaded_episode_indices
from pretrain_smolvla_low_k_action_steps.prepare import evaluation_plan
from pretrain_smolvla_low_k_action_steps.selection import build_manifest
from pretrain_smolvla_low_k_action_steps.trackio_report import first_outcome_media
from pretrain_smolvla_low_k_action_steps.training import build_command
import pretrain_smolvla_low_k_action_steps.training as training_module


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
        "source_revision": OFFICIAL_SOURCE_REVISION,
        "episodes": episodes,
    }


def test_selection_is_official_first_k():
    manifest = build_manifest(_conversion_manifest())
    assert len(manifest["tasks"]) == 10
    assert manifest["tasks"]["0"]["episodes"]["1"] == [0]
    assert manifest["tasks"]["0"]["episodes"]["3"] == [0, 1, 2]
    assert manifest["tasks"]["0"]["official_demos"]["2"] == [
        "demo_0",
        "demo_1",
    ]
    assert manifest["tasks"]["1"]["episodes"]["2"] == [50, 51]


def test_selection_rejects_broken_order_and_revision():
    manifest = _conversion_manifest()
    manifest["episodes"][0], manifest["episodes"][1] = (
        manifest["episodes"][1],
        manifest["episodes"][0],
    )
    with pytest.raises(ValueError, match="order broken"):
        build_manifest(manifest)
    manifest = _conversion_manifest()
    manifest["source_revision"] = "wrong"
    with pytest.raises(ValueError, match="source revision"):
        build_manifest(manifest)


def test_loaded_episode_verifier_rejects_silent_selection_bug():
    verify_loaded_episode_indices([50, 51], [50, 50, 51, 51])
    with pytest.raises(RuntimeError, match="expected exactly"):
        verify_loaded_episode_indices([50, 51], [0, 0, 1, 1])


def test_training_recipe_matches_low_k_and_is_not_triplicated(tmp_path: Path):
    manifest = build_manifest(_conversion_manifest())
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/episode_manifest.json").write_text(
        json.dumps(manifest)
    )
    command = build_command(tmp_path, 1, 2)
    assert "--policy.train_expert_only=true" in command
    assert "--policy.freeze_vision_encoder=true" in command
    assert "--policy.use_amp=false" in command
    assert f"--steps={TRAIN_STEPS}" in command
    assert "--dataset.episodes=[50,51]" in command
    assert not any(item.startswith("--policy.n_action_steps") for item in command)
    with pytest.raises(ValueError, match="zero-shot"):
        build_command(tmp_path, 0, 0)


def test_checkpoint_completeness_requires_weights_and_exact_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(training_module, "OUTPUT_ROOT", tmp_path)
    model = training_module.final_model(0, 1)
    state = (
        tmp_path
        / "task_0/k_1/checkpoints/002000/training_state/training_step.json"
    )
    model.mkdir(parents=True)
    state.parent.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"weights")
    state.write_text(json.dumps({"step": TRAIN_STEPS - 1}))
    assert training_module.training_complete(0, 1) is False
    state.write_text(json.dumps({"step": TRAIN_STEPS}))
    assert training_module.training_complete(0, 1) is True


def test_action_step_and_seed_protocol():
    assert ACTION_STEPS == (1, 10, 25)
    assert DEMO_BUDGETS == (1, 2, 3)
    assert EVAL_BATCH_SIZE == 1
    assert [noise_seed(index) for index in range(EVAL_EPISODES)] == list(
        range(MASTER_SEED, MASTER_SEED + EVAL_EPISODES)
    )
    assert TARGET_ENV_TASK_IDS == {index: index for index in range(10)}
    assert str(result_path(Path("raw"), 2, 3, 10)) == "raw/task_2/k_3/n_10.json"


def test_frozen_evaluation_plan_has_90_points_and_1800_videos():
    mapping = [
        {
            "logical_task_id": task_id,
            "env_task_id": task_id,
            "name": f"task_{task_id}",
            "instruction": instruction,
        }
        for task_id, instruction in TARGET_INSTRUCTIONS.items()
    ]
    plan = evaluation_plan(mapping)
    assert plan["training_jobs"] == 30
    assert plan["evaluation_points"] == 90
    assert plan["main_rollout_videos"] == 1800
    assert plan["maximum_policy_invocations"] == 205200
    assert len({point["label"] for point in plan["points"]}) == 90


def _determinism_payload(success: bool = True) -> dict:
    return {
        "logical_task_id": 0,
        "demo_budget": 1,
        "n_action_steps": 10,
        "model_safetensors_sha256": "abc",
        "successes": int(success),
        "per_episode": [
            {
                "episode_ix": 0,
                "env_seed": 1000,
                "noise_seed": 1000,
                "seed": 1000,
                "success": success,
                "sum_reward": float(success),
                "max_reward": float(success),
                "video_path": "layout-specific.mp4",
            }
        ],
    }


def test_determinism_comparison_is_layout_path_independent():
    left = _determinism_payload()
    right = _determinism_payload()
    right["per_episode"][0]["video_path"] = "other.mp4"
    assert compare(left, right)["passed"] is True
    right["per_episode"][0]["success"] = False
    assert compare(left, right)["passed"] is False


def _prior() -> dict:
    return {
        "prior_n50_low_k_rates": {
            "harness": "old",
            "comparison_warning": "descriptive only",
            "mean_all_10": {str(k): 0.5 for k in DEMO_BUDGETS},
            "mean_tasks_0_2": {str(k): 0.5 for k in DEMO_BUDGETS},
            "per_task_successes_out_of_20": {},
        }
    }


def _write_synthetic_results(root: Path) -> None:
    for task_id, instruction in TARGET_INSTRUCTIONS.items():
        for budget in DEMO_BUDGETS:
            for action_steps in ACTION_STEPS:
                successes = {1: 0, 10: 1, 25: 2}[action_steps]
                episodes = [
                    {
                        "episode_ix": index,
                        "env_seed": MASTER_SEED + index,
                        "noise_seed": MASTER_SEED + index,
                        "success": index < successes,
                    }
                    for index in range(EVAL_EPISODES)
                ]
                payload = {
                    "instruction": instruction,
                    "demo_budget": budget,
                    "n_action_steps": action_steps,
                    "chunk_size": 50,
                    "n_episodes": EVAL_EPISODES,
                    "batch_size": 1,
                    "successes": successes,
                    "success_rate": successes / EVAL_EPISODES,
                    "model": f"task_{task_id}_k_{budget}",
                    "model_safetensors_sha256": f"sha-{task_id}-{budget}",
                    "per_episode": episodes,
                }
                path = result_path(root, task_id, budget, action_steps)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload))


def test_aggregate_pairs_same_weights_and_episode_seed_banks(tmp_path: Path):
    _write_synthetic_results(tmp_path)
    summary = aggregate(tmp_path, _prior())
    assert summary["means"]["mean_all_10"]["1"] == {
        "1": 0.0,
        "10": 0.05,
        "25": 0.1,
    }
    assert len(summary["paired_action_step_comparisons"]) == 90
    assert all(
        item["mcnemar_p"] >= 0
        for item in summary["paired_action_step_comparisons"]
    )


def test_aggregate_rejects_different_weights_between_action_steps(tmp_path: Path):
    _write_synthetic_results(tmp_path)
    path = result_path(tmp_path, 0, 1, 10)
    payload = json.loads(path.read_text())
    payload["model_safetensors_sha256"] = "different"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different weights"):
        aggregate(tmp_path, _prior())


def test_first_outcome_media_records_k_n_episode_and_seeds(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    for budget in DEMO_BUDGETS:
        for action_steps in ACTION_STEPS:
            for task_id, outcome in ((0, "failure"), (1, "success")):
                path = result_path(tmp_path / "raw", task_id, budget, action_steps)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "per_episode": [
                                {
                                    "outcome": outcome,
                                    "episode_ix": 0,
                                    "env_seed": 1000,
                                    "noise_seed": 1000,
                                    "video_path": str(video),
                                }
                            ]
                        }
                    )
                )
    items = first_outcome_media(tmp_path / "raw")
    assert len(items) == 18
    assert {(item["demo_budget"], item["n_action_steps"]) for item in items} == {
        (k, n) for k in DEMO_BUDGETS for n in ACTION_STEPS
    }
    assert all(item["env_seed"] == item["noise_seed"] == 1000 for item in items)


def test_wilson_interval_bounds():
    low, high = wilson_interval(20, 20)
    assert high == 1.0 and low == pytest.approx(0.8389, abs=1e-3)
