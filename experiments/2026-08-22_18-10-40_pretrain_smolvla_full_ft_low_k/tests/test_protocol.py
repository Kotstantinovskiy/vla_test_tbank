from __future__ import annotations

import json
from pathlib import Path

import pytest

from pretrain_smolvla_full_ft_low_k.aggregate import wilson_interval
from pretrain_smolvla_full_ft_low_k.audit import (
    INTENDED_FROZEN_PATTERNS,
    classify_trainable,
    intended_frozen,
)
from pretrain_smolvla_full_ft_low_k.constants import (
    DEMO_BUDGETS,
    EVAL_ACTION_STEPS,
    EVAL_BATCH_SIZE,
    EVAL_EPISODES,
    FREEZE_VISION_ENCODER,
    MASTER_SEED,
    NUM_VLM_LAYERS,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TRAIN_BATCH_SIZE,
    TRAIN_EXPERT_ONLY,
    TRAIN_STATE_PROJ,
    TRAIN_STEPS,
    TRAINED_ACTION_STEPS,
    noise_seed,
    result_path,
)
from pretrain_smolvla_full_ft_low_k.dataset_smoke import (
    verify_loaded_episode_indices,
)
from pretrain_smolvla_full_ft_low_k.determinism import compare
from pretrain_smolvla_full_ft_low_k.selection import build_manifest
from pretrain_smolvla_full_ft_low_k.training import build_command
from pretrain_smolvla_full_ft_low_k.trackio_report import first_outcome_gifs


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


def test_training_command_is_full_fine_tune(tmp_path: Path):
    manifest = build_manifest(_conversion_manifest())
    root = tmp_path
    (root / "artifacts").mkdir()
    (root / "artifacts/episode_manifest.json").write_text(json.dumps(manifest))
    runtime = root / "artifacts/runtime_base_checkpoint"
    runtime.mkdir()
    (runtime / "model.safetensors").write_bytes(b"test")
    command = build_command(root, 1, 2)
    assert f"--policy.path={runtime}" in command
    assert "--policy.train_expert_only=false" in command
    assert "--policy.freeze_vision_encoder=false" in command
    assert "--policy.train_state_proj=true" in command
    assert "--policy.use_amp=false" in command
    assert f"--steps={TRAIN_STEPS}" in command
    assert f"--seed={MASTER_SEED}" in command
    assert f"--batch_size={TRAIN_BATCH_SIZE}" in command
    assert any(item.startswith("--policy.vlm_model_name=/var/tmp/") for item in command)
    assert "--dataset.episodes=[50,51]" in command
    assert not any(item.startswith("--dataset.revision") for item in command)
    with pytest.raises(ValueError, match="zero-shot"):
        build_command(root, 0, 0)


def test_full_fine_tune_flags():
    assert FREEZE_VISION_ENCODER is False
    assert TRAIN_EXPERT_ONLY is False
    assert TRAIN_STATE_PROJ is True


def test_env_task_ids_are_identity():
    assert TARGET_ENV_TASK_IDS == {0: 0, 1: 1, 2: 2}
    assert sorted(DEMO_BUDGETS) == [1, 2, 3]


def test_eval_action_steps_variants():
    assert EVAL_ACTION_STEPS == (50, 25)
    assert TRAINED_ACTION_STEPS == 50
    assert EVAL_ACTION_STEPS[0] == TRAINED_ACTION_STEPS
    path = result_path(Path("/r"), 1, 2, 25)
    assert path == Path("/r/task_1/k_2/n_25.json")


def test_seed_protocol_matches_deterministic_repro():
    """Seeds must equal the latest low-k runs (deterministic repro, seed 1000).

    Training seed 1000; per-episode env seed and SmolVLA flow-noise seed are
    both 1000 + episode_index; LIBERO init_state_id is the episode index.
    """

    assert MASTER_SEED == 1_000
    assert TRAIN_STEPS == 2_000
    assert EVAL_EPISODES == 20
    assert EVAL_BATCH_SIZE == 1
    assert [noise_seed(index) for index in range(EVAL_EPISODES)] == list(
        range(1_000, 1_020)
    )


def test_intended_frozen_patterns_cover_only_guard_tensors():
    assert (
        f".vlm.model.text_model.layers.{NUM_VLM_LAYERS - 1}."
        in INTENDED_FROZEN_PATTERNS
    )
    assert intended_frozen("model.vlm_with_expert.vlm.lm_head.weight")
    assert intended_frozen("model.vlm_with_expert.vlm.model.text_model.norm.weight")
    assert intended_frozen(
        f"model.vlm_with_expert.vlm.model.text_model.layers.{NUM_VLM_LAYERS - 1}.mlp.up_proj.weight"
    )
    assert intended_frozen("model.vlm_with_expert.lm_expert.lm_head.weight")
    assert not intended_frozen("model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.up_proj.weight")
    assert not intended_frozen(
        f"model.vlm_with_expert.vlm.model.text_model.layers.{NUM_VLM_LAYERS - 10}.mlp.up_proj.weight"
    )
    assert not intended_frozen("model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.fc1.weight")
    assert not intended_frozen("model.vlm_with_expert.lm_expert.layers.0.mlp.up_proj.weight")


def test_classify_trainable_covers_vlm_and_expert_groups():
    names = [
        "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.fc1.weight",
        "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight",
        "model.vlm_with_expert.vlm.model.text_model.layers.3.self_attn.q_proj.weight",
        "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight",
        "model.vlm_with_expert.lm_expert.layers.2.mlp.down_proj.weight",
        "model.state_proj.weight",
        "model.action_in_proj.weight",
        "model.action_out_proj.weight",
        "model.action_time_mlp_in.weight",
    ]
    groups = classify_trainable(names)
    assert all(len(members) == 1 for members in groups.values()), groups


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
        "demo_budget": 1,
        "n_action_steps": 50,
        "model_safetensors_sha256": "abc",
        "successes": 10,
        "per_episode": episodes,
    }
    assert compare(payload, json.loads(json.dumps(payload)))["passed"]
    other_variant = json.loads(json.dumps(payload))
    other_variant["n_action_steps"] = 25
    with pytest.raises(ValueError, match="n_action_steps"):
        compare(payload, other_variant)
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
            for action_steps in EVAL_ACTION_STEPS:
                payload = {
                    "per_episode": [
                        {"outcome": outcome, "video_path": str(video)}
                        for outcome in outcomes
                    ]
                }
                path = results / f"task_{task_id}" / f"k_{budget}" / f"n_{action_steps}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload))
    items = first_outcome_gifs(results, gifs)
    by_key = {
        (item["budget"], item["action_steps"], item["outcome"]): item for item in items
    }
    assert by_key[(1, 50, "failure")]["task_id"] == 0
    assert by_key[(1, 50, "failure")]["episode_index"] == 0
    assert by_key[(1, 25, "success")]["task_id"] == 2
    assert len(items) == 6 * len(EVAL_ACTION_STEPS)
    assert all(Path(item["gif"]).is_file() for item in items)
