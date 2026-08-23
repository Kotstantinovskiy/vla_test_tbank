from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pretrain_smolvla_bundle_all_k.aggregate import wilson_interval
from pretrain_smolvla_bundle_all_k.constants import (
    DEMO_BUDGETS,
    EVAL_ACTION_STEPS,
    EVAL_BATCH_SIZE,
    EVAL_EPISODES,
    EVAL_SLOTS_PER_GPU,
    FREEZE_VISION_ENCODER,
    MASTER_SEED,
    NOISE_STREAM_SEED,
    STATE_NOISE_ALPHA,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TRAIN_BATCH_SIZE,
    TRAIN_EXPERT_ONLY,
    TRAIN_SLOTS_PER_GPU,
    TRAIN_STEPS_BY_BUDGET,
    TRAINED_ACTION_STEPS,
    noise_seed,
    result_path,
)
from pretrain_smolvla_bundle_all_k.dataset_smoke import verify_loaded_episode_indices
from pretrain_smolvla_bundle_all_k.determinism import compare
from pretrain_smolvla_bundle_all_k.selection import build_manifest
from pretrain_smolvla_bundle_all_k.training import build_cli_args, install_state_noise


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


def test_bundle_axes():
    assert DEMO_BUDGETS == (1, 2, 3, 5, 10, 25)
    assert EVAL_ACTION_STEPS == (50, 35, 25)
    assert EVAL_ACTION_STEPS[0] == TRAINED_ACTION_STEPS
    assert TRAIN_STEPS_BY_BUDGET == {1: 1_000, 2: 1_500, 3: 2_000, 5: 2_000, 10: 2_000, 25: 2_000}
    assert STATE_NOISE_ALPHA == 0.10
    assert TRAIN_SLOTS_PER_GPU >= 1 and EVAL_SLOTS_PER_GPU >= 1
    assert result_path(Path("/r"), 1, 25, 35) == Path("/r/task_1/k_25/n_35.json")


def test_seed_protocol_matches_siblings():
    assert MASTER_SEED == 1_000
    assert EVAL_EPISODES == 20
    assert EVAL_BATCH_SIZE == 1
    assert [noise_seed(index) for index in range(EVAL_EPISODES)] == list(
        range(1_000, 1_020)
    )


def test_selection_is_official_first_k():
    manifest = build_manifest(_conversion_manifest())
    task0 = manifest["tasks"]["0"]
    assert task0["episodes"]["1"] == [0]
    assert task0["episodes"]["25"] == list(range(25))
    assert manifest["tasks"]["1"]["episodes"]["5"] == [50, 51, 52, 53, 54]


def test_cli_args_bundle_recipe(tmp_path: Path):
    manifest = build_manifest(_conversion_manifest())
    root = tmp_path
    (root / "artifacts").mkdir()
    (root / "artifacts/episode_manifest.json").write_text(json.dumps(manifest))
    runtime = root / "artifacts/runtime_base_checkpoint"
    runtime.mkdir()
    (runtime / "model.safetensors").write_bytes(b"test")
    args = build_cli_args(root, 1, 2)
    assert "--policy.train_expert_only=false" in args
    assert "--policy.freeze_vision_encoder=false" in args
    assert "--dataset.image_transforms.enable=true" in args
    assert "--steps=1500" in args
    assert f"--seed={MASTER_SEED}" in args
    assert f"--batch_size={TRAIN_BATCH_SIZE}" in args
    assert "--dataset.episodes=[50,51]" in args
    assert "--steps=1000" in build_cli_args(root, 0, 1)
    assert "--steps=2000" in build_cli_args(root, 2, 3)
    assert "--steps=2000" in build_cli_args(root, 2, 25)
    with pytest.raises(ValueError, match="zero-shot"):
        build_cli_args(root, 0, 0)


class _RecordingPolicy:
    def __init__(self):
        self.seen = []

    def forward(self, batch, *args, **kwargs):
        self.seen.append(batch["observation.state"])
        return torch.tensor(0.0), None


def test_install_state_noise_perturbs_only_state_deterministically():
    torch.manual_seed(7)
    state = torch.randn(4, 8)
    batch = {"observation.state": state, "action": state.clone()}
    policy_a = install_state_noise(_RecordingPolicy(), STATE_NOISE_ALPHA)
    policy_a.forward(dict(batch))
    policy_b = install_state_noise(_RecordingPolicy(), STATE_NOISE_ALPHA)
    policy_b.forward(dict(batch))
    assert torch.equal(policy_a.seen[0], policy_b.seen[0])
    delta = policy_a.seen[0] - state
    assert delta.std() == pytest.approx(STATE_NOISE_ALPHA, rel=0.35)
    assert torch.equal(batch["observation.state"], state)
    assert torch.equal(batch["action"], state)
    torch.manual_seed(NOISE_STREAM_SEED)
    assert not torch.equal(delta / STATE_NOISE_ALPHA, torch.randn(state.shape))


def test_installed_lerobot_image_transform_defaults_match_protocol():
    from lerobot.transforms.transforms import ImageTransformsConfig

    from pretrain_smolvla_bundle_all_k.constants import (
        IMAGE_TRANSFORMS_EXPECTED,
        IMAGE_TRANSFORMS_MAX_NUM,
    )

    cfg = ImageTransformsConfig()
    assert cfg.enable is False
    assert cfg.max_num_transforms == IMAGE_TRANSFORMS_MAX_NUM
    assert set(cfg.tfs) == set(IMAGE_TRANSFORMS_EXPECTED)
    for name, (expected_type, expected_kwargs) in IMAGE_TRANSFORMS_EXPECTED.items():
        transform = cfg.tfs[name]
        assert transform.type == expected_type, name
        assert transform.kwargs == expected_kwargs, name


def test_env_task_ids_are_identity():
    assert TARGET_ENV_TASK_IDS == {0: 0, 1: 1, 2: 2}
    assert FREEZE_VISION_ENCODER is False and TRAIN_EXPERT_ONLY is False


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
        "n_action_steps": 35,
        "model_safetensors_sha256": "abc",
        "successes": 10,
        "per_episode": episodes,
    }
    assert compare(payload, json.loads(json.dumps(payload)))["passed"]
    other = json.loads(json.dumps(payload))
    other["n_action_steps"] = 50
    with pytest.raises(ValueError, match="n_action_steps"):
        compare(payload, other)
    changed = json.loads(json.dumps(payload))
    changed["per_episode"][3]["sum_reward"] = 1.0
    assert compare(payload, changed)["mismatched_episode_indices"] == [3]


def test_wilson_interval_bounds():
    low, high = wilson_interval(20, 20)
    assert high == 1.0 and low == pytest.approx(0.8389, abs=1e-3)
