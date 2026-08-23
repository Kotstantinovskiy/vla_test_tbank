from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pretrain_smolvla_state_noise_k1_1k.aggregate import wilson_interval
from pretrain_smolvla_state_noise_k1_1k.constants import (
    ALPHAS,
    DEMO_BUDGET,
    DEMO_BUDGETS,
    EVAL_ACTION_STEPS,
    EVAL_BATCH_SIZE,
    EVAL_EPISODES,
    FREEZE_VISION_ENCODER,
    MASTER_SEED,
    NOISE_STREAM_SEED,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TRAIN_BATCH_SIZE,
    TRAIN_EXPERT_ONLY,
    TRAIN_STEPS,
    TRAINED_ACTION_STEPS,
    alpha_tag,
    noise_seed,
    result_path,
)
from pretrain_smolvla_state_noise_k1_1k.dataset_smoke import (
    verify_loaded_episode_indices,
)
from pretrain_smolvla_state_noise_k1_1k.determinism import compare
from pretrain_smolvla_state_noise_k1_1k.selection import build_manifest
from pretrain_smolvla_state_noise_k1_1k.training import build_cli_args, install_state_noise


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


def test_axis_is_alpha_at_k1():
    assert DEMO_BUDGET == 1 and DEMO_BUDGETS == (1,)
    assert ALPHAS == (0.0, 0.08, 0.1, 0.2)
    assert EVAL_ACTION_STEPS == (50,)
    assert EVAL_ACTION_STEPS[0] == TRAINED_ACTION_STEPS
    assert alpha_tag(0.0) == "0.00" and alpha_tag(0.1) == "0.10" and alpha_tag(0.2) == "0.20"
    with pytest.raises(ValueError):
        alpha_tag(0.05)
    assert result_path(Path("/r"), 2, 0.08, 50) == Path("/r/task_2/alpha_0.08/n_50.json")


def test_training_recipe_is_full_ft_with_same_seeds():
    assert FREEZE_VISION_ENCODER is False
    assert TRAIN_EXPERT_ONLY is False
    assert MASTER_SEED == 1_000
    assert TRAIN_STEPS == 1_000
    assert EVAL_EPISODES == 20
    assert EVAL_BATCH_SIZE == 1
    assert [noise_seed(index) for index in range(EVAL_EPISODES)] == list(
        range(1_000, 1_020)
    )


def test_selection_is_official_first_demo():
    manifest = build_manifest(_conversion_manifest())
    assert manifest["tasks"]["0"]["episodes"]["1"] == [0]
    assert manifest["tasks"]["1"]["episodes"]["1"] == [50]
    assert manifest["tasks"]["2"]["official_demos"]["1"] == ["demo_0"]


def test_cli_args_full_ft_and_alpha_axis(tmp_path: Path):
    manifest = build_manifest(_conversion_manifest())
    root = tmp_path
    (root / "artifacts").mkdir()
    (root / "artifacts/episode_manifest.json").write_text(json.dumps(manifest))
    runtime = root / "artifacts/runtime_base_checkpoint"
    runtime.mkdir()
    (runtime / "model.safetensors").write_bytes(b"test")
    args = build_cli_args(root, 1, 0.08)
    assert "--policy.train_expert_only=false" in args
    assert "--policy.freeze_vision_encoder=false" in args
    assert "--policy.use_amp=false" in args
    assert f"--seed={MASTER_SEED}" in args
    assert f"--steps={TRAIN_STEPS}" in args
    assert f"--batch_size={TRAIN_BATCH_SIZE}" in args
    assert "--dataset.episodes=[50]" in args
    assert any("alpha_0.08" in item for item in args)
    assert not any(item.startswith("--peft") for item in args)
    with pytest.raises(ValueError, match="alpha"):
        build_cli_args(root, 0, 0.05)


class _RecordingPolicy:
    def __init__(self):
        self.seen = []

    def forward(self, batch, *args, **kwargs):
        self.seen.append(batch["observation.state"])
        return torch.tensor(0.0), None


def test_install_state_noise_perturbs_only_state_deterministically():
    torch.manual_seed(123)
    state = torch.randn(4, 8)
    image = torch.randn(4, 3, 8, 8)
    batch = {"observation.state": state, "observation.images.top": image, "action": state.clone()}

    policy_a = install_state_noise(_RecordingPolicy(), 0.05)
    policy_a.forward(dict(batch))
    policy_b = install_state_noise(_RecordingPolicy(), 0.05)
    policy_b.forward(dict(batch))
    noisy_a = policy_a.seen[0]
    noisy_b = policy_b.seen[0]
    # deterministic dedicated stream: two fresh wrappers produce identical noise
    assert torch.equal(noisy_a, noisy_b)
    assert not torch.equal(noisy_a, state)
    delta = noisy_a - state
    assert delta.abs().max() < 0.05 * 6  # ~N(0, 0.05^2), 6 sigma bound
    assert delta.std() == pytest.approx(0.05, rel=0.3)
    # the original batch tensors are untouched
    assert torch.equal(batch["observation.state"], state)
    assert torch.equal(batch["action"], state)

    # successive forwards draw fresh noise from the same stream
    policy_a.forward(dict(batch))
    assert not torch.equal(policy_a.seen[0], policy_a.seen[1])

    # the wrapper seed does not touch the global torch RNG
    torch.manual_seed(NOISE_STREAM_SEED)
    reference = torch.randn(state.shape)
    assert not torch.equal(delta / 0.05, reference)


def test_env_task_ids_are_identity():
    assert TARGET_ENV_TASK_IDS == {0: 0, 1: 1, 2: 2}


def test_dataset_smoke_rejects_wrong_loaded_episode():
    verify_loaded_episode_indices([5], [5, 5])
    with pytest.raises(RuntimeError, match="expected exactly"):
        verify_loaded_episode_indices([5], [0])


def test_determinism_comparison_uses_alpha_identity():
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
        "state_noise_alpha": 0.0,
        "n_action_steps": 50,
        "model_safetensors_sha256": "abc",
        "successes": 10,
        "per_episode": episodes,
    }
    assert compare(payload, json.loads(json.dumps(payload)))["passed"]
    other = json.loads(json.dumps(payload))
    other["state_noise_alpha"] = 0.03
    with pytest.raises(ValueError, match="state_noise_alpha"):
        compare(payload, other)
    changed = json.loads(json.dumps(payload))
    changed["per_episode"][3]["sum_reward"] = 1.0
    assert compare(payload, changed)["mismatched_episode_indices"] == [3]


def test_wilson_interval_bounds():
    low, high = wilson_interval(20, 20)
    assert high == 1.0 and low == pytest.approx(0.8389, abs=1e-3)
