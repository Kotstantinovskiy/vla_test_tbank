from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .constants import (
    DEMO_BUDGETS,
    EVAL_BATCH_SIZE,
    EVAL_EPISODES,
    EVAL_HORIZON,
    EXPERIMENT_NAME,
    MASTER_SEED,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
    TRAINED_ACTION_STEPS,
    TRAINED_CHUNK_SIZE,
    VLM_BACKBONE,
    experiment_root,
    noise_seed,
    result_path,
)
from .libero_setup import ensure_libero_config

# Detached entrypoints must never reach LIBERO's first-run stdin prompt.
os.environ.setdefault(
    "LIBERO_CONFIG_PATH", str(experiment_root() / "artifacts/libero_config")
)
ensure_libero_config()

import torch
from lerobot.configs import FeatureType
from lerobot.envs.configs import LiberoEnv as LiberoConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.random_utils import set_seed

from .training import adapted_manifest_path, final_model


def camera_mapping(cfg: SmolVLAConfig) -> tuple[dict[str, str], int, int]:
    visual = {
        key: feature
        for key, feature in cfg.input_features.items()
        if feature.type is FeatureType.VISUAL
    }
    names = {key.rsplit(".", 1)[-1] for key in visual}
    if names != {"top", "wrist_image"}:
        raise ValueError(f"Expected top/wrist_image schema, got {sorted(names)}")
    shape = next(iter(visual.values())).shape
    return {
        "agentview_image": "top",
        "robot0_eye_in_hand_image": "wrist_image",
    }, int(shape[-2]), int(shape[-1])


def reseed_policy_noise(seed: int) -> None:
    """Pin SmolVLA flow-sampling noise independently for one episode."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pin_libero_init_state(env: Any, episode_index: int) -> None:
    """Tie LIBERO's state-bank index to the logical evaluation episode.

    LeRobot's LIBERO wrapper advances ``init_state_id`` on every reset and does
    not derive it from the Gym seed.  Without pinning it here, evaluating the
    same logical episodes in a different order silently assigns different
    initial states to them.
    """

    env.set_attr("init_state_id", episode_index)
    actual = tuple(int(value) for value in env.get_attr("init_state_id"))
    if actual != (episode_index,):
        raise RuntimeError(
            f"Failed to pin LIBERO init_state_id={episode_index}: actual={actual}"
        )


def run(
    task_id: int,
    budget: int,
    model: Path,
    results_root: Path,
    device: str,
    episode_indices: list[int] | None = None,
) -> dict[str, Any]:
    if task_id not in TARGET_INSTRUCTIONS:
        raise ValueError(f"Unknown task_id {task_id}")
    if budget not in DEMO_BUDGETS:
        raise ValueError(f"Unsupported budget {budget}")
    if EVAL_BATCH_SIZE != 1:
        raise ValueError("Per-episode policy-noise seeding requires batch_size=1")
    order = list(range(EVAL_EPISODES)) if episode_indices is None else episode_indices
    if sorted(order) != list(range(EVAL_EPISODES)):
        raise ValueError(f"Episode order must be a permutation of 0..{EVAL_EPISODES - 1}")

    set_seed(MASTER_SEED)
    cfg = SmolVLAConfig.from_pretrained(model)
    if Path(cfg.vlm_model_name).resolve() != VLM_BACKBONE.resolve():
        raise ValueError(f"Adapted checkpoint uses unpinned VLM: {cfg.vlm_model_name}")
    if cfg.chunk_size != TRAINED_CHUNK_SIZE:
        raise ValueError(f"Expected chunk_size={TRAINED_CHUNK_SIZE}, got {cfg.chunk_size}")
    if cfg.n_action_steps != TRAINED_ACTION_STEPS:
        raise ValueError(
            f"Expected n_action_steps={TRAINED_ACTION_STEPS}, got {cfg.n_action_steps}"
        )
    cfg.pretrained_path = str(model)
    cfg.pretrained_revision = None
    cfg.device = device
    policy = make_policy(
        cfg=cfg,
        env_cfg=LiberoConfig(task=TARGET_SUITE, task_ids=[0]),
        rename_map={"schema": "resolved below"},
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(model),
        pretrained_revision=None,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    mapping, height, width = camera_mapping(cfg)
    env_task_id = TARGET_ENV_TASK_IDS[task_id]
    env_cfg = LiberoConfig(
        task=TARGET_SUITE,
        task_ids=[env_task_id],
        observation_height=height,
        observation_width=width,
        camera_name_mapping=mapping,
        max_parallel_tasks=1,
        episode_length=EVAL_HORIZON,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=cfg
    )
    env = env_cfg.create_envs(n_envs=1, use_async_envs=False)[TARGET_SUITE][env_task_id]
    expected = TARGET_INSTRUCTIONS[task_id]
    output = result_path(results_root, task_id, budget)
    videos_dir = results_root / "videos" / f"task_{task_id}" / f"k_{budget}"
    videos_dir.mkdir(parents=True, exist_ok=True)
    per_episode: list[dict[str, Any]] = []
    try:
        descriptions = tuple(env.call("task_description"))
        if descriptions != (expected,):
            raise RuntimeError(
                f"Environment task mismatch: expected={expected!r}, actual={descriptions!r}"
            )
        context = (
            torch.autocast(device_type=torch.device(device).type)
            if cfg.use_amp
            else nullcontext()
        )
        with torch.no_grad(), context:
            for episode_index in order:
                env_seed = MASTER_SEED + episode_index
                policy_noise_seed = noise_seed(episode_index)
                pin_libero_init_state(env, episode_index)
                reseed_policy_noise(policy_noise_seed)
                scratch = videos_dir / f"_episode_{episode_index}"
                info = eval_policy(
                    env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=1,
                    max_episodes_rendered=1,
                    videos_dir=scratch,
                    return_episode_data=False,
                    start_seed=env_seed,
                )
                episodes = info.get("per_episode", [])
                raw_videos = [Path(path) for path in info.get("video_paths", [])]
                if len(episodes) != 1 or len(raw_videos) != 1:
                    raise RuntimeError(
                        f"Expected one episode/video, got {len(episodes)}/{len(raw_videos)}"
                    )
                if not raw_videos[0].is_file():
                    raise FileNotFoundError(raw_videos[0])
                episode = dict(episodes[0])
                if int(episode["seed"]) != env_seed:
                    raise RuntimeError(
                        f"Episode seed mismatch: expected {env_seed}, got {episode['seed']}"
                    )
                destination = videos_dir / f"eval_episode_{episode_index}.mp4"
                destination.unlink(missing_ok=True)
                shutil.move(str(raw_videos[0]), destination)
                shutil.rmtree(scratch, ignore_errors=True)
                episode.update(
                    {
                        "episode_ix": episode_index,
                        "env_seed": env_seed,
                        "noise_seed": policy_noise_seed,
                        "init_state_id": episode_index,
                        "outcome": "success" if episode["success"] else "failure",
                        "video_path": str(destination),
                    }
                )
                per_episode.append(episode)
    finally:
        env.close()

    per_episode.sort(key=lambda item: int(item["episode_ix"]))

    manifest_file = adapted_manifest_path(experiment_root(), task_id, budget)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Adapted checkpoint manifest missing: {manifest_file}")
    checkpoint_manifest = json.loads(manifest_file.read_text())
    outcomes = [bool(item["success"]) for item in per_episode]
    result = {
        "experiment": EXPERIMENT_NAME,
        "model": str(model),
        "model_safetensors_sha256": checkpoint_manifest["model_safetensors_sha256"],
        "suite": TARGET_SUITE,
        "logical_task_id": task_id,
        "env_task_id": env_task_id,
        "instruction": expected,
        "demo_budget": budget,
        "chunk_size": cfg.chunk_size,
        "n_action_steps": cfg.n_action_steps,
        "training_seed": MASTER_SEED,
        "master_seed": MASTER_SEED,
        "noise_seeding": (
            "batch=1; torch/CUDA seed = master_seed + episode_index before "
            "each eval_policy call"
        ),
        "init_state_seeding": (
            "LIBERO init_state_id = episode_index before each eval_policy call"
        ),
        "n_episodes": EVAL_EPISODES,
        "batch_size": EVAL_BATCH_SIZE,
        "successes": sum(outcomes),
        "success_rate": sum(outcomes) / len(outcomes),
        "per_episode": per_episode,
        "video_paths": [item["video_path"] for item in per_episode],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def result_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    result = json.loads(path.read_text())
    episodes = result.get("per_episode", [])
    expected_seeds = list(range(MASTER_SEED, MASTER_SEED + EVAL_EPISODES))
    return (
        len(episodes) == EVAL_EPISODES
        and result.get("batch_size") == 1
        and bool(result.get("model_safetensors_sha256"))
        and [item.get("env_seed") for item in episodes] == expected_seeds
        and [item.get("noise_seed") for item in episodes] == expected_seeds
        and [item.get("init_state_id") for item in episodes]
        == list(range(EVAL_EPISODES))
        and all(
            item.get("outcome") in {"success", "failure"}
            and Path(item.get("video_path", "")).is_file()
            for item in episodes
        )
    )


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Evaluate one adapted model")
    parser.add_argument("task_id", type=int)
    parser.add_argument("budget", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-root", type=Path, default=root / "results/raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    model = final_model(args.task_id, args.budget)
    output = result_path(args.results_root, args.task_id, args.budget)
    if not args.force and result_complete(output):
        print(json.dumps({"state": "already_complete", "output": str(output)}))
        return
    result = run(args.task_id, args.budget, model, args.results_root, args.device)
    print(json.dumps({"output": str(output), "success_rate": result["success_rate"]}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
