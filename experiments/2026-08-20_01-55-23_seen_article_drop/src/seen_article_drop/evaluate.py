from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

from .constants import (
    CHECKPOINT_PATH,
    EVAL_BATCH_SIZE,
    EXPERIMENT_NAME,
    MASTER_SEED,
    N_EVAL_EPISODES,
    SUITE,
    experiment_root,
    noise_seed,
)
from .prepare import ensure_libero_config

os.environ.setdefault(
    "LIBERO_CONFIG_PATH", str(experiment_root() / "artifacts/libero_config")
)
ensure_libero_config()

import torch
from lerobot.configs import FeatureType
from lerobot.envs.configs import LiberoEnv as LiberoConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import _get_suite
from lerobot.envs.utils import _LazyAsyncVectorEnv, parse_camera_names
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.random_utils import set_seed

from .predicate_env import PredicateLiberoEnv


class _PromptOverrideVectorEnv:
    def __init__(self, env: Any, prompt: str):
        self._env = env
        self._prompt = prompt

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "task_description":
            return tuple(self._prompt for _ in range(self._env.num_envs))
        return self._env.call(name, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


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


def load_policy(device: str):
    cfg = SmolVLAConfig.from_pretrained(CHECKPOINT_PATH)
    cfg.pretrained_path = str(CHECKPOINT_PATH)
    cfg.pretrained_revision = None
    cfg.device = device
    policy = make_policy(
        cfg=cfg,
        env_cfg=LiberoConfig(task=SUITE, task_ids=[0]),
        rename_map={"schema": "resolved below"},
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(CHECKPOINT_PATH),
        pretrained_revision=None,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, cfg


def make_predicate_vec_env(env_cfg: LiberoConfig, point: dict) -> _LazyAsyncVectorEnv:
    if EVAL_BATCH_SIZE != 1:
        raise ValueError("Deterministic per-episode noise protocol requires batch=1")
    suite = _get_suite(SUITE)
    camera_names = parse_camera_names(env_cfg.camera_name)
    gym_kwargs = dict(env_cfg.gym_kwargs)
    gym_kwargs.pop("task_ids", None)
    fns = [
        partial(
            PredicateLiberoEnv,
            task_suite=suite,
            task_id=point["env_task_id"],
            task_suite_name=SUITE,
            camera_name=camera_names,
            init_states=env_cfg.init_states,
            episode_length=env_cfg.episode_length,
            episode_index=0,
            n_envs=1,
            control_mode=env_cfg.control_mode,
            camera_name_mapping=env_cfg.camera_name_mapping,
            is_libero_plus=env_cfg.is_libero_plus,
            prompted_goal_states=point["prompted_goal_states"],
            expect_env_equivalent=point["expect_env_equivalent"],
            **gym_kwargs,
        )
    ]
    return _LazyAsyncVectorEnv(fns)


def reseed_policy_noise(seed: int) -> None:
    """Make flow-sampling noise a deterministic function of the episode."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_point(
    point: dict,
    policy_bundle,
    root: Path,
    out_dir: Path,
    checkpoint_sha: str,
) -> dict:
    policy, preprocessor, postprocessor, cfg = policy_bundle
    mapping, height, width = camera_mapping(cfg)
    env_cfg = LiberoConfig(
        task=SUITE,
        task_ids=[point["env_task_id"]],
        observation_height=height,
        observation_width=width,
        camera_name_mapping=mapping,
        max_parallel_tasks=1,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=cfg
    )
    base_env = make_predicate_vec_env(env_cfg, point)
    videos_dir = out_dir / "videos" / point["label"]
    videos_dir.mkdir(parents=True, exist_ok=True)
    per_episode: list[dict[str, Any]] = []
    video_paths: list[Path] = []
    try:
        descriptions = tuple(base_env.call("task_description"))
        if any(item != point["env_instruction"] for item in descriptions):
            raise RuntimeError(
                f"Env language mismatch for {point['label']}: expected "
                f"{point['env_instruction']!r}, got {descriptions!r}"
            )
        env = _PromptOverrideVectorEnv(base_env, point["prompt"])
        device = torch.device(cfg.device)
        amp = torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext()
        with torch.no_grad(), amp:
            for episode_index in range(N_EVAL_EPISODES):
                env_seed = MASTER_SEED + episode_index
                policy_noise_seed = noise_seed(episode_index)
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
                raw_videos = [Path(path) for path in info.get("video_paths", [])]
                if len(raw_videos) != 1 or not raw_videos[0].is_file():
                    raise RuntimeError(
                        f"{point['label']} episode {episode_index}: expected one video, "
                        f"got {raw_videos}"
                    )
                destination = videos_dir / f"eval_episode_{episode_index}.mp4"
                destination.unlink(missing_ok=True)
                shutil.move(str(raw_videos[0]), destination)
                shutil.rmtree(scratch, ignore_errors=True)
                episode = dict(info["per_episode"][0])
                if int(episode["seed"]) != env_seed:
                    raise RuntimeError(
                        f"Episode seed mismatch: expected {env_seed}, got {episode['seed']}"
                    )
                episode.update(
                    {
                        "episode_ix": episode_index,
                        "env_seed": env_seed,
                        "noise_seed": policy_noise_seed,
                        "video_path": str(destination),
                    }
                )
                per_episode.append(episode)
                video_paths.append(destination)
        raw_records = [
            record
            for sub_env_records in base_env.call("get_predicate_records")
            for record in sub_env_records
        ]
    finally:
        base_env.close()

    expected_seeds = {MASTER_SEED + index for index in range(N_EVAL_EPISODES)}
    records = [record for record in raw_records if record.get("seed") in expected_seeds]
    records_by_seed = {record["seed"]: record for record in records}
    if set(records_by_seed) != expected_seeds or len(records_by_seed) != len(records):
        raise RuntimeError(
            f"{point['label']}: predicate records do not match expected episode seeds"
        )
    consistency_violations = 0
    for episode in per_episode:
        record = records_by_seed[episode["env_seed"]]
        episode.update(
            {
                "env_task_success": record["env_task_success"],
                "env_first_step": record["env_first_step"],
                "prompted_success": record["prompted_success"],
                "prompted_first_step": record["prompted_first_step"],
                "prompted_satisfied_at_reset": record["prompted_satisfied_at_reset"],
                "steps": record["steps"],
            }
        )
        if record["prompted_satisfied_at_reset"]:
            raise RuntimeError(
                f"{point['label']}: prompted predicate already satisfied at "
                f"reset (env seed {episode['env_seed']}) — the point would be "
                "trivially successful; fix the plan instead of running it"
            )
        consistency_violations += record["consistency_violations"]
        if point["prompted_goal_states"] is not None and bool(
            episode["success"]
        ) != bool(record["prompted_success"]):
            raise RuntimeError(
                f"{point['label']}: eval success disagrees with prompted predicate"
            )
        episode["outcome"] = "success" if episode["success"] else "failure"
    if point["expect_env_equivalent"] and consistency_violations:
        raise RuntimeError(
            f"{point['label']}: {consistency_violations} predicate consistency violations"
        )

    outcomes = [bool(episode["success"]) for episode in per_episode]
    env_outcomes = [bool(episode["env_task_success"]) for episode in per_episode]
    result = {
        "experiment": EXPERIMENT_NAME,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_sha256": checkpoint_sha,
        **point,
        "success_metric": (
            "prompted_predicate"
            if point["prompted_goal_states"] is not None
            else "env_task"
        ),
        "master_seed": MASTER_SEED,
        "noise_seeding": "batch=1; torch seed = master_seed + episode_index",
        "n_episodes": N_EVAL_EPISODES,
        "successes": sum(outcomes),
        "success_rate": sum(outcomes) / len(outcomes),
        "env_task_successes": sum(env_outcomes),
        "env_task_success_rate": sum(env_outcomes) / len(env_outcomes),
        "consistency_violations": consistency_violations,
        "per_episode": per_episode,
        "video_paths": [str(path) for path in video_paths],
    }
    output = out_dir / f"{point['label']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def result_complete(out_dir: Path, label: str) -> bool:
    path = out_dir / f"{label}.json"
    if not path.is_file():
        return False
    result = json.loads(path.read_text())
    episodes = result.get("per_episode", [])
    return len(episodes) == N_EVAL_EPISODES and all(
        Path(episode.get("video_path", "")).is_file() for episode in episodes
    )


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Evaluate frozen prompt plan")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=root / "results/raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    ensure_libero_config()
    plan = json.loads((root / "artifacts/eval_plan.json").read_text())
    checkpoint_sha = json.loads(
        (root / "artifacts/checkpoint_manifest.json").read_text()
    )["model_safetensors_sha256"]
    all_points = plan["points"]
    if args.labels:
        wanted = set(args.labels)
        points = [point for point in all_points if point["label"] in wanted]
        missing = wanted - {point["label"] for point in points}
        if missing:
            raise ValueError(f"Unknown labels: {sorted(missing)}")
    else:
        points = all_points

    set_seed(MASTER_SEED)
    bundle = load_policy(args.device)
    summary = {}
    for point in points:
        if not args.force and result_complete(args.out_dir, point["label"]):
            summary[point["label"]] = "already_complete"
            continue
        result = run_point(point, bundle, root, args.out_dir, checkpoint_sha)
        summary[point["label"]] = result["success_rate"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
