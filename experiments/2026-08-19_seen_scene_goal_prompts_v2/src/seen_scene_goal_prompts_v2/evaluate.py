from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

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

from .constants import (
    CHECKPOINT_PATH,
    EVAL_BATCH_SIZE,
    MASTER_SEED,
    N_EVAL_EPISODES,
    SUITE,
    experiment_root,
)
from .predicate_env import PredicateLiberoEnv
from .prepare import ensure_libero_config


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
    """Mirror lerobot's create_libero_envs for a single task, swapping in
    PredicateLiberoEnv.  Construction matches _make_env_fns +
    _LazyAsyncVectorEnv (forkserver, shared memory, NEXT_STEP autoreset), so
    seeding, init-state striding and stepping replicate v1 exactly."""

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
            episode_index=episode_index,
            n_envs=EVAL_BATCH_SIZE,
            control_mode=env_cfg.control_mode,
            camera_name_mapping=env_cfg.camera_name_mapping,
            is_libero_plus=env_cfg.is_libero_plus,
            prompted_goal_states=point["prompted_goal_states"],
            expect_env_equivalent=point["expect_env_equivalent"],
            **gym_kwargs,
        )
        for episode_index in range(EVAL_BATCH_SIZE)
    ]
    return _LazyAsyncVectorEnv(fns)


def run_point(point: dict, policy_bundle, root: Path, checkpoint_sha: str) -> dict:
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
    try:
        descriptions = tuple(base_env.call("task_description"))
        if any(item != point["env_instruction"] for item in descriptions):
            raise RuntimeError(
                f"Env language mismatch for {point['label']}: expected "
                f"{point['env_instruction']!r}, got {descriptions!r}"
            )
        env = _PromptOverrideVectorEnv(base_env, point["prompt"])
        videos_dir = root / "results/raw/videos" / point["label"]
        device = torch.device(cfg.device)
        amp = torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext()
        with torch.no_grad(), amp:
            info = eval_policy(
                env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=N_EVAL_EPISODES,
                max_episodes_rendered=N_EVAL_EPISODES,
                videos_dir=videos_dir,
                return_episode_data=False,
                start_seed=MASTER_SEED,
            )
        records = [
            record
            for sub_env_records in base_env.call("get_predicate_records")
            for record in sub_env_records
        ]
    finally:
        base_env.close()

    video_paths = [Path(path) for path in info.get("video_paths", [])]
    if len(video_paths) != N_EVAL_EPISODES or any(
        not path.is_file() for path in video_paths
    ):
        raise RuntimeError(
            f"{point['label']}: expected {N_EVAL_EPISODES} videos, got {len(video_paths)}"
        )

    # Join predicate records to eval_policy's per_episode list by episode seed
    # (unique: start_seed + episode index).
    records_by_seed = {record["seed"]: record for record in records}
    if len(records_by_seed) != len(records):
        raise RuntimeError(f"{point['label']}: duplicate episode seeds in records")
    per_episode = info["per_episode"]
    if len(per_episode) != N_EVAL_EPISODES:
        raise RuntimeError(
            f"{point['label']}: expected {N_EVAL_EPISODES} episodes, got {len(per_episode)}"
        )
    consistency_violations = 0
    for episode in per_episode:
        record = records_by_seed.get(episode["seed"])
        if record is None:
            raise RuntimeError(
                f"{point['label']}: no predicate record for seed {episode['seed']}"
            )
        episode["env_task_success"] = record["env_task_success"]
        episode["env_first_step"] = record["env_first_step"]
        episode["prompted_success"] = record["prompted_success"]
        episode["prompted_first_step"] = record["prompted_first_step"]
        episode["steps"] = record["steps"]
        consistency_violations += record["consistency_violations"]
        if point["prompted_goal_states"] is not None and bool(
            episode["success"]
        ) != bool(record["prompted_success"]):
            raise RuntimeError(
                f"{point['label']}: eval_policy success disagrees with the "
                f"predicate record on seed {episode['seed']}"
            )
    if point["expect_env_equivalent"] and consistency_violations:
        raise RuntimeError(
            f"{point['label']}: {consistency_violations} step(s) where the "
            "external predicate evaluation disagreed with env.check_success()"
        )

    outcomes = [bool(episode["success"]) for episode in per_episode]
    env_outcomes = [bool(episode["env_task_success"]) for episode in per_episode]
    result = {
        "experiment": "seen_scene_goal_prompts_v2",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_sha256": checkpoint_sha,
        **point,
        "success_metric": (
            "prompted_predicate"
            if point["prompted_goal_states"] is not None
            else "env_task"
        ),
        "seed": MASTER_SEED,
        "n_episodes": N_EVAL_EPISODES,
        "successes": sum(outcomes),
        "success_rate": sum(outcomes) / len(outcomes),
        "env_task_successes": sum(env_outcomes),
        "env_task_success_rate": sum(env_outcomes) / len(env_outcomes),
        "consistency_violations": consistency_violations,
        **info,
    }
    for episode, video_path in zip(result["per_episode"], video_paths, strict=True):
        episode["video_path"] = str(video_path)
        episode["outcome"] = "success" if episode["success"] else "failure"
    output = root / "results/raw" / f"{point['label']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def result_complete(root: Path, label: str) -> bool:
    path = root / "results/raw" / f"{label}.json"
    if not path.is_file():
        return False
    result = json.loads(path.read_text())
    episodes = result.get("per_episode", [])
    return len(episodes) == N_EVAL_EPISODES and all(
        Path(episode.get("video_path", "")).is_file() for episode in episodes
    )


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Evaluate v2 plan points")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    ensure_libero_config()
    plan = json.loads((root / "artifacts/eval_plan.json").read_text())
    checkpoint_sha = json.loads(
        (root / "artifacts/checkpoint_manifest.json").read_text()
    )["model_safetensors_sha256"]
    points = plan["points"]
    if args.labels:
        wanted = set(args.labels)
        points = [point for point in points if point["label"] in wanted]
        missing = wanted - {point["label"] for point in points}
        if missing:
            raise ValueError(f"Unknown labels: {sorted(missing)}")

    set_seed(MASTER_SEED)
    bundle = load_policy(args.device)
    summary = {}
    for point in points:
        if result_complete(root, point["label"]):
            summary[point["label"]] = "already_complete"
            continue
        result = run_point(point, bundle, root, checkpoint_sha)
        summary[point["label"]] = result["success_rate"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
