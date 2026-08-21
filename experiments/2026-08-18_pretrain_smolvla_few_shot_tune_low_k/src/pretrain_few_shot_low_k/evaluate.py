from __future__ import annotations

import argparse
import importlib.util
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml
from lerobot.configs import FeatureType
from lerobot.envs.configs import LiberoEnv as LiberoConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.random_utils import set_seed

from .constants import (
    DEMO_BUDGETS,
    EVAL_BATCH_SIZE,
    EVAL_EPISODES,
    MASTER_SEED,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
    experiment_root,
)
from .training import final_model


def ensure_libero_config() -> Path:
    config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("LIBERO is not installed in the active environment")
    package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    benchmark_root = package_root / "libero"
    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(package_root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    required = ("benchmark_root", "bddl_files", "init_states")
    missing = [name for name in required if not Path(paths[name]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing LIBERO resource directories: {missing}")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(paths, sort_keys=False))
    return config_path


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


def run(task_id: int, budget: int, model: Path, output: Path, device: str) -> dict[str, Any]:
    if task_id not in TARGET_INSTRUCTIONS or budget not in DEMO_BUDGETS:
        raise ValueError("Invalid task/budget")
    ensure_libero_config()
    set_seed(MASTER_SEED)
    cfg = SmolVLAConfig.from_pretrained(model)
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
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=cfg
    )
    env = env_cfg.create_envs(n_envs=EVAL_BATCH_SIZE, use_async_envs=True)[
        TARGET_SUITE
    ][env_task_id]
    expected = TARGET_INSTRUCTIONS[task_id]
    try:
        descriptions = tuple(env.call("task_description"))
        if len(descriptions) != EVAL_BATCH_SIZE or any(
            item != expected for item in descriptions
        ):
            raise RuntimeError(
                f"Environment task mismatch: expected={expected!r}, actual={descriptions!r}"
            )
        videos_dir = output.parent.parent / "videos" / f"task_{task_id}" / f"k_{budget}"
        context = (
            torch.autocast(device_type=torch.device(device).type)
            if cfg.use_amp
            else nullcontext()
        )
        with torch.no_grad(), context:
            info = eval_policy(
                env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=EVAL_EPISODES,
                max_episodes_rendered=EVAL_EPISODES,
                videos_dir=videos_dir,
                return_episode_data=False,
                start_seed=MASTER_SEED,
            )
    finally:
        env.close()
    result = {
        "experiment": "pretrain_smolvla_few_shot_tune_low_k",
        "model": str(model),
        "suite": TARGET_SUITE,
        "logical_task_id": task_id,
        "env_task_id": env_task_id,
        "instruction": expected,
        "demo_budget": budget,
        "seed": MASTER_SEED,
        "n_episodes": EVAL_EPISODES,
        "batch_size": EVAL_BATCH_SIZE,
        **info,
    }
    outcomes = [bool(item["success"]) for item in info["per_episode"]]
    video_paths = [Path(path) for path in info.get("video_paths", [])]
    if len(video_paths) != EVAL_EPISODES or any(not path.is_file() for path in video_paths):
        raise RuntimeError(
            f"Expected {EVAL_EPISODES} saved rollout videos, got {len(video_paths)}"
        )
    for episode, video_path in zip(result["per_episode"], video_paths, strict=True):
        episode["video_path"] = str(video_path)
        episode["outcome"] = "success" if episode["success"] else "failure"
    result["successes"] = sum(outcomes)
    result["success_rate"] = sum(outcomes) / len(outcomes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def result_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    result = json.loads(path.read_text())
    episodes = result.get("per_episode", [])
    if len(episodes) != EVAL_EPISODES:
        return False
    return all(
        episode.get("outcome") in {"success", "failure"}
        and Path(episode.get("video_path", "")).is_file()
        for episode in episodes
    )


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Evaluate one adapted model")
    parser.add_argument("task_id", type=int)
    parser.add_argument("budget", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = final_model(args.task_id, args.budget)
    output = root / "results/raw" / f"task_{args.task_id}" / f"k_{args.budget}.json"
    if result_complete(output):
        print(json.dumps({"state": "already_complete", "output": str(output)}))
        return
    result = run(args.task_id, args.budget, model, output, args.device)
    print(json.dumps({"output": str(output), "success_rate": result["success_rate"]}))


if __name__ == "__main__":
    main()
