from __future__ import annotations

"""Positive control: evaluate the frozen checkpoint on a SEEN libero_90 task.

The earlier zero-shot floors (0/20 on held-out tasks) were uninterpretable
because no run ever demonstrated that the frozen-checkpoint evaluation
pipeline — LIBERO config, camera mapping, orientation, normalization — can
produce successes at all.  This entrypoint closes that gap: it rolls out the
final mirror-corrected checkpoint on a task the model was trained on, with
production episode count and media settings.  A near-baseline success rate
validates the pipeline; only then does a zero-shot floor on held-out tasks
carry information.

Run it AFTER training completes:

    scripts/eval_seen_control.sh [GPU]
"""

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
    SEED,
    SEEN_CONTROL_BATCH_SIZE,
    SEEN_CONTROL_ENV_TASK_ID,
    SEEN_CONTROL_EPISODES,
    SEEN_CONTROL_SUITE,
    experiment_root,
)


def ensure_libero_config() -> Path:
    """Idempotently write the experiment-local LIBERO config before env use."""

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
        raise ValueError(
            f"Expected the top/wrist_image checkpoint schema, got {sorted(names)}"
        )
    shape = next(iter(visual.values())).shape
    # This assignment (env agentview -> dataset `top`, env eye-in-hand ->
    # dataset `wrist_image`) is validated against official HDF5 frames by
    # verify_conversion.py; see artifacts/conversion_verification_libero_90.json.
    return {
        "agentview_image": "top",
        "robot0_eye_in_hand_image": "wrist_image",
    }, int(shape[-2]), int(shape[-1])


def seen_task_instructions() -> set[str]:
    import pandas as pd

    data_root = Path(
        os.environ.get(
            "VLA_OFFICIAL_SEEN_DATA_ROOT",
            "/var/tmp/vla_libero_official_rot180/libero_90",
        )
    )
    table = pd.read_parquet(data_root / "meta/tasks.parquet")
    return set(table.index.astype(str))


def default_model() -> Path:
    output_root = Path(
        os.environ.get(
            "VLA_OFFICIAL_OUTPUT_ROOT",
            "/var/tmp/vla_outputs/seen_libero90_official_20260817",
        )
    )
    return output_root / "checkpoints/last/pretrained_model"


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_libero_config()
    if not (args.model / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"No trained checkpoint at {args.model}; run training first"
        )
    set_seed(args.seed)
    cfg = SmolVLAConfig.from_pretrained(args.model)
    cfg.pretrained_path = str(args.model)
    cfg.pretrained_revision = None
    cfg.device = args.device
    policy = make_policy(
        cfg=cfg,
        env_cfg=LiberoConfig(task=SEEN_CONTROL_SUITE, task_ids=[0]),
        rename_map={"schema": "resolved below"},
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(args.model),
        pretrained_revision=None,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    mapping, height, width = camera_mapping(cfg)
    env_cfg = LiberoConfig(
        task=SEEN_CONTROL_SUITE,
        task_ids=[args.env_task_id],
        observation_height=height,
        observation_width=width,
        camera_name_mapping=mapping,
        max_parallel_tasks=1,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=cfg
    )
    env = env_cfg.create_envs(
        n_envs=args.batch_size, use_async_envs=args.batch_size > 1
    )[SEEN_CONTROL_SUITE][args.env_task_id]
    try:
        descriptions = tuple(env.call("task_description"))
        if len(set(descriptions)) != 1:
            raise RuntimeError(f"Inconsistent task descriptions: {descriptions}")
        instruction = descriptions[0]
        seen = seen_task_instructions()
        if instruction not in seen:
            raise RuntimeError(
                "Positive control requires a SEEN task; env task "
                f"{args.env_task_id} ({instruction!r}) is not among the 90 "
                "training task instructions"
            )
        videos_dir = args.output.parent / "videos" / f"seen_task_{args.env_task_id}"
        context = (
            torch.autocast(device_type=torch.device(args.device).type)
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
                n_episodes=args.n_episodes,
                max_episodes_rendered=args.n_episodes,
                videos_dir=videos_dir,
                return_episode_data=False,
                start_seed=args.seed,
            )
    finally:
        env.close()
    outcomes = [bool(episode["success"]) for episode in info["per_episode"]]
    video_paths = [Path(path) for path in info.get("video_paths", [])]
    if len(video_paths) != args.n_episodes or any(
        not path.is_file() for path in video_paths
    ):
        raise RuntimeError(
            f"Expected {args.n_episodes} saved rollout videos, got {len(video_paths)}"
        )
    result = {
        "experiment": "smolvla_pretrain_libero",
        "purpose": "positive control on a seen training task",
        "model": str(args.model),
        "suite": SEEN_CONTROL_SUITE,
        "env_task_id": args.env_task_id,
        "instruction": instruction,
        "seed": args.seed,
        "n_episodes": args.n_episodes,
        "batch_size": args.batch_size,
        "successes": sum(outcomes),
        "success_rate": sum(outcomes) / len(outcomes),
        **info,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=default_model())
    parser.add_argument("--env-task-id", type=int, default=SEEN_CONTROL_ENV_TASK_ID)
    parser.add_argument("--n-episodes", type=int, default=SEEN_CONTROL_EPISODES)
    parser.add_argument("--batch-size", type=int, default=SEEN_CONTROL_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results/raw/seen_control.json",
    )
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "instruction": result["instruction"],
                "success_rate": result["success_rate"],
            }
        )
    )


if __name__ == "__main__":
    main()
