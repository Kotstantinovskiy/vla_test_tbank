from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import FeatureType
from lerobot.envs.configs import LiberoEnv as LiberoConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.random_utils import set_seed

from .constants import (
    EVAL_BATCH_SIZE,
    MASTER_SEED,
    NONSENSE_PROMPT,
    N_EVAL_EPISODES,
    SEEN_REPO,
    SEEN_REVISION,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
)


class _PromptOverrideVectorEnv:
    """Delegate an env while overriding only the language seen by the policy.

    LeRobot's evaluator obtains language through ``env.call('task_description')``.
    Intercepting that call keeps task dynamics and initial states unchanged and
    also works with LeRobot's lazy vector-env, which intentionally has no
    ``set_attr`` method.
    """

    def __init__(self, env: Any, prompt: str):
        self._env = env
        self._prompt = prompt

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "task_description":
            return tuple(self._prompt for _ in range(self._env.num_envs))
        return self._env.call(name, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def prompt_for(condition: str, task_id: int) -> str:
    if condition == "true":
        return TARGET_INSTRUCTIONS[task_id]
    if condition == "wrong":
        return TARGET_INSTRUCTIONS[(task_id + 1) % len(TARGET_INSTRUCTIONS)]
    if condition == "nonsense":
        return NONSENSE_PROMPT
    raise ValueError(f"Unknown condition: {condition}")


def assert_environment_instruction(
    env: Any, logical_task_id: int, env_task_id: int
) -> str:
    """Fail before rollout if the suite task does not match our logical label."""
    descriptions = tuple(env.call("task_description"))
    expected = TARGET_INSTRUCTIONS[logical_task_id]
    if len(descriptions) != env.num_envs or any(
        description != expected for description in descriptions
    ):
        raise RuntimeError(
            "LIBERO task mapping mismatch: "
            f"logical_task_id={logical_task_id}, env_task_id={env_task_id}, "
            f"expected={expected!r}, actual={descriptions!r}"
        )
    return descriptions[0]


def _camera_mapping(cfg: SmolVLAConfig) -> tuple[dict[str, str], int, int]:
    visual = {
        key: feature
        for key, feature in cfg.input_features.items()
        if feature.type is FeatureType.VISUAL
    }
    names = {key.rsplit(".", 1)[-1] for key in visual}
    if names == {"top", "wrist_image"}:
        mapping = {
            "agentview_image": "top",
            "robot0_eye_in_hand_image": "wrist_image",
        }
    elif names == {"image", "image2"}:
        mapping = {
            "agentview_image": "image",
            "robot0_eye_in_hand_image": "image2",
        }
    else:
        raise ValueError(f"Unsupported checkpoint camera features: {sorted(visual)}")
    shape = next(iter(visual.values())).shape
    return mapping, int(shape[-2]), int(shape[-1])


def load_policy_and_processors(
    model: str, revision: str | None, device: str
) -> tuple[Any, Any, Any, SmolVLAConfig]:
    cfg = SmolVLAConfig.from_pretrained(model, revision=revision)
    cfg.pretrained_path = model
    cfg.pretrained_revision = revision
    cfg.device = device
    policy = make_policy(
        cfg=cfg,
        env_cfg=LiberoConfig(task=TARGET_SUITE, task_ids=[0]),
        rename_map={"schema": "resolved below"},
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=model,
        pretrained_revision=revision,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, cfg


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    policy, preprocessor, postprocessor, policy_cfg = load_policy_and_processors(
        args.model, args.revision, args.device
    )
    camera_mapping, height, width = _camera_mapping(policy_cfg)
    results: dict[str, Any] = {
        "model": args.model,
        "revision": args.revision,
        "condition": args.condition,
        "suite": TARGET_SUITE,
        "seed": args.seed,
        "n_episodes": args.n_episodes,
        "batch_size": args.batch_size,
        "tasks": {},
    }

    device = torch.device(args.device)
    amp = (
        torch.autocast(device_type=device.type)
        if policy_cfg.use_amp
        else nullcontext()
    )
    with torch.no_grad(), amp:
        for logical_task_id in args.task_ids:
            env_task_id = TARGET_ENV_TASK_IDS[logical_task_id]
            env_cfg = LiberoConfig(
                task=TARGET_SUITE,
                task_ids=[env_task_id],
                observation_height=height,
                observation_width=width,
                camera_name_mapping=camera_mapping,
                max_parallel_tasks=1,
            )
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=env_cfg, policy_cfg=policy_cfg
            )
            base_env = env_cfg.create_envs(
                n_envs=args.batch_size, use_async_envs=args.batch_size > 1
            )[TARGET_SUITE][env_task_id]
            try:
                environment_instruction = assert_environment_instruction(
                    base_env, logical_task_id, env_task_id
                )
                policy_prompt = prompt_for(args.condition, logical_task_id)
                env = _PromptOverrideVectorEnv(base_env, policy_prompt)
                videos_dir = None
                max_rendered = 0
                if args.videos > 0:
                    videos_dir = args.output.parent / "videos"
                    if args.video_tag:
                        videos_dir /= args.video_tag
                    videos_dir = (
                        videos_dir / args.condition / f"task_{logical_task_id}"
                    )
                    max_rendered = args.videos
                info = eval_policy(
                    env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=args.n_episodes,
                    max_episodes_rendered=max_rendered,
                    videos_dir=videos_dir,
                    return_episode_data=False,
                    start_seed=args.seed,
                )
            finally:
                base_env.close()
            results["tasks"][str(logical_task_id)] = {
                "logical_task_id": logical_task_id,
                "env_task_id": env_task_id,
                "environment_instruction": environment_instruction,
                "policy_prompt": policy_prompt,
                **info,
            }

    successes = [
        episode["success"]
        for task in results["tasks"].values()
        for episode in task["per_episode"]
    ]
    results["mean_success"] = sum(successes) / len(successes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=SEEN_REPO)
    parser.add_argument("--revision", default=SEEN_REVISION)
    parser.add_argument("--condition", choices=("true", "wrong", "nonsense"), default="true")
    parser.add_argument("--task-ids", type=int, nargs="+", default=sorted(TARGET_INSTRUCTIONS))
    parser.add_argument("--n-episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--videos", type=int, default=0)
    parser.add_argument(
        "--video-tag",
        help="Optional subdirectory (for example k_5) that prevents rollout videos from being overwritten",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.revision is not None and args.revision.lower() in {"", "none", "null"}:
        args.revision = None
    logging.basicConfig(level=logging.INFO)
    result = run(args)
    print(json.dumps({"output": str(args.output), "mean_success": result["mean_success"]}))


if __name__ == "__main__":
    main()
