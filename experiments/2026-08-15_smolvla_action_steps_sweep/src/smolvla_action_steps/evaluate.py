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
    ACTION_STEPS,
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    EVAL_BATCH_SIZE,
    MASTER_SEED,
    NONSENSE_PROMPT,
    N_EVAL_EPISODES,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
)


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


def prompt_for(condition: str, task_id: int) -> str:
    if condition == "true":
        return TARGET_INSTRUCTIONS[task_id]
    if condition == "wrong":
        return TARGET_INSTRUCTIONS[(task_id + 1) % len(TARGET_INSTRUCTIONS)]
    if condition == "nonsense":
        return NONSENSE_PROMPT
    raise ValueError(f"Unknown prompt condition: {condition}")


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
    model: str, revision: str | None, device: str, initial_action_steps: int
) -> tuple[Any, Any, Any, SmolVLAConfig, int]:
    cfg = SmolVLAConfig.from_pretrained(model, revision=revision)
    original_action_steps = cfg.n_action_steps
    if max(ACTION_STEPS) > cfg.chunk_size:
        raise ValueError(
            f"Sweep exceeds checkpoint chunk_size={cfg.chunk_size}: {ACTION_STEPS}"
        )
    cfg.n_action_steps = initial_action_steps
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
    return policy, preprocessor, postprocessor, cfg, original_action_steps


def _load_existing(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.resume or not args.output.exists():
        return None
    result = json.loads(args.output.read_text())
    expected = {
        "model": args.model,
        "revision": args.revision,
        "task_id": args.task_id,
        "logical_task_id": args.task_id,
        "env_task_id": TARGET_ENV_TASK_IDS[args.task_id],
        "demo_budget": args.demo_budget,
        "condition": args.condition,
        "seed": args.seed,
        "n_episodes": args.n_episodes,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"Cannot resume {args.output}: {key}={result.get(key)!r}, expected {value!r}"
            )
    return result


def _save(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)


def run(args: argparse.Namespace) -> dict[str, Any]:
    requested_steps = list(dict.fromkeys(args.action_steps))
    invalid = [step for step in requested_steps if step not in ACTION_STEPS]
    if invalid:
        raise ValueError(f"Action steps outside the locked sweep: {invalid}")

    existing = _load_existing(args)
    completed = set(existing.get("sweep", {})) if existing else set()
    pending = [step for step in requested_steps if str(step) not in completed]
    if not pending:
        return existing

    set_seed(args.seed)
    policy, preprocessor, postprocessor, policy_cfg, original_action_steps = (
        load_policy_and_processors(
            args.model, args.revision, args.device, pending[0]
        )
    )
    camera_mapping, height, width = _camera_mapping(policy_cfg)
    result = existing or {
        "experiment": "smolvla_action_steps_sweep",
        "model": args.model,
        "revision": args.revision,
        "task_id": args.task_id,
        "logical_task_id": args.task_id,
        "env_task_id": TARGET_ENV_TASK_IDS[args.task_id],
        "demo_budget": args.demo_budget,
        "condition": args.condition,
        "environment_instruction": TARGET_INSTRUCTIONS[args.task_id],
        "policy_prompt": prompt_for(args.condition, args.task_id),
        "suite": TARGET_SUITE,
        "seed": args.seed,
        "n_episodes": args.n_episodes,
        "batch_size": args.batch_size,
        "weights_modified": False,
        "chunk_size": policy_cfg.chunk_size,
        "checkpoint_n_action_steps": original_action_steps,
        "rng_protocol": "global seed reset before every action-step point",
        "sweep": {},
    }

    env_cfg = LiberoConfig(
        task=TARGET_SUITE,
        task_ids=[TARGET_ENV_TASK_IDS[args.task_id]],
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
    )[TARGET_SUITE][TARGET_ENV_TASK_IDS[args.task_id]]

    device = torch.device(args.device)
    amp = (
        torch.autocast(device_type=device.type)
        if policy_cfg.use_amp
        else nullcontext()
    )
    try:
        result["environment_instruction"] = assert_environment_instruction(
            base_env, args.task_id, TARGET_ENV_TASK_IDS[args.task_id]
        )
        env = _PromptOverrideVectorEnv(base_env, result["policy_prompt"])
        with torch.no_grad(), amp:
            for action_steps in pending:
                set_seed(args.seed)
                policy.config.n_action_steps = action_steps
                policy.reset()
                videos_dir = None
                max_rendered = 0
                if args.videos > 0 and action_steps in args.video_action_steps:
                    videos_dir = (
                        args.output.parent
                        / "videos"
                        / args.output.stem
                        / args.condition
                        / f"n_action_steps_{action_steps}"
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
                result["sweep"][str(action_steps)] = {
                    "n_action_steps": action_steps,
                    "replan_every_environment_steps": action_steps,
                    **info,
                }
                _save(result, args.output)
    finally:
        base_env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen SmolVLA weights across action execution horizons"
    )
    parser.add_argument("--model", default=CHECKPOINT_REPO)
    parser.add_argument("--revision", default=CHECKPOINT_REVISION)
    parser.add_argument("--task-id", type=int, choices=sorted(TARGET_INSTRUCTIONS), required=True)
    parser.add_argument("--demo-budget", type=int, required=True)
    parser.add_argument("--condition", choices=("true", "wrong", "nonsense"), default="true")
    parser.add_argument("--action-steps", type=int, nargs="+", default=list(ACTION_STEPS))
    parser.add_argument("--n-episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--videos", type=int, default=1)
    parser.add_argument(
        "--video-action-steps", type=int, nargs="+", default=list(ACTION_STEPS)
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.revision is not None and args.revision.lower() in {"", "none", "null"}:
        args.revision = None
    logging.basicConfig(level=logging.INFO)
    result = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "completed_action_steps": sorted(map(int, result["sweep"])),
            }
        )
    )


if __name__ == "__main__":
    main()
