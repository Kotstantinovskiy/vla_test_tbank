from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
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
    CHECKPOINT_PATH,
    CHECKPOINT_PROVENANCE,
    EVAL_BATCH_SIZE,
    MASTER_SEED,
    N_EVAL_EPISODES,
    PROMPT_CONDITIONS,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
    experiment_root,
    noise_seed,
    prompt_for,
)
from .prepare import write_libero_config


def ensure_libero_config() -> None:
    write_libero_config(Path(os.environ["LIBERO_CONFIG_PATH"]))


def all_labels() -> list[str]:
    return [
        f"{condition}__task_{task_id}"
        for condition in PROMPT_CONDITIONS
        for task_id in sorted(TARGET_INSTRUCTIONS)
    ]


def parse_label(label: str) -> tuple[str, int]:
    condition, _, task = label.partition("__task_")
    if condition not in PROMPT_CONDITIONS or not task.isdigit():
        raise ValueError(f"Bad label: {label!r}")
    task_id = int(task)
    if task_id not in TARGET_INSTRUCTIONS:
        raise ValueError(f"Unknown task id in label: {label!r}")
    return condition, task_id


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


def _camera_mapping(cfg: SmolVLAConfig) -> tuple[dict[str, str], int, int]:
    visual = {
        key: feature
        for key, feature in cfg.input_features.items()
        if feature.type is FeatureType.VISUAL
    }
    names = {key.rsplit(".", 1)[-1] for key in visual}
    if names != {"top", "wrist_image"}:
        raise ValueError(f"Unsupported checkpoint camera features: {sorted(visual)}")
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
        env_cfg=LiberoConfig(task=TARGET_SUITE, task_ids=[0]),
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


def _reseed_policy_noise(seed: int) -> None:
    """Pin the policy's sampling noise for one episode.

    The env's own randomness is seeded separately via eval_policy's
    start_seed; this call makes the flow-matching noise a deterministic
    function of the episode, independent of process history."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_point(
    label: str, policy_bundle, root: Path, out_dir: Path, checkpoint: dict
) -> dict:
    condition, task_id = parse_label(label)
    policy, preprocessor, postprocessor, cfg = policy_bundle
    mapping, height, width = _camera_mapping(cfg)
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
    base_env = env_cfg.create_envs(
        n_envs=EVAL_BATCH_SIZE, use_async_envs=EVAL_BATCH_SIZE > 1
    )[TARGET_SUITE][env_task_id]
    videos_dir = out_dir / "videos" / label
    videos_dir.mkdir(parents=True, exist_ok=True)
    per_episode: list[dict[str, Any]] = []
    video_paths: list[Path] = []
    try:
        descriptions = tuple(base_env.call("task_description"))
        expected = TARGET_INSTRUCTIONS[task_id]
        if any(item != expected for item in descriptions):
            raise RuntimeError(
                f"LIBERO task mapping mismatch for {label}: expected "
                f"{expected!r}, got {descriptions!r}"
            )
        prompt = prompt_for(condition, task_id)
        env = _PromptOverrideVectorEnv(base_env, prompt)
        device = torch.device(cfg.device)
        amp = torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext()
        with torch.no_grad(), amp:
            for episode in range(N_EVAL_EPISODES):
                _reseed_policy_noise(noise_seed(episode))
                scratch = videos_dir / f"_ep_{episode}"
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
                    start_seed=MASTER_SEED + episode,
                )
                raw_videos = [Path(path) for path in info.get("video_paths", [])]
                if len(raw_videos) != 1 or not raw_videos[0].is_file():
                    raise RuntimeError(
                        f"{label} episode {episode}: expected 1 video, got {raw_videos}"
                    )
                destination = videos_dir / f"eval_episode_{episode}.mp4"
                shutil.move(str(raw_videos[0]), destination)
                shutil.rmtree(scratch, ignore_errors=True)
                record = dict(info["per_episode"][0])
                record["episode_ix"] = episode
                record["seed"] = MASTER_SEED + episode
                record["noise_seed"] = noise_seed(episode)
                record["video_path"] = str(destination)
                record["outcome"] = "success" if record["success"] else "failure"
                per_episode.append(record)
                video_paths.append(destination)
    finally:
        base_env.close()

    outcomes = [bool(record["success"]) for record in per_episode]
    result = {
        "experiment": "pretrain_smolvla_prompt_only_3",
        **checkpoint,
        "label": label,
        "condition": condition,
        "logical_task_id": task_id,
        "env_task_id": env_task_id,
        "environment_instruction": TARGET_INSTRUCTIONS[task_id],
        "policy_prompt": prompt_for(condition, task_id),
        "seed": MASTER_SEED,
        "noise_seeding": "per-episode torch.manual_seed(noise_seed(episode)), batch=1",
        "n_episodes": N_EVAL_EPISODES,
        "successes": sum(outcomes),
        "success_rate": sum(outcomes) / len(outcomes),
        "per_episode": per_episode,
        "video_paths": [str(path) for path in video_paths],
    }
    output = out_dir / f"{label}.json"
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
    parser = argparse.ArgumentParser(description="Evaluate prompt-only points (v3)")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override results directory (used by the determinism check)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    ensure_libero_config()
    out_dir = args.out_dir if args.out_dir is not None else root / "results/raw"
    checkpoint = json.loads(
        (root / "artifacts/checkpoint_manifest.json").read_text()
    )
    checkpoint_fields = {
        "checkpoint_path": checkpoint["checkpoint_path"],
        "checkpoint_sha256": checkpoint["model_safetensors_sha256"],
        "provenance": checkpoint.get("provenance", CHECKPOINT_PROVENANCE),
    }

    labels = args.labels or all_labels()
    unknown = [label for label in labels if label not in all_labels()]
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")

    set_seed(MASTER_SEED)
    bundle = load_policy(args.device)
    summary = {}
    for label in labels:
        if not args.force and result_complete(out_dir, label):
            summary[label] = "already_complete"
            continue
        result = run_point(label, bundle, root, out_dir, checkpoint_fields)
        summary[label] = result["success_rate"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
