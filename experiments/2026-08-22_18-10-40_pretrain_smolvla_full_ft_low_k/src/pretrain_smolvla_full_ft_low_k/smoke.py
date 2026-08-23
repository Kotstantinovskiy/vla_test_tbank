from __future__ import annotations

import argparse
import json
import os

from .constants import (
    EVAL_HORIZON,
    MASTER_SEED,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
    experiment_root,
)
from .libero_setup import ensure_libero_config

os.environ.setdefault(
    "LIBERO_CONFIG_PATH", str(experiment_root() / "artifacts/libero_config")
)
ensure_libero_config()

from lerobot.envs.configs import LiberoEnv as LiberoConfig


def main() -> None:
    """Create real target envs, assert their descriptions, reset, and close."""

    root = experiment_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-ids", type=int, nargs="+", default=sorted(TARGET_INSTRUCTIONS)
    )
    args = parser.parse_args()

    checked: dict[str, dict[str, object]] = {}
    for logical_task_id in args.task_ids:
        env_task_id = TARGET_ENV_TASK_IDS[logical_task_id]
        env_cfg = LiberoConfig(
            task=TARGET_SUITE,
            task_ids=[env_task_id],
            max_parallel_tasks=1,
            episode_length=EVAL_HORIZON,
        )
        env = env_cfg.create_envs(n_envs=1, use_async_envs=False)[TARGET_SUITE][
            env_task_id
        ]
        try:
            descriptions = tuple(env.call("task_description"))
            expected = TARGET_INSTRUCTIONS[logical_task_id]
            if descriptions != (expected,):
                raise RuntimeError(
                    f"Task mapping mismatch: {logical_task_id} -> {descriptions!r}"
                )
            observations, _ = env.reset(seed=MASTER_SEED)
            if not observations:
                raise RuntimeError("Environment reset returned no observations")
        finally:
            env.close()
        checked[str(logical_task_id)] = {
            "env_task_id": env_task_id,
            "instruction": expected,
            "reset_seed": MASTER_SEED,
        }
    payload = {
        "passed": True,
        "suite": TARGET_SUITE,
        "validated_tasks": checked,
    }
    output = root / "artifacts/env_smoke.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
