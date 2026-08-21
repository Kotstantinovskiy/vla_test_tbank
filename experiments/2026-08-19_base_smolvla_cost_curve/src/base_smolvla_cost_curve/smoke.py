from __future__ import annotations

import argparse
import json

from lerobot.envs.configs import LiberoEnv as LiberoConfig

from .constants import TARGET_ENV_TASK_IDS, TARGET_INSTRUCTIONS, TARGET_SUITE
from .evaluate import ensure_libero_config


def main() -> None:
    """Create one real env per target task, assert its description, reset it."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-ids", type=int, nargs="+", default=sorted(TARGET_INSTRUCTIONS)
    )
    args = parser.parse_args()

    ensure_libero_config()
    checked: dict[str, str] = {}
    for logical_task_id in args.task_ids:
        env_task_id = TARGET_ENV_TASK_IDS[logical_task_id]
        env_cfg = LiberoConfig(
            task=TARGET_SUITE, task_ids=[env_task_id], max_parallel_tasks=1
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
            observations, _ = env.reset(seed=0)
            if not observations:
                raise RuntimeError("Environment reset returned no observations")
        finally:
            env.close()
        checked[str(logical_task_id)] = expected
    print(json.dumps({"suite": TARGET_SUITE, "validated_tasks": checked}, indent=2))


if __name__ == "__main__":
    main()
