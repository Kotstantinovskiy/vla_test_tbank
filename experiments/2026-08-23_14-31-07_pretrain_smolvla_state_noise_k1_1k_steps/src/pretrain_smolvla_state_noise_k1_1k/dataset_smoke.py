from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .constants import (
    DEMO_BUDGET,
    DEMO_BUDGETS,
    TARGET_DATASET_REPO,
    TARGET_DATASET_ROOT,
    TARGET_INSTRUCTIONS,
    experiment_root,
)
from .selection import episodes_from_manifest


def verify_loaded_episode_indices(expected: list[int], loaded: list[int]) -> None:
    unique = sorted(set(int(value) for value in loaded))
    if unique != expected:
        raise RuntimeError(
            f"LeRobot loaded episode indices {unique}, expected exactly {expected}"
        )


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(
        description="Instantiate every task/k dataset and verify loaded episode IDs"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "artifacts/episode_manifest.json",
    )
    args = parser.parse_args()
    checks = []
    for task_id in TARGET_INSTRUCTIONS:
        for budget in DEMO_BUDGETS:
            expected = episodes_from_manifest(args.manifest, task_id, budget)
            dataset = LeRobotDataset(
                repo_id=TARGET_DATASET_REPO,
                root=TARGET_DATASET_ROOT,
                episodes=expected,
                download_videos=False,
                video_backend="pyav",
            )
            if dataset.episodes != expected:
                raise RuntimeError(
                    f"Dataset retained episodes {dataset.episodes}, expected {expected}"
                )
            loaded = [int(value) for value in dataset.hf_dataset["episode_index"]]
            verify_loaded_episode_indices(expected, loaded)
            if dataset.num_episodes != budget:
                raise RuntimeError(
                    f"Dataset reports {dataset.num_episodes} episodes for k={budget}"
                )
            checks.append(
                {
                    "task_id": task_id,
                    "demo_budget": budget,
                    "requested_episode_indices": expected,
                    "loaded_episode_indices": sorted(set(loaded)),
                    "loaded_frames": dataset.num_frames,
                }
            )
    payload = {
        "passed": True,
        "lerobot_dataset_version_contract": "0.6.1",
        "checks": checks,
    }
    output = root / "artifacts/dataset_selection_smoke.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
