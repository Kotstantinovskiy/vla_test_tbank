from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

from .constants import (
    DEMO_BUDGETS,
    TARGET_DATASET_REPO,
    TARGET_DATASET_REVISION,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
)


def select_first_k(
    episode_tasks: Sequence[Sequence[str]], instruction: str, k: int
) -> list[int]:
    """Return the first k global episode IDs whose task list contains instruction."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    matches = [i for i, tasks in enumerate(episode_tasks) if instruction in tasks]
    if len(matches) < k:
        raise ValueError(
            f"Task {instruction!r} has only {len(matches)} episodes, cannot select {k}"
        )
    return matches[:k]


def build_manifest(
    episode_tasks: Sequence[Sequence[str]],
    *,
    dataset_repo: str = TARGET_DATASET_REPO,
    dataset_revision: str = TARGET_DATASET_REVISION,
    budgets: Iterable[int] = DEMO_BUDGETS,
) -> dict[str, Any]:
    budgets = sorted(set(int(k) for k in budgets))
    if not budgets:
        raise ValueError("At least one demonstration budget is required")

    tasks: dict[str, Any] = {}
    for task_id, instruction in TARGET_INSTRUCTIONS.items():
        all_selected = select_first_k(episode_tasks, instruction, max(budgets))
        tasks[str(task_id)] = {
            "logical_task_id": task_id,
            "env_task_id": TARGET_ENV_TASK_IDS[task_id],
            "instruction": instruction,
            "available_episodes": sum(instruction in row for row in episode_tasks),
            "episodes": {str(k): all_selected[:k] for k in budgets},
        }
    return {
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "selection_rule": "first k matching episodes in global dataset order",
        "tasks": tasks,
    }


def discover_manifest(
    *,
    dataset_repo: str = TARGET_DATASET_REPO,
    dataset_revision: str = TARGET_DATASET_REVISION,
    root: str | Path | None = None,
) -> dict[str, Any]:
    meta = LeRobotDatasetMetadata(
        dataset_repo,
        root=root,
        revision=dataset_revision,
    )
    return build_manifest(
        meta.episodes["tasks"],
        dataset_repo=dataset_repo,
        dataset_revision=dataset_revision,
    )


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def episodes_from_manifest(path: str | Path, task_id: int, k: int) -> list[int]:
    manifest = json.loads(Path(path).read_text())
    return list(manifest["tasks"][str(task_id)]["episodes"][str(k)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/episode_manifest.json"))
    parser.add_argument("--root", type=Path, default=Path("/var/tmp/vla_target_9176"))
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument(
        "--print-episodes",
        action="store_true",
        help="Read an existing manifest and print a JSON episode list.",
    )
    args = parser.parse_args()

    if args.print_episodes:
        if args.task_id is None or args.k is None:
            parser.error("--print-episodes requires --task-id and --k")
        print(json.dumps(episodes_from_manifest(args.manifest, args.task_id, args.k)))
        return

    manifest = discover_manifest(root=args.root)
    save_manifest(manifest, args.manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
