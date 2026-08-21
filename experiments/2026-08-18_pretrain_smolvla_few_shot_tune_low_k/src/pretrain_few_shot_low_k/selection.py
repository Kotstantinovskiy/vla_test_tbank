from __future__ import annotations

"""Deterministic demo selection: official demo_0..demo_{k-1} per task.

The converted libero_goal dataset preserves official order (verified by the
conversion round-trip), so "the first k episodes of a task in dataset order"
and "the official first k demonstrations" coincide.  This module builds the
frozen episode manifest from the conversion manifest and asserts that
official demo indices are contiguous from zero.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from .constants import (
    DEMO_BUDGETS,
    TARGET_DATASET_REPO,
    TARGET_DATASET_ROOT,
    TARGET_INSTRUCTIONS,
    TARGET_ENV_TASK_IDS,
    experiment_root,
)


def build_manifest(conversion_manifest: dict[str, Any]) -> dict[str, Any]:
    if conversion_manifest["repo_id"] != TARGET_DATASET_REPO:
        raise ValueError(
            f"Unexpected dataset {conversion_manifest['repo_id']!r}; "
            f"expected {TARGET_DATASET_REPO!r}"
        )
    episodes = conversion_manifest["episodes"]
    tasks: dict[str, Any] = {}
    for task_id, instruction in TARGET_INSTRUCTIONS.items():
        matching = [item for item in episodes if item["task"] == instruction]
        if len(matching) < max(DEMO_BUDGETS):
            raise ValueError(
                f"Task {instruction!r} has only {len(matching)} episodes"
            )
        demo_indices = [item["official_demo_index"] for item in matching]
        if demo_indices != list(range(len(matching))):
            raise ValueError(
                f"Official demo order broken for {instruction!r}: {demo_indices[:5]}..."
            )
        selected = [item["episode_index"] for item in matching]
        tasks[str(task_id)] = {
            "logical_task_id": task_id,
            "env_task_id": TARGET_ENV_TASK_IDS[task_id],
            "instruction": instruction,
            "official_file": matching[0]["official_file"],
            "available_episodes": len(matching),
            "episodes": {str(k): selected[:k] for k in DEMO_BUDGETS},
            "official_demos": {
                str(k): [item["official_demo"] for item in matching[:k]]
                for k in DEMO_BUDGETS
            },
        }
    return {
        "dataset_repo": TARGET_DATASET_REPO,
        "dataset_root": str(TARGET_DATASET_ROOT),
        "source_repo": conversion_manifest["source_repo"],
        "source_revision": conversion_manifest["source_revision"],
        "selection_rule": (
            "first k episodes of the task in dataset order == official "
            "demo_0..demo_{k-1} (order asserted against the conversion manifest)"
        ),
        "demo_budgets": list(DEMO_BUDGETS),
        "tasks": tasks,
    }


def episodes_from_manifest(path: Path, task_id: int, k: int) -> list[int]:
    manifest = json.loads(path.read_text())
    return list(manifest["tasks"][str(task_id)]["episodes"][str(k)])


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conversion-manifest",
        type=Path,
        default=TARGET_DATASET_ROOT / "conversion_manifest.json",
    )
    parser.add_argument(
        "--manifest", type=Path, default=root / "artifacts/episode_manifest.json"
    )
    args = parser.parse_args()
    manifest = build_manifest(json.loads(args.conversion_manifest.read_text()))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                task_id: task["official_demos"]["5"]
                for task_id, task in manifest["tasks"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
