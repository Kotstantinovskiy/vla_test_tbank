from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .constants import TARGET_DATASET_REPO, TARGET_DATASET_REVISION
from .dataset_repair import download_candidate_shards, repair_selected_metadata
from .schema_adapter import adapt_checkpoint
from .selection import discover_manifest, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/var/tmp/vla_target_9176"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/episode_manifest.json"))
    parser.add_argument(
        "--seen-output", type=Path, default=Path("artifacts/seen_image_schema")
    )
    args = parser.parse_args()

    manifest = discover_manifest(root=args.dataset_root)
    save_manifest(manifest, args.manifest)
    union = sorted(
        {
            episode
            for task in manifest["tasks"].values()
            for episode in task["episodes"]["25"]
        }
    )
    candidate_shards = download_candidate_shards(
        args.dataset_root, TARGET_DATASET_REPO, TARGET_DATASET_REVISION, union
    )
    repair = repair_selected_metadata(args.dataset_root, union)
    dataset = LeRobotDataset(
        TARGET_DATASET_REPO,
        root=args.dataset_root,
        episodes=union,
        revision=TARGET_DATASET_REVISION,
        video_backend="pyav",
    )
    if dataset.num_episodes != len(union):
        raise RuntimeError(
            f"Expected {len(union)} selected episodes, loaded {dataset.num_episodes}"
        )
    adapter = adapt_checkpoint(args.seen_output.resolve())
    print(
        json.dumps(
            {
                "dataset_root": str(dataset.root),
                "selected_episode_count": dataset.num_episodes,
                "selected_frame_count": dataset.num_frames,
                "frame_bounds_repair": {
                    "candidate_shards": [min(candidate_shards), max(candidate_shards)],
                    "changed_bounds_count": repair["changed_bounds_count"],
                    "changed_file_count": repair["changed_file_count"],
                    "audit": str(args.dataset_root / "meta/FRAME_BOUNDS_REPAIR.json"),
                },
                "manifest": str(args.manifest),
                "seen_adapter": adapter,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
