from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .constants import TARGET_DATASET_REPO, TARGET_DATASET_REVISION
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
                "manifest": str(args.manifest),
                "seen_adapter": adapter,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
