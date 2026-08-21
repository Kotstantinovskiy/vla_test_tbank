from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .constants import CHECKPOINT_PATH, MASTER_SEED, TRACKIO_PROJECT, experiment_root

DEFAULT_RUN_NAME = "2026-08-19-prompt-only-3"


def main() -> None:
    import trackio

    root = experiment_root()
    parser = argparse.ArgumentParser(description="Log prompt-only v3 to Trackio")
    parser.add_argument("--project", default=os.environ.get("TRACKIO_PROJECT", TRACKIO_PROJECT))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--summary", type=Path, default=root / "results/summary/summary.json")
    parser.add_argument("--raw-dir", type=Path, default=root / "results/raw")
    parser.add_argument("--plot", type=Path, default=root / "results/summary/prompt_only_3.png")
    parser.add_argument("--report", type=Path, default=root / "reports/REPORT.md")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    trackio.init(
        project=args.project,
        name=args.run_name,
        group="zero-shot",
        config={
            "checkpoint": str(CHECKPOINT_PATH),
            "seed": MASTER_SEED,
            "noise_seeding": summary["noise_seeding"],
        },
        auto_log_gpu=False,
        auto_log_cpu=False,
    )
    try:
        columns = [
            "condition", "task_id", "prompt", "successes", "trials",
            "success_rate", "ci95_low", "ci95_high", "v2_successes",
        ]
        rows = [[row[c] for c in columns] for row in summary["rows"]]
        payload: dict[str, Any] = {
            "tables/points": trackio.Table(columns=columns, data=rows),
            "plots/prompt_only_3": trackio.Image(
                args.plot, caption="Zero-shot per task, _3 (per-episode noise) vs _2"
            ),
            "reports/report": trackio.Markdown(args.report.read_text()),
        }
        # Media policy: episode 0 of every point.
        for row in summary["rows"]:
            raw = json.loads((args.raw_dir / f"{row['label']}.json").read_text())
            episode = raw["per_episode"][0]
            video = Path(episode["video_path"])
            if not video.is_file():
                raise FileNotFoundError(video)
            payload[f"rollouts/{row['label']}"] = trackio.Video(
                video,
                caption=(
                    f"{row['condition']} task {row['task_id']} episode 0 "
                    f"({episode['outcome']}); point {row['successes']}/{row['trials']}"
                ),
            )
        for condition, stats in summary["conditions"].items():
            trackio.log({f"success/{condition}_pooled": stats["success_rate"]})
        trackio.log(payload)
    finally:
        trackio.finish()
    print("logged", len(summary["rows"]), "points")


if __name__ == "__main__":
    main()
