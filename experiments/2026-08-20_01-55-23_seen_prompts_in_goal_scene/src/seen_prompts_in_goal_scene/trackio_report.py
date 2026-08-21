from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .constants import EXPERIMENT_NAME, TRACKIO_PROJECT, experiment_root


def main() -> None:
    import trackio

    root = experiment_root()
    parser = argparse.ArgumentParser(description="Log completed prompt experiment")
    parser.add_argument("--project", default=os.environ.get("TRACKIO_PROJECT", TRACKIO_PROJECT))
    parser.add_argument("--run-name", default=root.name)
    parser.add_argument("--space-id", default=os.environ.get("TRACKIO_SPACE_ID"))
    args = parser.parse_args()
    summary_path = root / "results/summary/summary.json"
    summary = json.loads(summary_path.read_text())
    init = {
        "project": args.project,
        "name": args.run_name,
        "group": "language-probes",
        "config": {"experiment": EXPERIMENT_NAME, "points": len(summary["rows"])},
        "auto_log_gpu": False,
        "auto_log_cpu": False,
    }
    if args.space_id:
        init["space_id"] = args.space_id
    trackio.init(**init)
    media = []
    try:
        for step, row in enumerate(summary["rows"]):
            trackio.log({f"success/{row['label']}": row["success_rate"]}, step=step)
        columns = [
            "label",
            "block",
            "env_instruction",
            "prompt",
            "successes",
            "trials",
            "success_rate",
            "ci95_low",
            "ci95_high",
        ]
        payload = {
            "tables/points": trackio.Table(
                columns=columns,
                data=[[row[column] for column in columns] for row in summary["rows"]],
            ),
            "plots/success_rates": trackio.Image(
                root / "results/summary/success_rates.png",
                caption="Prompted-predicate success (Wilson 95% CI)",
            ),
            "reports/report": trackio.Markdown((root / "reports/REPORT.md").read_text()),
        }
        for row in summary["rows"]:
            raw = json.loads(
                (root / "results/raw" / f"{row['label']}.json").read_text()
            )
            episode = raw["per_episode"][0]
            video = Path(episode["video_path"])
            if not video.is_file():
                raise FileNotFoundError(video)
            payload[f"rollouts/{row['label']}"] = trackio.Video(
                video,
                caption=f"episode 0: {episode['outcome']}; {row['successes']}/{row['trials']}",
            )
            media.append(
                {
                    "label": row["label"],
                    "episode_index": 0,
                    "outcome": episode["outcome"],
                    "video": str(video),
                }
            )
        trackio.log(payload, step=len(summary["rows"]))
    finally:
        trackio.finish()
    manifest = {
        "project": args.project,
        "run": args.run_name,
        "tables": ["tables/points"],
        "plot": str(root / "results/summary/success_rates.png"),
        "media": media,
    }
    output = root / "results/summary/trackio_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"project": args.project, "run": args.run_name}, indent=2))


if __name__ == "__main__":
    main()
