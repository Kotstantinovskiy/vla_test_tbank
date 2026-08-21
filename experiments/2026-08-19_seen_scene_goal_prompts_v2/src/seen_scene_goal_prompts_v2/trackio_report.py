from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .constants import CHECKPOINT_PATH, MASTER_SEED, SUITE, TRACKIO_PROJECT, experiment_root

DEFAULT_RUN_NAME = "2026-08-19-seen-scene-goal-prompts-v2"


def log_to_trackio(
    summary: dict[str, Any],
    raw_dir: Path,
    plot: Path,
    report: Path,
    project: str,
    run_name: str,
    space_id: str | None,
) -> list[dict[str, Any]]:
    import trackio

    init_args: dict[str, Any] = {
        "project": project,
        "name": run_name,
        "group": "language-probes",
        "config": {
            "checkpoint": str(CHECKPOINT_PATH),
            "suite": SUITE,
            "seed": MASTER_SEED,
            "points": len(summary["rows"]),
        },
        "auto_log_gpu": False,
        "auto_log_cpu": False,
    }
    if space_id:
        init_args["space_id"] = space_id
    trackio.init(**init_args)
    media: list[dict[str, Any]] = []
    try:
        for step, row in enumerate(summary["rows"]):
            trackio.log(
                {f"success/{row['block']}/{row['label']}": row["success_rate"]},
                step=step,
            )
        point_columns = [
            "label", "block", "env_instruction", "prompt", "success_metric",
            "successes", "trials", "success_rate", "ci95_low", "ci95_high",
            "env_task_successes", "env_task_success_rate",
        ]
        point_rows = [[row[c] for c in point_columns] for row in summary["rows"]]
        paired_columns = [
            "env_instruction", "block", "prompt", "trained", "condition",
            "delta", "discordant_only_trained", "discordant_only_condition",
            "mcnemar_p",
        ]
        paired_rows = [[row[c] for c in paired_columns] for row in summary["paired"]]
        slice_columns = ["goal_id", "prompt", "status", "relationship", "successes", "trials"]
        slice_rows = [
            [item.get(c) for c in slice_columns] for item in summary["goal_slice"]
        ]
        payload: dict[str, Any] = {
            "tables/points": trackio.Table(columns=point_columns, data=point_rows),
            "tables/paired_mcnemar": trackio.Table(columns=paired_columns, data=paired_rows),
            "tables/goal_slice": trackio.Table(columns=slice_columns, data=slice_rows),
            "plots/prompt_transfer": trackio.Image(
                plot, caption="Prompted-task predicate success by block (Wilson 95% CI)"
            ),
            "reports/report": trackio.Markdown(report.read_text()),
        }
        # Media policy: representative media only — episode 0 of every point.
        for row in summary["rows"]:
            raw = json.loads((raw_dir / f"{row['label']}.json").read_text())
            episode = raw["per_episode"][0]
            video = Path(episode["video_path"])
            if not video.is_file():
                raise FileNotFoundError(video)
            payload[f"rollouts/{row['label']}"] = trackio.Video(
                video,
                caption=(
                    f"{row['block']}: env '{row['env_instruction']}' / prompt "
                    f"'{row['prompt']}' — episode 0 ({episode['outcome']}); "
                    f"point {row['successes']}/{row['trials']}"
                ),
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
    return media


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Log prompt-transfer results to Trackio")
    parser.add_argument("--project", default=os.environ.get("TRACKIO_PROJECT", TRACKIO_PROJECT))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--space-id", default=os.environ.get("TRACKIO_SPACE_ID"))
    parser.add_argument("--summary", type=Path, default=root / "results/summary/summary.json")
    parser.add_argument("--raw-dir", type=Path, default=root / "results/raw")
    parser.add_argument("--plot", type=Path, default=root / "results/summary/prompt_transfer_v2.png")
    parser.add_argument("--report", type=Path, default=root / "reports/REPORT.md")
    parser.add_argument("--manifest", type=Path, default=root / "results/summary/trackio_manifest.json")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    media = log_to_trackio(
        summary, args.raw_dir, args.plot, args.report,
        args.project, args.run_name, args.space_id,
    )
    manifest = {
        "project": args.project,
        "run": args.run_name,
        "plot": str(args.plot),
        "tables": ["tables/points", "tables/paired_mcnemar"],
        "media": media,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("project", "run", "plot", "tables")}, indent=2))
    print(f"media points logged: {len(media)}")


if __name__ == "__main__":
    main()
