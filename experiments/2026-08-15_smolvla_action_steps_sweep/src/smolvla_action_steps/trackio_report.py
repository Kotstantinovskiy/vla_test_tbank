from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image

from .aggregate import metric_rows
from .constants import ACTION_STEPS, DEMO_BUDGETS, TARGET_INSTRUCTIONS

DEFAULT_PROJECT = "smolvla-action-steps-sweep"
DEFAULT_RUN_NAME = "2026-08-15-action-steps-sweep"


def video_to_gif(source: Path, destination: Path, fps: int = 12) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(source)
    try:
        source_fps = float(reader.get_meta_data().get("fps") or fps)
        stride = max(1, round(source_fps / fps))
        frames = [
            Image.fromarray(frame).convert("RGB")
            for index, frame in enumerate(reader)
            if index % stride == 0
        ]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames decoded from {source}")
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, round(1000 / (source_fps / stride))),
        loop=0,
        optimize=True,
    )


def representative_media(
    summary: dict[str, Any], gifs_dir: Path, force: bool = False
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for task_id in TARGET_INSTRUCTIONS:
        budgets = DEMO_BUDGETS if task_id == 0 else (0, 25)
        for budget in budgets:
            budget_result = summary["tasks"][str(task_id)]["budgets"][str(budget)]
            selected = budget_result["selected_best_action_steps"]
            for step in sorted({selected, 50}):
                paths = budget_result["points"][str(step)]["video_paths"]
                if not paths:
                    continue
                source = Path(paths[0])
                if not source.exists():
                    continue
                destination = gifs_dir / f"task_{task_id}_k_{budget}_n_{step}.gif"
                if (
                    force
                    or not destination.exists()
                    or destination.stat().st_mtime < source.stat().st_mtime
                ):
                    video_to_gif(source, destination)
                items.append(
                    {
                        "task_id": task_id,
                        "demo_budget": budget,
                        "n_action_steps": step,
                        "source": source,
                        "gif": destination,
                    }
                )
    return items


def table_data(summary: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    columns = [
        "task_id",
        "demo_budget",
        "n_action_steps",
        "successes",
        "trials",
        "success_rate",
        "ci95_low",
        "ci95_high",
        "frozen_baseline_success_rate",
        "delta_vs_paired_50",
        "delta_vs_frozen_baseline",
    ]
    return columns, [[row[column] for column in columns] for row in metric_rows(summary)]


def best_table_data(summary: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    columns = [
        "task_id",
        "demo_budget",
        "best_n_action_steps",
        "success_rate",
        "delta_vs_paired_50",
    ]
    rows: list[list[Any]] = []
    for task_id, task in summary["tasks"].items():
        for budget, result in task["budgets"].items():
            selected = result["selected_best_action_steps"]
            point = result["points"][str(selected)]
            rows.append(
                [
                    int(task_id),
                    int(budget),
                    ",".join(map(str, result["best_action_steps"])),
                    point["success_rate"],
                    point["delta_vs_paired_50"],
                ]
            )
    return columns, rows


def log_to_trackio(
    summary: dict[str, Any],
    media: list[dict[str, Any]],
    summary_dir: Path,
    project: str,
    run_name: str,
    space_id: str | None,
) -> None:
    import trackio

    init_args: dict[str, Any] = {
        "project": project,
        "name": run_name,
        "group": "frozen-weights-inference-ablation",
        "config": {
            "action_steps": list(ACTION_STEPS),
            "demo_budgets": list(DEMO_BUDGETS),
            "weights_modified": False,
            "baseline_frozen": True,
            "episodes_per_point": 20,
        },
        "auto_log_gpu": False,
        "auto_log_cpu": False,
    }
    if space_id:
        init_args["space_id"] = space_id
    trackio.init(**init_args)
    try:
        means = summary["mean_success_by_budget_and_action_steps"]
        for step in ACTION_STEPS:
            metrics: dict[str, float] = {}
            for budget in DEMO_BUDGETS:
                metrics[f"mean_success/k_{budget}"] = means[str(budget)][str(step)]
                metrics[f"delta_vs_50/k_{budget}"] = summary[
                    "mean_delta_vs_paired_50"
                ][str(budget)][str(step)]
            for task_id in TARGET_INSTRUCTIONS:
                for budget in DEMO_BUDGETS:
                    metrics[f"task_{task_id}/k_{budget}/success"] = summary["tasks"][
                        str(task_id)
                    ]["budgets"][str(budget)]["points"][str(step)]["success_rate"]
            trackio.log(metrics, step=step)

        columns, rows = table_data(summary)
        best_columns, best_rows = best_table_data(summary)
        payload: dict[str, Any] = {
            "tables/all_metrics": trackio.Table(columns=columns, data=rows),
            "tables/best_horizons": trackio.Table(
                columns=best_columns, data=best_rows
            ),
            "plots/cost_curves": trackio.Image(
                summary_dir / "cost_curves_by_action_steps.png",
                caption="Cost curves for each inference horizon",
            ),
            "plots/action_steps_by_task": trackio.Image(
                summary_dir / "action_steps_by_task.png",
                caption="Per-task success versus n_action_steps",
            ),
            "reports/protocol": trackio.Markdown(
                "# Frozen-weight inference ablation\n\n"
                "All checkpoints are unchanged. `n_action_steps=50` is the paired rerun "
                "anchor; deltas versus it isolate the inference contribution."
            ),
        }
        for item in media:
            key = (
                f"task_{item['task_id']}/k_{item['demo_budget']}"
                f"/n_{item['n_action_steps']}"
            )
            caption = (
                f"Task {item['task_id']}, k={item['demo_budget']}, "
                f"n_action_steps={item['n_action_steps']}"
            )
            payload[f"rollouts/{key}"] = trackio.Video(
                item["source"], caption=caption
            )
            payload[f"rollout_gifs/{key}"] = trackio.Image(
                item["gif"], caption=f"{caption} GIF"
            )
        trackio.log(payload, step=max(ACTION_STEPS) + 1)
    finally:
        trackio.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Log action-step sweep to Trackio")
    parser.add_argument("--project", default=os.environ.get("TRACKIO_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--space-id", default=os.environ.get("TRACKIO_SPACE_ID"))
    parser.add_argument("--summary", type=Path, default=Path("results/summary/summary.json"))
    parser.add_argument("--summary-dir", type=Path, default=Path("results/summary"))
    parser.add_argument("--gifs-dir", type=Path, default=Path("results/media/gifs"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("results/summary/trackio_manifest.json")
    )
    parser.add_argument("--force-gifs", action="store_true")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    media = representative_media(summary, args.gifs_dir, args.force_gifs)
    log_to_trackio(
        summary, media, args.summary_dir, args.project, args.run_name, args.space_id
    )
    trackio_dir = Path(
        os.environ.get("TRACKIO_DIR", Path.home() / ".cache" / "huggingface" / "trackio")
    )
    try:
        display_dir = trackio_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        display_dir = trackio_dir
    manifest = {
        "project": args.project,
        "run": args.run_name,
        "space_id": args.space_id,
        "trackio_dir": str(display_dir),
        "logged_metric_steps": list(ACTION_STEPS),
        "tables": ["tables/all_metrics", "tables/best_horizons"],
        "plots": ["plots/cost_curves", "plots/action_steps_by_task"],
        "media": [
            {**{key: value for key, value in item.items() if key not in {"source", "gif"}},
             "source": str(item["source"]), "gif": str(item["gif"])}
            for item in media
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
