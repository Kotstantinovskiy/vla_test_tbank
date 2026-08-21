from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image

from .constants import (
    BASE_CHECKPOINT,
    BASE_PROVENANCE,
    DEMO_BUDGETS,
    MASTER_SEED,
    TARGET_INSTRUCTIONS,
    TRACKIO_PROJECT,
    experiment_root,
)

DEFAULT_RUN_NAME = "2026-08-18-pretrain-few-shot-low-k"


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


def first_outcome_gifs(
    results_root: Path, gifs_dir: Path, force: bool = False
) -> list[dict[str, Any]]:
    """Per budget k: GIF of the first success and the first failure.

    "First" scans tasks in id order, then episodes in index order, over the
    per-episode outcomes recorded by evaluation; every episode has a saved
    video, so the chosen episodes are guaranteed to exist on disk.
    """

    items: list[dict[str, Any]] = []
    for budget in DEMO_BUDGETS:
        chosen: dict[str, dict[str, Any]] = {}
        for task_id in sorted(TARGET_INSTRUCTIONS):
            payload_path = results_root / f"task_{task_id}" / f"k_{budget}.json"
            if not payload_path.is_file():
                continue
            payload = json.loads(payload_path.read_text())
            for episode_index, episode in enumerate(payload["per_episode"]):
                outcome = episode["outcome"]
                if outcome in chosen:
                    continue
                chosen[outcome] = {
                    "budget": budget,
                    "outcome": outcome,
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "video": episode["video_path"],
                }
            if {"success", "failure"} <= set(chosen):
                break
        for outcome, item in sorted(chosen.items()):
            destination = gifs_dir / f"k_{budget}_first_{outcome}.gif"
            source = Path(item["video"])
            if force or not destination.exists() or destination.stat().st_mtime < source.stat().st_mtime:
                video_to_gif(source, destination)
            item["gif"] = str(destination)
            items.append(item)
    return items


def log_to_trackio(
    summary: dict[str, Any],
    gifs: list[dict[str, Any]],
    plot: Path,
    project: str,
    run_name: str,
    space_id: str | None,
) -> None:
    import trackio

    init_args: dict[str, Any] = {
        "project": project,
        "name": run_name,
        "group": "few-shot-cost-curve",
        "config": {
            "base_checkpoint": str(BASE_CHECKPOINT),
            **BASE_PROVENANCE,
            "seed": MASTER_SEED,
            "budgets": list(DEMO_BUDGETS),
        },
        "auto_log_gpu": False,
        "auto_log_cpu": False,
    }
    if space_id:
        init_args["space_id"] = space_id
    trackio.init(**init_args)
    try:
        for task_id, task in summary["tasks"].items():
            for budget in DEMO_BUDGETS:
                trackio.log(
                    {
                        f"success/task_{task_id}": task["budgets"][str(budget)][
                            "success_rate"
                        ]
                    },
                    step=budget,
                )
        for budget in DEMO_BUDGETS:
            trackio.log(
                {
                    "success/mean_all_10": summary["cost_curve"]["mean_all_10"][str(budget)],
                    "success/mean_tasks_0_2": summary["cost_curve"]["mean_tasks_0_2"][str(budget)],
                },
                step=budget,
            )
        columns = ["task_id", "instruction", "k", "successes", "trials", "success_rate", "ci95_low", "ci95_high"]
        rows = [
            [
                int(task_id),
                task["instruction"],
                int(budget),
                metrics["successes"],
                metrics["trials"],
                metrics["success_rate"],
                metrics["ci95_low"],
                metrics["ci95_high"],
            ]
            for task_id, task in summary["tasks"].items()
            for budget, metrics in task["budgets"].items()
        ]
        payload: dict[str, Any] = {
            "tables/cost_curve": trackio.Table(columns=columns, data=rows),
            "plots/cost_curve": trackio.Image(plot, caption="Success vs demonstrations"),
            "reports/summary": trackio.Markdown(
                "# Few-shot cost curve (official-data pretrain)\n\n"
                "Expert-only adaptations from the frozen pretrain; demos are the "
                "official demo_0..demo_{k-1}. Full per-episode videos are stored on "
                "disk; this run logs the first success and first failure per budget."
            ),
        }
        for item in gifs:
            key = f"rollouts/k_{item['budget']}_first_{item['outcome']}"
            payload[key] = trackio.Video(
                Path(item["video"]),
                caption=(
                    f"k={item['budget']}: first {item['outcome']} "
                    f"(task {item['task_id']}, episode {item['episode_index']})"
                ),
            )
            payload[f"rollout_gifs/k_{item['budget']}_first_{item['outcome']}"] = trackio.Image(
                Path(item["gif"]),
                caption=f"k={item['budget']} first {item['outcome']} GIF",
            )
        trackio.log(payload, step=max(DEMO_BUDGETS))
    finally:
        trackio.finish()


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Log cost curve and outcome GIFs to Trackio")
    parser.add_argument("--project", default=os.environ.get("TRACKIO_PROJECT", TRACKIO_PROJECT))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--space-id", default=os.environ.get("TRACKIO_SPACE_ID"))
    parser.add_argument("--summary", type=Path, default=root / "results/summary/summary.json")
    parser.add_argument("--results-root", type=Path, default=root / "results/raw")
    parser.add_argument("--gifs-dir", type=Path, default=root / "results/media/gifs")
    parser.add_argument("--plot", type=Path, default=root / "results/summary/cost_curve.png")
    parser.add_argument("--manifest", type=Path, default=root / "results/summary/trackio_manifest.json")
    parser.add_argument("--force-gifs", action="store_true")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    gifs = first_outcome_gifs(args.results_root, args.gifs_dir, args.force_gifs)
    log_to_trackio(summary, gifs, args.plot, args.project, args.run_name, args.space_id)
    manifest = {
        "project": args.project,
        "run": args.run_name,
        "gifs": gifs,
        "plot": str(args.plot),
        "table": "tables/cost_curve",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
