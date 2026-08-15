from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image

from .aggregate import metric_rows
from .constants import (
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    MASTER_SEED,
    PROMPT_CONDITIONS,
)

DEFAULT_PROJECT = "smolvla-prompt-only"
DEFAULT_RUN_NAME = "2026-08-15-prompt-only"


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


def create_gifs(
    results_root: Path, output_dir: Path, task_ids: list[int], force: bool = False
) -> list[dict[str, Any]]:
    items = []
    for task_id in task_ids:
        source = (
            results_root
            / "videos"
            / "true"
            / f"task_{task_id}"
            / "eval_episode_0.mp4"
        )
        if not source.exists():
            continue
        destination = output_dir / f"task_{task_id}_true.gif"
        if (
            force
            or not destination.exists()
            or destination.stat().st_mtime < source.stat().st_mtime
        ):
            video_to_gif(source, destination)
        items.append({"task_id": task_id, "source": source, "gif": destination})
    return items


def table_data(summary: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    columns = [
        "task_id",
        "instruction",
        "condition",
        "successes",
        "trials",
        "success_rate",
        "ci95_low",
        "ci95_high",
    ]
    return columns, [[row[column] for column in columns] for row in metric_rows(summary)]


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
        "group": "prompt-only-evaluation",
        "config": {
            "checkpoint": CHECKPOINT_REPO,
            "revision": CHECKPOINT_REVISION,
            "seed": MASTER_SEED,
            "target_demonstrations": 0,
            "optimizer_steps": 0,
        },
        "auto_log_gpu": False,
        "auto_log_cpu": False,
    }
    if space_id:
        init_args["space_id"] = space_id
    trackio.init(**init_args)
    try:
        for task_id, task in summary["tasks"].items():
            trackio.log(
                {
                    f"success/{condition}": task["conditions"][condition]["success_rate"]
                    for condition in PROMPT_CONDITIONS
                },
                step=int(task_id),
            )
        columns, rows = table_data(summary)
        payload: dict[str, Any] = {
            "tables/prompt_metrics": trackio.Table(columns=columns, data=rows),
            "plots/prompt_controls": trackio.Image(
                plot, caption="Success by task and prompt condition"
            ),
            "reports/summary": trackio.Markdown(
                "# SmolVLA prompt-only\n\n"
                "Frozen LIBERO-90 checkpoint, zero target demonstrations, zero optimizer steps. "
                "The table reports true, wrong-task, and nonsense-prompt controls."
            ),
        }
        for item in gifs:
            payload[f"rollouts/task_{item['task_id']}_true"] = trackio.Video(
                item["source"], caption=f"Task {item['task_id']}, true prompt"
            )
            payload[f"rollout_gifs/task_{item['task_id']}_true"] = trackio.Image(
                item["gif"], caption=f"Task {item['task_id']}, true prompt GIF"
            )
        trackio.log(payload, step=len(summary["tasks"]))
    finally:
        trackio.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log prompt-only curves, table, plot, and rollout GIFs to Trackio"
    )
    parser.add_argument("--project", default=os.environ.get("TRACKIO_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--space-id", default=os.environ.get("TRACKIO_SPACE_ID"))
    parser.add_argument("--summary", type=Path, default=Path("results/summary/summary.json"))
    parser.add_argument("--results-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--gifs-dir", type=Path, default=Path("results/media/gifs"))
    parser.add_argument(
        "--plot", type=Path, default=Path("results/summary/prompt_controls.png")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("results/summary/trackio_manifest.json")
    )
    parser.add_argument("--force-gifs", action="store_true")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    task_ids = [int(task_id) for task_id in summary["tasks"]]
    gifs = create_gifs(args.results_root, args.gifs_dir, task_ids, args.force_gifs)
    log_to_trackio(
        summary, gifs, args.plot, args.project, args.run_name, args.space_id
    )

    trackio_dir = Path(
        os.environ.get(
            "TRACKIO_DIR", Path.home() / ".cache" / "huggingface" / "trackio"
        )
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
        "logged_metrics": [f"success/{condition}" for condition in PROMPT_CONDITIONS],
        "table": "tables/prompt_metrics",
        "plot": "plots/prompt_controls",
        "gifs": [
            {
                "task_id": item["task_id"],
                "source": str(item["source"]),
                "gif": str(item["gif"]),
            }
            for item in gifs
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
