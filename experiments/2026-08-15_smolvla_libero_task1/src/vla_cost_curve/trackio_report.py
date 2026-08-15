from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image

from .constants import (
    DEMO_BUDGETS,
    MASTER_SEED,
    SEEN_REPO,
    SEEN_REVISION,
    TARGET_DATASET_REPO,
    TARGET_DATASET_REVISION,
)

DEFAULT_PROJECT = "smolvla-baseline"
DEFAULT_RUN_PREFIX = "2026-08-15"
TRAIN_LOG_FREQUENCY = 25

_TRAIN_LINE = re.compile(r"ot_train\.py:641\s+(?P<body>[^\r\n]+)")
_METRIC = re.compile(r"(?P<key>[A-Za-z_/]+):(?P<value>[0-9.eE+-]+)")
_TRAIN_KEYS = {
    "loss": "train/loss",
    "grdn": "train/grad_norm",
    "lr": "train/learning_rate",
    "updt_s": "timing/update_seconds",
    "data_s": "timing/data_seconds",
    "smp/s": "throughput/samples_per_second",
    "mem_gb": "system/memory_gb",
    "losses_after_forward": "loss_components/after_forward",
    "losses_after_in_ep_bound": "loss_components/after_episode_boundaries",
    "losses_after_rm_padding": "loss_components/after_padding_removal",
}


def parse_training_log(path: Path, log_frequency: int = TRAIN_LOG_FREQUENCY) -> list[dict[str, float | int]]:
    """Recover exact-step training curves from LeRobot's periodic INFO rows."""

    text = path.read_text(errors="replace")
    points: list[dict[str, float | int]] = []
    for index, match in enumerate(_TRAIN_LINE.finditer(text), start=1):
        raw = {
            item.group("key"): float(item.group("value"))
            for item in _METRIC.finditer(match.group("body"))
        }
        point: dict[str, float | int] = {"step": index * log_frequency}
        point.update(
            {_TRAIN_KEYS[key]: raw[key] for key in _TRAIN_KEYS if key in raw}
        )
        points.append(point)
    return points


def cost_curve_points(summary: dict[str, Any]) -> list[dict[str, float | int]]:
    points = []
    for budget in (0, *DEMO_BUDGETS):
        point: dict[str, float | int] = {
            "step": budget,
            "demo_budget": budget,
            "success/mean": summary["mean_cost_curve"][str(budget)],
        }
        for task_id, task in summary["tasks"].items():
            metrics = (
                task["k0"]["true"]
                if budget == 0
                else task["adapted"][str(budget)]
            )
            point[f"success/task_{task_id}"] = metrics["success_rate"]
        points.append(point)
    return points


def cost_curve_table(summary: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    columns = [
        "task_id",
        "instruction",
        "demo_budget",
        "successes",
        "trials",
        "success_rate",
        "ci95_low",
        "ci95_high",
    ]
    rows = []
    for task_id, task in summary["tasks"].items():
        for budget in (0, *DEMO_BUDGETS):
            metrics = (
                task["k0"]["true"]
                if budget == 0
                else task["adapted"][str(budget)]
            )
            rows.append(
                [
                    int(task_id),
                    task["instruction"],
                    budget,
                    metrics["successes"],
                    metrics["trials"],
                    metrics["success_rate"],
                    metrics["ci95_low"],
                    metrics["ci95_high"],
                ]
            )
    return columns, rows


def language_control_table(summary: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
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
    rows = []
    for task_id, task in summary["tasks"].items():
        for condition, metrics in task["k0"].items():
            rows.append(
                [
                    int(task_id),
                    task["instruction"],
                    condition,
                    metrics["successes"],
                    metrics["trials"],
                    metrics["success_rate"],
                    metrics["ci95_low"],
                    metrics["ci95_high"],
                ]
            )
    return columns, rows


def video_specs(results_root: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe available representative rollouts without inventing missing media."""

    specs: list[dict[str, Any]] = []
    for task_id, task in summary["tasks"].items():
        zero = (
            results_root
            / "zero_shot"
            / "videos"
            / "true"
            / f"task_{task_id}"
            / "eval_episode_0.mp4"
        )
        if zero.exists():
            specs.append(
                {
                    "task_id": int(task_id),
                    "budget": 0,
                    "source": zero,
                    "success_rate": task["k0"]["true"]["success_rate"],
                }
            )

        task_video_root = results_root / "adapted" / f"task_{task_id}" / "videos"
        for budget in DEMO_BUDGETS:
            tagged = (
                task_video_root
                / f"k_{budget}"
                / "true"
                / f"task_{task_id}"
                / "eval_episode_0.mp4"
            )
            if tagged.exists():
                specs.append(
                    {
                        "task_id": int(task_id),
                        "budget": budget,
                        "source": tagged,
                        "success_rate": task["adapted"][str(budget)]["success_rate"],
                    }
                )

        # The completed baseline predates budget-tagged video directories.  Its
        # sequential runner left the final k=25 rollout at this legacy path.
        legacy = task_video_root / "true" / f"task_{task_id}" / "eval_episode_0.mp4"
        if legacy.exists() and not any(
            spec["task_id"] == int(task_id) and spec["budget"] == max(DEMO_BUDGETS)
            for spec in specs
        ):
            budget = max(DEMO_BUDGETS)
            specs.append(
                {
                    "task_id": int(task_id),
                    "budget": budget,
                    "source": legacy,
                    "success_rate": task["adapted"][str(budget)]["success_rate"],
                }
            )
    return specs


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
    actual_fps = source_fps / stride
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, round(1000 / actual_fps)),
        loop=0,
        optimize=True,
    )


def create_gifs(
    specs: list[dict[str, Any]], output_dir: Path, force: bool = False
) -> list[dict[str, Any]]:
    manifest = []
    for spec in specs:
        destination = output_dir / f"task_{spec['task_id']}_k_{spec['budget']}.gif"
        if force or not destination.exists() or destination.stat().st_mtime < spec["source"].stat().st_mtime:
            video_to_gif(spec["source"], destination)
        manifest.append({**spec, "gif": destination})
    return manifest


def _init_trackio(project: str, name: str, group: str, config: dict[str, Any], space_id: str | None):
    import trackio

    kwargs: dict[str, Any] = {
        "project": project,
        "name": name,
        "group": group,
        "config": config,
        "auto_log_gpu": False,
        "auto_log_cpu": False,
    }
    if space_id:
        kwargs["space_id"] = space_id
    return trackio.init(**kwargs)


def log_training_curves(
    logs_dir: Path, project: str, run_prefix: str, space_id: str | None
) -> list[dict[str, Any]]:
    import trackio

    logged = []
    logs = []
    for path in logs_dir.glob("task_*_k_*.log"):
        match = re.fullmatch(r"task_(\d+)_k_(\d+)\.log", path.name)
        if match is None:
            continue
        task_id, budget = map(int, match.groups())
        logs.append((task_id, budget, path))
    for task_id, budget, path in sorted(logs):
        points = parse_training_log(path)
        if not points:
            continue
        name = f"{run_prefix}-train-task-{task_id}-k-{budget}"
        _init_trackio(
            project,
            name,
            "naive-finetune",
            {
                "task_id": task_id,
                "demo_budget": budget,
                "seed": MASTER_SEED,
                "seen_checkpoint": SEEN_REPO,
                "seen_revision": SEEN_REVISION,
                "log_frequency": TRAIN_LOG_FREQUENCY,
            },
            space_id,
        )
        try:
            for point in points:
                trackio.log(
                    {key: value for key, value in point.items() if key != "step"},
                    step=int(point["step"]),
                )
        finally:
            trackio.finish()
        logged.append({"name": name, "task_id": task_id, "budget": budget, "points": len(points)})
    return logged


def log_summary(
    summary: dict[str, Any],
    gifs: list[dict[str, Any]],
    summary_dir: Path,
    project: str,
    run_prefix: str,
    space_id: str | None,
) -> str:
    import trackio

    name = f"{run_prefix}-cost-curve"
    _init_trackio(
        project,
        name,
        "evaluation",
        {
            "seed": MASTER_SEED,
            "demo_budgets": [0, *DEMO_BUDGETS],
            "seen_checkpoint": SEEN_REPO,
            "seen_revision": SEEN_REVISION,
            "target_dataset": TARGET_DATASET_REPO,
            "target_dataset_revision": TARGET_DATASET_REVISION,
        },
        space_id,
    )
    try:
        for point in cost_curve_points(summary):
            trackio.log(
                {key: value for key, value in point.items() if key != "step"},
                step=int(point["step"]),
            )

        cost_columns, cost_rows = cost_curve_table(summary)
        language_columns, language_rows = language_control_table(summary)
        payload: dict[str, Any] = {
            "tables/cost_curve": trackio.Table(columns=cost_columns, data=cost_rows),
            "tables/language_controls": trackio.Table(
                columns=language_columns, data=language_rows
            ),
            "plots/cost_curve": trackio.Image(
                summary_dir / "cost_curve.png",
                caption="Success rate by demonstration budget",
            ),
            "reports/baseline": trackio.Markdown(
                "# SmolVLA LIBERO baseline\n\n"
                "Cost curve: **"
                + " → ".join(
                    f"{summary['mean_cost_curve'][str(budget)]:.3f}"
                    for budget in (0, *DEMO_BUDGETS)
                )
                + "** for "
                "demo budgets 0, 5, 10, 25. Tables contain per-task Wilson 95% CIs "
                "and zero-shot language controls."
            ),
        }
        for item in gifs:
            key = f"rollouts/task_{item['task_id']}_k_{item['budget']}"
            payload[key] = trackio.Video(
                item["gif"],
                caption=(
                    f"Task {item['task_id']}, k={item['budget']}, "
                    f"success={item['success_rate']:.2f}"
                ),
            )
        trackio.log(payload, step=max(DEMO_BUDGETS))
    finally:
        trackio.finish()
    return name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish Task 1 curves, tables, and rollout GIFs to Trackio"
    )
    parser.add_argument("--project", default=os.environ.get("TRACKIO_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--space-id", default=os.environ.get("TRACKIO_SPACE_ID"))
    parser.add_argument("--summary", type=Path, default=Path("results/summary/summary.json"))
    parser.add_argument("--results-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--logs-dir", type=Path, default=Path("results/logs/train"))
    parser.add_argument("--gifs-dir", type=Path, default=Path("results/media/gifs"))
    parser.add_argument("--manifest", type=Path, default=Path("results/summary/trackio_manifest.json"))
    parser.add_argument("--force-gifs", action="store_true")
    parser.add_argument("--skip-training-curves", action="store_true")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    gifs = create_gifs(video_specs(args.results_root, summary), args.gifs_dir, args.force_gifs)
    training_runs = []
    if not args.skip_training_curves:
        training_runs = log_training_curves(
            args.logs_dir, args.project, args.run_prefix, args.space_id
        )
    summary_run = log_summary(
        summary, gifs, args.summary.parent, args.project, args.run_prefix, args.space_id
    )

    trackio_dir = Path(
        os.environ.get("TRACKIO_DIR", Path.home() / ".cache" / "huggingface" / "trackio")
    )
    try:
        trackio_dir_display = trackio_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        trackio_dir_display = trackio_dir
    manifest = {
        "project": args.project,
        "space_id": args.space_id,
        "trackio_dir": str(trackio_dir_display),
        "summary_run": summary_run,
        "training_runs": training_runs,
        "gifs": [
            {
                "task_id": item["task_id"],
                "demo_budget": item["budget"],
                "success_rate": item["success_rate"],
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
