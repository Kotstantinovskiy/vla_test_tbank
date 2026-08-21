from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import FULL_RUN_NAME, TRACKIO_GROUP, TRACKIO_PROJECT, experiment_root
from .runner import parse_training_metrics, paths


def collect_metrics(
    log_path: Path, previous_step: int | None = None
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        parsed = parse_training_metrics(line)
        if parsed is None:
            continue
        displayed_step, metrics = parsed
        if rows:
            step = int(rows[-1]["step"]) + 50
        elif previous_step is not None:
            step = previous_step + 50
        else:
            step = displayed_step
        rows.append({"step": float(step), **metrics})
    return rows


def collect_canonical_metrics(root: Path) -> list[dict[str, float]]:
    full_rows = collect_metrics(root / "results/logs/full.log")
    manifest_path = root / "results/resume_manifest.json"
    resume_log = root / "results/logs/resume.log"
    if not manifest_path.is_file() or not resume_log.is_file():
        return full_rows
    manifest = json.loads(manifest_path.read_text())
    resume_from_step = int(manifest["resume_from_step"])
    durable_prefix = [row for row in full_rows if row["step"] <= resume_from_step]
    return durable_prefix + collect_metrics(resume_log, previous_step=resume_from_step)


def write_csv(rows: list[dict[str, float]], destination: Path) -> list[str]:
    columns = ["step"] + sorted({key for row in rows for key in row if key != "step"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return columns


def create_plot(rows: list[dict[str, float]], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], "train/loss", "Loss"),
        (axes[0, 1], "train/grad_norm", "Gradient norm"),
        (axes[1, 0], "train/learning_rate", "Learning rate"),
    )
    for axis, key, title in panels:
        axis.plot(steps, [row.get(key, float("nan")) for row in rows], linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
    timing = axes[1, 1]
    timing.plot(
        steps,
        [row.get("perf/update_seconds", float("nan")) for row in rows],
        label="update",
    )
    timing.plot(
        steps,
        [row.get("perf/dataload_seconds", float("nan")) for row in rows],
        label="dataloader",
    )
    timing.set_title("Step timing")
    timing.set_xlabel("Optimizer step")
    timing.set_ylabel("seconds")
    timing.grid(alpha=0.25)
    timing.legend()
    figure.suptitle("SmolVLA full LIBERO-90 DDP training")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def checkpoint_rows(output_root: Path) -> list[list[Any]]:
    checkpoint_root = output_root / "checkpoints"
    if not checkpoint_root.is_dir():
        return []
    rows: list[list[Any]] = []
    for checkpoint in sorted(checkpoint_root.iterdir()):
        if checkpoint.name == "last" or checkpoint.is_symlink() or not checkpoint.is_dir():
            continue
        model_dir = checkpoint / "pretrained_model"
        weights = model_dir / "model.safetensors"
        rows.append(
            [
                checkpoint.name,
                str(model_dir),
                weights.is_file(),
                weights.stat().st_size if weights.is_file() else 0,
            ]
        )
    return rows


def summarize(log_path: Path, status_path: Path, output_dir: Path) -> dict[str, Any]:
    root = experiment_root()
    rows = (
        collect_canonical_metrics(root)
        if log_path.resolve() == (root / "results/logs/full.log").resolve()
        else collect_metrics(log_path)
    )
    if not rows:
        raise ValueError(f"No training metrics found in {log_path}")
    status = json.loads(status_path.read_text())
    metrics_csv = output_dir / "training_metrics.csv"
    plot_path = output_dir / "training_curves.png"
    columns = write_csv(rows, metrics_csv)
    create_plot(rows, plot_path)
    losses = [row["train/loss"] for row in rows if "train/loss" in row]
    speeds = [
        row["perf/samples_per_second"]
        for row in rows
        if "perf/samples_per_second" in row
    ]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "state": status.get("state"),
        "exit_code": status.get("exit_code"),
        "last_step": int(rows[-1]["step"]),
        "target_steps": status.get("target_steps"),
        "logged_points": len(rows),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_logged_loss": min(losses),
        "mean_logged_samples_per_second": sum(speeds) / len(speeds),
        "columns": columns,
        "metrics_csv": str(metrics_csv),
        "plot": str(plot_path),
        "final_checkpoint": str(paths()["output"] / "checkpoints/last/pretrained_model"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def log_summary(
    summary: dict[str, Any], rows: list[dict[str, float]], output_dir: Path
) -> None:
    import trackio

    columns = ["step", "loss", "grad_norm", "learning_rate", "samples_per_second"]
    table_rows = [
        [
            int(row["step"]),
            row.get("train/loss"),
            row.get("train/grad_norm"),
            row.get("train/learning_rate"),
            row.get("perf/samples_per_second"),
        ]
        for row in rows
    ]
    checkpoints = checkpoint_rows(paths()["output"])
    trackio.init(
        project=os.environ.get("TRACKIO_PROJECT", TRACKIO_PROJECT),
        name=f"{FULL_RUN_NAME}-summary",
        group=TRACKIO_GROUP,
        config={"source_run": FULL_RUN_NAME, "postprocess": True},
        auto_log_gpu=False,
        auto_log_cpu=False,
    )
    try:
        payload: dict[str, Any] = {
            "plots/final_training_curves": trackio.Image(
                output_dir / "training_curves.png",
                caption="Loss, gradient norm, learning rate, and throughput",
            ),
            "tables/training_metrics": trackio.Table(columns=columns, data=table_rows),
            "tables/checkpoints": trackio.Table(
                columns=["step", "model_dir", "weights_present", "weights_bytes"],
                data=checkpoints,
            ),
            "reports/final_summary": trackio.Markdown(
                "# Full LIBERO-90 training\n\n"
                f"State: **{summary['state']}**; final logged step: "
                f"**{summary['last_step']}**; final loss: **{summary['final_loss']:.4f}**."
            ),
            "summary/final_loss": summary["final_loss"],
            "summary/best_logged_loss": summary["best_logged_loss"],
            "summary/mean_samples_per_second": summary[
                "mean_logged_samples_per_second"
            ],
        }
        trackio.log(payload, step=summary["last_step"])
    finally:
        trackio.finish()


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Summarize a completed LIBERO-90 run")
    parser.add_argument("--log", type=Path, default=root / "results/logs/full.log")
    parser.add_argument("--status", type=Path, default=root / "results/status.json")
    parser.add_argument("--output", type=Path, default=root / "results/summary")
    parser.add_argument("--skip-trackio", action="store_true")
    args = parser.parse_args()
    summary = summarize(args.log, args.status, args.output)
    if not args.skip_trackio:
        log_summary(summary, collect_canonical_metrics(root), args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
