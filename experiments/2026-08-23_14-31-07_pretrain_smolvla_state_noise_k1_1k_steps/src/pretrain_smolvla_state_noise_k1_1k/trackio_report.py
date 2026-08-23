from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image

from .constants import (
    ALPHAS,
    BASE_CHECKPOINT,
    BASE_PROVENANCE,
    EVAL_ACTION_STEPS,
    EXPERIMENT_NAME,
    MASTER_SEED,
    TARGET_INSTRUCTIONS,
    TRACKIO_PROJECT,
    alpha_tag,
    experiment_root,
)

DEFAULT_RUN_NAME = EXPERIMENT_NAME.replace("_", "-")


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
    """Per alpha arm: GIF of the first success and the first failure."""

    action_steps = EVAL_ACTION_STEPS[0]
    items: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        tag = alpha_tag(alpha)
        chosen: dict[str, dict[str, Any]] = {}
        for task_id in sorted(TARGET_INSTRUCTIONS):
            payload_path = (
                results_root
                / f"task_{task_id}"
                / f"alpha_{tag}"
                / f"n_{action_steps}.json"
            )
            if not payload_path.is_file():
                continue
            payload = json.loads(payload_path.read_text())
            for episode_index, episode in enumerate(payload["per_episode"]):
                outcome = episode["outcome"]
                if outcome in chosen:
                    continue
                chosen[outcome] = {
                    "alpha": alpha,
                    "alpha_tag": tag,
                    "outcome": outcome,
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "env_seed": episode.get("env_seed"),
                    "noise_seed": episode.get("noise_seed"),
                    "video": episode["video_path"],
                }
            if {"success", "failure"} <= set(chosen):
                break
        for outcome, item in sorted(chosen.items()):
            destination = gifs_dir / f"alpha_{tag}_first_{outcome}.gif"
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
            "eval_batch_size": 1,
            "policy_noise_seeded_per_episode": True,
            "demo_budget": 1,
            "state_noise_alphas": list(ALPHAS),
            "eval_action_steps": list(EVAL_ACTION_STEPS),
        },
        "auto_log_gpu": False,
        "auto_log_cpu": False,
    }
    if space_id:
        init_args["space_id"] = space_id
    trackio.init(**init_args)
    try:
        for index, alpha in enumerate(ALPHAS):
            tag = alpha_tag(alpha)
            payload_step = {
                "alpha": alpha,
                "success/mean_tasks_0_2": summary["noise_curve"]["mean_tasks_0_2"][tag],
            }
            for task_id, task in summary["tasks"].items():
                payload_step[f"success/task_{task_id}"] = task["alphas"][tag]["success_rate"]
            trackio.log(payload_step, step=index)
        columns = [
            "task_id", "instruction", "alpha", "successes", "trials",
            "success_rate", "ci95_low", "ci95_high",
        ]
        rows = [
            [
                int(task_id),
                task["instruction"],
                metrics["alpha"],
                metrics["successes"],
                metrics["trials"],
                metrics["success_rate"],
                metrics["ci95_low"],
                metrics["ci95_high"],
            ]
            for task_id, task in summary["tasks"].items()
            for metrics in task["alphas"].values()
        ]
        payload: dict[str, Any] = {
            "tables/noise_curve": trackio.Table(columns=columns, data=rows),
            "plots/noise_curve": trackio.Image(plot, caption="Success vs state-noise alpha"),
            "reports/summary": trackio.Markdown(
                "# Proprioception-noise sweep (k=1, full FT)\n\n"
                "Training-only Gaussian noise alpha*eps on the normalized state; "
                "alpha=0.00 is the in-experiment control. Inference n_action_steps=50. "
                "Full per-episode videos are stored on disk; this run logs the first "
                "success and first failure per alpha."
            ),
        }
        for item in gifs:
            stem = f"alpha_{item['alpha_tag']}_first_{item['outcome']}"
            payload[f"rollouts/{stem}"] = trackio.Video(
                Path(item["video"]),
                caption=(
                    f"alpha={item['alpha_tag']}: first {item['outcome']} "
                    f"(task {item['task_id']}, episode {item['episode_index']})"
                ),
            )
            payload[f"rollout_gifs/{stem}"] = trackio.Image(
                Path(item["gif"]),
                caption=f"alpha={item['alpha_tag']} first {item['outcome']} GIF",
            )
        trackio.log(payload, step=len(ALPHAS) - 1)
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
    parser.add_argument("--plot", type=Path, default=root / "results/summary/noise_curve.png")
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
        "table": "tables/noise_curve",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
