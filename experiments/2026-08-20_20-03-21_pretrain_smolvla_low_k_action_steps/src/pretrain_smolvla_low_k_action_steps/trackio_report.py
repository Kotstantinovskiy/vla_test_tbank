from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .constants import (
    ACTION_STEPS,
    BASE_CHECKPOINT,
    BASE_PROVENANCE,
    DEMO_BUDGETS,
    MASTER_SEED,
    TARGET_INSTRUCTIONS,
    TRACKIO_PROJECT,
    experiment_root,
    result_path,
)

DEFAULT_RUN_NAME = "2026-08-20-20-03-21-low-k-action-steps"


def first_outcome_media(results_root: Path) -> list[dict[str, Any]]:
    """Choose first success and first failure for every (k, action_steps)."""

    items: list[dict[str, Any]] = []
    for budget in DEMO_BUDGETS:
        for action_steps in ACTION_STEPS:
            chosen: dict[str, dict[str, Any]] = {}
            for task_id in sorted(TARGET_INSTRUCTIONS):
                path = result_path(results_root, task_id, budget, action_steps)
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text())
                for episode in payload["per_episode"]:
                    outcome = episode["outcome"]
                    if outcome in chosen:
                        continue
                    video = Path(episode["video_path"])
                    if not video.is_file():
                        raise FileNotFoundError(video)
                    chosen[outcome] = {
                        "demo_budget": budget,
                        "n_action_steps": action_steps,
                        "outcome": outcome,
                        "task_id": task_id,
                        "episode_index": episode["episode_ix"],
                        "env_seed": episode["env_seed"],
                        "noise_seed": episode["noise_seed"],
                        "video": str(video),
                    }
                if {"success", "failure"} <= set(chosen):
                    break
            items.extend(chosen[key] for key in sorted(chosen))
    return items


def log_to_trackio(
    summary: dict[str, Any],
    media: list[dict[str, Any]],
    plot: Path,
    report: Path,
    project: str,
    run_name: str,
    space_id: str | None,
) -> None:
    import trackio

    init_args: dict[str, Any] = {
        "project": project,
        "name": run_name,
        "group": "low-k-action-step-screen",
        "config": {
            "base_checkpoint": str(BASE_CHECKPOINT),
            **BASE_PROVENANCE,
            "training_seed": MASTER_SEED,
            "demo_budgets": list(DEMO_BUDGETS),
            "action_steps": list(ACTION_STEPS),
            "eval_batch_size": 1,
            "policy_noise_seeded_per_episode": True,
        },
        "auto_log_gpu": False,
        "auto_log_cpu": False,
    }
    if space_id:
        init_args["space_id"] = space_id
    trackio.init(**init_args)
    try:
        step = 0
        for budget in DEMO_BUDGETS:
            for action_steps in ACTION_STEPS:
                trackio.log(
                    {
                        f"success/mean_all_10/n_{action_steps}": summary["means"][
                            "mean_all_10"
                        ][str(budget)][str(action_steps)],
                        f"success/mean_tasks_0_2/n_{action_steps}": summary[
                            "means"
                        ]["mean_tasks_0_2"][str(budget)][str(action_steps)],
                        "demo_budget": budget,
                        "n_action_steps": action_steps,
                    },
                    step=step,
                )
                step += 1
        columns = [
            "task_id",
            "instruction",
            "k",
            "n_action_steps",
            "successes",
            "trials",
            "success_rate",
            "ci95_low",
            "ci95_high",
        ]
        rows = []
        for task_id, task in summary["tasks"].items():
            for budget, budget_data in task["budgets"].items():
                for action_steps, metrics in budget_data["action_steps"].items():
                    rows.append(
                        [
                            int(task_id),
                            task["instruction"],
                            int(budget),
                            int(action_steps),
                            metrics["successes"],
                            metrics["trials"],
                            metrics["success_rate"],
                            metrics["ci95_low"],
                            metrics["ci95_high"],
                        ]
                    )
        payload: dict[str, Any] = {
            "tables/success_rates": trackio.Table(columns=columns, data=rows),
            "plots/action_steps_low_k": trackio.Image(
                plot, caption="Low-k success by action replanning interval"
            ),
            "reports/report": trackio.Markdown(report.read_text()),
        }
        for item in media:
            key = (
                f"rollouts/k_{item['demo_budget']}/n_{item['n_action_steps']}/"
                f"first_{item['outcome']}"
            )
            payload[key] = trackio.Video(
                Path(item["video"]),
                caption=(
                    f"k={item['demo_budget']}, n={item['n_action_steps']}, "
                    f"first {item['outcome']}: task {item['task_id']}, "
                    f"episode {item['episode_index']}"
                ),
            )
        trackio.log(payload, step=step)
    finally:
        trackio.finish()


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Log action-step screen to Trackio")
    parser.add_argument(
        "--project", default=os.environ.get("TRACKIO_PROJECT", TRACKIO_PROJECT)
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--space-id", default=os.environ.get("TRACKIO_SPACE_ID"))
    parser.add_argument(
        "--summary", type=Path, default=root / "results/summary/summary.json"
    )
    parser.add_argument("--results-root", type=Path, default=root / "results/raw")
    parser.add_argument(
        "--plot", type=Path, default=root / "results/summary/action_steps_low_k.png"
    )
    parser.add_argument("--report", type=Path, default=root / "reports/REPORT.md")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "results/summary/trackio_manifest.json",
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    media = first_outcome_media(args.results_root)
    log_to_trackio(
        summary,
        media,
        args.plot,
        args.report,
        args.project,
        args.run_name,
        args.space_id,
    )
    manifest = {
        "project": args.project,
        "run": args.run_name,
        "table": "tables/success_rates",
        "plot": str(args.plot),
        "media": media,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
