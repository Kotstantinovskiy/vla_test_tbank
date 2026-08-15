from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .constants import (
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    MASTER_SEED,
    PROMPT_CONDITIONS,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
)


def wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_episodes(task: dict[str, Any]) -> dict[str, Any]:
    outcomes = [bool(episode["success"]) for episode in task["per_episode"]]
    if not outcomes:
        raise ValueError("Evaluation result contains no episodes")
    successes = sum(outcomes)
    low, high = wilson_interval(successes, len(outcomes))
    return {
        "successes": successes,
        "trials": len(outcomes),
        "success_rate": successes / len(outcomes),
        "ci95_low": low,
        "ci95_high": high,
    }


def aggregate(results_root: Path) -> dict[str, Any]:
    raw = {
        condition: json.loads((results_root / f"{condition}.json").read_text())
        for condition in PROMPT_CONDITIONS
    }
    for condition, result in raw.items():
        if result["condition"] != condition:
            raise ValueError(f"Condition mismatch in {condition}.json")
        if result["revision"] != CHECKPOINT_REVISION:
            raise ValueError(f"Checkpoint revision mismatch in {condition}.json")

    summary: dict[str, Any] = {
        "experiment": "smolvla_prompt_only",
        "checkpoint": {
            "repo_id": CHECKPOINT_REPO,
            "revision": CHECKPOINT_REVISION,
        },
        "protocol": {
            "suite": TARGET_SUITE,
            "seed": MASTER_SEED,
            "target_demonstrations": 0,
            "optimizer_steps": 0,
        },
        "tasks": {},
    }
    for task_id, instruction in TARGET_INSTRUCTIONS.items():
        summary["tasks"][str(task_id)] = {
            "instruction": instruction,
            "conditions": {
                condition: summarize_episodes(result["tasks"][str(task_id)])
                for condition, result in raw.items()
            },
        }

    summary["condition_means"] = {
        condition: sum(
            task["conditions"][condition]["success_rate"]
            for task in summary["tasks"].values()
        )
        / len(summary["tasks"])
        for condition in PROMPT_CONDITIONS
    }
    return summary


def metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task_id, task in summary["tasks"].items():
        for condition in PROMPT_CONDITIONS:
            metrics = task["conditions"][condition]
            rows.append(
                {
                    "task_id": int(task_id),
                    "instruction": task["instruction"],
                    "condition": condition,
                    **metrics,
                }
            )
    return rows


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = metric_rows(summary)
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    task_ids = list(TARGET_INSTRUCTIONS)
    width = 0.24
    x_positions = list(range(len(task_ids)))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = {"true": "#3264a8", "wrong": "#d0802d", "nonsense": "#888888"}
    for offset, condition in enumerate(PROMPT_CONDITIONS, start=-1):
        values = [
            summary["tasks"][str(task_id)]["conditions"][condition]["success_rate"]
            for task_id in task_ids
        ]
        ax.bar(
            [position + offset * width for position in x_positions],
            values,
            width=width,
            label=condition,
            color=colors[condition],
        )
    ax.set(
        xlabel="LIBERO goal task ID",
        ylabel="Success rate",
        ylim=(0.0, 1.03),
        title="Prompt-only SmolVLA evaluation",
    )
    ax.set_xticks(x_positions, [str(task_id) for task_id in task_ids])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Prompt")
    fig.tight_layout()
    fig.savefig(output_dir / "prompt_controls.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate prompt-only rollouts into JSON, CSV, and a plot"
    )
    parser.add_argument("--results-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/summary"))
    args = parser.parse_args()
    summary = aggregate(args.results_root)
    write_outputs(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
