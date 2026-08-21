from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import (
    BASE_CHECKPOINT,
    BASE_PROVENANCE,
    DEMO_BUDGETS,
    EVAL_EPISODES,
    MASTER_SEED,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
    experiment_root,
)

# k=0 reference from the prompt-only experiment on the same checkpoint (mean
# over the same ten tasks, same seed/episodes).  Normalization caveat: the
# k=0 rollouts used the checkpoint's pretraining statistics while every k>0
# point uses target-dataset statistics (LeRobot swaps them at fine-tune time).
ZERO_SHOT_REFERENCE = {
    "experiment": "2026-08-18_pretrain_smolvla_prompt_only",
    "mean_success": 0.005,
    "per_task_successes": {"4": 1},
}


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def aggregate(results_root: Path) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task_id, instruction in TARGET_INSTRUCTIONS.items():
        budgets = {}
        for budget in DEMO_BUDGETS:
            payload = json.loads(
                (results_root / f"task_{task_id}" / f"k_{budget}.json").read_text()
            )
            if payload["instruction"] != instruction:
                raise ValueError(f"Instruction mismatch for task {task_id} k={budget}")
            if payload["n_episodes"] != EVAL_EPISODES:
                raise ValueError(f"Episode count mismatch for task {task_id} k={budget}")
            low, high = wilson_interval(payload["successes"], payload["n_episodes"])
            budgets[str(budget)] = {
                "successes": payload["successes"],
                "trials": payload["n_episodes"],
                "success_rate": payload["success_rate"],
                "ci95_low": low,
                "ci95_high": high,
                "model": payload["model"],
            }
        tasks[str(task_id)] = {"instruction": instruction, "budgets": budgets}

    def mean_over(task_ids: list[int], budget: int) -> float:
        return sum(
            tasks[str(task_id)]["budgets"][str(budget)]["success_rate"]
            for task_id in task_ids
        ) / len(task_ids)

    all_ids = sorted(TARGET_INSTRUCTIONS)
    summary = {
        "experiment": "pretrain_smolvla_naive_baseline_few_shot_tune",
        "base_checkpoint": {"path": str(BASE_CHECKPOINT), "provenance": BASE_PROVENANCE},
        "protocol": {
            "suite": TARGET_SUITE,
            "seed": MASTER_SEED,
            "eval_episodes": EVAL_EPISODES,
            "budgets": list(DEMO_BUDGETS),
        },
        "zero_shot_reference": ZERO_SHOT_REFERENCE,
        "tasks": tasks,
        "cost_curve": {
            "mean_all_10": {str(k): mean_over(all_ids, k) for k in DEMO_BUDGETS},
            "mean_tasks_0_2": {str(k): mean_over([0, 1, 2], k) for k in DEMO_BUDGETS},
        },
    }
    return summary


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = []
    for task_id, task in summary["tasks"].items():
        for budget, metrics in task["budgets"].items():
            rows.append(
                {
                    "task_id": int(task_id),
                    "instruction": task["instruction"],
                    "k": int(budget),
                    "successes": metrics["successes"],
                    "trials": metrics["trials"],
                    "success_rate": metrics["success_rate"],
                    "ci95_low": metrics["ci95_low"],
                    "ci95_high": metrics["ci95_high"],
                }
            )
    with (output_dir / "cost_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, (ax_tasks, ax_mean) = plt.subplots(1, 2, figsize=(14, 5))
    ks = [0] + list(DEMO_BUDGETS)
    for task_id, task in sorted(summary["tasks"].items(), key=lambda x: int(x[0])):
        zero = 0.05 if task_id in summary["zero_shot_reference"]["per_task_successes"] else 0.0
        values = [zero] + [task["budgets"][str(k)]["success_rate"] for k in DEMO_BUDGETS]
        ax_tasks.plot(ks, values, marker="o", alpha=0.7, label=f"task {task_id}")
    ax_tasks.set(
        xlabel="demonstrations k", ylabel="success rate", ylim=(-0.03, 1.03),
        title="Per-task cost curves (k=0 from prompt-only reference)",
    )
    ax_tasks.grid(alpha=0.25)
    ax_tasks.legend(fontsize=8, ncol=2)

    mean_all = [summary["zero_shot_reference"]["mean_success"]] + [
        summary["cost_curve"]["mean_all_10"][str(k)] for k in DEMO_BUDGETS
    ]
    mean_02 = [1 / 60] + [
        summary["cost_curve"]["mean_tasks_0_2"][str(k)] for k in DEMO_BUDGETS
    ]
    # k=0 for tasks 0-2 was 0/60 in the prompt-only run; plot as 0.
    mean_02[0] = 0.0
    ax_mean.plot(ks, mean_all, marker="o", linewidth=2, label="mean over 10 tasks")
    ax_mean.plot(ks, mean_02, marker="s", linewidth=2, label="mean over tasks 0-2")
    ax_mean.set(
        xlabel="demonstrations k", ylabel="success rate", ylim=(-0.03, 1.03),
        title="Cost curve: success vs demonstrations",
    )
    ax_mean.grid(alpha=0.25)
    ax_mean.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cost_curve.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Few-shot cost curve on the official-data pretrain",
        "",
        "Every task/budget adaptation starts from the frozen pretrain checkpoint",
        "independently (expert-only: vision encoder and VLM frozen; action expert,",
        "state/action projections trainable). Demos are the official",
        "demo_0..demo_{k-1} of each task from the in-repo libero_goal conversion.",
        "",
        "Normalization disclosure: k>0 points run under target-dataset statistics",
        "(LeRobot swaps normalizer stats at fine-tune time), while the k=0",
        "reference ran under pretraining statistics; both come from the same",
        "conversion pipeline.",
        "",
        "| task | instruction | k=5 | k=10 | k=25 |",
        "|---:|---|---:|---:|---:|",
    ]
    for task_id, task in sorted(summary["tasks"].items(), key=lambda x: int(x[0])):
        cells = [
            f"{task['budgets'][str(k)]['successes']}/{task['budgets'][str(k)]['trials']}"
            f" ({task['budgets'][str(k)]['success_rate']:.2f})"
            for k in DEMO_BUDGETS
        ]
        lines.append(f"| {task_id} | `{task['instruction']}` | " + " | ".join(cells) + " |")
    curve = summary["cost_curve"]
    lines.extend(
        [
            "",
            "## Cost curve",
            "",
            f"k=0 (prompt-only reference): {summary['zero_shot_reference']['mean_success']:.3f} over 10 tasks.",
            "",
            "| mean | k=5 | k=10 | k=25 |",
            "|---|---:|---:|---:|",
            "| all 10 tasks | "
            + " | ".join(f"{curve['mean_all_10'][str(k)]:.3f}" for k in DEMO_BUDGETS)
            + " |",
            "| tasks 0-2 (assignment) | "
            + " | ".join(f"{curve['mean_tasks_0_2'][str(k)]:.3f}" for k in DEMO_BUDGETS)
            + " |",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Aggregate few-shot cost curve")
    parser.add_argument("--results-root", type=Path, default=root / "results/raw")
    parser.add_argument("--output-dir", type=Path, default=root / "results/summary")
    parser.add_argument("--report", type=Path, default=root / "reports/REPORT.md")
    args = parser.parse_args()
    summary = aggregate(args.results_root)
    write_outputs(summary, args.output_dir)
    write_report(summary, args.report)
    print(json.dumps(summary["cost_curve"], indent=2))


if __name__ == "__main__":
    main()
