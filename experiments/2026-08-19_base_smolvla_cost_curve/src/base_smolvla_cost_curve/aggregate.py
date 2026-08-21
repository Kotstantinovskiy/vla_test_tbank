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

# No k=0 point for plain smolvla_base: the LIBERO state/action projections
# are (re-)initialized only at fine-tune time, so an untrained evaluation
# would act through random projections (documented in constants).
#
# Frozen reference: the SAME recipe run from the libero_90 pretrain
# (2026-08-18_pretrain_smolvla_naive_baseline_few_shot_tune + _low_k,
# single seed, identical demos/eval).  The gap between the two curves is the
# value of the in-domain pretrain, expressed in demonstrations.
PRETRAINED_CURVE_REFERENCE = {
    "experiments": [
        "2026-08-18_pretrain_smolvla_few_shot_tune_low_k",
        "2026-08-18_pretrain_smolvla_naive_baseline_few_shot_tune",
    ],
    "mean_all_10": {"1": 0.55, "2": 0.705, "3": 0.78, "5": 0.83, "10": 0.875, "25": 0.85},
    "mean_tasks_0_2": {"1": 0.567, "2": 0.717, "3": 0.717, "5": 0.95, "10": 0.95, "25": 0.90},
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
        "experiment": "base_smolvla_cost_curve",
        "base_checkpoint": {"path": str(BASE_CHECKPOINT), "provenance": BASE_PROVENANCE},
        "protocol": {
            "suite": TARGET_SUITE,
            "seed": MASTER_SEED,
            "eval_episodes": EVAL_EPISODES,
            "budgets": list(DEMO_BUDGETS),
        },
        "pretrained_curve_reference": PRETRAINED_CURVE_REFERENCE,
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
    ks = list(DEMO_BUDGETS)
    for task_id, task in sorted(summary["tasks"].items(), key=lambda x: int(x[0])):
        values = [task["budgets"][str(k)]["success_rate"] for k in DEMO_BUDGETS]
        ax_tasks.plot(ks, values, marker="o", alpha=0.7, label=f"task {task_id}")
    ax_tasks.set(
        xlabel="demonstrations k", ylabel="success rate", ylim=(-0.03, 1.03),
        title="Per-task cost curves (plain smolvla_base, no k=0 by design)",
    )
    ax_tasks.grid(alpha=0.25)
    ax_tasks.legend(fontsize=8, ncol=2)

    reference = summary["pretrained_curve_reference"]
    mean_all = [summary["cost_curve"]["mean_all_10"][str(k)] for k in DEMO_BUDGETS]
    mean_02 = [summary["cost_curve"]["mean_tasks_0_2"][str(k)] for k in DEMO_BUDGETS]
    ax_mean.plot(ks, mean_all, marker="o", linewidth=2, color="#3264a8",
                 label="base: mean over 10 tasks")
    ax_mean.plot(ks, mean_02, marker="s", linewidth=2, color="#d0802d",
                 label="base: mean over tasks 0-2")
    ax_mean.plot(ks, [reference["mean_all_10"][str(k)] for k in DEMO_BUDGETS],
                 marker="o", linewidth=1.5, linestyle="--", color="#3264a8", alpha=0.5,
                 label="pretrained (frozen ref): 10 tasks")
    ax_mean.plot(ks, [reference["mean_tasks_0_2"][str(k)] for k in DEMO_BUDGETS],
                 marker="s", linewidth=1.5, linestyle="--", color="#d0802d", alpha=0.5,
                 label="pretrained (frozen ref): tasks 0-2")
    ax_mean.set(
        xlabel="demonstrations k", ylabel="success rate", ylim=(-0.03, 1.03),
        title="Value of the libero_90 pretrain: base vs pretrained curves",
    )
    ax_mean.grid(alpha=0.25)
    ax_mean.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cost_curve.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Cost curve from PLAIN smolvla_base (no libero_90 pretrain)",
        "",
        "Ablation of the in-domain pretrain: every task/budget adaptation starts",
        "from schema-adapted lerobot/smolvla_base (community SO-100 pretraining,",
        "never saw LIBERO/Franka) with the byte-identical recipe of the",
        "pretrained cost curve (expert-only, 2000 steps, batch 32, official",
        "demo_0..demo_{k-1}, same eval). The gap to the frozen pretrained-curve",
        "reference is the value of the libero_90 pretrain in demonstrations.",
        "",
        "No k=0 point by design: LIBERO state/action projections are initialized",
        "only at fine-tune time (an untrained eval would act through random",
        "projections).",
        "",
        "| task | instruction | " + " | ".join(f"k={k}" for k in DEMO_BUDGETS) + " |",
        "|---:|---|" + "---:|" * len(DEMO_BUDGETS),
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
            "Frozen pretrained-curve reference (same recipe from the libero_90 "
            "pretrain): mean-10 "
            + ", ".join(
                f"k={k}: {summary['pretrained_curve_reference']['mean_all_10'][str(k)]:.3f}"
                for k in DEMO_BUDGETS
            )
            + ".",
            "",
            "| mean | " + " | ".join(f"k={k}" for k in DEMO_BUDGETS) + " |",
            "|---|" + "---:|" * len(DEMO_BUDGETS),
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
