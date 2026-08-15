from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .constants import (
    ACTION_STEPS,
    ADAPTED_BUDGETS,
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    DEMO_BUDGETS,
    EVAL_BATCH_SIZE,
    MASTER_SEED,
    N_EVAL_EPISODES,
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


def summarize_point(point: dict[str, Any]) -> dict[str, Any]:
    episodes = point.get("per_episode", [])
    if not episodes:
        raise ValueError("Evaluation point contains no episodes")
    successes = sum(bool(episode["success"]) for episode in episodes)
    trials = len(episodes)
    low, high = wilson_interval(successes, trials)
    return {
        "successes": successes,
        "trials": trials,
        "success_rate": successes / trials,
        "ci95_low": low,
        "ci95_high": high,
        "eval_seconds": point.get("aggregated", {}).get("eval_s"),
        "video_paths": [str(path) for path in point.get("video_paths", [])],
    }


def result_path(results_root: Path, task_id: int, budget: int) -> Path:
    if budget == 0:
        return results_root / "zero_shot" / f"task_{task_id}.json"
    return results_root / "adapted" / f"task_{task_id}" / f"k_{budget}.json"


def _validate_result(
    result: dict[str, Any], path: Path, task_id: int, budget: int, condition: str
) -> None:
    expected = {
        "task_id": task_id,
        "demo_budget": budget,
        "condition": condition,
        "seed": MASTER_SEED,
        "n_episodes": N_EVAL_EPISODES,
        "batch_size": EVAL_BATCH_SIZE,
        "suite": TARGET_SUITE,
        "weights_modified": False,
        "chunk_size": 50,
        "checkpoint_n_action_steps": 50,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"Protocol mismatch in {path}: {key}={result.get(key)!r}, "
                f"expected {value!r}"
            )
    missing = [str(step) for step in ACTION_STEPS if str(step) not in result["sweep"]]
    if missing:
        raise ValueError(f"Missing action-step points in {path}: {missing}")
    expected_seeds = list(range(MASTER_SEED, MASTER_SEED + N_EVAL_EPISODES))
    for step in ACTION_STEPS:
        point = result["sweep"][str(step)]
        if point.get("n_action_steps") != step:
            raise ValueError(f"Action-step label mismatch in {path}: {step}")
        seeds = [episode.get("seed") for episode in point.get("per_episode", [])]
        if seeds != expected_seeds:
            raise ValueError(f"Episode seed mismatch in {path} at n_action_steps={step}")
    if budget == 0 and result.get("revision") != CHECKPOINT_REVISION:
        raise ValueError(f"Zero-shot revision mismatch in {path}")


def load_primary(results_root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    raw: dict[tuple[int, int], dict[str, Any]] = {}
    for task_id in TARGET_INSTRUCTIONS:
        for budget in DEMO_BUDGETS:
            path = result_path(results_root, task_id, budget)
            result = json.loads(path.read_text())
            _validate_result(result, path, task_id, budget, "true")
            raw[(task_id, budget)] = result
    return raw


def load_controls(results_root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    controls: dict[str, dict[int, dict[str, Any]]] = {}
    for condition in ("wrong", "nonsense"):
        condition_results: dict[int, dict[str, Any]] = {}
        for task_id in TARGET_INSTRUCTIONS:
            path = (
                results_root
                / "zero_shot_controls"
                / condition
                / f"task_{task_id}.json"
            )
            if not path.exists():
                condition_results = {}
                break
            result = json.loads(path.read_text())
            _validate_result(result, path, task_id, 0, condition)
            condition_results[task_id] = result
        if condition_results:
            controls[condition] = condition_results
    return controls


def _best_steps(points: dict[str, dict[str, Any]]) -> tuple[list[int], int]:
    best_rate = max(point["success_rate"] for point in points.values())
    tied = sorted(
        int(step)
        for step, point in points.items()
        if point["success_rate"] == best_rate
    )
    return tied, tied[0]


def aggregate(
    results_root: Path, baseline_reference: Path
) -> dict[str, Any]:
    raw = load_primary(results_root)
    frozen = json.loads(baseline_reference.read_text())
    summary: dict[str, Any] = {
        "experiment": "smolvla_action_steps_sweep",
        "intervention": "inference_only_n_action_steps",
        "weights_modified": False,
        "zero_shot_checkpoint": {
            "repo_id": CHECKPOINT_REPO,
            "revision": CHECKPOINT_REVISION,
        },
        "protocol": {
            "suite": TARGET_SUITE,
            "seed": MASTER_SEED,
            "episodes_per_point": N_EVAL_EPISODES,
            "total_primary_points": (
                len(TARGET_INSTRUCTIONS) * len(DEMO_BUDGETS) * len(ACTION_STEPS)
            ),
            "total_primary_episodes": (
                len(TARGET_INSTRUCTIONS)
                * len(DEMO_BUDGETS)
                * len(ACTION_STEPS)
                * N_EVAL_EPISODES
            ),
            "action_steps": list(ACTION_STEPS),
            "demo_budgets": list(DEMO_BUDGETS),
            "paired_initial_states": True,
            "tie_break": "report all ties; select smallest n_action_steps",
        },
        "frozen_baseline_reference": frozen,
        "tasks": {},
    }
    for task_id, instruction in TARGET_INSTRUCTIONS.items():
        task_summary: dict[str, Any] = {
            "instruction": instruction,
            "budgets": {},
        }
        for budget in DEMO_BUDGETS:
            result = raw[(task_id, budget)]
            points = {
                str(step): summarize_point(result["sweep"][str(step)])
                for step in ACTION_STEPS
            }
            anchor_rate = points["50"]["success_rate"]
            frozen_rate = frozen["task_success_rates"][str(task_id)][str(budget)]
            for point in points.values():
                point["delta_vs_paired_50"] = point["success_rate"] - anchor_rate
                point["delta_vs_frozen_baseline"] = (
                    point["success_rate"] - frozen_rate
                )
            ties, selected = _best_steps(points)
            task_summary["budgets"][str(budget)] = {
                "frozen_baseline_success_rate": frozen_rate,
                "rerun_50_success_rate": anchor_rate,
                "rerun_50_minus_frozen": anchor_rate - frozen_rate,
                "best_action_steps": ties,
                "selected_best_action_steps": selected,
                "points": points,
            }
        summary["tasks"][str(task_id)] = task_summary

    means: dict[str, dict[str, float]] = {}
    for budget in DEMO_BUDGETS:
        means[str(budget)] = {}
        for step in ACTION_STEPS:
            means[str(budget)][str(step)] = sum(
                summary["tasks"][str(task_id)]["budgets"][str(budget)]["points"][
                    str(step)
                ]["success_rate"]
                for task_id in TARGET_INSTRUCTIONS
            ) / len(TARGET_INSTRUCTIONS)
    summary["mean_success_by_budget_and_action_steps"] = means
    summary["mean_delta_vs_paired_50"] = {
        str(budget): {
            str(step): means[str(budget)][str(step)] - means[str(budget)]["50"]
            for step in ACTION_STEPS
        }
        for budget in DEMO_BUDGETS
    }
    summary["mean_delta_vs_frozen_baseline"] = {
        str(budget): {
            str(step): means[str(budget)][str(step)]
            - frozen["mean_cost_curve"][str(budget)]
            for step in ACTION_STEPS
        }
        for budget in DEMO_BUDGETS
    }

    controls = load_controls(results_root)
    summary["zero_shot_any_true_success"] = any(
        summary["tasks"][str(task_id)]["budgets"]["0"]["points"][str(step)][
            "successes"
        ]
        > 0
        for task_id in TARGET_INSTRUCTIONS
        for step in ACTION_STEPS
    )
    summary["language_controls"] = {
        "required": summary["zero_shot_any_true_success"],
        "status": (
            "complete"
            if controls
            else "pending"
            if summary["zero_shot_any_true_success"]
            else "skipped_floor"
        ),
        "conditions": {},
    }
    for condition, task_results in controls.items():
        summary["language_controls"]["conditions"][condition] = {
            str(task_id): {
                str(step): summarize_point(task_results[task_id]["sweep"][str(step)])
                for step in ACTION_STEPS
            }
            for task_id in TARGET_INSTRUCTIONS
        }
    return summary


def metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id, task in summary["tasks"].items():
        for budget, budget_result in task["budgets"].items():
            for step, point in budget_result["points"].items():
                rows.append(
                    {
                        "task_id": int(task_id),
                        "instruction": task["instruction"],
                        "demo_budget": int(budget),
                        "n_action_steps": int(step),
                        "successes": point["successes"],
                        "trials": point["trials"],
                        "success_rate": point["success_rate"],
                        "ci95_low": point["ci95_low"],
                        "ci95_high": point["ci95_high"],
                        "frozen_baseline_success_rate": budget_result[
                            "frozen_baseline_success_rate"
                        ],
                        "delta_vs_paired_50": point["delta_vs_paired_50"],
                        "delta_vs_frozen_baseline": point[
                            "delta_vs_frozen_baseline"
                        ],
                    }
                )
    return rows


def control_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, tasks in summary["language_controls"]["conditions"].items():
        for task_id, points in tasks.items():
            for step, metrics in points.items():
                rows.append(
                    {
                        "condition": condition,
                        "task_id": int(task_id),
                        "n_action_steps": int(step),
                        "successes": metrics["successes"],
                        "trials": metrics["trials"],
                        "success_rate": metrics["success_rate"],
                        "ci95_low": metrics["ci95_low"],
                        "ci95_high": metrics["ci95_high"],
                    }
                )
    return rows


def _plot_cost_curves(summary: dict[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = plt.cm.viridis_r([0.05, 0.25, 0.5, 0.72, 0.92])
    for color, step in zip(colors, ACTION_STEPS):
        ax.plot(
            DEMO_BUDGETS,
            [
                summary["mean_success_by_budget_and_action_steps"][str(budget)][
                    str(step)
                ]
                for budget in DEMO_BUDGETS
            ],
            marker="o",
            linewidth=2,
            color=color,
            label=f"n_action_steps={step}",
        )
    ax.plot(
        DEMO_BUDGETS,
        [
            summary["frozen_baseline_reference"]["mean_cost_curve"][str(budget)]
            for budget in DEMO_BUDGETS
        ],
        linestyle="--",
        color="#777777",
        linewidth=1.8,
        label="frozen reported baseline",
    )
    ax.set(
        xlabel="Target-task demonstrations",
        ylabel="Mean success across tasks",
        ylim=(-0.03, 1.03),
        title="Cost curves with frozen weights and new inference horizons",
    )
    ax.set_xticks(DEMO_BUDGETS)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_action_steps(summary: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), sharey=True)
    for task_id, ax in zip(TARGET_INSTRUCTIONS, axes):
        for budget in DEMO_BUDGETS:
            ax.plot(
                ACTION_STEPS,
                [
                    summary["tasks"][str(task_id)]["budgets"][str(budget)][
                        "points"
                    ][str(step)]["success_rate"]
                    for step in ACTION_STEPS
                ],
                marker="o",
                label=f"k={budget}",
            )
        ax.set_xscale("log")
        ax.set_xticks(ACTION_STEPS, [str(step) for step in ACTION_STEPS])
        ax.set_title(f"Task {task_id}")
        ax.set_xlabel("n_action_steps")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[-1].legend(title="Demos")
    fig.suptitle("Replanning sweep by held-out task")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, Any], path: Path) -> None:
    means = summary["mean_success_by_budget_and_action_steps"]
    adapted_mean_by_step = {
        step: sum(means[str(budget)][str(step)] for budget in ADAPTED_BUDGETS)
        / len(ADAPTED_BUDGETS)
        for step in ACTION_STEPS
    }
    best_fixed_step = max(
        ACTION_STEPS, key=lambda step: (adapted_mean_by_step[step], -step)
    )
    lines = [
        "# Action-step sweep report",
        "",
        "This report is generated from the completed rollout JSON files. The original",
        "Task 1 baseline is frozen; every new point uses the same checkpoint weights and",
        "changes only `n_action_steps` at inference.",
        "",
        "## Mean success",
        "",
        "| demos | " + " | ".join(f"n={step}" for step in ACTION_STEPS) + " |",
        "|---:" + "|---:" * len(ACTION_STEPS) + "|",
    ]
    for budget in DEMO_BUDGETS:
        lines.append(
            f"| {budget} | "
            + " | ".join(f"{means[str(budget)][str(step)]:.3f}" for step in ACTION_STEPS)
            + " |"
        )
    lines.extend(["", "## Best horizons by task and budget", ""])
    lines.append("| task | demos | best n_action_steps | success | delta vs paired n=50 |")
    lines.append("|---:|---:|---|---:|---:|")
    for task_id, task in summary["tasks"].items():
        for budget, result in task["budgets"].items():
            selected = result["selected_best_action_steps"]
            point = result["points"][str(selected)]
            ties = ", ".join(map(str, result["best_action_steps"]))
            lines.append(
                f"| {task_id} | {budget} | {ties} | {point['success_rate']:.3f} | "
                f"{point['delta_vs_paired_50']:+.3f} |"
            )
    task_0 = summary["tasks"]["0"]["budgets"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The prediction of the largest gain on task 1 was not supported. Tasks 1 and 2 "
            "remained at zero for every budget and horizon; only the already learned drawer "
            "task 0 responded to the inference change. This suggests that tasks 1 and 2 fail "
            "before open-loop compounding error becomes the main bottleneck.",
            "",
            f"For task 0, k=5 peaks at n=10 ({task_0['5']['points']['10']['success_rate']:.2f}, "
            f"{task_0['5']['points']['10']['delta_vs_paired_50']:+.2f} versus paired n=50), "
            f"k=10 peaks at n=25 ({task_0['10']['points']['25']['success_rate']:.2f}, "
            f"{task_0['10']['points']['25']['delta_vs_paired_50']:+.2f}), while k=25 is best "
            f"at n=50 ({task_0['25']['points']['50']['success_rate']:.2f}). The effect is "
            "therefore not uniform across k, and n=1 is never uniquely best.",
            "",
            f"Across the three adapted budgets, the best single fixed horizon is n={best_fixed_step} "
            f"with mean success {adapted_mean_by_step[best_fixed_step]:.3f}, versus "
            f"{adapted_mean_by_step[50]:.3f} for paired n=50. This aggregate gain is driven "
            "entirely by task 0 and should not be presented as recovery of cross-task "
            "generalization.",
            "",
            "The paired n=50 rerun differs from the frozen historical result at task 0, k=5 "
            f"({task_0['5']['rerun_50_success_rate']:.2f} versus "
            f"{task_0['5']['frozen_baseline_success_rate']:.2f}). The new sweep resets the "
            "policy RNG immediately before every horizon to pair flow-matching samples; the "
            "historical evaluator did not use that exact RNG protocol. Consequently, delta "
            "versus paired n=50 is the primary inference estimate, while delta versus frozen "
            "baseline is reported separately for protocol transparency.",
        ]
    )
    if summary["zero_shot_any_true_success"]:
        control_note = (
            "At least one true-prompt zero-shot rollout succeeded, so wrong and nonsense "
            "prompt controls are required by the locked protocol."
        )
    else:
        control_note = (
            "All true-prompt zero-shot points stayed at the success floor; the locked "
            "protocol therefore skips additional language controls."
        )
    lines.extend(["", "## Language-control gate", "", control_note, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def write_outputs(summary: dict[str, Any], output_dir: Path, report: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = metric_rows(summary)
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    controls = control_rows(summary)
    if controls:
        with (output_dir / "language_controls.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(controls[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(controls)
    _plot_cost_curves(summary, output_dir / "cost_curves_by_action_steps.png")
    _plot_action_steps(summary, output_dir / "action_steps_by_task.png")
    write_report(summary, report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the action-step sweep")
    parser.add_argument("--results-root", type=Path, default=Path("results/raw"))
    parser.add_argument(
        "--baseline-reference",
        type=Path,
        default=Path("artifacts/frozen_baseline_reference.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/summary"))
    parser.add_argument("--report", type=Path, default=Path("reports/REPORT.md"))
    args = parser.parse_args()
    summary = aggregate(args.results_root, args.baseline_reference)
    write_outputs(summary, args.output_dir, args.report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
