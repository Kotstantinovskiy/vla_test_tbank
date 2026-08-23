from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import (
    ACTION_STEPS,
    BASE_CHECKPOINT,
    BASE_PROVENANCE,
    DEMO_BUDGETS,
    EVAL_EPISODES,
    EXPERIMENT_NAME,
    MASTER_SEED,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
    TRAINED_CHUNK_SIZE,
    experiment_root,
    result_path,
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


def mcnemar_exact_p(b: int, c: int) -> float:
    total = b + c
    if total == 0:
        return 1.0
    lower = min(b, c)
    tail = sum(math.comb(total, index) for index in range(lower + 1)) / 2**total
    return min(1.0, 2 * tail)


def _episode_keys(payload: dict) -> list[tuple[int, int, int, int]]:
    return [
        (
            item["episode_ix"],
            item["env_seed"],
            item["noise_seed"],
            item["init_state_id"],
        )
        for item in payload["per_episode"]
    ]


def aggregate(results_root: Path, prior: dict) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    payloads: dict[tuple[int, int, int], dict] = {}
    expected_seeds = list(range(MASTER_SEED, MASTER_SEED + EVAL_EPISODES))
    for task_id, instruction in TARGET_INSTRUCTIONS.items():
        budgets: dict[str, Any] = {}
        for budget in DEMO_BUDGETS:
            action_metrics: dict[str, Any] = {}
            checkpoint_hashes = set()
            for action_steps in ACTION_STEPS:
                path = result_path(results_root, task_id, budget, action_steps)
                payload = json.loads(path.read_text())
                payloads[(task_id, budget, action_steps)] = payload
                if payload["instruction"] != instruction:
                    raise ValueError(
                        f"Instruction mismatch for task {task_id}, k={budget}, "
                        f"n={action_steps}"
                    )
                if payload["demo_budget"] != budget:
                    raise ValueError(f"Budget mismatch in {path}")
                if payload["n_action_steps"] != action_steps:
                    raise ValueError(f"Action-step mismatch in {path}")
                if payload["chunk_size"] != TRAINED_CHUNK_SIZE:
                    raise ValueError(f"Chunk-size mismatch in {path}")
                if payload["n_episodes"] != EVAL_EPISODES:
                    raise ValueError(f"Episode-count mismatch in {path}")
                if payload["batch_size"] != 1:
                    raise ValueError(f"Evaluation was not batch=1 in {path}")
                if [x[1] for x in _episode_keys(payload)] != expected_seeds:
                    raise ValueError(f"Environment seed bank mismatch in {path}")
                if [x[2] for x in _episode_keys(payload)] != expected_seeds:
                    raise ValueError(f"Policy-noise seed bank mismatch in {path}")
                if [x[0] for x in _episode_keys(payload)] != list(
                    range(EVAL_EPISODES)
                ):
                    raise ValueError(f"Logical episode order mismatch in {path}")
                if [x[3] for x in _episode_keys(payload)] != list(
                    range(EVAL_EPISODES)
                ):
                    raise ValueError(f"LIBERO init-state bank mismatch in {path}")
                checkpoint_hashes.add(payload["model_safetensors_sha256"])
                low, high = wilson_interval(
                    payload["successes"], payload["n_episodes"]
                )
                action_metrics[str(action_steps)] = {
                    "successes": payload["successes"],
                    "trials": payload["n_episodes"],
                    "success_rate": payload["success_rate"],
                    "ci95_low": low,
                    "ci95_high": high,
                    "model": payload["model"],
                    "model_safetensors_sha256": payload[
                        "model_safetensors_sha256"
                    ],
                }
            if len(checkpoint_hashes) != 1:
                raise ValueError(
                    f"Action-step conditions used different weights: task={task_id}, "
                    f"k={budget}"
                )
            budgets[str(budget)] = {"action_steps": action_metrics}
        tasks[str(task_id)] = {"instruction": instruction, "budgets": budgets}

    paired = []
    for task_id in TARGET_INSTRUCTIONS:
        for budget in DEMO_BUDGETS:
            for left_n, right_n in combinations(ACTION_STEPS, 2):
                left = payloads[(task_id, budget, left_n)]
                right = payloads[(task_id, budget, right_n)]
                if _episode_keys(left) != _episode_keys(right):
                    raise ValueError(
                        f"Unpaired seed banks: task={task_id}, k={budget}, "
                        f"n={left_n}/{right_n}"
                    )
                left_outcomes = [bool(x["success"]) for x in left["per_episode"]]
                right_outcomes = [bool(x["success"]) for x in right["per_episode"]]
                only_left = sum(
                    x and not y
                    for x, y in zip(left_outcomes, right_outcomes, strict=True)
                )
                only_right = sum(
                    y and not x
                    for x, y in zip(left_outcomes, right_outcomes, strict=True)
                )
                paired.append(
                    {
                        "task_id": task_id,
                        "demo_budget": budget,
                        "left_n_action_steps": left_n,
                        "right_n_action_steps": right_n,
                        "left_successes": left["successes"],
                        "right_successes": right["successes"],
                        "delta_right_minus_left": (
                            right["success_rate"] - left["success_rate"]
                        ),
                        "discordant_only_left": only_left,
                        "discordant_only_right": only_right,
                        "mcnemar_p": mcnemar_exact_p(only_left, only_right),
                    }
                )

    def mean_over(task_ids: list[int], budget: int, action_steps: int) -> float:
        return sum(
            tasks[str(task_id)]["budgets"][str(budget)]["action_steps"][
                str(action_steps)
            ]["success_rate"]
            for task_id in task_ids
        ) / len(task_ids)

    target_ids = sorted(TARGET_INSTRUCTIONS)
    means_02 = {
        str(k): {str(n): mean_over(target_ids, k, n) for n in ACTION_STEPS}
        for k in DEMO_BUDGETS
    }
    old_50 = prior["prior_n50_low_k_rates"]
    descriptive_delta = {
        "mean_tasks_0_2": {
            str(k): {
                str(n): means_02[str(k)][str(n)]
                - old_50["mean_tasks_0_2"][str(k)]
                for n in ACTION_STEPS
            }
            for k in DEMO_BUDGETS
        },
        "warning": old_50["comparison_warning"],
    }
    return {
        "experiment": EXPERIMENT_NAME,
        "base_checkpoint": {
            "path": str(BASE_CHECKPOINT),
            "provenance": BASE_PROVENANCE,
        },
        "protocol": {
            "suite": TARGET_SUITE,
            "training_seed": MASTER_SEED,
            "eval_episodes": EVAL_EPISODES,
            "demo_budgets": list(DEMO_BUDGETS),
            "action_steps": list(ACTION_STEPS),
            "chunk_size": TRAINED_CHUNK_SIZE,
            "eval_batch_size": 1,
            "policy_noise_seeded_per_episode": True,
            "libero_init_state_pinned_per_episode": True,
        },
        "tasks": tasks,
        "means": {"mean_tasks_0_2": means_02},
        "paired_action_step_comparisons": paired,
        "prior_n50_reference": old_50,
        "descriptive_delta_vs_prior_n50": descriptive_delta,
        "limitations": [
            "single adaptation seed (1000); exploratory screen, not a final cost-curve claim",
            "prior n=50 reference used batch=4 and stream RNG; its deltas are descriptive only",
        ],
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = []
    for task_id, task in summary["tasks"].items():
        for budget, budget_data in task["budgets"].items():
            for action_steps, metrics in budget_data["action_steps"].items():
                rows.append(
                    {
                        "task_id": int(task_id),
                        "instruction": task["instruction"],
                        "k": int(budget),
                        "n_action_steps": int(action_steps),
                        "successes": metrics["successes"],
                        "trials": metrics["trials"],
                        "success_rate": metrics["success_rate"],
                        "ci95_low": metrics["ci95_low"],
                        "ci95_high": metrics["ci95_high"],
                    }
                )
    with (output_dir / "success_rates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axis = plt.subplots(figsize=(7, 5))
    for action_steps in ACTION_STEPS:
        axis.plot(
            DEMO_BUDGETS,
            [
                summary["means"]["mean_tasks_0_2"][str(k)][str(action_steps)]
                for k in DEMO_BUDGETS
            ],
            marker="o",
            linewidth=2,
            label=f"n_action_steps={action_steps}",
        )
    axis.set(
        xlabel="target demonstrations k",
        ylabel="success rate",
        ylim=(-0.03, 1.03),
        xticks=list(DEMO_BUDGETS),
        title="Mean over assignment tasks 0–2",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "action_steps_low_k.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Low-k action-step sweep on the official-data pretrain",
        "",
        "All action-step conditions for a task/budget use the same adapted",
        "checkpoint and the same per-episode environment/noise/init-state bank.",
        "Only binary success is aggregated; all rollout videos remain on disk.",
        "",
        "| task set | k | n=1 | n=10 | n=25 | prior n=50* |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    prior = summary["prior_n50_reference"]
    for budget in DEMO_BUDGETS:
        means = summary["means"]["mean_tasks_0_2"][str(budget)]
        lines.append(
            f"| tasks 0–2 | {budget} | "
            + " | ".join(f"{means[str(n)]:.3f}" for n in ACTION_STEPS)
            + f" | {prior['mean_tasks_0_2'][str(budget)]:.3f} |"
        )
    lines.extend(
        [
            "",
            "*The previous n=50 result is descriptive only: it used batch=4",
            "and one process RNG stream, whereas this experiment uses batch=1",
            "and explicit per-episode policy-noise seeds. A claimed improvement",
            "must be confirmed with a paired n=50 rerun and a second training seed.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Aggregate low-k action-step sweep")
    parser.add_argument("--results-root", type=Path, default=root / "results/raw")
    parser.add_argument("--output-dir", type=Path, default=root / "results/summary")
    parser.add_argument("--report", type=Path, default=root / "reports/REPORT.md")
    args = parser.parse_args()
    prior = json.loads(
        (root / "artifacts/prior_action_steps_evidence.json").read_text()
    )
    summary = aggregate(args.results_root, prior)
    write_outputs(summary, args.output_dir)
    write_report(summary, args.report)
    print(json.dumps(summary["means"], indent=2))


if __name__ == "__main__":
    main()
