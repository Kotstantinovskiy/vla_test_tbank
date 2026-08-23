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
    EVAL_ACTION_STEPS,
    EVAL_BATCH_SIZE,
    EVAL_EPISODES,
    EXPERIMENT_NAME,
    MASTER_SEED,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
    TRAINED_ACTION_STEPS,
    experiment_root,
    noise_seed,
)

# k=0 reference from the prompt-only experiment on the same checkpoint (mean
# over assignment tasks 0-2, same seed/episodes).  Normalization caveat: the
# k=0 rollouts used the checkpoint's pretraining statistics while every k>0
# point uses target-dataset statistics (LeRobot swaps them at fine-tune time).
ZERO_SHOT_REFERENCE = {
    "experiment": "2026-08-19_pretrain_smolvla_prompt_only_3",
    "mean_success": 0.0,
    "per_task_successes": {},
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
        budgets: dict[str, Any] = {}
        for budget in DEMO_BUDGETS:
            variants: dict[str, Any] = {}
            for action_steps in EVAL_ACTION_STEPS:
                payload = json.loads(
                    (
                        results_root
                        / f"task_{task_id}"
                        / f"k_{budget}"
                        / f"n_{action_steps}.json"
                    ).read_text()
                )
                label = f"task {task_id} k={budget} n={action_steps}"
                if payload["instruction"] != instruction:
                    raise ValueError(f"Instruction mismatch for {label}")
                if payload["n_episodes"] != EVAL_EPISODES:
                    raise ValueError(f"Episode count mismatch for {label}")
                if payload.get("batch_size") != EVAL_BATCH_SIZE:
                    raise ValueError(f"Evaluation batch mismatch for {label}")
                if payload.get("n_action_steps") != action_steps:
                    raise ValueError(f"Action-steps mismatch for {label}")
                if payload.get("trained_n_action_steps") != TRAINED_ACTION_STEPS:
                    raise ValueError(f"Trained action-steps mismatch for {label}")
                episodes = payload.get("per_episode", [])
                expected = [noise_seed(index) for index in range(EVAL_EPISODES)]
                if [item.get("env_seed") for item in episodes] != expected:
                    raise ValueError(f"Environment seed bank mismatch for {label}")
                if [item.get("noise_seed") for item in episodes] != expected:
                    raise ValueError(f"Policy-noise seed bank mismatch for {label}")
                if [item.get("init_state_id") for item in episodes] != list(range(EVAL_EPISODES)):
                    raise ValueError(f"LIBERO init-state bank mismatch for {label}")
                if not payload.get("model_safetensors_sha256"):
                    raise ValueError(f"Missing model SHA-256 for {label}")
                if any(not Path(item.get("video_path", "")).is_file() for item in episodes):
                    raise ValueError(f"Missing rollout video for {label}")
                low, high = wilson_interval(payload["successes"], payload["n_episodes"])
                variants[str(action_steps)] = {
                    "successes": payload["successes"],
                    "trials": payload["n_episodes"],
                    "success_rate": payload["success_rate"],
                    "ci95_low": low,
                    "ci95_high": high,
                    "model": payload["model"],
                    "model_safetensors_sha256": payload["model_safetensors_sha256"],
                }
            shas = {variant["model_safetensors_sha256"] for variant in variants.values()}
            if len(shas) != 1:
                raise ValueError(
                    f"Action-steps variants of task {task_id} k={budget} used different checkpoints"
                )
            budgets[str(budget)] = {"action_steps": variants}
        tasks[str(task_id)] = {"instruction": instruction, "budgets": budgets}

    def mean_over(task_ids: list[int], budget: int, action_steps: int) -> float:
        return sum(
            tasks[str(task_id)]["budgets"][str(budget)]["action_steps"][str(action_steps)][
                "success_rate"
            ]
            for task_id in task_ids
        ) / len(task_ids)

    target_ids = sorted(TARGET_INSTRUCTIONS)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "base_checkpoint": {"path": str(BASE_CHECKPOINT), "provenance": BASE_PROVENANCE},
        "protocol": {
            "suite": TARGET_SUITE,
            "seed": MASTER_SEED,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "noise_seeding": "per-episode torch/CUDA seed = 1000 + episode_index",
            "eval_episodes": EVAL_EPISODES,
            "budgets": list(DEMO_BUDGETS),
            "eval_action_steps": list(EVAL_ACTION_STEPS),
            "trained_n_action_steps": TRAINED_ACTION_STEPS,
        },
        "zero_shot_reference": ZERO_SHOT_REFERENCE,
        "tasks": tasks,
        "cost_curve": {
            "mean_tasks_0_2": {
                str(action_steps): {
                    str(k): mean_over(target_ids, k, action_steps) for k in DEMO_BUDGETS
                }
                for action_steps in EVAL_ACTION_STEPS
            },
        },
    }
    return summary


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = []
    for task_id, task in summary["tasks"].items():
        for budget, point in task["budgets"].items():
            for action_steps, metrics in point["action_steps"].items():
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
    with (output_dir / "cost_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, (ax_tasks, ax_mean) = plt.subplots(1, 2, figsize=(14, 5))
    ks = [0] + list(DEMO_BUDGETS)
    styles = {str(EVAL_ACTION_STEPS[0]): "-", str(EVAL_ACTION_STEPS[1]): "--"}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, (task_id, task) in enumerate(
        sorted(summary["tasks"].items(), key=lambda x: int(x[0]))
    ):
        for action_steps, style in styles.items():
            values = [0.0] + [
                task["budgets"][str(k)]["action_steps"][action_steps]["success_rate"]
                for k in DEMO_BUDGETS
            ]
            ax_tasks.plot(
                ks,
                values,
                style,
                marker="o",
                alpha=0.7,
                color=colors[index % len(colors)],
                label=f"task {task_id} n={action_steps}",
            )
    ax_tasks.set(
        xlabel="demonstrations k", ylabel="success rate", ylim=(-0.03, 1.03),
        title="Per-task cost curves (solid n=50, dashed n=25; k=0 prompt-only ref)",
    )
    ax_tasks.grid(alpha=0.25)
    ax_tasks.legend(fontsize=8, ncol=2)

    for action_steps, style in styles.items():
        mean_02 = [summary["zero_shot_reference"]["mean_success"]] + [
            summary["cost_curve"]["mean_tasks_0_2"][action_steps][str(k)]
            for k in DEMO_BUDGETS
        ]
        ax_mean.plot(
            ks, mean_02, style, marker="s", linewidth=2,
            label=f"mean tasks 0-2, n={action_steps}",
        )
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
        "# Full-fine-tune few-shot cost curve on the official-data pretrain",
        "",
        "Every task/budget adaptation starts from the frozen pretrain checkpoint",
        "independently and fine-tunes the whole policy (VLM text tower, vision",
        "encoder, connector, action expert, projections; only LeRobot's",
        "unused-by-design guard tensors stay frozen). Demos are the official",
        "demo_0..demo_{k-1} of each task from the in-repo libero_goal conversion.",
        "Each adapted checkpoint is evaluated at inference n_action_steps=50",
        "(trained default) and 25 on identical per-episode seeds/init states.",
        "",
        "Normalization disclosure: k>0 points run under target-dataset statistics",
        "(LeRobot swaps normalizer stats at fine-tune time), while the k=0",
        "reference ran under pretraining statistics; both come from the same",
        "conversion pipeline.",
        "",
    ]
    for action_steps in EVAL_ACTION_STEPS:
        lines.extend(
            [
                f"## Inference n_action_steps = {action_steps}",
                "",
                "| task | instruction | " + " | ".join(f"k={k}" for k in DEMO_BUDGETS) + " |",
                "|---:|---|" + "---:|" * len(DEMO_BUDGETS),
            ]
        )
        for task_id, task in sorted(summary["tasks"].items(), key=lambda x: int(x[0])):
            cells = []
            for k in DEMO_BUDGETS:
                metrics = task["budgets"][str(k)]["action_steps"][str(action_steps)]
                cells.append(
                    f"{metrics['successes']}/{metrics['trials']} ({metrics['success_rate']:.2f})"
                )
            lines.append(
                f"| {task_id} | `{task['instruction']}` | " + " | ".join(cells) + " |"
            )
        lines.append("")
    curve = summary["cost_curve"]
    lines.extend(
        [
            "## Cost curve",
            "",
            f"k=0 (prompt-only reference): {summary['zero_shot_reference']['mean_success']:.3f} over tasks 0-2.",
            "",
            "| mean tasks 0-2 | " + " | ".join(f"k={k}" for k in DEMO_BUDGETS) + " |",
            "|---|" + "---:|" * len(DEMO_BUDGETS),
        ]
    )
    for action_steps in EVAL_ACTION_STEPS:
        lines.append(
            f"| n={action_steps} | "
            + " | ".join(
                f"{curve['mean_tasks_0_2'][str(action_steps)][str(k)]:.3f}"
                for k in DEMO_BUDGETS
            )
            + " |"
        )
    lines.append("")
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
