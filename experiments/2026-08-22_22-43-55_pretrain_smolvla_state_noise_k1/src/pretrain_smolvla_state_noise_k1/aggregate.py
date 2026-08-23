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
    ALPHAS,
    BASE_CHECKPOINT,
    BASE_PROVENANCE,
    DEMO_BUDGET,
    EVAL_ACTION_STEPS,
    EVAL_BATCH_SIZE,
    EVAL_EPISODES,
    EXPERIMENT_NAME,
    MASTER_SEED,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
    TRAINED_ACTION_STEPS,
    alpha_tag,
    experiment_root,
    noise_seed,
)

# External reference: the full-FT experiment's k=1 / n=50 points
# (2026-08-22_18-10-40_pretrain_smolvla_full_ft_low_k, same recipe as the
# alpha=0.00 arm, same seeds/episodes).  A rerun is not bit-identical on GPU,
# so alpha=0.00 here is the in-experiment control and this table is context.
FULL_FT_K1_REFERENCE = {
    "experiment": "2026-08-22_18-10-40_pretrain_smolvla_full_ft_low_k",
    "per_task_successes": {"0": 2, "1": 19, "2": 14},
    "mean_success": 0.5833333333333334,
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
    action_steps = EVAL_ACTION_STEPS[0]
    tasks: dict[str, Any] = {}
    for task_id, instruction in TARGET_INSTRUCTIONS.items():
        arms: dict[str, Any] = {}
        for alpha in ALPHAS:
            tag = alpha_tag(alpha)
            payload = json.loads(
                (
                    results_root
                    / f"task_{task_id}"
                    / f"alpha_{tag}"
                    / f"n_{action_steps}.json"
                ).read_text()
            )
            label = f"task {task_id} alpha={tag}"
            if payload["instruction"] != instruction:
                raise ValueError(f"Instruction mismatch for {label}")
            if payload["n_episodes"] != EVAL_EPISODES:
                raise ValueError(f"Episode count mismatch for {label}")
            if payload.get("batch_size") != EVAL_BATCH_SIZE:
                raise ValueError(f"Evaluation batch mismatch for {label}")
            if payload.get("demo_budget") != DEMO_BUDGET:
                raise ValueError(f"Demo budget mismatch for {label}")
            if payload.get("state_noise_alpha") != alpha:
                raise ValueError(f"Alpha mismatch for {label}")
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
            arms[tag] = {
                "alpha": alpha,
                "successes": payload["successes"],
                "trials": payload["n_episodes"],
                "success_rate": payload["success_rate"],
                "ci95_low": low,
                "ci95_high": high,
                "model": payload["model"],
                "model_safetensors_sha256": payload["model_safetensors_sha256"],
            }
        tasks[str(task_id)] = {"instruction": instruction, "alphas": arms}

    def mean_over(alpha: float) -> float:
        tag = alpha_tag(alpha)
        return sum(
            tasks[str(task_id)]["alphas"][tag]["success_rate"]
            for task_id in TARGET_INSTRUCTIONS
        ) / len(TARGET_INSTRUCTIONS)

    summary = {
        "experiment": EXPERIMENT_NAME,
        "base_checkpoint": {"path": str(BASE_CHECKPOINT), "provenance": BASE_PROVENANCE},
        "protocol": {
            "suite": TARGET_SUITE,
            "seed": MASTER_SEED,
            "demo_budget": DEMO_BUDGET,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "noise_seeding": "per-episode torch/CUDA seed = 1000 + episode_index",
            "eval_episodes": EVAL_EPISODES,
            "eval_action_steps": list(EVAL_ACTION_STEPS),
            "state_noise_alphas": list(ALPHAS),
            "state_noise_location": (
                "normalized observation.state (MEAN_STD) in policy.forward, training only"
            ),
        },
        "full_ft_k1_reference": FULL_FT_K1_REFERENCE,
        "tasks": tasks,
        "noise_curve": {
            "mean_tasks_0_2": {alpha_tag(alpha): mean_over(alpha) for alpha in ALPHAS},
        },
    }
    return summary


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = []
    for task_id, task in summary["tasks"].items():
        for tag, metrics in task["alphas"].items():
            rows.append(
                {
                    "task_id": int(task_id),
                    "instruction": task["instruction"],
                    "k": DEMO_BUDGET,
                    "alpha": metrics["alpha"],
                    "n_action_steps": EVAL_ACTION_STEPS[0],
                    "successes": metrics["successes"],
                    "trials": metrics["trials"],
                    "success_rate": metrics["success_rate"],
                    "ci95_low": metrics["ci95_low"],
                    "ci95_high": metrics["ci95_high"],
                }
            )
    with (output_dir / "noise_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, (ax_tasks, ax_mean) = plt.subplots(1, 2, figsize=(13, 5))
    alphas = list(ALPHAS)
    for task_id, task in sorted(summary["tasks"].items(), key=lambda x: int(x[0])):
        values = [task["alphas"][alpha_tag(alpha)]["success_rate"] for alpha in alphas]
        ax_tasks.plot(alphas, values, marker="o", alpha=0.8, label=f"task {task_id}")
        reference = FULL_FT_K1_REFERENCE["per_task_successes"].get(task_id)
        if reference is not None:
            ax_tasks.scatter(
                [0.0], [reference / EVAL_EPISODES], marker="x", s=70, alpha=0.8,
                color=ax_tasks.lines[-1].get_color(),
            )
    ax_tasks.set(
        xlabel="state-noise alpha", ylabel="success rate", ylim=(-0.03, 1.03),
        title="Per-task success vs alpha (k=1, n=50; x = full-FT reference)",
    )
    ax_tasks.grid(alpha=0.25)
    ax_tasks.legend(fontsize=9)

    mean_values = [summary["noise_curve"]["mean_tasks_0_2"][alpha_tag(a)] for a in alphas]
    ax_mean.plot(alphas, mean_values, marker="s", linewidth=2, label="mean tasks 0-2")
    ax_mean.axhline(
        FULL_FT_K1_REFERENCE["mean_success"], linestyle=":", alpha=0.7,
        label="full-FT k=1 reference",
    )
    ax_mean.set(
        xlabel="state-noise alpha", ylabel="success rate", ylim=(-0.03, 1.03),
        title="Mean success vs alpha (k=1, n=50)",
    )
    ax_mean.grid(alpha=0.25)
    ax_mean.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "noise_curve.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Proprioception-noise augmentation at k=1 (full fine-tune)",
        "",
        "Each (task, alpha) adaptation is an independent full fine-tune of the",
        "pinned pretrain on the task's official demo_0, with additive zero-mean",
        "Gaussian noise `alpha * eps` applied to the normalized proprioceptive",
        "state inside policy.forward during training only (STATE is normalized",
        "MEAN_STD, so alpha equals sigma_i = alpha * Std(s_i) in raw units).",
        "Actions and images are untouched; evaluation (n_action_steps=50, 20",
        "episodes, seed bank 1000..1019) applies no noise. alpha=0.00 is the",
        "in-experiment control (strict no-op, full-FT k=1 recipe).",
        "",
        "| task | instruction | "
        + " | ".join(f"α={alpha_tag(a)}" for a in ALPHAS)
        + " | full-FT ref |",
        "|---:|---|" + "---:|" * (len(ALPHAS) + 1),
    ]
    for task_id, task in sorted(summary["tasks"].items(), key=lambda x: int(x[0])):
        cells = []
        for alpha in ALPHAS:
            metrics = task["alphas"][alpha_tag(alpha)]
            cells.append(
                f"{metrics['successes']}/{metrics['trials']} ({metrics['success_rate']:.2f})"
            )
        reference = FULL_FT_K1_REFERENCE["per_task_successes"].get(task_id, "—")
        lines.append(
            f"| {task_id} | `{task['instruction']}` | "
            + " | ".join(cells)
            + f" | {reference}/20 |"
        )
    curve = summary["noise_curve"]["mean_tasks_0_2"]
    lines.extend(
        [
            "",
            "## Mean over tasks 0-2",
            "",
            "| | " + " | ".join(f"α={alpha_tag(a)}" for a in ALPHAS) + " |",
            "|---|" + "---:|" * len(ALPHAS),
            "| mean success | "
            + " | ".join(f"{curve[alpha_tag(a)]:.3f}" for a in ALPHAS)
            + " |",
            "",
            f"Full-FT k=1 reference mean (external): {FULL_FT_K1_REFERENCE['mean_success']:.3f}.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Aggregate the state-noise sweep")
    parser.add_argument("--results-root", type=Path, default=root / "results/raw")
    parser.add_argument("--output-dir", type=Path, default=root / "results/summary")
    parser.add_argument("--report", type=Path, default=root / "reports/REPORT.md")
    args = parser.parse_args()
    summary = aggregate(args.results_root)
    write_outputs(summary, args.output_dir)
    write_report(summary, args.report)
    print(json.dumps(summary["noise_curve"], indent=2))


if __name__ == "__main__":
    main()
