from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .constants import (
    CHECKPOINT_PATH,
    CHECKPOINT_PROVENANCE,
    MASTER_SEED,
    PROMPT_CONDITIONS,
    TARGET_ENV_TASK_IDS,
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
    checkpoint_hashes = set()
    for condition, result in raw.items():
        if result["condition"] != condition:
            raise ValueError(f"Condition mismatch in {condition}.json")
        if result["model"] != str(CHECKPOINT_PATH):
            raise ValueError(f"Checkpoint path mismatch in {condition}.json")
        checkpoint_hashes.add(result["checkpoint"]["model_safetensors_sha256"])
        for task_id, instruction in TARGET_INSTRUCTIONS.items():
            task = result["tasks"][str(task_id)]
            expected = {
                "logical_task_id": task_id,
                "env_task_id": TARGET_ENV_TASK_IDS[task_id],
                "environment_instruction": instruction,
            }
            for key, value in expected.items():
                if task.get(key) != value:
                    raise ValueError(
                        f"Task mapping mismatch in {condition}.json: "
                        f"{key}={task.get(key)!r}, expected {value!r}"
                    )
    if len(checkpoint_hashes) != 1:
        raise ValueError(
            f"Conditions were evaluated on different weights: {checkpoint_hashes}"
        )

    summary: dict[str, Any] = {
        "experiment": "pretrain_smolvla_prompt_only_2",
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "model_safetensors_sha256": next(iter(checkpoint_hashes)),
            "provenance": CHECKPOINT_PROVENANCE,
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
    fig, ax = plt.subplots(figsize=(13.0, 4.8))
    colors = {"true": "#3264a8", "wrong": "#d0802d", "nonsense": "#888888"}
    for offset, condition in enumerate(PROMPT_CONDITIONS, start=-1):
        metrics = [
            summary["tasks"][str(task_id)]["conditions"][condition]
            for task_id in task_ids
        ]
        values = [item["success_rate"] for item in metrics]
        lower_errors = [
            value - item["ci95_low"] for value, item in zip(values, metrics)
        ]
        upper_errors = [
            item["ci95_high"] - value for value, item in zip(values, metrics)
        ]
        ax.errorbar(
            [position + offset * width for position in x_positions],
            values,
            yerr=[lower_errors, upper_errors],
            fmt="o",
            markersize=6,
            capsize=3,
            label=condition,
            color=colors[condition],
        )
    ax.set(
        xlabel="LIBERO goal task ID",
        ylabel="Success rate",
        ylim=(-0.03, 1.03),
        title="Official-pretrain prompt-only success on all 10 goal tasks (Wilson 95% CI)",
    )
    ax.set_xticks(x_positions, [str(task_id) for task_id in task_ids])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Prompt")
    fig.tight_layout()
    fig.savefig(output_dir / "prompt_controls.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Official-pretrain prompt-only report",
        "",
        "The frozen official-data pretrain checkpoint (30k DDP steps from",
        "`lerobot/smolvla_base`, seen positive control 20/20) is evaluated with",
        "zero target demonstrations and zero optimizer steps on all ten",
        "suite-local `libero_goal` tasks. Logical task IDs equal environment IDs.",
        "",
        "| task | instruction | true | wrong | nonsense |",
        "|---:|---|---:|---:|---:|",
    ]
    for task_id, task in summary["tasks"].items():
        cells = []
        for condition in PROMPT_CONDITIONS:
            metric = task["conditions"][condition]
            cells.append(
                f"{metric['successes']}/{metric['trials']} ({metric['success_rate']:.3f})"
            )
        lines.append(
            f"| {task_id} | `{task['instruction']}` | " + " | ".join(cells) + " |"
        )

    means = summary["condition_means"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Mean success by prompt condition: "
            + ", ".join(
                f"{condition}={means[condition]:.3f}"
                for condition in PROMPT_CONDITIONS
            )
            + ".",
            "",
            "Unlike every earlier zero-shot run in this repository, the evaluation",
            "pipeline behind these numbers has a passing seen-task positive control",
            "(20/20), so floor results here measure generalization, not pipeline",
            "defects.",
        ]
    )
    all_successes = [
        task["conditions"][condition]["successes"]
        for task in summary["tasks"].values()
        for condition in PROMPT_CONDITIONS
    ]
    if not any(all_successes):
        lines.append(
            "All conditions are at the binary-success floor. Equal zero rates do not "
            "establish that the model ignores language; this control is non-identifying "
            "at the floor."
        )
    else:
        lines.append(
            "The controls are identifiable above the floor; compare the per-task true, "
            "wrong-task, and nonsense rates rather than only their global means."
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate prompt-only rollouts into JSON, CSV, and a plot"
    )
    parser.add_argument("--results-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/summary"))
    parser.add_argument("--report", type=Path, default=Path("reports/REPORT.md"))
    args = parser.parse_args()
    summary = aggregate(args.results_root)
    write_outputs(summary, args.output_dir)
    write_report(summary, args.report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
