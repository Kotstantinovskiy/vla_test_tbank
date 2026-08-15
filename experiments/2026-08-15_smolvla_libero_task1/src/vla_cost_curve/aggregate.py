from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .constants import DEMO_BUDGETS, TARGET_ENV_TASK_IDS, TARGET_INSTRUCTIONS


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _summary(task: dict[str, Any]) -> dict[str, Any]:
    outcomes = [bool(ep["success"]) for ep in task["per_episode"]]
    count = sum(outcomes)
    low, high = wilson_interval(count, len(outcomes))
    return {
        "successes": count,
        "trials": len(outcomes),
        "success_rate": count / len(outcomes),
        "ci95_low": low,
        "ci95_high": high,
    }


def _validate_task(task: dict[str, Any], task_id: int, path: Path) -> None:
    expected = {
        "logical_task_id": task_id,
        "env_task_id": TARGET_ENV_TASK_IDS[task_id],
        "environment_instruction": TARGET_INSTRUCTIONS[task_id],
    }
    for key, value in expected.items():
        if task.get(key) != value:
            raise ValueError(
                f"Task mapping mismatch in {path}: {key}={task.get(key)!r}, "
                f"expected {value!r}"
            )


def aggregate(results_root: Path) -> dict[str, Any]:
    zero_paths = {
        condition: results_root / "zero_shot" / f"{condition}.json"
        for condition in ("true", "wrong", "nonsense")
    }
    zero = {condition: _read(path) for condition, path in zero_paths.items()}
    result: dict[str, Any] = {"tasks": {}}
    for task_id, instruction in TARGET_INSTRUCTIONS.items():
        for condition, data in zero.items():
            _validate_task(data["tasks"][str(task_id)], task_id, zero_paths[condition])
        task_result: dict[str, Any] = {
            "instruction": instruction,
            "k0": {
                condition: _summary(data["tasks"][str(task_id)])
                for condition, data in zero.items()
            },
            "adapted": {},
        }
        for k in DEMO_BUDGETS:
            path = results_root / "adapted" / f"task_{task_id}" / f"k_{k}.json"
            data = _read(path)
            _validate_task(data["tasks"][str(task_id)], task_id, path)
            task_result["adapted"][str(k)] = _summary(data["tasks"][str(task_id)])
        result["tasks"][str(task_id)] = task_result

    points = [0, *DEMO_BUDGETS]
    means = {}
    for k in points:
        rates = []
        for task in result["tasks"].values():
            rates.append(
                task["k0"]["true"]["success_rate"]
                if k == 0
                else task["adapted"][str(k)]["success_rate"]
            )
        means[str(k)] = sum(rates) / len(rates)
    result["mean_cost_curve"] = means
    return result


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = []
    for task_id, task in summary["tasks"].items():
        row: dict[str, Any] = {"task_id": task_id, "instruction": task["instruction"]}
        for condition, metrics in task["k0"].items():
            row[f"k0_{condition}"] = metrics["success_rate"]
        for k, metrics in task["adapted"].items():
            row[f"k{k}"] = metrics["success_rate"]
        rows.append(row)
    with (output_dir / "cost_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    xs = [0, *DEMO_BUDGETS]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for task_id, task in summary["tasks"].items():
        ys = [task["k0"]["true"]["success_rate"]] + [
            task["adapted"][str(k)]["success_rate"] for k in DEMO_BUDGETS
        ]
        ax.plot(xs, ys, marker="o", alpha=0.65, label=f"Task {task_id}")
    ax.plot(
        xs,
        [summary["mean_cost_curve"][str(k)] for k in xs],
        marker="o",
        linewidth=3,
        color="black",
        label="Mean",
    )
    ax.set(xlabel="Number of demonstrations", ylabel="Success rate", ylim=(-0.03, 1.03))
    ax.set_xticks(xs)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cost_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/summary"))
    args = parser.parse_args()
    summary = aggregate(args.results_root)
    write_outputs(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
