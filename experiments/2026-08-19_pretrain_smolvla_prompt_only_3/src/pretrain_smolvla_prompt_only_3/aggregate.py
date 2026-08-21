from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import (
    N_EVAL_EPISODES,
    PROMPT_CONDITIONS,
    TARGET_INSTRUCTIONS,
    experiment_root,
)

# Frozen reference: the _2 run of the same protocol under process-global
# noise (kept for comparison; _2's numbers are never modified).
V2_RESULTS_ROOT = (
    experiment_root().parent
    / "2026-08-18_pretrain_smolvla_prompt_only_2/results/raw"
)


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


def load_v2_reference() -> dict[tuple[str, int], dict[str, Any]]:
    reference: dict[tuple[str, int], dict[str, Any]] = {}
    for condition in PROMPT_CONDITIONS:
        path = V2_RESULTS_ROOT / f"{condition}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        for task_key, task in payload["tasks"].items():
            episodes = task["per_episode"]
            successes = sum(bool(episode["success"]) for episode in episodes)
            reference[(condition, int(task_key))] = {
                "successes": successes,
                "trials": len(episodes),
                "success_episodes": [
                    index
                    for index, episode in enumerate(episodes)
                    if episode["success"]
                ],
            }
    return reference


def aggregate(results_root: Path) -> dict[str, Any]:
    reference = load_v2_reference()
    rows = []
    for condition in PROMPT_CONDITIONS:
        for task_id in sorted(TARGET_INSTRUCTIONS):
            label = f"{condition}__task_{task_id}"
            payload = json.loads((results_root / f"{label}.json").read_text())
            low, high = wilson_interval(payload["successes"], payload["n_episodes"])
            v2 = reference.get((condition, task_id))
            rows.append(
                {
                    "label": label,
                    "condition": condition,
                    "task_id": task_id,
                    "instruction": payload["environment_instruction"],
                    "prompt": payload["policy_prompt"],
                    "successes": payload["successes"],
                    "trials": payload["n_episodes"],
                    "success_rate": payload["success_rate"],
                    "ci95_low": low,
                    "ci95_high": high,
                    "success_episodes": [
                        episode["episode_ix"]
                        for episode in payload["per_episode"]
                        if episode["success"]
                    ],
                    "v2_successes": None if v2 is None else v2["successes"],
                    "delta_vs_v2": (
                        None
                        if v2 is None
                        else (payload["successes"] - v2["successes"]) / payload["n_episodes"]
                    ),
                }
            )

    conditions = {}
    for condition in PROMPT_CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        successes = sum(row["successes"] for row in condition_rows)
        trials = sum(row["trials"] for row in condition_rows)
        low, high = wilson_interval(successes, trials)
        v2_successes = [row["v2_successes"] for row in condition_rows]
        conditions[condition] = {
            "successes": successes,
            "trials": trials,
            "success_rate": successes / trials,
            "ci95_low": low,
            "ci95_high": high,
            "v2_successes": (
                None if any(value is None for value in v2_successes) else sum(v2_successes)
            ),
        }

    return {
        "experiment": "pretrain_smolvla_prompt_only_3",
        "noise_seeding": "per-episode (noise_seed = 1000 + episode), batch=1",
        "rows": rows,
        "conditions": conditions,
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    tasks = sorted(TARGET_INSTRUCTIONS)
    for ax, condition in zip(axes, PROMPT_CONDITIONS):
        rows = [
            row for row in summary["rows"] if row["condition"] == condition
        ]
        xs = list(range(len(tasks)))
        v3 = [row["success_rate"] for row in rows]
        v2 = [
            (row["v2_successes"] or 0) / row["trials"] if row["v2_successes"] is not None else 0
            for row in rows
        ]
        err_low = [row["success_rate"] - row["ci95_low"] for row in rows]
        err_high = [row["ci95_high"] - row["success_rate"] for row in rows]
        ax.bar([x - 0.2 for x in xs], v3, width=0.38, label="_3 (per-episode noise)", color="#3264a8")
        ax.errorbar([x - 0.2 for x in xs], v3, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=2, linewidth=1)
        ax.bar([x + 0.2 for x in xs], v2, width=0.38, label="_2 (stream noise)", color="#c9c9c9")
        ax.set_xticks(xs, [str(task) for task in tasks], fontsize=8)
        ax.set_title(f"{condition} (pooled {summary['conditions'][condition]['successes']}/{summary['conditions'][condition]['trials']})")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("success rate")
    axes[0].legend(fontsize=8)
    fig.suptitle("Zero-shot prompt-only, reproducible per-episode noise (_3) vs _2")
    fig.tight_layout()
    fig.savefig(output_dir / "prompt_only_3.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Prompt-only v3: reproducible per-episode sampling noise",
        "",
        "Same protocol as _2 (frozen pretrain, 10 goal tasks, true/wrong/",
        "nonsense, seed 1000, all videos) with ONE change: the policy's flow",
        "noise is reseeded from the episode seed (batch=1), so results are",
        "reproducible regardless of process layout.",
        "",
        "| condition | task | prompted with | _3 succ | 95% CI | _2 succ | success episodes (_3) |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['condition']} | {row['task_id']} | `{row['prompt'][:40]}` | "
            f"{row['successes']}/{row['trials']} | [{row['ci95_low']:.2f}, {row['ci95_high']:.2f}] | "
            f"{'—' if row['v2_successes'] is None else row['v2_successes']}/{row['trials']} | "
            f"{row['success_episodes'] or '—'} |"
        )
    lines += ["", "## Pooled per condition", ""]
    for condition, stats in summary["conditions"].items():
        lines.append(
            f"- **{condition}**: {stats['successes']}/{stats['trials']} "
            f"[{stats['ci95_low']:.3f}, {stats['ci95_high']:.3f}] "
            f"(_2: {stats['v2_successes']}/{stats['trials']})"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Aggregate prompt-only v3 results")
    parser.add_argument("--results-root", type=Path, default=root / "results/raw")
    parser.add_argument("--output-dir", type=Path, default=root / "results/summary")
    parser.add_argument("--report", type=Path, default=root / "reports/REPORT.md")
    args = parser.parse_args()
    summary = aggregate(args.results_root)
    write_outputs(summary, args.output_dir)
    write_report(summary, args.report)
    print(json.dumps(summary["conditions"], indent=2))


if __name__ == "__main__":
    main()
