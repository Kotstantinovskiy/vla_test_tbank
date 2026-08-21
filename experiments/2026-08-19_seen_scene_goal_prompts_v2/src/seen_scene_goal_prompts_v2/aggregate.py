from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import experiment_root


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


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant pairs (binomial test, p=0.5)."""

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


POINT_KEYS = (
    "label",
    "block",
    "env_task_id",
    "env_name",
    "env_instruction",
    "scene",
    "prompt",
    "prompted_source",
    "expect_env_equivalent",
)


def aggregate(results_root: Path, plan: dict) -> dict[str, Any]:
    rows = []
    outcomes: dict[str, list[bool]] = {}
    for point in plan["points"]:
        payload = json.loads((results_root / f"{point['label']}.json").read_text())
        episode_outcomes = [bool(episode["success"]) for episode in payload["per_episode"]]
        outcomes[point["label"]] = episode_outcomes
        low, high = wilson_interval(payload["successes"], payload["n_episodes"])
        first_steps = [
            episode["prompted_first_step"]
            for episode in payload["per_episode"]
            if episode.get("prompted_first_step") is not None
        ]
        rows.append(
            {
                **{key: point[key] for key in POINT_KEYS},
                "success_metric": payload["success_metric"],
                "successes": payload["successes"],
                "trials": payload["n_episodes"],
                "success_rate": payload["success_rate"],
                "ci95_low": low,
                "ci95_high": high,
                "env_task_successes": payload["env_task_successes"],
                "env_task_success_rate": payload["env_task_success_rate"],
                "median_prompted_first_step": (
                    sorted(first_steps)[len(first_steps) // 2] if first_steps else None
                ),
                "consistency_violations": payload["consistency_violations"],
            }
        )

    # Paired comparisons: every non-trained point vs the trained point of the
    # same env (identical seeds and init states -> exact McNemar), on the
    # PRIMARY (prompted-predicate) metric.
    trained_by_env = {
        row["env_task_id"]: row for row in rows if row["block"] == "trained"
    }
    paired = []
    for row in rows:
        if row["block"] == "trained":
            continue
        base = trained_by_env.get(row["env_task_id"])
        if base is None:
            continue
        a = outcomes[base["label"]]
        b = outcomes[row["label"]]
        only_base = sum(1 for x, y in zip(a, b) if x and not y)
        only_cond = sum(1 for x, y in zip(a, b) if y and not x)
        paired.append(
            {
                "env_instruction": row["env_instruction"],
                "block": row["block"],
                "prompt": row["prompt"],
                "trained": f"{base['successes']}/{base['trials']}",
                "condition": f"{row['successes']}/{row['trials']}",
                "delta": row["success_rate"] - base["success_rate"],
                "discordant_only_trained": only_base,
                "discordant_only_condition": only_cond,
                "mcnemar_p": mcnemar_exact_p(only_base, only_cond),
            }
        )

    # The goal-prompt slice: one row per libero_goal instruction.
    row_by_label = {row["label"]: row for row in rows}
    goal_slice = []
    for entry in plan["notes"].get("goal_slice", []):
        item = dict(entry)
        label = entry.get("label") or entry.get("alias_of")
        if label is not None:
            source = row_by_label[label]
            item["env_instruction"] = source["env_instruction"]
            item["successes"] = source["successes"]
            item["trials"] = source["trials"]
            item["success_rate"] = source["success_rate"]
            low, high = wilson_interval(source["successes"], source["trials"])
            item["ci95_low"], item["ci95_high"] = low, high
        goal_slice.append(item)

    return {
        "experiment": "seen_scene_goal_prompts_v2",
        "rows": rows,
        "paired": paired,
        "goal_slice": goal_slice,
        "notes": plan.get("notes", {}),
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = summary["rows"]
    envs = sorted({row["env_task_id"] for row in rows})
    blocks = ["trained", "paraphrase", "cross", "goal", "nonsense"]
    colors = {
        "trained": "#3264a8",
        "paraphrase": "#2e9e60",
        "cross": "#d0802d",
        "goal": "#a04ac2",
        "nonsense": "#888888",
    }
    fig, ax = plt.subplots(figsize=(14, 5))
    width = 0.16
    for offset, block in enumerate(blocks):
        xs, ys, los, his = [], [], [], []
        for position, env in enumerate(envs):
            row = next(
                (r for r in rows if r["env_task_id"] == env and r["block"] == block),
                None,
            )
            if row is None:
                continue
            xs.append(position + (offset - 2) * width)
            ys.append(row["success_rate"])
            los.append(row["success_rate"] - row["ci95_low"])
            his.append(row["ci95_high"] - row["success_rate"])
        if xs:
            ax.bar(xs, ys, width=width, label=block, color=colors[block])
            ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor="black", capsize=3, linewidth=1)
    labels = []
    for env in envs:
        instruction = next(r["env_instruction"] for r in rows if r["env_task_id"] == env)
        words = instruction.split()
        labels.append(
            "\n".join(" ".join(words[i : i + 3]) for i in range(0, len(words), 3))
        )
    ax.set_xticks(range(len(envs)), labels, fontsize=7)
    ax.set(ylabel="Success rate (prompted-task predicate)", ylim=(0, 1.05),
           title="v2: frozen pretrain in SEEN scenes, success = PROMPTED task's predicate (Wilson 95% CI)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Prompt condition")
    fig.tight_layout()
    fig.savefig(output_dir / "prompt_transfer_v2.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Seen-scene x goal-prompts v2: prompted-predicate success",
        "",
        "Frozen official-data pretrain in SEEN libero_90 scenes; only the",
        "prompt varies. v2 primary success = the PROMPTED task's goal",
        "predicate, evaluated per step on the running scene (episodes also",
        "terminate on prompted success). `env succ` is the env task's own",
        "predicate — the v1-comparable secondary metric. Same seeds and init",
        "states within an env -> paired exact McNemar.",
        "",
        "| env instruction | block | prompt | prompted succ | 95% CI | env succ |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in summary["rows"]:
        primary = f"{row['successes']}/{row['trials']}"
        if row["success_metric"] == "env_task":
            primary += " (env)"
        lines.append(
            f"| `{row['env_instruction']}` | {row['block']} | `{row['prompt']}` | "
            f"{primary} | [{row['ci95_low']:.2f}, {row['ci95_high']:.2f}] | "
            f"{row['env_task_successes']}/{row['trials']} |"
        )
    lines += [
        "",
        "## Goal-prompt slice (all 10 libero_goal instructions)",
        "",
        "| goal id | prompt | status | host env | relationship | prompted succ |",
        "|---:|---|---|---|---|---:|",
    ]
    for item in summary["goal_slice"]:
        if item["status"] == "skipped":
            lines.append(
                f"| {item['goal_id']} | `{item['prompt']}` | skipped | — | — | — |"
            )
        else:
            label = item.get("label") or item.get("alias_of")
            lines.append(
                f"| {item['goal_id']} | `{item['prompt']}` | {item['status']} "
                f"(`{label}`) | `{item['env_instruction']}` | "
                f"{item['relationship']} | {item['successes']}/{item['trials']} |"
            )
    lines += [
        "",
        "## Paired vs trained prompt (same env, same init states; primary metric)",
        "",
        "| env | block | delta | discordant (trained-only / condition-only) | McNemar p |",
        "|---|---|---:|---|---:|",
    ]
    for pair in summary["paired"]:
        lines.append(
            f"| `{pair['env_instruction']}` | {pair['block']} | {pair['delta']:+.2f} | "
            f"{pair['discordant_only_trained']} / {pair['discordant_only_condition']} | {pair['mcnemar_p']:.3f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Aggregate v2 prompt-transfer results")
    parser.add_argument("--results-root", type=Path, default=root / "results/raw")
    parser.add_argument("--output-dir", type=Path, default=root / "results/summary")
    parser.add_argument("--report", type=Path, default=root / "reports/REPORT.md")
    args = parser.parse_args()
    plan = json.loads((root / "artifacts/eval_plan.json").read_text())
    summary = aggregate(args.results_root, plan)
    write_outputs(summary, args.output_dir)
    write_report(summary, args.report)
    print(json.dumps(summary["paired"], indent=2))


if __name__ == "__main__":
    main()
