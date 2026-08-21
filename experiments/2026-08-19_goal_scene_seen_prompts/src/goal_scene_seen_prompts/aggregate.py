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


def median(values: list[float]) -> float | None:
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


POINT_KEYS = (
    "label",
    "block",
    "env_task_id",
    "env_name",
    "env_instruction",
    "prompt",
    "prompted_source",
    "expect_env_equivalent",
    "behavior_target",
)


def aggregate(results_root: Path, plan: dict) -> dict[str, Any]:
    rows = []
    outcomes: dict[str, list[bool]] = {}
    for point in plan["points"]:
        payload = json.loads((results_root / f"{point['label']}.json").read_text())
        episodes = payload["per_episode"]
        outcomes[point["label"]] = [bool(episode["success"]) for episode in episodes]
        low, high = wilson_interval(payload["successes"], payload["n_episodes"])
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
                "consistency_violations": payload["consistency_violations"],
                "median_min_eef_target_dist": median(
                    [episode["min_eef_target_dist"] for episode in episodes]
                ),
                "median_max_target_displacement": median(
                    [episode["max_target_displacement"] for episode in episodes]
                ),
                "episodes_target_moved_5cm": sum(
                    1
                    for episode in episodes
                    if (episode["max_target_displacement"] or 0) > 0.05
                ),
            }
        )

    # Paired comparisons vs the TRUE point of the same env (identical seeds
    # and init states -> exact McNemar on the primary metric).
    true_by_env = {row["env_task_id"]: row for row in rows if row["block"] == "true"}
    paired = []
    for row in rows:
        if row["block"] == "true":
            continue
        base = true_by_env.get(row["env_task_id"])
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
                "true": f"{base['successes']}/{base['trials']}",
                "condition": f"{row['successes']}/{row['trials']}",
                "delta": row["success_rate"] - base["success_rate"],
                "discordant_only_true": only_base,
                "discordant_only_condition": only_cond,
                "mcnemar_p": mcnemar_exact_p(only_base, only_cond),
                "delta_median_min_dist": (
                    None
                    if row["median_min_eef_target_dist"] is None
                    or base["median_min_eef_target_dist"] is None
                    else row["median_min_eef_target_dist"]
                    - base["median_min_eef_target_dist"]
                ),
            }
        )

    return {
        "experiment": "goal_scene_seen_prompts",
        "rows": rows,
        "paired": paired,
        "notes": plan.get("notes", {}),
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = summary["rows"]
    envs = sorted({row["env_task_id"] for row in rows})
    blocks = ["true", "seen_twin", "seen_cross", "nonsense"]
    colors = {
        "true": "#3264a8",
        "seen_twin": "#2e9e60",
        "seen_cross": "#d0802d",
        "nonsense": "#888888",
    }
    fig, (ax_succ, ax_beh) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    width = 0.2
    for offset, block in enumerate(blocks):
        xs, succ, dist = [], [], []
        for position, env in enumerate(envs):
            row = next(
                (r for r in rows if r["env_task_id"] == env and r["block"] == block),
                None,
            )
            if row is None:
                continue
            xs.append(position + (offset - 1.5) * width)
            succ.append(row["success_rate"])
            dist.append(row["median_min_eef_target_dist"] or 0)
        if xs:
            ax_succ.bar(xs, succ, width=width, label=block, color=colors[block])
            ax_beh.bar(xs, dist, width=width, label=block, color=colors[block])
    labels = []
    for env in envs:
        instruction = next(r["env_instruction"] for r in rows if r["env_task_id"] == env)
        words = instruction.split()
        labels.append(
            "\n".join(" ".join(words[i : i + 3]) for i in range(0, len(words), 3))
        )
    ax_beh.set_xticks(range(len(envs)), labels, fontsize=8)
    ax_succ.set(ylabel="prompted-predicate success", ylim=(0, 1.05),
                title="Frozen pretrain in the GOAL scene under seen-trained prompt strings")
    ax_beh.set(ylabel="median min eef->target dist (m)",
               title="Behavioral: closest approach to the env task's object")
    for ax in (ax_succ, ax_beh):
        ax.grid(axis="y", alpha=0.25)
    ax_succ.legend(title="Prompt condition")
    fig.tight_layout()
    fig.savefig(output_dir / "goal_scene_probe.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal-scene x seen-prompts (probe 3): trained strings in a novel scene",
        "",
        "Frozen official-data pretrain in the GOAL scene (libero_goal); prompts",
        "are verbatim libero_90-trained strings vs the goal instructions.",
        "Primary success = the PROMPTED task's predicate; behavioral metrics",
        "discriminate conditions at the floor.",
        "",
        "| env instruction | block | prompt | prompted succ | 95% CI | med min dist (m) | moved>5cm |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for row in summary["rows"]:
        primary = f"{row['successes']}/{row['trials']}"
        if row["success_metric"] == "env_task":
            primary += " (env)"
        dist = row["median_min_eef_target_dist"]
        lines.append(
            f"| `{row['env_instruction']}` | {row['block']} | `{row['prompt']}` | "
            f"{primary} | [{row['ci95_low']:.2f}, {row['ci95_high']:.2f}] | "
            f"{dist:.3f} | {row['episodes_target_moved_5cm']}/{row['trials']} |"
            if dist is not None
            else f"| `{row['env_instruction']}` | {row['block']} | `{row['prompt']}` | "
            f"{primary} | [{row['ci95_low']:.2f}, {row['ci95_high']:.2f}] | — | "
            f"{row['episodes_target_moved_5cm']}/{row['trials']} |"
        )
    lines += [
        "",
        "## Paired vs the true goal prompt (same env, same init states)",
        "",
        "| env | block | prompt | true → condition | Δ succ | Δ med min dist | McNemar p |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for pair in summary["paired"]:
        d_dist = pair["delta_median_min_dist"]
        lines.append(
            f"| `{pair['env_instruction'][:40]}` | {pair['block']} | `{pair['prompt'][:40]}` | "
            f"{pair['true']} → {pair['condition']} | {pair['delta']:+.2f} | "
            f"{'—' if d_dist is None else f'{d_dist:+.3f}'} | {pair['mcnemar_p']:.3g} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Aggregate probe-3 results")
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
