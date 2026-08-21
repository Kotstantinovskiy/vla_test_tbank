from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .constants import EXPERIMENT_NAME, experiment_root


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
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def aggregate(results_root: Path, plan: dict) -> dict[str, Any]:
    rows = []
    payloads = {}
    for point in plan["points"]:
        payload = json.loads((results_root / f"{point['label']}.json").read_text())
        payloads[point["label"]] = payload
        low, high = wilson_interval(payload["successes"], payload["n_episodes"])
        rows.append(
            {
                "label": point["label"],
                "block": point["block"],
                "pair_id": point.get("pair_id"),
                "reference_label": point.get("reference_label"),
                "logical_task_id": point.get("logical_task_id"),
                "env_task_id": point["env_task_id"],
                "env_name": point["env_name"],
                "env_instruction": point["env_instruction"],
                "prompt": point["prompt"],
                "prompted_source": point["prompted_source"],
                "success_metric": payload["success_metric"],
                "successes": payload["successes"],
                "trials": payload["n_episodes"],
                "success_rate": payload["success_rate"],
                "ci95_low": low,
                "ci95_high": high,
                "env_task_successes": payload["env_task_successes"],
                "env_task_success_rate": payload["env_task_success_rate"],
                "consistency_violations": payload["consistency_violations"],
            }
        )
    row_by_label = {row["label"]: row for row in rows}
    paired = []
    for row in rows:
        reference_label = row.get("reference_label")
        if not reference_label:
            continue
        reference = row_by_label[reference_label]
        left = payloads[reference_label]
        right = payloads[row["label"]]
        left_episodes = left["per_episode"]
        right_episodes = right["per_episode"]
        left_keys = [(item["env_seed"], item["noise_seed"]) for item in left_episodes]
        right_keys = [(item["env_seed"], item["noise_seed"]) for item in right_episodes]
        if left_keys != right_keys:
            raise ValueError(f"Unpaired seed banks for {row['label']}")
        a = [bool(item["success"]) for item in left_episodes]
        b = [bool(item["success"]) for item in right_episodes]
        only_reference = sum(x and not y for x, y in zip(a, b, strict=True))
        only_condition = sum(y and not x for x, y in zip(a, b, strict=True))
        paired.append(
            {
                "pair_id": row["pair_id"],
                "reference_label": reference_label,
                "condition_label": row["label"],
                "env_instruction": row["env_instruction"],
                "reference_prompt": reference["prompt"],
                "condition_prompt": row["prompt"],
                "reference": f"{reference['successes']}/{reference['trials']}",
                "condition": f"{row['successes']}/{row['trials']}",
                "delta": row["success_rate"] - reference["success_rate"],
                "discordant_only_reference": only_reference,
                "discordant_only_condition": only_condition,
                "mcnemar_p": mcnemar_exact_p(only_reference, only_condition),
            }
        )
    return {
        "experiment": EXPERIMENT_NAME,
        "rows": rows,
        "paired": paired,
        "notes": plan.get("notes", {}),
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = summary["rows"]
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.55), 5))
    colors = {
        "exact": "#3264a8",
        "seen": "#3264a8",
        "goal": "#a04ac2",
        "paraphrase": "#2e9e60",
        "article_drop": "#d0802d",
        "nonsense": "#888888",
        "true_goal": "#a04ac2",
        "seen_prompt": "#3264a8",
    }
    xs = list(range(len(rows)))
    ys = [row["success_rate"] for row in rows]
    yerr = [
        [row["success_rate"] - row["ci95_low"] for row in rows],
        [row["ci95_high"] - row["success_rate"] for row in rows],
    ]
    ax.bar(xs, ys, color=[colors.get(row["block"], "#666666") for row in rows])
    ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(xs, [row["label"] for row in rows], rotation=70, ha="right", fontsize=7)
    ax.set(
        ylabel="Success rate",
        ylim=(0, 1.05),
        title=f"{EXPERIMENT_NAME}: prompted-predicate success (Wilson 95% CI)",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "success_rates.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        f"# {EXPERIMENT_NAME}: prompted-predicate success",
        "",
        "Only binary success is aggregated. Every rollout video remains on disk",
        "for manual behavior inspection. Environment and policy-noise seeds are",
        "recorded per episode.",
        "",
        "| label | env instruction | block | policy prompt | success | 95% CI | env success |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| `{row['label']}` | `{row['env_instruction']}` | {row['block']} | "
            f"`{row['prompt']}` | {row['successes']}/{row['trials']} | "
            f"[{row['ci95_low']:.2f}, {row['ci95_high']:.2f}] | "
            f"{row['env_task_successes']}/{row['trials']} |"
        )
    if summary["paired"]:
        lines += [
            "",
            "## Paired comparisons",
            "",
            "| pair | reference -> condition | delta | discordant ref/condition | McNemar p |",
            "|---|---|---:|---|---:|",
        ]
        for pair in summary["paired"]:
            lines.append(
                f"| `{pair['pair_id']}` | {pair['reference']} -> {pair['condition']} | "
                f"{pair['delta']:+.2f} | {pair['discordant_only_reference']}/"
                f"{pair['discordant_only_condition']} | {pair['mcnemar_p']:.4f} |"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Aggregate prompt experiment")
    parser.add_argument("--results-root", type=Path, default=root / "results/raw")
    parser.add_argument("--output-dir", type=Path, default=root / "results/summary")
    parser.add_argument("--report", type=Path, default=root / "reports/REPORT.md")
    args = parser.parse_args()
    plan = json.loads((root / "artifacts/eval_plan.json").read_text())
    summary = aggregate(args.results_root, plan)
    write_outputs(summary, args.output_dir)
    write_report(summary, args.report)
    print(json.dumps({"rows": len(summary["rows"]), "paired": summary["paired"]}, indent=2))


if __name__ == "__main__":
    main()
