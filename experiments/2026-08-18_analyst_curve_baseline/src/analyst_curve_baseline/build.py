from __future__ import annotations

"""Assemble the full naive-baseline cost curve and its AUC metrics.

Analysis-only experiment: consumes the published `results/summary/summary.json`
artifacts of three sibling experiments (no code imports, no new rollouts):

- k=0        <- 2026-08-18_pretrain_smolvla_prompt_only (canonical `true` run)
- k=1/2/3    <- 2026-08-18_pretrain_smolvla_few_shot_tune_low_k
- k=5/10/25  <- 2026-08-18_pretrain_smolvla_naive_baseline_few_shot_tune

Outputs: combined CSV, curve plot, AUC metrics (per task, mean over all ten
tasks, mean over assignment tasks 0-2), and a short report.

AUC definition: trapezoidal area under success(k) over k in [0, 25] using the
measured budgets as nodes. Reported both raw (units: success x demos, max 25)
and normalized by the k-range (25), i.e. the linear-interpolation average
success over the range - a single scalar in [0, 1] summarizing how far LEFT
the curve sits. Low-k points dominate raw trapezoids less than the wide
5..25 segments, so a log2-scale variant (nodes weighted by log2(1+k)) is also
reported; it emphasizes exactly the cheap-demo regime the assignment targets.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

TASK_IDS = list(range(10))
ASSIGNMENT_TASKS = [0, 1, 2]
BUDGET_SOURCES: dict[int, str] = {0: "prompt_only", 1: "low_k", 2: "low_k", 3: "low_k", 5: "few_shot", 10: "few_shot", 25: "few_shot"}
BUDGETS = sorted(BUDGET_SOURCES)

DEFAULT_SOURCES = {
    "prompt_only": "experiments/2026-08-18_pretrain_smolvla_prompt_only/results/summary/summary.json",
    "low_k": "experiments/2026-08-18_pretrain_smolvla_few_shot_tune_low_k/results/summary/summary.json",
    "few_shot": "experiments/2026-08-18_pretrain_smolvla_naive_baseline_few_shot_tune/results/summary/summary.json",
}
EXPECTED_CHECKPOINT = (
    "/var/tmp/vla_outputs/seen_libero90_official_20260817/checkpoints/030000/pretrained_model"
)


def load_sources(repo_root: Path, overrides: dict[str, Path] | None = None) -> dict[str, Any]:
    sources = {}
    for name, default in DEFAULT_SOURCES.items():
        path = (overrides or {}).get(name) or repo_root / default
        sources[name] = {"path": str(path), "data": json.loads(Path(path).read_text())}
    prompt = sources["prompt_only"]["data"]
    if prompt["checkpoint"]["path"] != EXPECTED_CHECKPOINT:
        raise ValueError(f"prompt_only evaluated {prompt['checkpoint']['path']}")
    for name in ("low_k", "few_shot"):
        base = sources[name]["data"]["base_checkpoint"]["path"]
        if base != EXPECTED_CHECKPOINT:
            raise ValueError(f"{name} adapted from {base}, expected {EXPECTED_CHECKPOINT}")
    return sources


def per_task_rates(sources: dict[str, Any]) -> dict[int, dict[int, float]]:
    """{task_id: {k: success_rate}} across all seven budgets."""

    rates: dict[int, dict[int, float]] = {task_id: {} for task_id in TASK_IDS}
    prompt = sources["prompt_only"]["data"]
    for task_id in TASK_IDS:
        rates[task_id][0] = prompt["tasks"][str(task_id)]["conditions"]["true"]["success_rate"]
    for name in ("low_k", "few_shot"):
        data = sources[name]["data"]
        for task_id in TASK_IDS:
            for budget, metrics in data["tasks"][str(task_id)]["budgets"].items():
                rates[task_id][int(budget)] = metrics["success_rate"]
    for task_id in TASK_IDS:
        missing = [k for k in BUDGETS if k not in rates[task_id]]
        if missing:
            raise ValueError(f"Task {task_id} is missing budgets {missing}")
    return rates


def trapezoid_auc(points: dict[int, float]) -> dict[str, float]:
    ks = sorted(points)
    raw = 0.0
    for a, b in zip(ks, ks[1:]):
        raw += (points[a] + points[b]) / 2 * (b - a)
    log_raw = 0.0
    log_span = math.log2(1 + ks[-1]) - math.log2(1 + ks[0])
    for a, b in zip(ks, ks[1:]):
        width = math.log2(1 + b) - math.log2(1 + a)
        log_raw += (points[a] + points[b]) / 2 * width
    return {
        "auc_raw": raw,
        "auc_normalized": raw / (ks[-1] - ks[0]),
        "auc_log2_normalized": log_raw / log_span,
    }


def mean_curve(rates: dict[int, dict[int, float]], task_ids: list[int]) -> dict[int, float]:
    return {
        k: sum(rates[task_id][k] for task_id in task_ids) / len(task_ids)
        for k in BUDGETS
    }


def build(repo_root: Path, output_dir: Path, report_path: Path) -> dict[str, Any]:
    sources = load_sources(repo_root)
    rates = per_task_rates(sources)
    curve_all = mean_curve(rates, TASK_IDS)
    curve_assignment = mean_curve(rates, ASSIGNMENT_TASKS)

    summary: dict[str, Any] = {
        "experiment": "analyst_curve_baseline",
        "checkpoint": EXPECTED_CHECKPOINT,
        "sources": {name: item["path"] for name, item in sources.items()},
        "budgets": BUDGETS,
        "curves": {
            "mean_all_10": {str(k): v for k, v in curve_all.items()},
            "mean_tasks_0_2": {str(k): v for k, v in curve_assignment.items()},
            "per_task": {
                str(task_id): {str(k): v for k, v in rates[task_id].items()}
                for task_id in TASK_IDS
            },
        },
        "auc": {
            "definition": (
                "trapezoid over measured budgets k in [0, 25]; raw units are "
                "success x demos (max 25); normalized = raw / 25; log2 variant "
                "uses node spacing log2(1+k) to weight the cheap-demo regime"
            ),
            "mean_all_10": trapezoid_auc(curve_all),
            "mean_tasks_0_2": trapezoid_auc(curve_assignment),
            "per_task": {
                str(task_id): trapezoid_auc(rates[task_id]) for task_id in TASK_IDS
            },
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (output_dir / "combined_curve.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["series", *[f"k={k}" for k in BUDGETS], "auc_raw", "auc_normalized", "auc_log2_normalized"])
        for label, curve, auc in (
            ("mean_all_10", curve_all, summary["auc"]["mean_all_10"]),
            ("mean_tasks_0_2", curve_assignment, summary["auc"]["mean_tasks_0_2"]),
            *(
                (f"task_{task_id}", rates[task_id], summary["auc"]["per_task"][str(task_id)])
                for task_id in TASK_IDS
            ),
        ):
            writer.writerow(
                [label]
                + [f"{curve[k]:.3f}" for k in BUDGETS]
                + [f"{auc['auc_raw']:.3f}", f"{auc['auc_normalized']:.4f}", f"{auc['auc_log2_normalized']:.4f}"]
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_lin, ax_task) = plt.subplots(1, 2, figsize=(14, 5))
    ax_lin.plot(BUDGETS, [curve_all[k] for k in BUDGETS], marker="o", linewidth=2, label="mean over 10 tasks")
    ax_lin.plot(BUDGETS, [curve_assignment[k] for k in BUDGETS], marker="s", linewidth=2, label="mean over tasks 0-2")
    auc_all = summary["auc"]["mean_all_10"]["auc_normalized"]
    auc_02 = summary["auc"]["mean_tasks_0_2"]["auc_normalized"]
    ax_lin.fill_between(BUDGETS, [curve_all[k] for k in BUDGETS], alpha=0.12)
    ax_lin.set(xlabel="demonstrations k", ylabel="success rate", ylim=(-0.03, 1.03),
               title=f"Naive baseline, official pretrain (nAUC: all-10 {auc_all:.3f}, tasks 0-2 {auc_02:.3f})")
    ax_lin.set_xticks(BUDGETS)
    ax_lin.grid(alpha=0.25)
    ax_lin.legend()

    for task_id in TASK_IDS:
        ax_task.plot(BUDGETS, [rates[task_id][k] for k in BUDGETS], marker=".", alpha=0.65, label=f"task {task_id}")
    ax_task.set(xlabel="demonstrations k", ylabel="success rate", ylim=(-0.03, 1.03), title="Per-task curves")
    ax_task.set_xticks(BUDGETS)
    ax_task.grid(alpha=0.25)
    ax_task.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "combined_curve.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Combined naive-baseline cost curve and AUC",
        "",
        "Single frozen recipe (expert-only, 2000 steps, seed 1000, official",
        "demo_0..demo_{k-1}) measured across seven budgets by three experiments;",
        "this experiment only aggregates their published summaries.",
        "",
        "| series | " + " | ".join(f"k={k}" for k in BUDGETS) + " | AUC raw | nAUC | nAUC(log2) |",
        "|---|" + "---:|" * (len(BUDGETS) + 3),
    ]
    for label, curve, auc in (
        ("mean all 10", curve_all, summary["auc"]["mean_all_10"]),
        ("mean tasks 0-2", curve_assignment, summary["auc"]["mean_tasks_0_2"]),
    ):
        lines.append(
            f"| {label} | "
            + " | ".join(f"{curve[k]:.3f}" for k in BUDGETS)
            + f" | {auc['auc_raw']:.2f} | {auc['auc_normalized']:.3f} | {auc['auc_log2_normalized']:.3f} |"
        )
    lines += [
        "",
        "AUC is the trapezoid under success(k) for k in [0, 25]. `nAUC` divides",
        "by the range (25): the linear-interpolation average success over the",
        "whole budget range. `nAUC(log2)` spaces nodes by log2(1+k), so the",
        "cheap-demo regime the assignment targets dominates the score; it is",
        "the recommended headline scalar for comparing adaptation methods",
        "(a method that lifts k=1..3 moves it far more than one that lifts k=25).",
        "",
        "Per-task curves and AUCs: `results/summary/combined_curve.csv`.",
        "Caveats inherited from the sources: single training seed; k=0 ran under",
        "pretraining normalization statistics while k>0 used target statistics.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    experiment_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=experiment_root.parents[1])
    parser.add_argument("--output-dir", type=Path, default=experiment_root / "results/summary")
    parser.add_argument("--report", type=Path, default=experiment_root / "reports/REPORT.md")
    args = parser.parse_args()
    summary = build(args.repo_root, args.output_dir, args.report)
    print(json.dumps({"auc": summary["auc"]["mean_all_10"], "auc_tasks_0_2": summary["auc"]["mean_tasks_0_2"]}, indent=2))


if __name__ == "__main__":
    main()
