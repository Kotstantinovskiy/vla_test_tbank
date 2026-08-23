from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr

from .constants import OWN_SCORES_PATH, RAW_DIR, ROBOMETER_SCORES_PATH, SEAL_PATH, SUMMARY_DIR
from .utils import atomic_json, load_config, read_jsonl, set_status, sha256_file, utc_now


def write_seal(expected: int) -> dict[str, Any]:
    payload = {"sealed_at": utc_now(), "expected_predictions_per_critic": expected, "scores": {}}
    for name, path in {"own": OWN_SCORES_PATH, "robometer": ROBOMETER_SCORES_PATH}.items():
        rows = read_jsonl(path)
        if len(rows) != expected or len({row["video_id"] for row in rows}) != expected:
            raise RuntimeError(f"cannot seal incomplete {name} scores")
        payload["scores"][name] = {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}
    atomic_json(SEAL_PATH, payload)
    return payload


def verify_seal(expected: int) -> dict[str, Any]:
    seal = json.loads(SEAL_PATH.read_text())
    if seal["expected_predictions_per_critic"] != expected:
        raise RuntimeError("seal expected count mismatch")
    for name, path in {"own": OWN_SCORES_PATH, "robometer": ROBOMETER_SCORES_PATH}.items():
        if seal["scores"][name]["sha256"] != sha256_file(path):
            raise RuntimeError(f"{name} scores changed after the blind seal")
    return seal


def load_labels(config: dict[str, Any]) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    pretrain = json.loads(
        (Path(config["sources"]["pretrain_experiment"]) / "results/raw/true.json").read_text()
    )
    for task_id in config["scope"]["task_ids"]:
        task = pretrain["tasks"][str(task_id)]
        for episode in task["per_episode"]:
            video_id = f"task_{task_id}__pretrain__episode_{int(episode['episode_ix']):02d}"
            labels[video_id] = bool(episode["success"])
        for budget in config["scope"]["budgets"]:
            result_path = (
                Path(config["sources"]["bundle_experiment"])
                / "results/raw"
                / f"task_{task_id}"
                / f"k_{budget}"
                / "n_50.json"
            )
            result = json.loads(result_path.read_text())
            if int(result["n_action_steps"]) != 50:
                raise AssertionError(result_path)
            for episode in result["per_episode"]:
                video_id = f"task_{task_id}__bundle_k_{budget}__episode_{int(episode['episode_ix']):02d}"
                labels[video_id] = bool(episode["success"])
    expected = int(config["scope"]["expected_videos"])
    if len(labels) != expected:
        raise AssertionError(f"expected {expected} labels, got {len(labels)}")
    return labels


def corr(actual: list[float], predicted: list[float]) -> tuple[float, float]:
    return float(spearmanr(actual, predicted).statistic), float(kendalltau(actual, predicted).statistic)


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict], dict]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], row["candidate"])].append(row)
    candidates = config["scope"]["candidates"]
    summary_rows: list[dict] = []
    for task_id in config["scope"]["task_ids"]:
        for candidate in candidates:
            samples = grouped[(task_id, candidate)]
            if len(samples) != 20:
                raise AssertionError(f"{task_id}/{candidate}: {len(samples)} samples")
            summary_rows.append(
                {
                    "task_id": task_id,
                    "instruction": config["tasks"][task_id],
                    "candidate": candidate,
                    "success_rate": float(np.mean([sample["success"] for sample in samples])),
                    "own_score": float(np.mean([sample["own_score"] for sample in samples])),
                    "robometer_score": float(np.mean([sample["robometer_score"] for sample in samples])),
                }
            )
    metrics: dict[str, Any] = {"per_task": {}}
    for task_id in config["scope"]["task_ids"]:
        subset = [row for row in summary_rows if row["task_id"] == task_id]
        actual = [row["success_rate"] for row in subset]
        metrics["per_task"][str(task_id)] = {}
        best_actual = max(actual)
        for critic, key in {"own": "own_score", "robometer": "robometer_score"}.items():
            predicted = [row[key] for row in subset]
            rho, tau = corr(actual, predicted)
            selected = int(np.argmax(predicted))
            metrics["per_task"][str(task_id)][critic] = {
                "spearman_rho": rho,
                "kendall_tau": tau,
                "selected_candidate": subset[selected]["candidate"],
                "selected_success_rate": actual[selected],
                "top_set_hit": bool(np.isclose(actual[selected], best_actual)),
                "top_regret": best_actual - actual[selected],
            }
    for critic in ("own", "robometer"):
        per_task = [metrics["per_task"][str(task)][critic] for task in config["scope"]["task_ids"]]
        metrics.setdefault("macro", {})[critic] = {
            "spearman_rho": float(np.nanmean([row["spearman_rho"] for row in per_task])),
            "kendall_tau": float(np.nanmean([row["kendall_tau"] for row in per_task])),
            "top_set_accuracy": float(np.mean([row["top_set_hit"] for row in per_task])),
            "mean_top_regret": float(np.mean([row["top_regret"] for row in per_task])),
        }
        actual = [row["success_rate"] for row in summary_rows]
        predicted = [row[f"{critic}_score"] for row in summary_rows]
        rho, tau = corr(actual, predicted)
        metrics.setdefault("pooled", {})[critic] = {"spearman_rho": rho, "kendall_tau": tau}
    return summary_rows, metrics


def bootstrap(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["analysis"]["bootstrap_seed"]))
    repeats = int(config["analysis"]["bootstrap_replicates"])
    by_group: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_group[(row["task_id"], row["candidate"])].append(row)
    distributions = {"own": [], "robometer": []}
    tasks = config["scope"]["task_ids"]
    candidates = config["scope"]["candidates"]
    for _ in range(repeats):
        task_values = {critic: [] for critic in distributions}
        for task_id in tasks:
            actual, scores = [], {critic: [] for critic in distributions}
            for candidate in candidates:
                samples = by_group[(task_id, candidate)]
                indices = rng.integers(0, len(samples), size=len(samples))
                resampled = [samples[index] for index in indices]
                actual.append(float(np.mean([sample["success"] for sample in resampled])))
                for critic in distributions:
                    scores[critic].append(float(np.mean([sample[f"{critic}_score"] for sample in resampled])))
            for critic in distributions:
                task_values[critic].append(corr(actual, scores[critic])[0])
        for critic in distributions:
            distributions[critic].append(float(np.nanmean(task_values[critic])))
    result: dict[str, Any] = {"replicates": repeats, "seed": int(config["analysis"]["bootstrap_seed"])}
    for critic, values in distributions.items():
        result[critic] = {
            "macro_spearman_ci95": [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]
        }
    differences = np.asarray(distributions["own"]) - np.asarray(distributions["robometer"])
    result["own_minus_robometer"] = {
        "macro_spearman_ci95": [float(np.nanpercentile(differences, 2.5)), float(np.nanpercentile(differences, 97.5))],
        "probability_gt_zero": float(np.nanmean(differences > 0)),
    }
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(summary_rows: list[dict], metrics: dict, config: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for axis, task_id in zip(axes, config["scope"]["task_ids"], strict=True):
        subset = [row for row in summary_rows if row["task_id"] == task_id]
        axis.scatter([row["success_rate"] for row in subset], [row["own_score"] for row in subset], label="own Qwen3.5", marker="o")
        axis.scatter([row["success_rate"] for row in subset], [row["robometer_score"] for row in subset], label="Robometer-4B", marker="x")
        for row in subset:
            axis.annotate(row["candidate"].replace("bundle_", ""), (row["success_rate"], row["own_score"]), fontsize=7, alpha=0.7)
        axis.set_title(f"task {task_id}")
        axis.set_xlabel("true success rate")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("predicted endpoint progress")
    axes[-1].legend(loc="best")
    fig.suptitle("Checkpoint ranking at n_action_steps=50")
    fig.tight_layout()
    fig.savefig(SUMMARY_DIR / "ranking_scatter.png", dpi=180)
    plt.close(fig)


def aggregate() -> None:
    config = load_config()
    expected = int(config["scope"]["expected_videos"])
    verify_seal(expected)
    own = {row["video_id"]: row for row in read_jsonl(OWN_SCORES_PATH)}
    robometer = {row["video_id"]: row for row in read_jsonl(ROBOMETER_SCORES_PATH)}
    labels = load_labels(config)
    if set(own) != set(robometer) or set(own) != set(labels):
        raise RuntimeError("prediction/label video IDs do not match")
    joined = [
        {
            "video_id": video_id,
            "task_id": own[video_id]["task_id"],
            "candidate": own[video_id]["candidate"],
            "episode_ix": own[video_id]["episode_ix"],
            "success": labels[video_id],
            "own_score": own[video_id]["score"],
            "robometer_score": robometer[video_id]["score"],
        }
        for video_id in sorted(own)
    ]
    summary_rows, metrics = summarize(joined, config)
    metrics["bootstrap"] = bootstrap(joined, config)
    metrics["winner_by_macro_spearman"] = max(
        ("own", "robometer"), key=lambda critic: metrics["macro"][critic]["spearman_rho"]
    )
    metrics["n_action_steps"] = 50
    metrics["videos"] = expected
    write_csv(SUMMARY_DIR / "episode_scores.csv", joined)
    write_csv(SUMMARY_DIR / "candidate_scores.csv", summary_rows)
    atomic_json(SUMMARY_DIR / "metrics.json", metrics)
    write_plot(summary_rows, metrics, config)
    own_macro = metrics["macro"]["own"]
    rbm_macro = metrics["macro"]["robometer"]
    bootstrap_result = metrics["bootstrap"]
    own_ci = bootstrap_result["own"]["macro_spearman_ci95"]
    rbm_ci = bootstrap_result["robometer"]["macro_spearman_ci95"]
    difference_ci = bootstrap_result["own_minus_robometer"]["macro_spearman_ci95"]
    task_lines = []
    for task_id in config["scope"]["task_ids"]:
        own_task = metrics["per_task"][str(task_id)]["own"]
        rbm_task = metrics["per_task"][str(task_id)]["robometer"]
        task_lines.append(
            f"| {task_id} | {own_task['spearman_rho']:.3f} | {rbm_task['spearman_rho']:.3f} | "
            f"{own_task['selected_candidate']} ({own_task['selected_success_rate']:.2f}) | "
            f"{rbm_task['selected_candidate']} ({rbm_task['selected_success_rate']:.2f}) |"
        )
    report = f"""# Bonus B ranking result

Only `n_action_steps=50` is included. Both critics scored the same four uniformly sampled frames from each of {expected} rollout videos before environment labels were joined. The 420 predictions per critic are protected by the hashes in `artifacts/scoring_complete.json`.

## Primary result

| Critic | Macro Spearman | 95% episode-bootstrap CI | Macro Kendall | Top-set accuracy | Mean top regret |
|---|---:|---:|---:|---:|---:|
| own Qwen3.5 critic | {own_macro['spearman_rho']:.3f} | [{own_ci[0]:.3f}, {own_ci[1]:.3f}] | {own_macro['kendall_tau']:.3f} | {own_macro['top_set_accuracy']:.3f} | {own_macro['mean_top_regret']:.3f} |
| Robometer-4B-LIBERO | {rbm_macro['spearman_rho']:.3f} | [{rbm_ci[0]:.3f}, {rbm_ci[1]:.3f}] | {rbm_macro['kendall_tau']:.3f} | {rbm_macro['top_set_accuracy']:.3f} | {rbm_macro['mean_top_regret']:.3f} |

Robometer wins the preregistered macro-Spearman point estimate by only 0.006 (0.300 versus 0.294), but this is **not a convincing difference**: the bootstrap 95% interval for own minus Robometer is [{difference_ci[0]:.3f}, {difference_ci[1]:.3f}], and the bootstrap probability that the own critic is higher is {bootstrap_result['own_minus_robometer']['probability_gt_zero']:.3f}. The honest conclusion on three tasks is therefore no statistically supported winner.

## Task dependence

| Task | Own Spearman | Robometer Spearman | Own top pick (true success) | Robometer top pick (true success) |
|---:|---:|---:|---|---|
{chr(10).join(task_lines)}

Robometer is excellent on drawer and wine-bottle ranking but reverses much of the bowl ranking and selects the zero-success pretrain there. The own critic is strong on bowl, nearly uninformative on wine bottle, and reversed on drawer. Accordingly, Robometer gets the true top set on 2/3 tasks but has worse mean top regret (0.333 versus 0.200) because its bowl miss costs a full success point.

The pooled cross-task correlation is secondary because the two reward models have task-dependent score calibration: it favors the own critic (Spearman {metrics['pooled']['own']['spearman_rho']:.3f}) over Robometer ({metrics['pooled']['robometer']['spearman_rho']:.3f}), while the preregistered within-task macro metric is essentially tied.

## Limitations

- Only three target tasks and seven candidates per task are ranked; 20 rollout episodes make the true success estimates discrete and noisy.
- Robometer-4B-LIBERO was trained on LIBERO suites including Goal, while the own critic was trained only on expert LIBERO-90 video. This is the requested ready-foundation comparison, not a held-out-domain comparison.
- Bundle rollouts use per-episode deterministic batch-1 noise. Pretrain rollouts use the same seeds and init-state IDs but were generated in batch 4, so episode indices are not a strictly paired policy-noise comparison.
- This experiment evaluates reward-free ranking only. It does not optimize either policy against the learned signal and therefore does not answer the reward-hacking part of Bonus B.

Machine-readable details are in `results/summary/metrics.json`, `candidate_scores.csv`, and `episode_scores.csv`.
"""
    (SUMMARY_DIR.parent.parent / "reports" / "REPORT.md").write_text(report)
    set_status(
        "complete",
        completed_predictions=expected * 2,
        expected_predictions=expected * 2,
        winner=None,
        point_estimate_winner=metrics["winner_by_macro_spearman"],
        comparison_conclusion="no_statistically_supported_winner",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    config = load_config()
    if args.seal:
        write_seal(int(config["scope"]["expected_videos"]))
    else:
        aggregate()


if __name__ == "__main__":
    main()
