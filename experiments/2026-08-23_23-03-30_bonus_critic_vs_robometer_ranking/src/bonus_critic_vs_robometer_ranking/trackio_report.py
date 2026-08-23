from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .constants import ARTIFACTS_DIR, MANIFEST_PATH, SUMMARY_DIR
from .utils import atomic_json, load_config, read_jsonl, utc_now


def log_results() -> None:
    import trackio

    config = load_config()
    metrics = json.loads((SUMMARY_DIR / "metrics.json").read_text())
    report_path = SUMMARY_DIR.parent.parent / "reports" / "REPORT.md"
    report = report_path.read_text()
    with (SUMMARY_DIR / "candidate_scores.csv").open() as handle:
        candidate_rows = list(csv.DictReader(handle))
    columns = [
        "task_id",
        "instruction",
        "candidate",
        "success_rate",
        "own_score",
        "robometer_score",
    ]
    table_rows = [
        [
            int(row["task_id"]),
            row["instruction"],
            row["candidate"],
            float(row["success_rate"]),
            float(row["own_score"]),
            float(row["robometer_score"]),
        ]
        for row in candidate_rows
    ]
    os.environ.setdefault("TRACKIO_DIR", str(ARTIFACTS_DIR / "trackio"))
    project = os.environ.get("TRACKIO_PROJECT", "bonus-critic-vs-robometer-ranking")
    run_name = "n50-checkpoint-ranking-seed-1000"
    trackio.init(
        project=project,
        name=run_name,
        group="bonus-reward-free-ranking",
        config={
            "seed": config["experiment"]["seed"],
            "n_action_steps": 50,
            "tasks": config["scope"]["task_ids"],
            "candidates": config["scope"]["candidates"],
            "videos_per_critic": config["scope"]["expected_videos"],
            "own_base_revision": config["own_critic"]["base_revision"],
            "robometer_revision": config["robometer"]["revision"],
        },
        resume="allow",
        auto_log_gpu=False,
        auto_log_cpu=False,
    )
    try:
        for task_id in config["scope"]["task_ids"]:
            task = metrics["per_task"][str(task_id)]
            trackio.log(
                {
                    "ranking/own_spearman": task["own"]["spearman_rho"],
                    "ranking/robometer_spearman": task["robometer"]["spearman_rho"],
                    "ranking/own_kendall": task["own"]["kendall_tau"],
                    "ranking/robometer_kendall": task["robometer"]["kendall_tau"],
                    "selection/own_top_regret": task["own"]["top_regret"],
                    "selection/robometer_top_regret": task["robometer"]["top_regret"],
                },
                step=int(task_id),
            )
        payload = {
            "ranking/macro_own_spearman": metrics["macro"]["own"]["spearman_rho"],
            "ranking/macro_robometer_spearman": metrics["macro"]["robometer"]["spearman_rho"],
            "ranking/macro_own_kendall": metrics["macro"]["own"]["kendall_tau"],
            "ranking/macro_robometer_kendall": metrics["macro"]["robometer"]["kendall_tau"],
            "tables/candidate_ranking": trackio.Table(columns=columns, data=table_rows),
            "plots/ranking_scatter": trackio.Image(
                SUMMARY_DIR / "ranking_scatter.png",
                caption="Predicted endpoint progress versus true success, n_action_steps=50",
            ),
            "reports/result": trackio.Markdown(report),
        }
        # Representative inputs only: episode 0 of the pretrain candidate for
        # each task. All 420 source videos remain reviewable at their source paths.
        manifest = read_jsonl(MANIFEST_PATH)
        media = [
            row
            for row in manifest
            if row["candidate"] == "pretrain" and row["episode_ix"] == 0
        ]
        for row in media:
            payload[f"rollouts/task_{row['task_id']}_pretrain_episode_0"] = trackio.Video(
                Path(row["video_path"]),
                caption=f"task {row['task_id']} pretrain episode 0; input to both critics",
            )
        trackio.log(payload, step=3)
    finally:
        trackio.finish()
    atomic_json(
        SUMMARY_DIR / "trackio_manifest.json",
        {
            "created_at": utc_now(),
            "project": project,
            "run": run_name,
            "table": "tables/candidate_ranking",
            "plot": "plots/ranking_scatter",
            "representative_videos": 3,
        },
    )


def main() -> None:
    log_results()


if __name__ == "__main__":
    main()
