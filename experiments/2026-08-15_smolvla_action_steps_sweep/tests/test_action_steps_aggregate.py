from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from smolvla_action_steps.aggregate import aggregate, summarize_point, wilson_interval
from smolvla_action_steps.constants import (
    ACTION_STEPS,
    CHECKPOINT_REVISION,
    DEMO_BUDGETS,
    MASTER_SEED,
    TARGET_INSTRUCTIONS,
    TARGET_SUITE,
)


def _result(task_id: int, budget: int) -> dict:
    sweep = {}
    for step in ACTION_STEPS:
        successes = 1 if task_id == 1 and budget == 5 and step in {5, 10} else 0
        episodes = [
            {"success": index < successes, "seed": MASTER_SEED + index}
            for index in range(2)
        ]
        sweep[str(step)] = {
            "per_episode": episodes,
            "aggregated": {"eval_s": 1.0},
            "video_paths": [],
        }
    return {
        "model": "checkpoint",
        "revision": CHECKPOINT_REVISION if budget == 0 else None,
        "task_id": task_id,
        "demo_budget": budget,
        "condition": "true",
        "suite": TARGET_SUITE,
        "seed": MASTER_SEED,
        "weights_modified": False,
        "sweep": sweep,
    }


def _write_fixture(root: Path) -> Path:
    for task_id in TARGET_INSTRUCTIONS:
        for budget in DEMO_BUDGETS:
            if budget == 0:
                path = root / "zero_shot" / f"task_{task_id}.json"
            else:
                path = root / "adapted" / f"task_{task_id}" / f"k_{budget}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_result(task_id, budget)))
    baseline = root.parent / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "task_success_rates": {
                    str(task): {str(budget): 0.0 for budget in DEMO_BUDGETS}
                    for task in TARGET_INSTRUCTIONS
                },
                "mean_cost_curve": {str(budget): 0.0 for budget in DEMO_BUDGETS},
            }
        )
    )
    return baseline


def test_wilson_and_episode_summary() -> None:
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)
    summary = summarize_point(
        {"per_episode": [{"success": True}, {"success": False}], "video_paths": []}
    )
    assert summary["success_rate"] == 0.5
    assert summary["ci95_low"] < 0.5 < summary["ci95_high"]


def test_aggregate_keeps_all_ties_and_selects_smallest(tmp_path: Path) -> None:
    results = tmp_path / "raw"
    baseline = _write_fixture(results)
    summary = aggregate(results, baseline)
    best = summary["tasks"]["1"]["budgets"]["5"]
    assert best["best_action_steps"] == [5, 10]
    assert best["selected_best_action_steps"] == 5
    assert best["points"]["5"]["delta_vs_paired_50"] == 0.5
    assert summary["zero_shot_any_true_success"] is False
