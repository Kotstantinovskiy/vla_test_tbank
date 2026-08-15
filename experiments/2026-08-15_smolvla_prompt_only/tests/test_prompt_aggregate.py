import json
from pathlib import Path

import pytest

from smolvla_prompt_only.aggregate import aggregate, metric_rows, wilson_interval
from smolvla_prompt_only.constants import CHECKPOINT_REVISION, TARGET_INSTRUCTIONS


def _raw(condition: str, successes: dict[int, int]) -> dict:
    return {
        "revision": CHECKPOINT_REVISION,
        "condition": condition,
        "tasks": {
            str(task_id): {
                "per_episode": [
                    {"success": index < successes[task_id]} for index in range(2)
                ]
            }
            for task_id in TARGET_INSTRUCTIONS
        },
    }


def test_wilson_interval_at_zero_is_bounded():
    low, high = wilson_interval(0, 20)
    assert low == 0
    assert high == pytest.approx(0.1611251581)


def test_aggregate_covers_every_task_and_prompt(tmp_path: Path):
    for condition, counts in {
        "true": {0: 2, 1: 1, 2: 0},
        "wrong": {0: 0, 1: 0, 2: 0},
        "nonsense": {0: 1, 1: 1, 2: 1},
    }.items():
        (tmp_path / f"{condition}.json").write_text(
            json.dumps(_raw(condition, counts))
        )

    summary = aggregate(tmp_path)

    assert len(summary["tasks"]) == 3
    assert len(metric_rows(summary)) == 9
    assert summary["tasks"]["0"]["conditions"]["true"]["success_rate"] == 1
    assert summary["condition_means"]["true"] == pytest.approx(0.5)
