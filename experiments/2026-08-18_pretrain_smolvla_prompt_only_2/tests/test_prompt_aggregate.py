import json
from pathlib import Path

import pytest

from pretrain_smolvla_prompt_only_2.aggregate import (
    aggregate,
    metric_rows,
    wilson_interval,
    write_report,
)
from pretrain_smolvla_prompt_only_2.constants import (
    CHECKPOINT_PATH,
    TARGET_ENV_TASK_IDS,
    TARGET_INSTRUCTIONS,
)

FAKE_SHA = "0" * 64


def _raw(condition: str, successes: dict[int, int]) -> dict:
    return {
        "model": str(CHECKPOINT_PATH),
        "checkpoint": {"model_safetensors_sha256": FAKE_SHA},
        "condition": condition,
        "tasks": {
            str(task_id): {
                "logical_task_id": task_id,
                "env_task_id": TARGET_ENV_TASK_IDS[task_id],
                "environment_instruction": TARGET_INSTRUCTIONS[task_id],
                "per_episode": [
                    {"success": index < successes.get(task_id, 0)}
                    for index in range(2)
                ],
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
        "true": {0: 2, 1: 1},
        "wrong": {},
        "nonsense": {5: 1},
    }.items():
        (tmp_path / f"{condition}.json").write_text(
            json.dumps(_raw(condition, counts))
        )

    summary = aggregate(tmp_path)

    assert len(summary["tasks"]) == 10
    assert len(metric_rows(summary)) == 30
    assert summary["tasks"]["0"]["conditions"]["true"]["success_rate"] == 1
    assert summary["condition_means"]["true"] == pytest.approx(0.15)

    report = tmp_path / "REPORT.md"
    write_report(summary, report)
    assert "all ten" in report.read_text()


def test_aggregate_rejects_mixed_weights(tmp_path: Path):
    payloads = {
        "true": _raw("true", {}),
        "wrong": _raw("wrong", {}),
        "nonsense": _raw("nonsense", {}),
    }
    payloads["wrong"]["checkpoint"]["model_safetensors_sha256"] = "1" * 64
    for condition, payload in payloads.items():
        (tmp_path / f"{condition}.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="different weights"):
        aggregate(tmp_path)
