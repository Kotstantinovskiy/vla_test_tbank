from __future__ import annotations

import pytest

from seen_article_drop.constants import EVAL_BATCH_SIZE, N_EVAL_EPISODES, noise_seed
from seen_article_drop.plan import ANCHORS, build_plan, drop_first_the


def _listing():
    return [
        {
            "env_task_id": index,
            "name": name,
            "language": exact,
            "predicate_vocab": [f"object_{index}"],
            "goal_state": [["done", f"object_{index}"]],
        }
        for index, (name, exact, _) in enumerate(ANCHORS)
    ]


def test_every_edit_is_exactly_first_article_deletion():
    assert len(ANCHORS) == 10
    for _, exact, modified in ANCHORS:
        assert drop_first_the(exact) == modified
        assert len(exact.split()) == len(modified.split()) + 1


def test_drop_requires_article():
    with pytest.raises(ValueError):
        drop_first_the("open microwave")


def test_plan_is_paired_and_predicate_equivalent():
    plan = build_plan(_listing())
    assert len(plan["points"]) == 20
    by_label = {point["label"]: point for point in plan["points"]}
    for index in range(10):
        exact = by_label[f"exact__task_{index}"]
        changed = by_label[f"article_drop__task_{index}"]
        assert changed["reference_label"] == exact["label"]
        assert changed["prompted_goal_states"] == exact["prompted_goal_states"]


def test_seed_protocol():
    assert EVAL_BATCH_SIZE == 1
    assert len({noise_seed(i) for i in range(N_EVAL_EPISODES)}) == N_EVAL_EPISODES
