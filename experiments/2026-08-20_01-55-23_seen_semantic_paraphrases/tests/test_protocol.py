from __future__ import annotations

from seen_semantic_paraphrases.constants import (
    EVAL_BATCH_SIZE,
    MASTER_SEED,
    N_EVAL_EPISODES,
    noise_seed,
)
from seen_semantic_paraphrases.plan import ANCHORS, build_plan


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


def test_ten_frozen_nonidentical_paraphrases():
    assert len(ANCHORS) == 10
    assert all(exact != paraphrase for _, exact, paraphrase in ANCHORS)
    assert len({name for name, _, _ in ANCHORS}) == 10


def test_plan_is_paired_and_predicate_equivalent():
    plan = build_plan(_listing())
    assert len(plan["points"]) == 20
    by_label = {point["label"]: point for point in plan["points"]}
    for index in range(10):
        exact = by_label[f"exact__task_{index}"]
        paraphrase = by_label[f"paraphrase__task_{index}"]
        assert paraphrase["reference_label"] == exact["label"]
        assert paraphrase["env_task_id"] == exact["env_task_id"]
        assert paraphrase["prompted_goal_states"] == exact["prompted_goal_states"]
        assert paraphrase["expect_env_equivalent"] is True


def test_seed_protocol():
    assert EVAL_BATCH_SIZE == 1
    assert [noise_seed(i) for i in range(N_EVAL_EPISODES)] == list(
        range(MASTER_SEED, MASTER_SEED + N_EVAL_EPISODES)
    )
