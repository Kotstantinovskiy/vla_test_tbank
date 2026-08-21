from __future__ import annotations

import pytest

from goal_prompts_in_seen_hosts.aggregate import mcnemar_exact_p, wilson_interval
from goal_prompts_in_seen_hosts.constants import (
    EVAL_BATCH_SIZE,
    MASTER_SEED,
    N_EVAL_EPISODES,
    noise_seed,
)
from goal_prompts_in_seen_hosts.determinism import compare
from goal_prompts_in_seen_hosts.plan import HOSTS, build_plan


def _listing():
    rows = []
    for index, spec in enumerate(HOSTS):
        goal_state = [["native", f"fixture_{index}"]]
        vocab = {f"fixture_{index}"}
        vocab.update(arg for state in spec["goal_state"] for arg in state[1:])
        rows.append(
            {
                "env_task_id": index,
                "name": spec["host_name"],
                "language": spec["host_instruction"],
                "predicate_vocab": sorted(vocab),
                "goal_state": goal_state,
            }
        )
    return rows


def _goal_listing():
    return [
        {
            "env_task_id": spec["goal_id"],
            "name": f"goal_task_{spec['goal_id']}",
            "language": spec["goal_prompt"],
        }
        for spec in HOSTS
    ]


def test_plan_has_three_conditions_per_goal():
    plan = build_plan(_listing(), _goal_listing())
    assert len(plan["points"]) == 9
    assert {point["block"] for point in plan["points"]} == {
        "seen",
        "goal",
        "nonsense",
    }
    assert {point["logical_task_id"] for point in plan["points"]} == {0, 1, 2}


def test_semantic_mappings_are_frozen():
    plan = build_plan(_listing(), _goal_listing())
    goal = {
        point["logical_task_id"]: point
        for point in plan["points"]
        if point["block"] == "goal"
    }
    assert goal[1]["prompted_goal_states"] == [
        ["on", "white_bowl_1", "flat_stove_1_cook_region"]
    ]
    assert goal[2]["prompted_goal_states"] == [
        ["on", "wine_bottle_1", "white_cabinet_1_top_side"]
    ]


def test_missing_predicate_argument_fails():
    listing = _listing()
    listing[1]["predicate_vocab"].remove("white_bowl_1")
    with pytest.raises(ValueError, match="missing predicate args"):
        build_plan(listing, _goal_listing())


def test_first_three_goal_prompts_are_verified_against_suite_order():
    goals = _goal_listing()
    goals[0]["language"] = "wrong goal prompt"
    with pytest.raises(ValueError, match="Goal prompt mismatch"):
        build_plan(_listing(), goals)


def test_episode_and_noise_seed_protocol():
    assert EVAL_BATCH_SIZE == 1
    assert [noise_seed(i) for i in range(N_EVAL_EPISODES)] == list(
        range(MASTER_SEED, MASTER_SEED + N_EVAL_EPISODES)
    )


def test_determinism_comparison_ignores_video_path():
    episode = {
        "episode_ix": 0,
        "env_seed": 1000,
        "noise_seed": 1000,
        "success": True,
        "prompted_success": True,
        "prompted_first_step": 10,
        "env_task_success": False,
        "env_first_step": None,
        "steps": 10,
    }
    left = {
        "label": "x",
        "checkpoint_sha256": "abc",
        "successes": 1,
        "per_episode": [{**episode, "video_path": "a.mp4"}],
    }
    right = {
        "label": "x",
        "checkpoint_sha256": "abc",
        "successes": 1,
        "per_episode": [{**episode, "video_path": "b.mp4"}],
    }
    assert compare(left, right)["passed"] is True


def test_statistics():
    low, high = wilson_interval(1, 20)
    assert low == pytest.approx(0.0089, abs=1e-3)
    assert high == pytest.approx(0.2359, abs=1e-3)
    assert mcnemar_exact_p(5, 0) == pytest.approx(0.0625)
