from __future__ import annotations

import pytest

from goal_scene_seen_prompts.aggregate import mcnemar_exact_p, median, wilson_interval
from goal_scene_seen_prompts.constants import NONSENSE_PROMPT, TWIN_PAIRS
from goal_scene_seen_prompts.plan import behavior_target_of, build_plan


def _goal_listing() -> list[dict]:
    vocab = [
        "akita_black_bowl_1", "plate_1", "wine_bottle_1", "wine_rack_1",
        "wine_rack_1_top_region", "flat_stove_1", "wooden_cabinet_1",
        "wooden_cabinet_1_top_side", "wooden_cabinet_1_top_region",
    ]
    rows = [
        (3, "open_the_top_drawer_and_put_the_bowl_inside",
         "open the top drawer and put the bowl inside",
         [["in", "akita_black_bowl_1", "wooden_cabinet_1_top_region"]]),
        (4, "put_the_bowl_on_top_of_the_cabinet",
         "put the bowl on top of the cabinet",
         [["on", "akita_black_bowl_1", "wooden_cabinet_1_top_side"]]),
        (7, "turn_on_the_stove", "turn on the stove",
         [["turnon", "flat_stove_1"]]),
        (8, "put_the_bowl_on_the_plate", "put the bowl on the plate",
         [["on", "akita_black_bowl_1", "plate_1"]]),
        (9, "put_the_wine_bottle_on_the_rack", "put the wine bottle on the rack",
         [["on", "wine_bottle_1", "wine_rack_1_top_region"]]),
    ]
    return [
        {
            "env_task_id": tid,
            "name": name,
            "language": lang,
            "predicate_vocab": vocab,
            "goal_state": goal,
        }
        for tid, name, lang, goal in rows
    ]


def _seen_languages() -> set[str]:
    return {pair["seen_twin"] for pair in TWIN_PAIRS} | {"turn on the stove"}


def test_plan_structure():
    plan = build_plan(_goal_listing(), _seen_languages())
    blocks: dict[str, list[dict]] = {}
    for point in plan["points"]:
        blocks.setdefault(point["block"], []).append(point)
    assert len(blocks["true"]) == 5
    assert len(blocks["seen_twin"]) == 4
    assert len(blocks["seen_cross"]) == 2
    assert len(blocks["nonsense"]) == 2
    labels = [point["label"] for point in plan["points"]]
    assert len(labels) == len(set(labels))
    for point in blocks["nonsense"]:
        assert point["prompt"] == NONSENSE_PROMPT
        assert point["prompted_goal_states"] is None
    # seen_twin scores the env's own predicate under the trained string.
    for pair, point in zip(TWIN_PAIRS, blocks["seen_twin"]):
        assert point["prompt"] == pair["seen_twin"]
        assert point["env_instruction"] == pair["goal"]
        assert point["expect_env_equivalent"] is True


def test_seen_cross_swaps_predicates():
    plan = build_plan(_goal_listing(), _seen_languages())
    cross = {p["env_instruction"]: p for p in plan["points"] if p["block"] == "seen_cross"}
    plate = cross["put the bowl on the plate"]
    cabinet = cross["put the bowl on top of the cabinet"]
    assert plate["prompt"] == "put the black bowl on top of the cabinet"
    assert plate["prompted_goal_states"] == [
        ["on", "akita_black_bowl_1", "wooden_cabinet_1_top_side"]
    ]
    assert plate["expect_env_equivalent"] is False
    assert cabinet["prompt"] == "put the black bowl on the plate"
    assert cabinet["prompted_goal_states"] == [
        ["on", "akita_black_bowl_1", "plate_1"]
    ]


def test_behavior_target_fixed_per_env():
    plan = build_plan(_goal_listing(), _seen_languages())
    by_env: dict[int, set[str]] = {}
    for point in plan["points"]:
        by_env.setdefault(point["env_task_id"], set()).add(point["behavior_target"])
    for env, targets in by_env.items():
        assert len(targets) == 1, (env, targets)
    assert behavior_target_of([["turnon", "flat_stove_1"]]) == "flat_stove_1"


def test_untrained_twin_fails():
    with pytest.raises(ValueError, match="not a libero_90 instruction"):
        build_plan(_goal_listing(), {"turn on the stove"})


def test_stats_helpers():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(5, 0) == pytest.approx(2 * 0.5**5)
    low, high = wilson_interval(0, 20)
    assert low == 0.0 and high == pytest.approx(0.161, abs=1e-3)
    assert median([3.0, None, 1.0]) == 2.0
    assert median([]) is None
