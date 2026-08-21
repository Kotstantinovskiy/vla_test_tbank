from __future__ import annotations

import pytest

from seen_scene_goal_prompts_v2.aggregate import mcnemar_exact_p, wilson_interval
from seen_scene_goal_prompts_v2.constants import NONSENSE_PROMPT, PARAPHRASE_PAIRS
from seen_scene_goal_prompts_v2.plan import build_plan, scene_of


def _listing() -> list[dict]:
    rows = []

    def add(idx, name, language, vocab, goal):
        rows.append(
            {
                "env_task_id": idx,
                "name": name,
                "language": language,
                "predicate_vocab": vocab,
                "goal_state": goal,
            }
        )

    stove_vocab = ["flat_stove_1", "flat_stove_1_cook_region", "chefmate_8_frypan_1", "moka_pot_1"]
    add(0, "KITCHEN_SCENE3_turn_on_the_stove", "turn on the stove",
        stove_vocab, [["turnon", "flat_stove_1"]])
    add(1, "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove", "put the frying pan on the stove",
        stove_vocab, [["on", "chefmate_8_frypan_1", "flat_stove_1_cook_region"]])
    cabinet_vocab = [
        "akita_black_bowl_1", "plate_1", "wooden_cabinet_1",
        "wooden_cabinet_1_top_side", "wooden_cabinet_1_top_region",
        "wooden_cabinet_1_middle_region",
    ]
    add(2, "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
        "put the black bowl on top of the cabinet", cabinet_vocab,
        [["on", "akita_black_bowl_1", "wooden_cabinet_1_top_side"]])
    add(3, "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
        "put the black bowl on the plate", cabinet_vocab,
        [["on", "akita_black_bowl_1", "plate_1"]])
    add(4, "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
        "put the wine bottle on the wine rack",
        ["wine_bottle_1", "wine_rack_1", "wine_rack_1_top_region"],
        [["on", "wine_bottle_1", "wine_rack_1_top_region"]])
    add(5, "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
        "open the top drawer of the cabinet and put the bowl in it", cabinet_vocab,
        [["open", "wooden_cabinet_1_top_region"],
         ["in", "akita_black_bowl_1", "wooden_cabinet_1_top_region"]])
    add(6, "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
        "close the top drawer of the cabinet", cabinet_vocab,
        [["close", "wooden_cabinet_1_top_region"]])
    return rows


def _goal_tasks() -> list[dict]:
    return [
        {"goal_id": 0, "language": "open the middle drawer of the cabinet",
         "goal_state": [["open", "wooden_cabinet_1_middle_region"]]},
        {"goal_id": 1, "language": "put the bowl on the stove",
         "goal_state": [["on", "akita_black_bowl_1", "flat_stove_1_cook_region"]]},
        {"goal_id": 3, "language": "open the top drawer and put the bowl inside",
         "goal_state": [["in", "akita_black_bowl_1", "wooden_cabinet_1_top_region"]]},
        {"goal_id": 4, "language": "put the bowl on top of the cabinet",
         "goal_state": [["on", "akita_black_bowl_1", "wooden_cabinet_1_top_side"]]},
        {"goal_id": 7, "language": "turn on the stove",
         "goal_state": [["turnon", "flat_stove_1"]]},
        {"goal_id": 8, "language": "put the bowl on the plate",
         "goal_state": [["on", "akita_black_bowl_1", "plate_1"]]},
        {"goal_id": 9, "language": "put the wine bottle on the rack",
         "goal_state": [["on", "wine_bottle_1", "wine_rack_1_top_region"]]},
    ]


def test_scene_parsing():
    assert scene_of("KITCHEN_SCENE10_close_the_top_drawer") == "KITCHEN_SCENE10"
    with pytest.raises(ValueError):
        scene_of("no_scene_here")


def test_plan_structure():
    plan = build_plan(_listing(), _goal_tasks())
    labels = [point["label"] for point in plan["points"]]
    assert len(labels) == len(set(labels))
    blocks: dict[str, list[dict]] = {}
    for point in plan["points"]:
        blocks.setdefault(point["block"], []).append(point)
    assert len(blocks["paraphrase"]) == 4
    assert len(blocks["cross"]) == 2
    assert len(blocks["nonsense"]) == 2
    assert "absent" not in blocks
    # goal 0 lands on the lowest-id evaluable env (KITCHEN_SCENE5, id 2 in the
    # fake listing); goals 3/4/7/8/9 alias existing points; goal 1 is skipped.
    assert len(blocks["goal"]) == 1
    goal_point = blocks["goal"][0]
    assert goal_point["prompt"] == "open the middle drawer of the cabinet"
    assert goal_point["env_task_id"] == 2
    # trained baselines: 4 paraphrase envs + 2 cross envs (goal 0 reuses the
    # SCENE5 paraphrase env, so no extra trained point).
    assert len(blocks["trained"]) == 6

    slice_by_id = {item["goal_id"]: item for item in plan["notes"]["goal_slice"]}
    assert slice_by_id[1]["status"] == "skipped"
    assert slice_by_id[7]["status"] == "alias"
    assert slice_by_id[7]["alias_of"] == "trained__turn_on_the_stove"
    assert slice_by_id[7]["relationship"] == "verbatim_trained"
    assert slice_by_id[4]["status"] == "alias"
    assert slice_by_id[4]["relationship"] == "paraphrase_of_trained"
    assert slice_by_id[0]["status"] == "point"
    assert slice_by_id[0]["relationship"] == "novel_string"


def test_prompted_predicates_and_equivalence():
    plan = build_plan(_listing(), _goal_tasks())
    by_label = {point["label"]: point for point in plan["points"]}
    trained = by_label["trained__turn_on_the_stove"]
    assert trained["prompted_goal_states"] == [["turnon", "flat_stove_1"]]
    assert trained["expect_env_equivalent"] is True
    cross = by_label["cross__turn_on_the_stove"]
    assert cross["prompt"] == "put the frying pan on the stove"
    assert cross["prompted_goal_states"] == [
        ["on", "chefmate_8_frypan_1", "flat_stove_1_cook_region"]
    ]
    assert cross["expect_env_equivalent"] is False
    # Paraphrase of the drawer task: goal predicate is a strict subset of the
    # seen env's two-condition goal -> not equivalent.
    para = by_label["paraphrase__open_the_top_drawer_of_the_cabinet_and_p"]
    assert para["expect_env_equivalent"] is False
    for point in plan["points"]:
        if point["block"] == "nonsense":
            assert point["prompt"] == NONSENSE_PROMPT
            assert point["prompted_goal_states"] is None


def test_paraphrase_pairs_use_goal_language():
    plan = build_plan(_listing(), _goal_tasks())
    para = [p for p in plan["points"] if p["block"] == "paraphrase"]
    for pair, point in zip(PARAPHRASE_PAIRS, para):
        assert point["prompt"] == pair["prompt"]
        assert point["env_instruction"] == pair["seen"]
        assert point["prompted_source"] == f"goal_task_{pair['goal_ref']}"


def test_unevaluable_prompt_in_chosen_env_fails():
    listing = _listing()
    goal_tasks = _goal_tasks()
    # Break the wine env's vocabulary: its own paraphrase predicate becomes
    # unevaluable there -> plan construction must fail loudly.
    listing[4]["predicate_vocab"] = ["wine_bottle_1"]
    with pytest.raises(ValueError, match="not in scene vocabulary"):
        build_plan(listing, goal_tasks)


def test_missing_seen_instruction_fails():
    listing = [row for row in _listing() if "wine" not in row["name"].lower()]
    with pytest.raises(ValueError, match="not in benchmark"):
        build_plan(listing, _goal_tasks())


def test_mcnemar_and_wilson():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(5, 0) == pytest.approx(2 * 0.5**5)
    low, high = wilson_interval(20, 20)
    assert (low, high) == (pytest.approx(0.8389, abs=1e-3), 1.0)
