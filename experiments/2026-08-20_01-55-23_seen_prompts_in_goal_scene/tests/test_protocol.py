from __future__ import annotations

import pytest

from seen_prompts_in_goal_scene.constants import EVAL_BATCH_SIZE, N_EVAL_EPISODES, noise_seed
from seen_prompts_in_goal_scene.plan import EXCLUDED, HOST_INSTRUCTION, HOST_NAME, SEEN_PROMPTS, build_plan


def _listing():
    vocab = sorted(
        {
            arg
            for _, _, states in SEEN_PROMPTS
            for state in states
            for arg in state[1:]
        }
        | {"wooden_cabinet_1_middle_region"}
    )
    return [
        {
            "env_task_id": 0,
            "name": HOST_NAME,
            "language": HOST_INSTRUCTION,
            "predicate_vocab": vocab,
            "goal_state": [["open", "wooden_cabinet_1_middle_region"]],
        }
    ]


def _seen_listing():
    return [
        {
            "env_task_id": index,
            "name": name,
            "language": prompt,
            "goal_state": states,
        }
        for index, (name, prompt, states) in enumerate(SEEN_PROMPTS)
    ]


def test_nine_nontrivial_seen_prompts_plus_controls():
    assert len(SEEN_PROMPTS) == 9
    assert len(EXCLUDED) == 3
    plan = build_plan(_listing(), _seen_listing())
    assert len(plan["points"]) == 11
    assert sum(point["block"] == "seen_prompt" for point in plan["points"]) == 9
    assert {point["block"] for point in plan["points"]} == {
        "true_goal",
        "seen_prompt",
        "nonsense",
    }


def test_trivial_and_ambiguous_prompts_are_excluded():
    selected = {prompt for _, prompt, _ in SEEN_PROMPTS}
    excluded = {item["prompt"] for item in EXCLUDED}
    assert selected.isdisjoint(excluded)
    assert "close the top drawer of the cabinet" in excluded
    assert "turn off the stove" in excluded


def test_missing_goal_scene_object_fails():
    listing = _listing()
    listing[0]["predicate_vocab"].remove("wine_bottle_1")
    with pytest.raises(ValueError, match="missing args"):
        build_plan(listing, _seen_listing())


def test_seen_source_instruction_and_predicate_are_verified():
    sources = _seen_listing()
    sources[0]["goal_state"] = [["wrong", "object"]]
    with pytest.raises(ValueError, match="Seen predicate mismatch"):
        build_plan(_listing(), sources)


def test_seed_protocol():
    assert EVAL_BATCH_SIZE == 1
    assert len({noise_seed(i) for i in range(N_EVAL_EPISODES)}) == N_EVAL_EPISODES
