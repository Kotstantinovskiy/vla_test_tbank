from __future__ import annotations

from typing import Any, Sequence


ANCHORS = (
    ("KITCHEN_SCENE3_turn_on_the_stove", "turn on the stove", "turn on stove"),
    (
        "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
        "put the frying pan on the stove",
        "put frying pan on the stove",
    ),
    (
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
        "close the top drawer of the cabinet",
        "close top drawer of the cabinet",
    ),
    (
        "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
        "put the black bowl on top of the cabinet",
        "put black bowl on top of the cabinet",
    ),
    (
        "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
        "put the black bowl on the plate",
        "put black bowl on the plate",
    ),
    (
        "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
        "put the wine bottle on the wine rack",
        "put wine bottle on the wine rack",
    ),
    (
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
        "open the top drawer of the cabinet and put the bowl in it",
        "open top drawer of the cabinet and put the bowl in it",
    ),
    ("KITCHEN_SCENE7_open_the_microwave", "open the microwave", "open microwave"),
    (
        "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "pick up the alphabet soup and put it in the basket",
        "pick up alphabet soup and put it in the basket",
    ),
    (
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "pick up the book and place it in the left compartment of the caddy",
        "pick up book and place it in the left compartment of the caddy",
    ),
)


def drop_first_the(text: str) -> str:
    words = text.split()
    index = words.index("the")
    return " ".join(words[:index] + words[index + 1 :])


def build_plan(listing: Sequence[dict]) -> dict[str, Any]:
    by_name = {row["name"]: row for row in listing}
    points = []
    for logical_id, (name, exact_prompt, modified_prompt) in enumerate(ANCHORS):
        env = by_name.get(name)
        if env is None:
            raise ValueError(f"Anchor task not found: {name}")
        if env["language"] != exact_prompt:
            raise ValueError(
                f"Language mismatch for {name}: {env['language']!r} != {exact_prompt!r}"
            )
        if drop_first_the(exact_prompt) != modified_prompt:
            raise ValueError(f"Task {logical_id} is not exactly one first-'the' deletion")
        common = {
            "logical_task_id": logical_id,
            "env_task_id": env["env_task_id"],
            "env_name": env["name"],
            "env_instruction": env["language"],
            "pair_id": f"task_{logical_id}",
            "prompted_goal_states": env["goal_state"],
            "prompted_source": "host_env_bddl",
            "expect_env_equivalent": True,
        }
        exact_label = f"exact__task_{logical_id}"
        points.extend(
            [
                {
                    **common,
                    "label": exact_label,
                    "block": "exact",
                    "prompt": exact_prompt,
                },
                {
                    **common,
                    "label": f"article_drop__task_{logical_id}",
                    "block": "article_drop",
                    "prompt": modified_prompt,
                    "reference_label": exact_label,
                },
            ]
        )
    labels = [point["label"] for point in points]
    if len(labels) != 20 or len(labels) != len(set(labels)):
        raise ValueError("Expected 20 unique evaluation points")
    return {
        "points": points,
        "notes": {
            "edit_rule": "delete exactly the first standalone token 'the'",
            "other_edits": "none",
        },
    }
