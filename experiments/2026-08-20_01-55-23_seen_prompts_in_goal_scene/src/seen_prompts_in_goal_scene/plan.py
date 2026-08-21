from __future__ import annotations

from typing import Any, Sequence

from .constants import NONSENSE_PROMPT

HOST_NAME = "open_the_middle_drawer_of_the_cabinet"
HOST_INSTRUCTION = "open the middle drawer of the cabinet"

SEEN_PROMPTS = (
    (
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it",
        "close the top drawer of the cabinet and put the black bowl on top of it",
        [
            ["close", "wooden_cabinet_1_top_region"],
            ["on", "akita_black_bowl_1", "wooden_cabinet_1_top_side"],
        ],
    ),
    (
        "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
        "put the black bowl in the top drawer of the cabinet",
        [["in", "akita_black_bowl_1", "wooden_cabinet_1_top_region"]],
    ),
    (
        "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
        "open the bottom drawer of the cabinet",
        [["open", "wooden_cabinet_1_bottom_region"]],
    ),
    (
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
        "open the top drawer of the cabinet",
        [["open", "wooden_cabinet_1_top_region"]],
    ),
    (
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
        "open the top drawer of the cabinet and put the bowl in it",
        [
            ["open", "wooden_cabinet_1_top_region"],
            ["in", "akita_black_bowl_1", "wooden_cabinet_1_top_region"],
        ],
    ),
    (
        "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
        "put the black bowl on the plate",
        [["on", "akita_black_bowl_1", "plate_1"]],
    ),
    (
        "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
        "put the black bowl on top of the cabinet",
        [["on", "akita_black_bowl_1", "wooden_cabinet_1_top_side"]],
    ),
    (
        "KITCHEN_SCENE3_turn_on_the_stove",
        "turn on the stove",
        [["turnon", "flat_stove_1"]],
    ),
    (
        "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
        "put the wine bottle on the wine rack",
        [["on", "wine_bottle_1", "wine_rack_1_top_region"]],
    ),
)

EXCLUDED = (
    {
        "prompt": "close the top drawer of the cabinet",
        "reason": "top drawer is closed at reset, so success is trivial",
    },
    {
        "prompt": "turn off the stove",
        "reason": "stove is off at reset, so success is trivial",
    },
    {
        "prompt": "put the black bowl at the front on the plate",
        "reason": "front selects one of three bowls in seen; goal scene has one bowl",
    },
)


def predicate_args(goal_states):
    return {arg for state in goal_states for arg in state[1:]}


def build_plan(
    listing: Sequence[dict], seen_listing: Sequence[dict]
) -> dict[str, Any]:
    matches = [row for row in listing if row["name"] == HOST_NAME]
    if len(matches) != 1:
        raise ValueError(f"Expected one fixed goal host {HOST_NAME}, got {len(matches)}")
    env = matches[0]
    if env["language"] != HOST_INSTRUCTION:
        raise ValueError(
            f"Host instruction mismatch: {env['language']!r} != {HOST_INSTRUCTION!r}"
        )
    vocab = set(env["predicate_vocab"])
    seen_by_name = {row["name"]: row for row in seen_listing}
    if len(seen_by_name) != len(seen_listing):
        raise ValueError("Duplicate LIBERO-90 task names")
    points = [
        {
            "label": "true_goal__host",
            "block": "true_goal",
            "logical_task_id": "host",
            "env_task_id": env["env_task_id"],
            "env_name": env["name"],
            "env_instruction": env["language"],
            "pair_id": None,
            "prompt": env["language"],
            "prompted_goal_states": env["goal_state"],
            "prompted_source": "host_env_bddl",
            "expect_env_equivalent": True,
        }
    ]
    for seen_id, (source_name, prompt, goal_state) in enumerate(SEEN_PROMPTS):
        source = seen_by_name.get(source_name)
        if source is None:
            raise ValueError(f"Seen source task not found: {source_name}")
        if source["language"] != prompt:
            raise ValueError(
                f"Seen prompt mismatch for {source_name}: "
                f"{source['language']!r} != {prompt!r}"
            )
        if source["goal_state"] != goal_state:
            raise ValueError(
                f"Seen predicate mismatch for {source_name}: "
                f"{source['goal_state']!r} != {goal_state!r}"
            )
        missing = predicate_args(goal_state) - vocab
        if missing:
            raise ValueError(f"Seen prompt {seen_id} missing args: {sorted(missing)}")
        points.append(
            {
                "label": f"seen_prompt__seen_{seen_id}",
                "block": "seen_prompt",
                "logical_task_id": seen_id,
                "env_task_id": env["env_task_id"],
                "env_name": env["name"],
                "env_instruction": env["language"],
                "pair_id": None,
                "prompt": prompt,
                "prompted_goal_states": goal_state,
                "prompted_source": "verbatim_libero_90_goal_state",
                "seen_source_env_task_id": source["env_task_id"],
                "seen_source_env_name": source["name"],
                "expect_env_equivalent": False,
            }
        )
    points.append(
        {
            "label": "nonsense__host",
            "block": "nonsense",
            "logical_task_id": "host",
            "env_task_id": env["env_task_id"],
            "env_name": env["name"],
            "env_instruction": env["language"],
            "pair_id": None,
            "prompt": NONSENSE_PROMPT,
            "prompted_goal_states": None,
            "prompted_source": "none",
            "expect_env_equivalent": False,
        }
    )
    labels = [point["label"] for point in points]
    if len(labels) != 11 or len(labels) != len(set(labels)):
        raise ValueError("Expected 11 unique points (9 seen + 2 controls)")
    return {
        "points": points,
        "notes": {
            "fixed_host": HOST_NAME,
            "candidate_count_before_exclusions": 12,
            "selected_seen_prompts": 9,
            "excluded": EXCLUDED,
        },
    }
