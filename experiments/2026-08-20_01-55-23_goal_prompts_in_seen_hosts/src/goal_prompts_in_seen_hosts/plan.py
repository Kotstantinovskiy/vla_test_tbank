from __future__ import annotations

import re
from typing import Any, Sequence

from .constants import NONSENSE_PROMPT


HOSTS = (
    {
        "goal_id": 0,
        "host_name": "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
        "host_instruction": "close the top drawer of the cabinet",
        "goal_prompt": "open the middle drawer of the cabinet",
        "goal_state": [["open", "wooden_cabinet_1_middle_region"]],
        "mapping": "exact fixture identity",
    },
    {
        "goal_id": 1,
        "host_name": "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
        "host_instruction": "put the white bowl on top of the cabinet",
        "goal_prompt": "put the bowl on the stove",
        "goal_state": [["on", "white_bowl_1", "flat_stove_1_cook_region"]],
        "mapping": "generic goal 'bowl' -> the host's white_bowl_1",
    },
    {
        "goal_id": 2,
        "host_name": "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
        "host_instruction": "put the wine bottle on the wine rack",
        "goal_prompt": "put the wine bottle on top of the cabinet",
        "goal_state": [["on", "wine_bottle_1", "white_cabinet_1_top_side"]],
        "mapping": "generic goal 'cabinet' -> the host's white_cabinet_1",
    },
)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def predicate_args(goal_states: Sequence[Sequence[str]]) -> set[str]:
    return {arg for state in goal_states for arg in state[1:]}


def build_plan(
    listing: Sequence[dict], goal_listing: Sequence[dict]
) -> dict[str, Any]:
    by_name = {row["name"]: row for row in listing}
    if len(by_name) != len(listing):
        raise ValueError("Duplicate LIBERO task names")
    goal_by_id = {row["env_task_id"]: row for row in goal_listing}
    points: list[dict[str, Any]] = []
    mappings = []
    for spec in HOSTS:
        goal_source = goal_by_id.get(spec["goal_id"])
        if goal_source is None:
            raise ValueError(f"Goal task not found: {spec['goal_id']}")
        if goal_source["language"] != spec["goal_prompt"]:
            raise ValueError(
                f"Goal prompt mismatch for task {spec['goal_id']}: "
                f"{goal_source['language']!r} != {spec['goal_prompt']!r}"
            )
        env = by_name.get(spec["host_name"])
        if env is None:
            raise ValueError(f"Host task not found: {spec['host_name']}")
        if env["language"] != spec["host_instruction"]:
            raise ValueError(
                f"Host language mismatch for {spec['host_name']}: "
                f"{env['language']!r}"
            )
        missing = predicate_args(spec["goal_state"]) - set(env["predicate_vocab"])
        if missing:
            raise ValueError(
                f"Goal {spec['goal_id']} missing predicate args in host: "
                f"{sorted(missing)}"
            )
        common = {
            "logical_task_id": spec["goal_id"],
            "env_task_id": env["env_task_id"],
            "env_name": env["name"],
            "env_instruction": env["language"],
            "pair_id": f"goal_{spec['goal_id']}",
        }
        points.extend(
            [
                {
                    **common,
                    "label": f"seen__goal_{spec['goal_id']}",
                    "block": "seen",
                    "prompt": env["language"],
                    "prompted_goal_states": env["goal_state"],
                    "prompted_source": "host_env_bddl",
                    "expect_env_equivalent": True,
                },
                {
                    **common,
                    "label": f"goal__goal_{spec['goal_id']}",
                    "block": "goal",
                    "prompt": spec["goal_prompt"],
                    "prompted_goal_states": spec["goal_state"],
                    "prompted_source": f"goal_task_{spec['goal_id']}_semantic_mapping",
                    "goal_source_env_task_id": goal_source["env_task_id"],
                    "goal_source_env_name": goal_source["name"],
                    "expect_env_equivalent": False,
                },
                {
                    **common,
                    "label": f"nonsense__goal_{spec['goal_id']}",
                    "block": "nonsense",
                    "prompt": NONSENSE_PROMPT,
                    "prompted_goal_states": None,
                    "prompted_source": "none",
                    "expect_env_equivalent": False,
                },
            ]
        )
        mappings.append(
            {
                "goal_id": spec["goal_id"],
                "goal_prompt": spec["goal_prompt"],
                "goal_source_name": goal_source["name"],
                "host_name": spec["host_name"],
                "mapping": spec["mapping"],
            }
        )
    labels = [point["label"] for point in points]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate evaluation labels")
    return {"points": points, "notes": {"semantic_mappings": mappings}}
