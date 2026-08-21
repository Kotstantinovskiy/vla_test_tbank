from __future__ import annotations

"""Pure plan-building logic for probe 3 (hermetically testable).

Inputs
------
goal_listing: libero_goal rows — ``env_task_id``, ``name``, ``language``,
    ``goal_state`` (own parsed goal predicate), ``predicate_vocab``.
seen_languages: the set of libero_90 instruction strings (used ONLY to
    validate that twin/anchor prompts are verbatim-trained).

Blocks
------
true: each probe env under its own goal instruction (k=0 baseline; prompted
    predicate = env predicate; consistency-checked).
seen_twin: the same env under the verbatim libero_90-trained string of the
    SAME skill (prompted predicate = env predicate).  The decisive cells:
    does the exact trained string unlock the skill in the novel scene?
seen_cross: the two bowl twins swapped between their goal envs, scored by the
    predicate of the PROMPTED skill (instruction following in a novel scene).
nonsense: env metric only.

Every point carries ``behavior_target`` — the env task's manipulated object
(first argument of its goal predicate) — fixed per env so behavioral metrics
compare across conditions.
"""

import re
from typing import Any, Sequence

from .constants import (
    ANCHOR_TRUE_TRAINED,
    NONSENSE_PROMPT,
    SEEN_CROSS_PAIR,
    TWIN_PAIRS,
)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def find_by_language(listing: Sequence[dict], language: str) -> dict:
    matches = [item for item in listing if item["language"] == language]
    if not matches:
        raise ValueError(f"Instruction not in benchmark: {language!r}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous instruction in libero_goal: {language!r}")
    return matches[0]


def behavior_target_of(goal_state: Sequence[Sequence[str]]) -> str:
    """The manipulated object of a goal predicate: first argument of the
    first condition (for unary predicates like turnon/open — the fixture or
    region itself)."""

    return goal_state[0][1]


def build_plan(
    goal_listing: Sequence[dict], seen_languages: set[str]
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    notes: dict[str, Any] = {}

    def add(
        block: str,
        env: dict,
        prompt: str,
        prompted_goal_states: list[list[str]] | None,
        prompted_source: str,
    ) -> dict:
        label = f"{block}__{slug(env['language'])[:40]}"
        if any(point["label"] == label for point in points):
            raise ValueError(f"Duplicate plan label {label}")
        if prompted_goal_states is not None:
            vocab = set(env["predicate_vocab"])
            missing = sorted(
                {
                    arg
                    for state in prompted_goal_states
                    for arg in state[1:]
                    if arg not in vocab
                }
            )
            if missing:
                raise ValueError(
                    f"{label}: prompted predicate args {missing} not in the "
                    f"goal scene vocabulary"
                )
        point = {
            "label": label,
            "block": block,
            "env_task_id": env["env_task_id"],
            "env_name": env["name"],
            "env_instruction": env["language"],
            "prompt": prompt,
            "prompted_goal_states": prompted_goal_states,
            "prompted_source": prompted_source,
            "expect_env_equivalent": (
                prompted_goal_states is not None
                and [list(s) for s in prompted_goal_states]
                == [list(s) for s in env["goal_state"]]
            ),
            "behavior_target": behavior_target_of(env["goal_state"]),
        }
        points.append(point)
        return point

    # Anchor: true instruction is itself a verbatim trained string.
    anchor = find_by_language(goal_listing, ANCHOR_TRUE_TRAINED)
    if ANCHOR_TRUE_TRAINED not in seen_languages:
        raise ValueError(
            f"Anchor {ANCHOR_TRUE_TRAINED!r} is not a libero_90 instruction"
        )
    add(
        "true",
        anchor,
        anchor["language"],
        [list(s) for s in anchor["goal_state"]],
        "env_bddl",
    )
    notes["anchor"] = {
        "env": anchor["name"],
        "note": "true == verbatim trained string -> pure scene effect",
    }

    # Twin pairs: true + seen_twin on each goal env.
    twin_records = []
    for pair in TWIN_PAIRS:
        env = find_by_language(goal_listing, pair["goal"])
        if pair["seen_twin"] not in seen_languages:
            raise ValueError(
                f"Twin {pair['seen_twin']!r} is not a libero_90 instruction"
            )
        goal_states = [list(s) for s in env["goal_state"]]
        add("true", env, env["language"], goal_states, "env_bddl")
        add("seen_twin", env, pair["seen_twin"], goal_states, "env_bddl_twin")
        twin_records.append({"goal": pair["goal"], "seen_twin": pair["seen_twin"]})
    notes["twin_pairs"] = twin_records

    # seen_cross: swap the two bowl twins between their goal envs; score by
    # the PROMPTED skill's predicate.
    env_a = find_by_language(goal_listing, SEEN_CROSS_PAIR[0])
    env_b = find_by_language(goal_listing, SEEN_CROSS_PAIR[1])
    twin_by_goal = {pair["goal"]: pair["seen_twin"] for pair in TWIN_PAIRS}
    add(
        "seen_cross",
        env_a,
        twin_by_goal[SEEN_CROSS_PAIR[1]],
        [list(s) for s in env_b["goal_state"]],
        f"goal_env_{env_b['env_task_id']}",
    )
    add(
        "seen_cross",
        env_b,
        twin_by_goal[SEEN_CROSS_PAIR[0]],
        [list(s) for s in env_a["goal_state"]],
        f"goal_env_{env_a['env_task_id']}",
    )
    notes["seen_cross"] = {
        "envs": [env_a["name"], env_b["name"]],
        "prompts": [twin_by_goal[SEEN_CROSS_PAIR[1]], twin_by_goal[SEEN_CROSS_PAIR[0]]],
    }

    # Nonsense: anchor env + the first twin env (env metric only).
    for env in (anchor, find_by_language(goal_listing, TWIN_PAIRS[0]["goal"])):
        add("nonsense", env, NONSENSE_PROMPT, None, "none")

    return {"points": points, "notes": notes}
