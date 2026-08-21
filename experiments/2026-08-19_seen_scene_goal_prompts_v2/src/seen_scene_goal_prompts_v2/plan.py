from __future__ import annotations

"""Pure plan-building logic (hermetically testable).

Inputs
------
listing: libero_90 rows — dicts with ``env_task_id``, ``name``,
    ``language``, ``predicate_vocab`` (objects+fixtures+regions declared in
    the scene BDDL) and ``goal_state`` (the env's own parsed goal predicate).
goal_tasks: libero_goal rows — dicts with ``goal_id``, ``language`` and
    ``goal_state``.

Output: frozen evaluation plan. Every point carries the PROMPTED task's
``prompted_goal_states`` (or None for nonsense), which redefines success in
v2, plus ``expect_env_equivalent`` when the prompted predicate is literally
the env's own goal predicate (free measurement cross-check).
"""

import re
from typing import Any, Sequence

from .constants import (
    CROSS_ANCHOR_INSTRUCTION,
    NONSENSE_PROMPT,
    PARAPHRASE_PAIRS,
)

SCENE_RE = re.compile(r"^([A-Z_]+_SCENE\d+)")


def scene_of(name: str) -> str:
    match = SCENE_RE.match(name)
    if match is None:
        raise ValueError(f"Cannot parse scene from task name {name!r}")
    return match.group(1)


def find_by_language(listing: Sequence[dict], language: str) -> list[dict]:
    return [item for item in listing if item["language"] == language]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def predicate_args(goal_states: Sequence[Sequence[str]]) -> set[str]:
    return {arg for state in goal_states for arg in state[1:]}


def evaluable_in(env: dict, goal_states: Sequence[Sequence[str]]) -> bool:
    return predicate_args(goal_states) <= set(env["predicate_vocab"])


def build_plan(listing: Sequence[dict], goal_tasks: Sequence[dict]) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    notes: dict[str, Any] = {}

    def find_point(env_task_id: int, prompt: str) -> dict | None:
        for point in points:
            if point["env_task_id"] == env_task_id and point["prompt"] == prompt:
                return point
        return None

    def add(
        block: str,
        env: dict,
        prompt: str,
        tag: str,
        prompted_goal_states: list[list[str]] | None,
        prompted_source: str,
    ) -> dict:
        label = f"{block}__{tag}"
        if any(point["label"] == label for point in points):
            raise ValueError(f"Duplicate plan label {label}")
        if prompted_goal_states is not None and not evaluable_in(
            env, prompted_goal_states
        ):
            missing = sorted(
                predicate_args(prompted_goal_states) - set(env["predicate_vocab"])
            )
            raise ValueError(
                f"{label}: prompted predicate args {missing} not in scene "
                f"vocabulary of {env['name']}"
            )
        point = {
            "label": label,
            "block": block,
            "env_task_id": env["env_task_id"],
            "env_name": env["name"],
            "env_instruction": env["language"],
            "scene": scene_of(env["name"]),
            "prompt": prompt,
            "prompted_goal_states": prompted_goal_states,
            "prompted_source": prompted_source,
            "expect_env_equivalent": (
                prompted_goal_states is not None
                and [list(s) for s in prompted_goal_states]
                == [list(s) for s in env["goal_state"]]
            ),
        }
        points.append(point)
        return point

    def ensure_trained(env: dict) -> dict:
        existing = find_point(env["env_task_id"], env["language"])
        if existing is not None:
            return existing
        return add(
            "trained",
            env,
            env["language"],
            slug(env["language"])[:40],
            [list(s) for s in env["goal_state"]],
            "env_bddl",
        )

    goal_by_language = {task["language"]: task for task in goal_tasks}

    # Block A: trained vs goal-paraphrase on the same env; the paraphrase
    # point is scored by the GOAL task's predicate (usually identical to the
    # env's own, recorded either way).
    for pair in PARAPHRASE_PAIRS:
        matches = find_by_language(listing, pair["seen"])
        if not matches:
            raise ValueError(f"Seen instruction not in benchmark: {pair['seen']!r}")
        env = matches[0]
        goal_task = goal_by_language.get(pair["prompt"])
        if goal_task is None or goal_task["goal_id"] != pair["goal_ref"]:
            raise ValueError(
                f"Paraphrase prompt {pair['prompt']!r} does not resolve to "
                f"libero_goal task {pair['goal_ref']}"
            )
        ensure_trained(env)
        add(
            "paraphrase",
            env,
            pair["prompt"],
            slug(pair["seen"])[:40],
            [list(s) for s in goal_task["goal_state"]],
            f"goal_task_{goal_task['goal_id']}",
        )
        notes.setdefault("paraphrase_matches", {})[pair["seen"]] = [
            item["name"] for item in matches
        ]

    # Block B: 2x2 cross-prompt inside the anchor's scene; each cross point is
    # scored by the PARTNER task's predicate (instruction-following accuracy).
    anchors = find_by_language(listing, CROSS_ANCHOR_INSTRUCTION)
    if not anchors:
        raise ValueError(f"Anchor instruction not found: {CROSS_ANCHOR_INSTRUCTION!r}")
    anchor = partner = None
    for candidate in anchors:
        scene = scene_of(candidate["name"])
        siblings = [
            item
            for item in listing
            if scene_of(item["name"]) == scene
            and item["env_task_id"] != candidate["env_task_id"]
        ]
        if siblings:
            anchor, partner = candidate, siblings[0]
            break
    if anchor is None or partner is None:
        raise ValueError("No multi-task scene found for the cross block")
    notes["cross_scene"] = {
        "scene": scene_of(anchor["name"]),
        "anchor": anchor["language"],
        "partner": partner["language"],
    }
    ensure_trained(anchor)
    ensure_trained(partner)
    add(
        "cross",
        anchor,
        partner["language"],
        slug(anchor["language"])[:40],
        [list(s) for s in partner["goal_state"]],
        f"partner_env_{partner['env_task_id']}",
    )
    add(
        "cross",
        partner,
        anchor["language"],
        slug(partner["language"])[:40],
        [list(s) for s in anchor["goal_state"]],
        f"partner_env_{anchor['env_task_id']}",
    )

    # Block C (new in v2): the full goal-prompt slice.  For every libero_goal
    # instruction pick one seen env where its predicate is evaluable:
    #   1. env whose trained instruction is the verbatim prompt;
    #   2. else the paraphrase twin's env (keeps v1 pairing);
    #   3. else the lowest evaluable env_task_id.
    # Rows whose predicate is evaluable nowhere are recorded as skipped (they
    # would be absent-object probes, dropped by design in v2).
    paraphrase_env_by_prompt = {
        pair["prompt"]: find_by_language(listing, pair["seen"])[0]
        for pair in PARAPHRASE_PAIRS
    }
    goal_slice: list[dict[str, Any]] = []
    for task in goal_tasks:
        goal_states = [list(s) for s in task["goal_state"]]
        candidates = [env for env in listing if evaluable_in(env, goal_states)]
        if not candidates:
            goal_slice.append(
                {
                    "goal_id": task["goal_id"],
                    "prompt": task["language"],
                    "status": "skipped",
                    "reason": "predicate not evaluable in any libero_90 scene "
                    "(absent objects/regions; absent probes dropped in v2)",
                }
            )
            continue
        verbatim = [env for env in candidates if env["language"] == task["language"]]
        if verbatim:
            env, relationship = verbatim[0], "verbatim_trained"
        elif task["language"] in paraphrase_env_by_prompt and evaluable_in(
            paraphrase_env_by_prompt[task["language"]], goal_states
        ):
            env, relationship = paraphrase_env_by_prompt[task["language"]], "paraphrase_of_trained"
        else:
            env = min(candidates, key=lambda item: item["env_task_id"])
            relationship = "novel_string"
        existing = find_point(env["env_task_id"], task["language"])
        if existing is not None:
            goal_slice.append(
                {
                    "goal_id": task["goal_id"],
                    "prompt": task["language"],
                    "status": "alias",
                    "alias_of": existing["label"],
                    "relationship": relationship,
                    "n_candidate_envs": len(candidates),
                }
            )
            continue
        ensure_trained(env)
        point = add(
            "goal",
            env,
            task["language"],
            slug(task["language"])[:40],
            goal_states,
            f"goal_task_{task['goal_id']}",
        )
        goal_slice.append(
            {
                "goal_id": task["goal_id"],
                "prompt": task["language"],
                "status": "point",
                "label": point["label"],
                "relationship": relationship,
                "n_candidate_envs": len(candidates),
            }
        )
    notes["goal_slice"] = goal_slice

    # Block D: nonsense control on two distinct scenes (no prompted predicate;
    # env-task metric only).
    probe_envs = [anchor, find_by_language(listing, PARAPHRASE_PAIRS[0]["seen"])[0]]
    for env in probe_envs:
        add(
            "nonsense",
            env,
            NONSENSE_PROMPT,
            slug(env["language"])[:40],
            None,
            "none",
        )

    return {"points": points, "notes": notes}
