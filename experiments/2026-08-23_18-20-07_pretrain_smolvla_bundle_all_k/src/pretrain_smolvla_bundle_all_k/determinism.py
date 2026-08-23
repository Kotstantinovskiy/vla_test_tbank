from __future__ import annotations

"""Forward/reverse determinism gate, parallelizable across action-steps variants.

``--variant N`` runs the production point (task 0 / k=1) forward (into
results/raw) and in reverse episode order (into results/determinism_check),
compares them per episode, and writes ``artifacts/production_smoke_n{N}.json``.
``--combine`` merges every variant verdict into ``artifacts/production_smoke.json``.
``production_smoke.sh`` runs the three variants concurrently on separate GPUs.
"""

import argparse
import hashlib
import json

from .constants import (
    EVAL_ACTION_STEPS,
    EVAL_EPISODES,
    PRODUCTION_SMOKE_POINT,
    experiment_root,
    result_path,
)
from .training import final_model

EPISODE_FIELDS = (
    "episode_ix",
    "env_seed",
    "noise_seed",
    "init_state_id",
    "seed",
    "success",
    "sum_reward",
    "max_reward",
)


def canonical_episodes(payload: dict) -> list[dict]:
    return [
        {field: episode.get(field) for field in EPISODE_FIELDS}
        for episode in payload["per_episode"]
    ]


def compare(left: dict, right: dict) -> dict:
    for field in ("logical_task_id", "demo_budget", "n_action_steps"):
        if left[field] != right[field]:
            raise ValueError(f"Determinism inputs differ on {field}")
    if left["model_safetensors_sha256"] != right["model_safetensors_sha256"]:
        raise ValueError("Determinism layouts used different adapted checkpoints")
    a = canonical_episodes(left)
    b = canonical_episodes(right)
    if len(a) != len(b):
        raise ValueError(f"Episode-count mismatch: {len(a)} != {len(b)}")
    mismatches = [
        index for index, (x, y) in enumerate(zip(a, b, strict=True)) if x != y
    ]
    encoded = json.dumps(a, sort_keys=True).encode()
    return {
        "passed": not mismatches,
        "n_action_steps": left["n_action_steps"],
        "layout_a_episode_order": list(range(EVAL_EPISODES)),
        "layout_b_episode_order": list(reversed(range(EVAL_EPISODES))),
        "model_safetensors_sha256": left["model_safetensors_sha256"],
        "episode_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "n_episodes": len(a),
        "left_successes": left["successes"],
        "right_successes": right["successes"],
        "mismatched_episode_indices": mismatches,
    }


def variant_output(action_steps: int):
    return experiment_root() / f"artifacts/production_smoke_n{action_steps}.json"


def run_variant(action_steps: int, device: str) -> dict:
    from .evaluate import run

    root = experiment_root()
    task_id = int(PRODUCTION_SMOKE_POINT["task_id"])
    budget = int(PRODUCTION_SMOKE_POINT["budget"])
    model = final_model(task_id, budget)
    left = run(task_id, budget, action_steps, model, root / "results/raw", device)
    right_root = root / "results/determinism_check/layout_b"
    right = run(
        task_id,
        budget,
        action_steps,
        model,
        right_root,
        device,
        episode_indices=list(reversed(range(EVAL_EPISODES))),
    )
    verdict = compare(left, right)
    verdict.update(
        {
            "layout_a_result": str(
                result_path(root / "results/raw", task_id, budget, action_steps)
            ),
            "layout_b_result": str(
                result_path(right_root, task_id, budget, action_steps)
            ),
        }
    )
    variant_output(action_steps).write_text(json.dumps(verdict, indent=2) + "\n")
    return verdict


def combine() -> dict:
    variants: dict[str, dict] = {}
    for action_steps in EVAL_ACTION_STEPS:
        path = variant_output(action_steps)
        if not path.is_file():
            raise FileNotFoundError(f"Missing determinism variant verdict: {path}")
        variants[str(action_steps)] = json.loads(path.read_text())
    result = {
        "passed": all(verdict["passed"] for verdict in variants.values()),
        "smoke_point": dict(PRODUCTION_SMOKE_POINT),
        "action_steps": list(EVAL_ACTION_STEPS),
        "variants": variants,
    }
    output = experiment_root() / "artifacts/production_smoke.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, choices=list(EVAL_ACTION_STEPS))
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.combine == (args.variant is not None):
        raise SystemExit("Pass exactly one of --variant N or --combine")
    if args.variant is not None:
        verdict = run_variant(args.variant, args.device)
        print(json.dumps(verdict, indent=2))
        if not verdict["passed"]:
            raise SystemExit(f"Determinism variant n={args.variant} failed")
        return
    result = combine()
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Per-episode determinism smoke failed")


if __name__ == "__main__":
    main()
