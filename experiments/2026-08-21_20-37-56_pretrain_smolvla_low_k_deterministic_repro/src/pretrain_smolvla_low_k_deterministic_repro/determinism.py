from __future__ import annotations

import argparse
import hashlib
import json

from .constants import EVAL_EPISODES, PRODUCTION_SMOKE_POINT, experiment_root, result_path
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
    for field in ("logical_task_id", "demo_budget"):
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
        "smoke_point": dict(PRODUCTION_SMOKE_POINT),
        "layout_a_episode_order": list(range(EVAL_EPISODES)),
        "layout_b_episode_order": list(reversed(range(EVAL_EPISODES))),
        "model_safetensors_sha256": left["model_safetensors_sha256"],
        "episode_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "n_episodes": len(a),
        "left_successes": left["successes"],
        "right_successes": right["successes"],
        "mismatched_episode_indices": mismatches,
    }


def main() -> None:
    from .evaluate import run

    root = experiment_root()
    parser = argparse.ArgumentParser(
        description="Run one production point in forward and reverse episode order"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    task_id = int(PRODUCTION_SMOKE_POINT["task_id"])
    budget = int(PRODUCTION_SMOKE_POINT["budget"])
    model = final_model(task_id, budget)

    left = run(task_id, budget, model, root / "results/raw", args.device)
    right_root = root / "results/determinism_check/layout_b"
    right = run(
        task_id,
        budget,
        model,
        right_root,
        args.device,
        episode_indices=list(reversed(range(EVAL_EPISODES))),
    )
    result = compare(left, right)
    result.update(
        {
            "layout_a_result": str(
                result_path(root / "results/raw", task_id, budget)
            ),
            "layout_b_result": str(result_path(right_root, task_id, budget)),
        }
    )
    output = root / "artifacts/production_smoke.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Per-episode determinism smoke failed")


if __name__ == "__main__":
    main()
