from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .constants import DETERMINISM_LABEL, experiment_root

EPISODE_FIELDS = (
    "episode_ix",
    "env_seed",
    "noise_seed",
    "prompted_satisfied_at_reset",
    "success",
    "prompted_success",
    "prompted_first_step",
    "env_task_success",
    "env_first_step",
    "steps",
)


def canonical_episodes(payload: dict) -> list[dict]:
    return [
        {field: episode.get(field) for field in EPISODE_FIELDS}
        for episode in payload["per_episode"]
    ]


def compare(left: dict, right: dict) -> dict:
    if left["checkpoint_sha256"] != right["checkpoint_sha256"]:
        raise ValueError("Determinism runs used different checkpoints")
    a = canonical_episodes(left)
    b = canonical_episodes(right)
    passed = a == b
    encoded = json.dumps(a, sort_keys=True).encode()
    return {
        "passed": passed,
        "label": left["label"],
        "checkpoint_sha256": left["checkpoint_sha256"],
        "episode_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "n_episodes": len(a),
        "left_successes": left["successes"],
        "right_successes": right["successes"],
        "mismatched_episode_indices": [
            index for index, (x, y) in enumerate(zip(a, b, strict=True)) if x != y
        ],
    }


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Verify per-episode determinism")
    parser.add_argument("--label", default=DETERMINISM_LABEL)
    parser.add_argument(
        "--left", type=Path, default=root / "results/determinism_check/a"
    )
    parser.add_argument(
        "--right", type=Path, default=root / "results/determinism_check/b"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts/determinism_check.json"
    )
    args = parser.parse_args()
    left = json.loads((args.left / f"{args.label}.json").read_text())
    right = json.loads((args.right / f"{args.label}.json").read_text())
    result = compare(left, right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Per-episode determinism check failed")


if __name__ == "__main__":
    main()
