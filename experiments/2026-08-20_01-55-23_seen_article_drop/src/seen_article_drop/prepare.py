from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import yaml

from .constants import (
    CHECKPOINT_PATH,
    CHECKPOINT_PROVENANCE,
    EXPECTED_SUITE_TASKS,
    SUITE,
    experiment_root,
)
from .plan import build_plan

REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "train_config.json",
)


def libero_paths() -> dict[str, str]:
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("LIBERO is not installed in the active environment")
    package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    benchmark_root = package_root / "libero"
    return {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(package_root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }


def write_libero_config(config_dir: Path) -> Path:
    paths = libero_paths()
    missing = [
        key
        for key in ("benchmark_root", "bddl_files", "init_states")
        if not Path(paths[key]).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing LIBERO resources: {missing}")
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "config.yaml"
    path.write_text(yaml.safe_dump(paths, sort_keys=False))
    return path


def ensure_libero_config() -> Path:
    config_dir = Path(
        os.environ.get(
            "LIBERO_CONFIG_PATH", experiment_root() / "artifacts/libero_config"
        )
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    return write_libero_config(config_dir)


def checkpoint_manifest() -> dict[str, object]:
    missing = [
        name for name in REQUIRED_CHECKPOINT_FILES if not (CHECKPOINT_PATH / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint, missing: {missing}")
    digest = hashlib.sha256()
    with (CHECKPOINT_PATH / "model.safetensors").open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "model_safetensors_sha256": digest.hexdigest(),
        "model_safetensors_bytes": (CHECKPOINT_PATH / "model.safetensors").stat().st_size,
        "provenance": CHECKPOINT_PROVENANCE,
    }


def _parse_bddl(path: Path) -> tuple[list[str], list[list[str]]]:
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    parsed = robosuite_parse_problem(str(path))
    vocab: set[str] = set(parsed["regions"].keys())
    for group in (parsed["objects"], parsed["fixtures"]):
        for names in group.values():
            vocab.update(names)
    return sorted(vocab), [list(state) for state in parsed["goal_state"]]


def benchmark_listing() -> list[dict]:
    ensure_libero_config()
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark

    benchmark = get_benchmark(SUITE)()
    bddl_root = Path(get_libero_path("bddl_files"))
    total = getattr(benchmark, "n_tasks", None) or benchmark.get_num_tasks()
    listing = []
    for env_task_id in range(total):
        task = benchmark.get_task(env_task_id)
        bddl = bddl_root / task.problem_folder / task.bddl_file
        vocab, goal_state = _parse_bddl(bddl)
        listing.append(
            {
                "env_task_id": env_task_id,
                "name": task.name,
                "language": task.language,
                "bddl_file": str(bddl),
                "predicate_vocab": vocab,
                "goal_state": goal_state,
            }
        )
    if len(listing) != EXPECTED_SUITE_TASKS:
        raise ValueError(
            f"Expected {EXPECTED_SUITE_TASKS} {SUITE} tasks, got {len(listing)}"
        )
    return listing


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(description="Freeze prompt evaluation plan")
    parser.add_argument("--plan", type=Path, default=root / "artifacts/eval_plan.json")
    args = parser.parse_args()
    config = ensure_libero_config()
    manifest = checkpoint_manifest()
    manifest_path = root / "artifacts/checkpoint_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    plan = build_plan(benchmark_listing())
    plan.update(
        {
            "experiment": root.name,
            "suite": SUITE,
            "checkpoint_path": str(CHECKPOINT_PATH),
        }
    )
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {
                "config": str(config),
                "manifest": str(manifest_path),
                "plan": str(args.plan),
                "points": len(plan["points"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
