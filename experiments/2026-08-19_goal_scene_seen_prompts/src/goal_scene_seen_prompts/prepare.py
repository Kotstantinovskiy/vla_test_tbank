from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import yaml

from .constants import CHECKPOINT_PATH, SEEN_SUITE, SUITE, experiment_root
from .plan import build_plan


def write_libero_config(config_dir: Path) -> Path:
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("LIBERO is not installed in the active environment")
    package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    benchmark_root = package_root / "libero"
    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(package_root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    required = ("benchmark_root", "bddl_files", "init_states")
    missing = [name for name in required if not Path(paths[name]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing LIBERO resource directories: {missing}")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(paths, sort_keys=False))
    return config_path


def ensure_libero_config() -> Path:
    return write_libero_config(Path(os.environ["LIBERO_CONFIG_PATH"]))


def _parse_bddl(bddl_path: Path) -> tuple[list[str], list[list[str]]]:
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    parsed = robosuite_parse_problem(str(bddl_path))
    vocab: set[str] = set()
    for group in (parsed["objects"], parsed["fixtures"]):
        for names in group.values():
            vocab.update(names)
    vocab.update(parsed["regions"].keys())
    return sorted(vocab), [list(state) for state in parsed["goal_state"]]


def goal_listing() -> list[dict]:
    ensure_libero_config()
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark

    bench = get_benchmark(SUITE)()
    bddl_root = Path(get_libero_path("bddl_files"))
    listing = []
    total = getattr(bench, "n_tasks", None) or bench.get_num_tasks()
    for index in range(total):
        task = bench.get_task(index)
        bddl_path = bddl_root / task.problem_folder / task.bddl_file
        if not bddl_path.is_file():
            raise FileNotFoundError(bddl_path)
        vocab, goal_state = _parse_bddl(bddl_path)
        listing.append(
            {
                "env_task_id": index,
                "name": task.name,
                "language": task.language,
                "predicate_vocab": vocab,
                "goal_state": goal_state,
            }
        )
    if len(listing) != 10:
        raise ValueError(f"Expected 10 libero_goal tasks, benchmark has {len(listing)}")
    return listing


def seen_language_set() -> set[str]:
    ensure_libero_config()
    from libero.libero.benchmark import get_benchmark

    bench = get_benchmark(SEEN_SUITE)()
    total = getattr(bench, "n_tasks", None) or bench.get_num_tasks()
    languages = {bench.get_task(index).language for index in range(total)}
    if total != 90:
        raise ValueError(f"Expected 90 libero_90 tasks, benchmark has {total}")
    return languages


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = experiment_root()
    parser = argparse.ArgumentParser(
        description="Resolve goal envs by instruction text and freeze the plan"
    )
    parser.add_argument("--plan", type=Path, default=root / "artifacts/eval_plan.json")
    args = parser.parse_args()

    if not (CHECKPOINT_PATH / "model.safetensors").is_file():
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT_PATH}")
    (root / "artifacts/checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(CHECKPOINT_PATH),
                "model_safetensors_sha256": sha256(
                    CHECKPOINT_PATH / "model.safetensors"
                ),
            },
            indent=2,
        )
        + "\n"
    )

    plan = build_plan(goal_listing(), seen_language_set())
    plan["suite"] = SUITE
    plan["checkpoint_path"] = str(CHECKPOINT_PATH)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {
                "points": len(plan["points"]),
                "labels": [point["label"] for point in plan["points"]],
                "notes": plan["notes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
