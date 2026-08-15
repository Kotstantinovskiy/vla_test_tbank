from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import yaml


def libero_paths() -> dict[str, str]:
    """Resolve LIBERO resources from this experiment's active Python environment."""

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an experiment-local LIBERO path configuration"
    )
    parser.add_argument(
        "--config-dir", type=Path, default=Path("artifacts/libero_config")
    )
    args = parser.parse_args()

    paths = libero_paths()
    # The PyPI build omits optional asset and demonstration directories.  They
    # still belong in LIBERO's five-key config, while prompt-only simulation
    # requires benchmark definitions, BDDL files, and initial states.
    required = ("benchmark_root", "bddl_files", "init_states")
    missing = [name for name in required if not Path(paths[name]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing LIBERO resource directories: {missing}")
    args.config_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(paths, sort_keys=False))
    print(json.dumps({"config": str(config_path), "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
