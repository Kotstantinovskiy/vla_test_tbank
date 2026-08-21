from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import yaml

from .constants import experiment_root


def ensure_libero_config() -> Path:
    """Write an experiment-local LIBERO config before importing libero.libero."""

    config_dir = Path(
        os.environ.get(
            "LIBERO_CONFIG_PATH", experiment_root() / "artifacts/libero_config"
        )
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
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
    missing = [
        name
        for name in ("benchmark_root", "bddl_files", "init_states")
        if not Path(paths[name]).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing LIBERO resource directories: {missing}")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(paths, sort_keys=False))
    return config_path
