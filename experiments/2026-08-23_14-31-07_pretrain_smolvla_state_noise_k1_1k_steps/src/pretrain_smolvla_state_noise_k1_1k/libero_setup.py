from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import yaml

from .constants import LIBERO_ASSETS_ROOT, experiment_root


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
    required_assets = (
        LIBERO_ASSETS_ROOT / "scenes/libero_tabletop_base_style.xml",
        LIBERO_ASSETS_ROOT / "articulated_objects",
        LIBERO_ASSETS_ROOT / "stable_scanned_objects",
    )
    missing_assets = [str(path) for path in required_assets if not path.exists()]
    if missing_assets:
        raise FileNotFoundError(f"Pinned LIBERO assets are incomplete: {missing_assets}")
    package_assets = benchmark_root / "assets"
    if package_assets.is_symlink():
        if package_assets.resolve() != LIBERO_ASSETS_ROOT.resolve():
            raise RuntimeError(f"LIBERO assets symlink points elsewhere: {package_assets}")
    elif package_assets.exists():
        raise FileExistsError(
            f"Refusing to replace unpinned package-local assets: {package_assets}"
        )
    else:
        package_assets.symlink_to(LIBERO_ASSETS_ROOT, target_is_directory=True)
    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(package_root / "datasets"),
        "assets": str(LIBERO_ASSETS_ROOT),
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
