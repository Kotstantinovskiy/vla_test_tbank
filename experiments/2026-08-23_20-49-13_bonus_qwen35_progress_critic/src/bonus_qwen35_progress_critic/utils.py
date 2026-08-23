from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_full_runtime(
    *,
    median_step_seconds: float,
    full_steps: int,
    validation_seconds_total: float,
    observed_validation_runs: int,
    observed_validation_samples: int,
    full_validation_runs: int,
    full_validation_samples: int,
) -> dict[str, float]:
    if min(
        full_steps,
        observed_validation_runs,
        observed_validation_samples,
        full_validation_runs,
        full_validation_samples,
    ) <= 0:
        raise ValueError("runtime-estimation counts must be positive")
    mean_validation_seconds_per_sample = (
        validation_seconds_total / observed_validation_runs / observed_validation_samples
    )
    estimated_full_validation_seconds = (
        mean_validation_seconds_per_sample * full_validation_samples * full_validation_runs
    )
    estimated_full_seconds = median_step_seconds * full_steps + estimated_full_validation_seconds
    return {
        "mean_validation_seconds_per_sample": mean_validation_seconds_per_sample,
        "estimated_full_validation_seconds": estimated_full_validation_seconds,
        "estimated_full_training_seconds": estimated_full_seconds,
        "estimated_full_training_hours": estimated_full_seconds / 3600.0,
    }
