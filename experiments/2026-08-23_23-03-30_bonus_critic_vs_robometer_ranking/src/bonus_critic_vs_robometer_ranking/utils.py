from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from .constants import CONFIG_PATH


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open() as handle:
        return yaml.safe_load(handle)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def set_status(state: str, **fields: Any) -> None:
    from .constants import STATUS_PATH

    previous: dict[str, Any] = {}
    if STATUS_PATH.exists():
        previous = json.loads(STATUS_PATH.read_text())
    previous.update(fields)
    previous.update({"state": state, "updated_at": utc_now()})
    atomic_json(STATUS_PATH, previous)
