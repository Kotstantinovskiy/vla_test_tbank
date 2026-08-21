from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import DETERMINISM_LABEL, EXPERIMENT_NAME, experiment_root


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def require_determinism_gate(root: Path) -> None:
    path = root / "artifacts/determinism_check.json"
    if not path.is_file():
        raise RuntimeError("Run scripts/determinism_smoke.sh before fan-out")
    payload = json.loads(path.read_text())
    if payload.get("passed") is not True or payload.get("label") != DETERMINISM_LABEL:
        raise RuntimeError(f"Invalid determinism gate: {payload}")


def main() -> None:
    root = experiment_root()
    require_determinism_gate(root)
    plan = json.loads((root / "artifacts/eval_plan.json").read_text())
    labels = [point["label"] for point in plan["points"]]
    gpu_ids = tuple(
        int(item) for item in os.environ.get("VLA_GPU_IDS", "0,1,2,3").split(",")
    )
    if not gpu_ids:
        raise ValueError("VLA_GPU_IDS must contain at least one GPU")
    status_path = root / "results/status.json"
    logs = root / "results/logs"
    logs.mkdir(parents=True, exist_ok=True)
    work: queue.Queue[str] = queue.Queue()
    for label in labels:
        work.put(label)
    lock = threading.Lock()
    state: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "state": "running",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "total_points": len(labels),
        "completed": 0,
        "failed": 0,
        "points": {label: {"state": "pending"} for label in labels},
    }
    atomic_json(status_path, state)

    def update(label: str, **values: Any) -> None:
        with lock:
            state["points"][label].update(values)
            state["completed"] = sum(
                item["state"] == "completed" for item in state["points"].values()
            )
            state["failed"] = sum(
                item["state"] == "failed" for item in state["points"].values()
            )
            state["updated_at"] = now_iso()
            atomic_json(status_path, state)

    def worker(gpu: int) -> None:
        while True:
            try:
                label = work.get_nowait()
            except queue.Empty:
                return
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            update(label, state="running", gpu=gpu)
            log_path = logs / f"{label}.log"
            with log_path.open("w") as stream:
                code = subprocess.Popen(
                    [sys.executable, "-m", f"{EXPERIMENT_NAME}.evaluate", "--labels", label],
                    cwd=root,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                ).wait()
            if code:
                update(label, state="failed", exit_code=code)
            else:
                result = json.loads(
                    (root / "results/raw" / f"{label}.json").read_text()
                )
                update(
                    label,
                    state="completed",
                    successes=result["successes"],
                    env_task_successes=result["env_task_successes"],
                    trials=result["n_episodes"],
                )
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state["state"] = "failed" if state["failed"] else "completed"
    state["finished_at"] = now_iso()
    atomic_json(status_path, state)
    if state["failed"]:
        raise SystemExit(f"{state['failed']} points failed")


if __name__ == "__main__":
    main()
