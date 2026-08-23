from __future__ import annotations

import fcntl
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import (
    DEMO_BUDGETS,
    EXPERIMENT_NAME,
    GPU_IDS,
    PRODUCTION_SMOKE_POINT,
    TARGET_INSTRUCTIONS,
    experiment_root,
    result_path,
)
from .evaluate import result_complete


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def require_preflight(root: Path) -> None:
    required = (
        "artifacts/base_checkpoint_manifest.json",
        "artifacts/runtime_base_checkpoint_manifest.json",
        "artifacts/episode_manifest.json",
        "artifacts/evaluation_plan.json",
        "artifacts/dataset_selection_smoke.json",
        "artifacts/trainable_parameters.json",
        "artifacts/env_smoke.json",
        "artifacts/production_smoke.json",
    )
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"Preflight artifacts missing: {missing}")
    audit = json.loads((root / "artifacts/trainable_parameters.json").read_text())
    if audit.get("forbidden_trainable"):
        raise RuntimeError("Trainable-parameter audit contains forbidden parameters")
    dataset_smoke = json.loads(
        (root / "artifacts/dataset_selection_smoke.json").read_text()
    )
    expected_jobs = len(TARGET_INSTRUCTIONS) * len(DEMO_BUDGETS)
    if dataset_smoke.get("passed") is not True or len(dataset_smoke["checks"]) != expected_jobs:
        raise RuntimeError("Dataset selection smoke did not validate every task/k set")
    env_smoke = json.loads((root / "artifacts/env_smoke.json").read_text())
    validated_ids = {
        int(task_id) for task_id in env_smoke.get("validated_tasks", {})
    }
    if env_smoke.get("passed") is not True or validated_ids != set(TARGET_INSTRUCTIONS):
        raise RuntimeError("Real-environment smoke did not validate the three assignment tasks")
    production = json.loads((root / "artifacts/production_smoke.json").read_text())
    if production.get("passed") is not True:
        raise RuntimeError("Production/determinism smoke has not passed")
    if production.get("smoke_point") != PRODUCTION_SMOKE_POINT:
        raise RuntimeError(f"Unexpected production smoke point: {production}")
    plan = json.loads((root / "artifacts/evaluation_plan.json").read_text())
    if plan.get("training_jobs") != expected_jobs or plan.get("evaluation_points") != expected_jobs:
        raise RuntimeError("Frozen evaluation plan has the wrong number of jobs")


def main() -> None:
    root = experiment_root()
    require_preflight(root)
    lock_path = root / "results/orchestrator.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another orchestrator process is already running") from error
    lock_stream.write(f"pid={os.getpid()}\n")
    lock_stream.flush()

    package = __package__
    status_path = root / "results/status.json"
    train_logs = root / "results/logs/train"
    eval_logs = root / "results/logs/eval"
    train_logs.mkdir(parents=True, exist_ok=True)
    eval_logs.mkdir(parents=True, exist_ok=True)
    jobs = [
        (task_id, budget)
        for budget in DEMO_BUDGETS
        for task_id in TARGET_INSTRUCTIONS
    ]
    work: queue.Queue[tuple[int, int]] = queue.Queue()
    for job in jobs:
        work.put(job)
    gpu_ids = tuple(
        int(value)
        for value in os.environ.get(
            "VLA_GPU_IDS", ",".join(str(value) for value in GPU_IDS)
        ).split(",")
        if value.strip()
    )
    if not gpu_ids:
        raise ValueError("VLA_GPU_IDS must contain at least one GPU")
    state_lock = threading.Lock()
    state: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "state": "running",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "total_training_jobs": len(jobs),
        "total_evaluation_points": len(jobs),
        "completed_training_jobs": 0,
        "completed_evaluation_points": 0,
        "failed_jobs": 0,
        "jobs": {
            f"task_{task_id}_k_{budget}": {
                "task_id": task_id,
                "budget": budget,
                "state": "pending",
                "training_state": "pending",
                "evaluation_state": "pending",
            }
            for task_id, budget in jobs
        },
    }
    atomic_json(status_path, state)

    def refresh_counts() -> None:
        state["completed_training_jobs"] = sum(
            job["training_state"] == "completed" for job in state["jobs"].values()
        )
        state["completed_evaluation_points"] = sum(
            job["evaluation_state"] == "completed" for job in state["jobs"].values()
        )
        state["failed_jobs"] = sum(
            job["state"] == "failed" for job in state["jobs"].values()
        )
        state["updated_at"] = now_iso()

    def update(key: str, **values: Any) -> None:
        with state_lock:
            state["jobs"][key].update(values)
            refresh_counts()
            atomic_json(status_path, state)

    def run_stage(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
        # Append so an idempotent scoped restart cannot erase a completed
        # training log when the training entrypoint reports already_complete.
        with log_path.open("a") as stream:
            return subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            ).wait()

    def worker(gpu: int) -> None:
        while True:
            try:
                task_id, budget = work.get_nowait()
            except queue.Empty:
                return
            key = f"task_{task_id}_k_{budget}"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            update(key, state="training", gpu=gpu, started_at=now_iso())
            train_log = train_logs / f"{key}.log"
            code = run_stage(
                [sys.executable, "-m", f"{package}.training", str(task_id), str(budget)],
                train_log,
                environment,
            )
            if code:
                update(
                    key,
                    state="failed",
                    training_state="failed",
                    failed_stage="training",
                    exit_code=code,
                    finished_at=now_iso(),
                    log=str(train_log.relative_to(root)),
                )
                work.task_done()
                continue
            update(
                key,
                state="evaluating",
                training_state="completed",
                train_log=str(train_log.relative_to(root)),
            )
            output = result_path(root / "results/raw", task_id, budget)
            if result_complete(output):
                result = json.loads(output.read_text())
                update(
                    key,
                    state="completed",
                    evaluation_state="completed",
                    success_rate=result["success_rate"],
                    successes=result["successes"],
                    trials=result["n_episodes"],
                    reused=True,
                    finished_at=now_iso(),
                )
                work.task_done()
                continue
            eval_log = eval_logs / f"{key}.log"
            code = run_stage(
                [sys.executable, "-m", f"{package}.evaluate", str(task_id), str(budget)],
                eval_log,
                environment,
            )
            if code:
                update(
                    key,
                    state="failed",
                    evaluation_state="failed",
                    failed_stage="evaluation",
                    exit_code=code,
                    finished_at=now_iso(),
                    log=str(eval_log.relative_to(root)),
                )
            else:
                result = json.loads(output.read_text())
                update(
                    key,
                    state="completed",
                    evaluation_state="completed",
                    success_rate=result["success_rate"],
                    successes=result["successes"],
                    trials=result["n_episodes"],
                    eval_log=str(eval_log.relative_to(root)),
                    finished_at=now_iso(),
                )
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if state["failed_jobs"]:
        state.update({"state": "failed", "finished_at": now_iso(), "updated_at": now_iso()})
        atomic_json(status_path, state)
        raise SystemExit(f"{state['failed_jobs']} jobs failed; inspect results/status.json")

    state.update({"state": "aggregating", "updated_at": now_iso()})
    atomic_json(status_path, state)
    aggregate_log = root / "results/logs/aggregate.log"
    if run_stage([sys.executable, "-m", f"{package}.aggregate"], aggregate_log, os.environ.copy()):
        raise SystemExit("Aggregation failed")
    trackio_log = root / "results/logs/trackio.log"
    if run_stage([sys.executable, "-m", f"{package}.trackio_report"], trackio_log, os.environ.copy()):
        raise SystemExit("Trackio finalization failed")
    state.update(
        {
            "state": "completed",
            "finished_at": now_iso(),
            "updated_at": now_iso(),
            "summary": "results/summary/summary.json",
        }
    )
    atomic_json(status_path, state)


if __name__ == "__main__":
    main()
