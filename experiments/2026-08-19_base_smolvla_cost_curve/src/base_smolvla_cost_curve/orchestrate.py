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

from .constants import DEMO_BUDGETS, GPU_IDS, JOBS_PER_GPU, TARGET_INSTRUCTIONS, experiment_root


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    root = experiment_root()
    status_path = root / "results/status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    train_logs = root / "results/logs/train"
    eval_logs = root / "results/logs/eval"
    train_logs.mkdir(parents=True, exist_ok=True)
    eval_logs.mkdir(parents=True, exist_ok=True)
    jobs = [(task_id, budget) for budget in DEMO_BUDGETS for task_id in TARGET_INSTRUCTIONS]
    work: queue.Queue[tuple[int, int]] = queue.Queue()
    for job in jobs:
        work.put(job)
    lock = threading.Lock()
    state: dict[str, Any] = {
        "state": "running",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "total_jobs": len(jobs),
        "completed_jobs": 0,
        "failed_jobs": 0,
        "jobs": {
            f"task_{task_id}_k_{budget}": {
                "task_id": task_id,
                "budget": budget,
                "state": "pending",
            }
            for task_id, budget in jobs
        },
    }
    atomic_json(status_path, state)

    def update(key: str, **values: Any) -> None:
        with lock:
            state["jobs"][key].update(values)
            state["completed_jobs"] = sum(
                item["state"] == "completed" for item in state["jobs"].values()
            )
            state["failed_jobs"] = sum(
                item["state"] == "failed" for item in state["jobs"].values()
            )
            state["updated_at"] = now_iso()
            atomic_json(status_path, state)

    def run_stage(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
        with log_path.open("w") as stream:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return process.wait()

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
            train_command = [
                sys.executable,
                "-m",
                "base_smolvla_cost_curve.training",
                str(task_id),
                str(budget),
            ]
            train_log = train_logs / f"task_{task_id}_k_{budget}.log"
            code = run_stage(train_command, train_log, environment)
            if code:
                update(
                    key,
                    state="failed",
                    stage="training",
                    exit_code=code,
                    finished_at=now_iso(),
                    log=str(train_log.relative_to(root)),
                )
                work.task_done()
                continue
            update(key, state="evaluating", train_log=str(train_log.relative_to(root)))
            eval_command = [
                sys.executable,
                "-m",
                "base_smolvla_cost_curve.evaluate",
                str(task_id),
                str(budget),
            ]
            eval_log = eval_logs / f"task_{task_id}_k_{budget}.log"
            code = run_stage(eval_command, eval_log, environment)
            if code:
                update(
                    key,
                    state="failed",
                    stage="evaluation",
                    exit_code=code,
                    finished_at=now_iso(),
                    log=str(eval_log.relative_to(root)),
                )
            else:
                result_path = root / "results/raw" / f"task_{task_id}" / f"k_{budget}.json"
                result = json.loads(result_path.read_text())
                update(
                    key,
                    state="completed",
                    success_rate=result["success_rate"],
                    successes=result["successes"],
                    trials=result["n_episodes"],
                    eval_log=str(eval_log.relative_to(root)),
                    finished_at=now_iso(),
                )
            work.task_done()

    # Co-location: training is dataloader-bound (updt_s~0.14s vs data_s
    # 0.6-1.8s), so several jobs share one GPU; commands are unchanged.
    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=False)
        for gpu in GPU_IDS
        for _ in range(JOBS_PER_GPU)
    ]
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
    if run_stage([sys.executable, "-m", "base_smolvla_cost_curve.aggregate"], aggregate_log, os.environ.copy()):
        raise SystemExit("Aggregation failed")
    trackio_log = root / "results/logs/trackio.log"
    if run_stage([sys.executable, "-m", "base_smolvla_cost_curve.trackio_report"], trackio_log, os.environ.copy()):
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
