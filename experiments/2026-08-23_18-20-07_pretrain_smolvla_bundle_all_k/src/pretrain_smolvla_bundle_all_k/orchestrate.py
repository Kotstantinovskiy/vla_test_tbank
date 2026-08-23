from __future__ import annotations

"""Slot-based orchestrator: concurrent trainings and evaluations per GPU.

Each GPU gets TRAIN_SLOTS_PER_GPU training workers and EVAL_SLOTS_PER_GPU
evaluation workers (144 GB cards hold ~30 GB per full-FT training and
~12 GB per evaluation).  Evaluations of a checkpoint are enqueued the moment
its training completes, so they overlap with the remaining trainings; once
the training queue drains, training workers convert to evaluation workers.
Shorter jobs (fewer steps) are scheduled first to start the evaluation
stream as early as possible.
"""

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
    EVAL_ACTION_STEPS,
    EVAL_SLOTS_PER_GPU,
    EXPERIMENT_NAME,
    GPU_IDS,
    PRODUCTION_SMOKE_POINT,
    TARGET_INSTRUCTIONS,
    TRAIN_SLOTS_PER_GPU,
    TRAIN_STEPS_BY_BUDGET,
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
    if audit.get("unexpectedly_frozen"):
        raise RuntimeError("Full fine-tune audit left unexpected parameters frozen")
    flags = audit.get("flags", {})
    if flags.get("train_expert_only") is not False or flags.get("freeze_vision_encoder") is not False:
        raise RuntimeError(f"Audit flags are not the full-fine-tune flags: {flags}")
    dataset_smoke = json.loads(
        (root / "artifacts/dataset_selection_smoke.json").read_text()
    )
    expected_jobs = len(TARGET_INSTRUCTIONS) * len(DEMO_BUDGETS)
    if dataset_smoke.get("passed") is not True or len(dataset_smoke["checks"]) != expected_jobs:
        raise RuntimeError("Dataset selection smoke did not validate every task/k set")
    env_smoke = json.loads((root / "artifacts/env_smoke.json").read_text())
    validated_ids = {int(task_id) for task_id in env_smoke.get("validated_tasks", {})}
    if env_smoke.get("passed") is not True or validated_ids != set(TARGET_INSTRUCTIONS):
        raise RuntimeError("Real-environment smoke did not validate the three assignment tasks")
    production = json.loads((root / "artifacts/production_smoke.json").read_text())
    if production.get("passed") is not True:
        raise RuntimeError("Production/determinism smoke has not passed")
    if production.get("smoke_point") != PRODUCTION_SMOKE_POINT:
        raise RuntimeError(f"Unexpected production smoke point: {production}")
    if production.get("action_steps") != list(EVAL_ACTION_STEPS):
        raise RuntimeError(
            f"Production smoke did not cover every action-steps variant: {production.get('action_steps')}"
        )
    plan = json.loads((root / "artifacts/evaluation_plan.json").read_text())
    expected_points = expected_jobs * len(EVAL_ACTION_STEPS)
    if plan.get("training_jobs") != expected_jobs or plan.get("evaluation_points") != expected_points:
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

    gpu_ids = tuple(
        int(value)
        for value in os.environ.get(
            "VLA_GPU_IDS", ",".join(str(value) for value in GPU_IDS)
        ).split(",")
        if value.strip()
    )
    if not gpu_ids:
        raise ValueError("VLA_GPU_IDS must contain at least one GPU")
    train_slots = int(os.environ.get("VLA_TRAIN_SLOTS", TRAIN_SLOTS_PER_GPU))
    eval_slots = int(os.environ.get("VLA_EVAL_SLOTS", EVAL_SLOTS_PER_GPU))

    # Shorter trainings first: their evaluations start streaming sooner.
    jobs = sorted(
        (
            (task_id, budget)
            for budget in DEMO_BUDGETS
            for task_id in TARGET_INSTRUCTIONS
        ),
        key=lambda job: (TRAIN_STEPS_BY_BUDGET[job[1]], job[1], job[0]),
    )

    state_lock = threading.Lock()
    state: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "state": "running",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "scheduler": {
            "gpus": list(gpu_ids),
            "train_slots_per_gpu": train_slots,
            "eval_slots_per_gpu": eval_slots,
        },
        "total_training_jobs": len(jobs),
        "total_evaluation_points": len(jobs) * len(EVAL_ACTION_STEPS),
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
                "evaluations": {
                    str(action_steps): {"state": "pending"}
                    for action_steps in EVAL_ACTION_STEPS
                },
            }
            for task_id, budget in jobs
        },
    }
    atomic_json(status_path, state)

    train_queue: queue.Queue[tuple[int, int]] = queue.Queue()
    for job in jobs:
        train_queue.put(job)
    eval_queue: queue.Queue[tuple[int, int, int]] = queue.Queue()
    evals_done = threading.Event()

    def refresh_counts() -> None:
        state["completed_training_jobs"] = sum(
            job["training_state"] == "completed" for job in state["jobs"].values()
        )
        state["completed_evaluation_points"] = sum(
            variant.get("state") == "completed"
            for job in state["jobs"].values()
            for variant in job.get("evaluations", {}).values()
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

    def update_variant(key: str, action_steps: int, **values: Any) -> None:
        with state_lock:
            state["jobs"][key]["evaluations"][str(action_steps)].update(values)
            refresh_counts()
            atomic_json(status_path, state)

    def refresh_job_completion(key: str) -> None:
        """Mark the job completed/failed once every eval variant is terminal."""

        with state_lock:
            job = state["jobs"][key]
            variants = job["evaluations"].values()
            if any(variant.get("state") == "failed" for variant in variants):
                job["state"] = "failed"
                job["evaluation_state"] = "failed"
                job.setdefault("finished_at", now_iso())
            elif all(variant.get("state") == "completed" for variant in variants):
                job["state"] = "completed"
                job["evaluation_state"] = "completed"
                job.setdefault("finished_at", now_iso())
            refresh_counts()
            atomic_json(status_path, state)
            all_terminal = all(
                variant.get("state") in {"completed", "failed"}
                for j in state["jobs"].values()
                if j["training_state"] in {"completed", "failed"}
                for variant in j["evaluations"].values()
            ) and all(
                j["training_state"] in {"completed", "failed"}
                for j in state["jobs"].values()
            )
            pending_failed = any(
                j["training_state"] == "failed" for j in state["jobs"].values()
            )
            if all_terminal:
                evals_done.set()
            _ = pending_failed

    def run_stage(command: list[str], log_path: Path, gpu: int) -> int:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log_path.open("a") as stream:
            return subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            ).wait()

    def enqueue_evals(task_id: int, budget: int) -> None:
        for action_steps in EVAL_ACTION_STEPS:
            eval_queue.put((task_id, budget, action_steps))

    def check_all_done() -> None:
        with state_lock:
            done = all(
                job["training_state"] in {"completed", "failed"}
                and all(
                    variant.get("state") in {"completed", "failed"}
                    for variant in job["evaluations"].values()
                )
                or job["training_state"] == "failed"
                for job in state["jobs"].values()
            )
        if done and train_queue.empty() and eval_queue.empty():
            evals_done.set()

    def do_training(gpu: int, task_id: int, budget: int) -> None:
        key = f"task_{task_id}_k_{budget}"
        update(key, state="training", training_state="training", gpu=gpu, started_at=now_iso())
        train_log = train_logs / f"{key}.log"
        code = run_stage(
            [sys.executable, "-m", f"{package}.training", str(task_id), str(budget)],
            train_log,
            gpu,
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
            check_all_done()
            return
        update(
            key,
            state="evaluating",
            training_state="completed",
            train_log=str(train_log.relative_to(root)),
        )
        enqueue_evals(task_id, budget)

    def do_eval(gpu: int, task_id: int, budget: int, action_steps: int) -> None:
        key = f"task_{task_id}_k_{budget}"
        output = result_path(root / "results/raw", task_id, budget, action_steps)
        if result_complete(output, action_steps):
            result = json.loads(output.read_text())
            update_variant(
                key,
                action_steps,
                state="completed",
                success_rate=result["success_rate"],
                successes=result["successes"],
                trials=result["n_episodes"],
                reused=True,
            )
            refresh_job_completion(key)
            return
        eval_log = eval_logs / f"{key}_n_{action_steps}.log"
        update_variant(key, action_steps, state="evaluating", gpu=gpu)
        code = run_stage(
            [
                sys.executable,
                "-m",
                f"{package}.evaluate",
                str(task_id),
                str(budget),
                str(action_steps),
            ],
            eval_log,
            gpu,
        )
        if code:
            update_variant(
                key,
                action_steps,
                state="failed",
                exit_code=code,
                log=str(eval_log.relative_to(root)),
            )
        else:
            result = json.loads(output.read_text())
            update_variant(
                key,
                action_steps,
                state="completed",
                success_rate=result["success_rate"],
                successes=result["successes"],
                trials=result["n_episodes"],
                eval_log=str(eval_log.relative_to(root)),
            )
        refresh_job_completion(key)

    def eval_loop(gpu: int) -> None:
        while not evals_done.is_set():
            try:
                task_id, budget, action_steps = eval_queue.get(timeout=5)
            except queue.Empty:
                check_all_done()
                continue
            do_eval(gpu, task_id, budget, action_steps)
            eval_queue.task_done()
            check_all_done()

    def train_worker(gpu: int) -> None:
        while True:
            try:
                task_id, budget = train_queue.get_nowait()
            except queue.Empty:
                break
            do_training(gpu, task_id, budget)
            train_queue.task_done()
        # Converted: help drain the evaluation queue.
        eval_loop(gpu)

    def eval_worker(gpu: int) -> None:
        eval_loop(gpu)

    threads: list[threading.Thread] = []
    for gpu in gpu_ids:
        for _ in range(train_slots):
            threads.append(threading.Thread(target=train_worker, args=(gpu,)))
        for _ in range(eval_slots):
            threads.append(threading.Thread(target=eval_worker, args=(gpu,)))
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
    if run_stage([sys.executable, "-m", f"{package}.aggregate"], aggregate_log, gpu_ids[0]):
        raise SystemExit("Aggregation failed")
    trackio_log = root / "results/logs/trackio.log"
    if run_stage([sys.executable, "-m", f"{package}.trackio_report"], trackio_log, gpu_ids[0]):
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
