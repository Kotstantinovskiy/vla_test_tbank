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
    ACTION_STEPS,
    DEMO_BUDGETS,
    EVAL_EPISODES,
    EXPERIMENT_NAME,
    GPU_IDS,
    MASTER_SEED,
    PRODUCTION_SMOKE_POINT,
    TARGET_INSTRUCTIONS,
    experiment_root,
    result_path,
)


def result_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    episodes = payload.get("per_episode", [])
    expected_seeds = list(range(MASTER_SEED, MASTER_SEED + EVAL_EPISODES))
    return (
        len(episodes) == EVAL_EPISODES
        and [item.get("env_seed") for item in episodes] == expected_seeds
        and [item.get("noise_seed") for item in episodes] == expected_seeds
        and all(Path(item.get("video_path", "")).is_file() for item in episodes)
    )


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
    if dataset_smoke.get("passed") is not True or len(dataset_smoke["checks"]) != 30:
        raise RuntimeError("Dataset selection smoke did not validate all 30 task/k sets")
    env_smoke = json.loads((root / "artifacts/env_smoke.json").read_text())
    if env_smoke.get("passed") is not True or len(env_smoke["validated_tasks"]) != 10:
        raise RuntimeError("Real-environment smoke did not validate all ten tasks")
    production = json.loads((root / "artifacts/production_smoke.json").read_text())
    if production.get("passed") is not True:
        raise RuntimeError("Production/determinism smoke has not passed")
    if production.get("smoke_point") != PRODUCTION_SMOKE_POINT:
        raise RuntimeError(f"Unexpected production smoke point: {production}")
    plan = json.loads((root / "artifacts/evaluation_plan.json").read_text())
    expected_points = len(TARGET_INSTRUCTIONS) * len(DEMO_BUDGETS) * len(ACTION_STEPS)
    if plan.get("evaluation_points") != expected_points:
        raise RuntimeError("Frozen evaluation plan has the wrong number of points")


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
        "total_evaluation_points": len(jobs) * len(ACTION_STEPS),
        "completed_training_jobs": 0,
        "completed_evaluation_points": 0,
        "failed_jobs": 0,
        "jobs": {
            f"task_{task_id}_k_{budget}": {
                "task_id": task_id,
                "budget": budget,
                "state": "pending",
                "evaluations": {
                    str(action_steps): {"state": "pending"}
                    for action_steps in ACTION_STEPS
                },
            }
            for task_id, budget in jobs
        },
    }
    atomic_json(status_path, state)

    def refresh_counts() -> None:
        state["completed_training_jobs"] = sum(
            job.get("training_state") == "completed"
            for job in state["jobs"].values()
        )
        state["completed_evaluation_points"] = sum(
            point["state"] == "completed"
            for job in state["jobs"].values()
            for point in job["evaluations"].values()
        )
        state["failed_jobs"] = sum(
            job["state"] == "failed" for job in state["jobs"].values()
        )
        state["updated_at"] = now_iso()

    def update_job(key: str, **values: Any) -> None:
        with state_lock:
            state["jobs"][key].update(values)
            refresh_counts()
            atomic_json(status_path, state)

    def update_eval(key: str, action_steps: int, **values: Any) -> None:
        with state_lock:
            state["jobs"][key]["evaluations"][str(action_steps)].update(values)
            refresh_counts()
            atomic_json(status_path, state)

    def run_stage(
        command: list[str], log_path: Path, environment: dict[str, str]
    ) -> int:
        with log_path.open("w") as stream:
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
            update_job(key, state="training", gpu=gpu, started_at=now_iso())
            train_log = train_logs / f"{key}.log"
            train_command = [
                sys.executable,
                "-m",
                "pretrain_smolvla_low_k_action_steps.training",
                str(task_id),
                str(budget),
            ]
            code = run_stage(train_command, train_log, environment)
            if code:
                update_job(
                    key,
                    state="failed",
                    stage="training",
                    exit_code=code,
                    finished_at=now_iso(),
                    log=str(train_log.relative_to(root)),
                )
                work.task_done()
                continue
            update_job(
                key,
                state="evaluating",
                training_state="completed",
                train_log=str(train_log.relative_to(root)),
            )

            failed = False
            for action_steps in ACTION_STEPS:
                output = result_path(
                    root / "results/raw", task_id, budget, action_steps
                )
                if result_complete(output):
                    result = json.loads(output.read_text())
                    update_eval(
                        key,
                        action_steps,
                        state="completed",
                        success_rate=result["success_rate"],
                        successes=result["successes"],
                        trials=result["n_episodes"],
                        reused=True,
                    )
                    continue
                update_eval(key, action_steps, state="running", started_at=now_iso())
                eval_log = eval_logs / f"{key}_n_{action_steps}.log"
                eval_command = [
                    sys.executable,
                    "-m",
                    "pretrain_smolvla_low_k_action_steps.evaluate",
                    str(task_id),
                    str(budget),
                    str(action_steps),
                ]
                code = run_stage(eval_command, eval_log, environment)
                if code:
                    update_eval(
                        key,
                        action_steps,
                        state="failed",
                        exit_code=code,
                        finished_at=now_iso(),
                        log=str(eval_log.relative_to(root)),
                    )
                    update_job(
                        key,
                        state="failed",
                        stage=f"evaluation_n_{action_steps}",
                        finished_at=now_iso(),
                    )
                    failed = True
                    break
                result = json.loads(output.read_text())
                update_eval(
                    key,
                    action_steps,
                    state="completed",
                    success_rate=result["success_rate"],
                    successes=result["successes"],
                    trials=result["n_episodes"],
                    finished_at=now_iso(),
                    log=str(eval_log.relative_to(root)),
                )
            if not failed:
                update_job(key, state="completed", finished_at=now_iso())
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if state["failed_jobs"]:
        state.update(
            {"state": "failed", "finished_at": now_iso(), "updated_at": now_iso()}
        )
        atomic_json(status_path, state)
        raise SystemExit(
            f"{state['failed_jobs']} jobs failed; inspect results/status.json"
        )

    state.update({"state": "aggregating", "updated_at": now_iso()})
    atomic_json(status_path, state)
    aggregate_log = root / "results/logs/aggregate.log"
    if run_stage(
        [sys.executable, "-m", "pretrain_smolvla_low_k_action_steps.aggregate"],
        aggregate_log,
        os.environ.copy(),
    ):
        state.update(
            {
                "state": "failed",
                "failed_stage": "aggregation",
                "finished_at": now_iso(),
                "updated_at": now_iso(),
            }
        )
        atomic_json(status_path, state)
        raise SystemExit("Aggregation failed")
    trackio_log = root / "results/logs/trackio.log"
    if run_stage(
        [sys.executable, "-m", "pretrain_smolvla_low_k_action_steps.trackio_report"],
        trackio_log,
        os.environ.copy(),
    ):
        state.update(
            {
                "state": "failed",
                "failed_stage": "trackio",
                "finished_at": now_iso(),
                "updated_at": now_iso(),
            }
        )
        atomic_json(status_path, state)
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
