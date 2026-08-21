from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from .constants import (
    FULL_RUN_NAME,
    RESUMED_LIVE_RUN_NAME,
    TRACKIO_GROUP,
    TRACKIO_PROJECT,
    experiment_root,
)
from .summarize import collect_canonical_metrics


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    import trackio

    root = experiment_root()
    training_status_path = root / "results/status.json"
    status_path = root / "results/live_trackio_status.json"
    trackio.init(
        project=os.environ.get("TRACKIO_PROJECT", TRACKIO_PROJECT),
        name=RESUMED_LIVE_RUN_NAME,
        group=TRACKIO_GROUP,
        config={
            "source_run": FULL_RUN_NAME,
            "resume_policy": "full log through durable checkpoint, then resume log",
            "step_reconstruction": "checkpoint step, then +50 per resume log record",
        },
        auto_log_gpu=False,
        auto_log_cpu=False,
    )
    logged = 0
    try:
        while True:
            rows = collect_canonical_metrics(root)
            for row in rows[logged:]:
                step = int(row["step"])
                trackio.log(
                    {key: value for key, value in row.items() if key != "step"},
                    step=step,
                )
            logged = len(rows)
            training_status = (
                json.loads(training_status_path.read_text())
                if training_status_path.is_file()
                else {"state": "starting"}
            )
            payload: dict[str, object] = {
                "state": training_status.get("state", "starting"),
                "pid": os.getpid(),
                "logged_points": logged,
                "exact_last_step": int(rows[-1]["step"]) if rows else 0,
                "resume_from_step": training_status.get("resume_from_step"),
                "updated_at": datetime.now(UTC).isoformat(),
            }
            atomic_json(status_path, payload)
            if training_status.get("state") in {"completed", "failed"}:
                trackio.log(
                    {"run/source_exit_code": training_status.get("exit_code", 1)},
                    step=payload["exact_last_step"] + 1,
                )
                break
            time.sleep(15)
    finally:
        trackio.finish()


if __name__ == "__main__":
    main()
