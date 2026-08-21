from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import (
    OFFICIAL_REPO,
    OFFICIAL_REVISION,
    SEEN_REPO_ID,
    EFFECTIVE_BATCH_SIZE,
    FULL_RUN_NAME,
    GPU_IDS,
    LEARNING_RATE,
    LOG_FREQ,
    NUM_WORKERS_PER_RANK,
    PER_RANK_BATCH_SIZE,
    SAVE_FREQ,
    SEED,
    SMOKE_EPISODES,
    SMOKE_RUN_NAME,
    SMOKE_STEPS,
    TRACKIO_GROUP,
    TRACKIO_PROJECT,
    TRAIN_STEPS,
    WORLD_SIZE,
    experiment_root,
)

TOKEN_RE = re.compile(r"(?P<key>[A-Za-z_/-]+):(?P<value>[^\s]+)")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def compact_int(value: str) -> int:
    multipliers = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}
    suffix = value[-1:].upper()
    if suffix in multipliers:
        return int(round(float(value[:-1]) * multipliers[suffix]))
    return int(value)


def parse_training_metrics(line: str) -> tuple[int, dict[str, float]] | None:
    if " step:" not in f" {line}" or " loss:" not in line:
        return None
    tokens = {match.group("key"): match.group("value") for match in TOKEN_RE.finditer(line)}
    if "step" not in tokens or "loss" not in tokens:
        return None
    try:
        step = compact_int(tokens["step"])
    except ValueError:
        return None
    mapping = {
        "loss": "train/loss",
        "grdn": "train/grad_norm",
        "lr": "train/learning_rate",
        "updt_s": "perf/update_seconds",
        "data_s": "perf/dataload_seconds",
        "smp/s": "perf/samples_per_second",
        "mem_gb": "system/gpu_memory_gb",
        "epch": "train/epoch",
    }
    metrics: dict[str, float] = {}
    for source, destination in mapping.items():
        if source not in tokens:
            continue
        try:
            metrics[destination] = float(tokens[source])
        except ValueError:
            continue
    for source, destination in (("smpl", "train/samples"), ("ep", "train/episodes")):
        if source in tokens:
            try:
                metrics[destination] = float(compact_int(tokens[source]))
            except ValueError:
                pass
    return step, metrics


def paths() -> dict[str, Path]:
    root = experiment_root()
    return {
        "root": root,
        "base": (root / "artifacts/base_model").resolve(),
        "dataset": Path(
            os.environ.get(
                "VLA_OFFICIAL_SEEN_DATA_ROOT",
                "/var/tmp/vla_libero_official_rot180/libero_90",
            )
        ).resolve(),
        "output": Path(
            os.environ.get(
                "VLA_OFFICIAL_OUTPUT_ROOT",
                "/var/tmp/vla_outputs/seen_libero90_official_20260817",
            )
        ).resolve(),
    }


def base_command(mode: str) -> list[str]:
    p = paths()
    env_root = Path(os.environ.get("VLA_ENV_ROOT", p["root"].parents[1] / ".venv"))
    torchrun = env_root / "bin/torchrun"
    # Stock lerobot-train: the converted dataset already stores frames in the
    # eval rot180 convention, so no runtime transform wrapper is needed.
    trainer = env_root / "bin/lerobot-train"
    if not torchrun.is_file() or not trainer.is_file():
        raise FileNotFoundError(
            "torchrun or lerobot-train is missing from the project uv environment"
        )

    command = [
        str(torchrun),
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
        str(trainer),
    ]
    if mode == "resume":
        config = p["output"] / "checkpoints/last/pretrained_model/train_config.json"
        if not config.is_file():
            raise FileNotFoundError(f"No resumable checkpoint config: {config}")
        return command + ["--resume=true", f"--config_path={config}"]

    smoke = mode == "smoke"
    output = p["output"].with_name(p["output"].name + "_smoke") if smoke else p["output"]
    steps = SMOKE_STEPS if smoke else TRAIN_STEPS
    save_freq = 0 if smoke else SAVE_FREQ
    log_freq = 1 if smoke else LOG_FREQ
    command.extend(
        [
            f"--policy.path={p['base']}",
            "--policy.freeze_vision_encoder=true",
            "--policy.train_expert_only=true",
            "--policy.train_state_proj=true",
            # Mixed precision is an opt-in, declared deviation from the fp32
            # reference preset: VLA_TRAIN_AMP=1 enables fp16 autocast via
            # Accelerate (~1.5-2x faster steps on H200).  The flag is stored in
            # the checkpoint config, so evaluation runs under the same autocast.
            f"--policy.use_amp={'true' if os.environ.get('VLA_TRAIN_AMP') == '1' else 'false'}",
            "--policy.push_to_hub=false",
            f"--dataset.repo_id={SEEN_REPO_ID}",
            f"--dataset.root={p['dataset']}",
            "--dataset.video_backend=pyav",
            "--dataset.use_imagenet_stats=true",
            "--dataset.return_uint8=false",
            "--dataset.image_transforms.enable=false",
            f"--output_dir={output}",
            f"--job_name={'smoke_' if smoke else ''}smolvla_pretrain_libero4",
            f"--steps={steps}",
            f"--batch_size={PER_RANK_BATCH_SIZE}",
            f"--num_workers={2 if smoke else NUM_WORKERS_PER_RANK}",
            "--prefetch_factor=4",
            "--persistent_workers=true",
            "--dataloader_multiprocessing_context=spawn",
            "--cudnn_deterministic=false",
            "--env_eval_freq=0",
            "--eval_steps=0",
            f"--save_checkpoint={'false' if smoke else 'true'}",
            f"--save_freq={save_freq}",
            f"--log_freq={log_freq}",
            "--use_policy_training_preset=true",
            f"--seed={SEED}",
            "--wandb.enable=false",
        ]
    )
    if smoke:
        episodes = ",".join(map(str, SMOKE_EPISODES))
        command.append(f"--dataset.episodes=[{episodes}]")
    return command


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run(mode: str, dry_run: bool = False) -> int:
    import trackio

    p = paths()
    root = p["root"]
    command = base_command(mode)
    if dry_run:
        print(json.dumps(command, indent=2))
        return 0
    if not (root / "artifacts/source_manifest.json").is_file():
        raise FileNotFoundError("Run scripts/prepare.sh before training")

    output = p["output"]
    if mode == "full" and output.exists():
        raise FileExistsError(
            f"Full output already exists: {output}. Use scripts/resume_ddp.sh after an interruption."
        )

    run_name = SMOKE_RUN_NAME if mode == "smoke" else FULL_RUN_NAME
    log_path = root / "results/logs" / f"{mode}.log"
    status_path = root / "results/status.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    trackio_dir = root / "artifacts/trackio"
    trackio_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "mode": mode,
        "dataset": f"{SEEN_REPO_ID} (local conversion of {OFFICIAL_REPO}@{OFFICIAL_REVISION})",
        "strategy": "ddp",
        "world_size": WORLD_SIZE,
        "per_rank_batch_size": PER_RANK_BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "steps": SMOKE_STEPS if mode == "smoke" else TRAIN_STEPS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "gpu_ids": list(GPU_IDS),
    }
    trackio.init(
        project=os.environ.get("TRACKIO_PROJECT", TRACKIO_PROJECT),
        name=run_name,
        group=TRACKIO_GROUP,
        config=config,
        resume="allow" if mode == "resume" else "never",
        auto_log_gpu=True,
        gpu_log_interval=30,
        auto_log_cpu=True,
        cpu_log_interval=30,
    )

    process: subprocess.Popen[str] | None = None
    checkpoint_step = 0
    if mode == "resume":
        last_checkpoint = p["output"] / "checkpoints/last"
        try:
            checkpoint_step = int(last_checkpoint.resolve().name)
        except (OSError, ValueError):
            checkpoint_step = 0
        atomic_json(
            root / "results/resume_manifest.json",
            {
                "resume_from_step": checkpoint_step,
                "checkpoint": str(last_checkpoint.resolve()),
                "started_at": now_iso(),
                "discarded_uncheckpointed_steps": {
                    "first": checkpoint_step + 1,
                    "last_observed": json.loads(status_path.read_text()).get(
                        "last_step", checkpoint_step
                    )
                    if status_path.is_file()
                    else checkpoint_step,
                },
            },
        )
    state: dict[str, Any] = {
        "state": "starting",
        "mode": mode,
        "run_name": run_name,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "pid": None,
        "last_step": checkpoint_step,
        "resume_from_step": checkpoint_step if mode == "resume" else None,
        "target_steps": SMOKE_STEPS if mode == "smoke" else TRAIN_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "command": command,
        "log": str(log_path.relative_to(root)),
    }
    atomic_json(status_path, state)

    def forward_signal(signum: int, _frame: Any) -> None:
        if process is not None and process.poll() is None:
            process.send_signal(signum)

    previous = {
        signum: signal.signal(signum, forward_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))
        environment["TRACKIO_DIR"] = str(trackio_dir)
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        state.update({"state": "running", "pid": process.pid, "updated_at": now_iso()})
        atomic_json(status_path, state)
        with log_path.open("a" if mode == "resume" else "w") as log_file:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
                log_file.flush()
                parsed = parse_training_metrics(line)
                if parsed is None:
                    continue
                displayed_step, metrics = parsed
                # LeRobot's console formatter rounds 1050...1450 to `1K`.
                # Training logs at a fixed cadence, so use the known cadence
                # after the first point instead of the lossy displayed value.
                interval = 1 if mode == "smoke" else LOG_FREQ
                step = (
                    displayed_step
                    if state["last_step"] == 0
                    else state["last_step"] + interval
                )
                trackio.log(metrics, step=step)
                state.update(
                    {
                        "last_step": step,
                        "last_metrics": metrics,
                        "updated_at": now_iso(),
                    }
                )
                atomic_json(status_path, state)
                loss = metrics.get("train/loss")
                if loss is not None and not math.isfinite(loss):
                    trackio.alert(
                        title="Non-finite training loss",
                        text=f"Loss became {loss} at step {step}",
                        level=trackio.AlertLevel.ERROR,
                    )
            return_code = process.wait()

        state.update(
            {
                "state": "completed" if return_code == 0 else "failed",
                "exit_code": return_code,
                "finished_at": now_iso(),
                "updated_at": now_iso(),
            }
        )
        atomic_json(status_path, state)
        trackio.log({"run/exit_code": return_code}, step=state["last_step"] + 1)
        if return_code:
            trackio.alert(
                title="Training process failed",
                text=f"torchrun exited with code {return_code}; see {log_path}",
                level=trackio.AlertLevel.ERROR,
            )
        else:
            trackio.alert(
                title="Training process completed",
                text=f"{mode} run finished at step {state['last_step']}",
                level=trackio.AlertLevel.INFO,
            )
        return return_code
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        trackio.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pinned LIBERO-90 DDP training")
    parser.add_argument("mode", choices=("smoke", "full", "resume"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.mode, args.dry_run))


if __name__ == "__main__":
    main()
