from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from .constants import ARTIFACTS_DIR, CONFIG_PATH, RESULTS_DIR
from .data import (
    ProgressCollator,
    VideoProgressDataset,
    build_validation_specs,
    load_episode_records,
)
from .model import audit_to_dict, audit_trainable_parameters, build_model
from .prepare import prepare
from .utils import atomic_json, estimate_full_runtime, load_config, now_iso, seed_everything


def move_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in inputs.items()}


def load_split_records(config: dict[str, Any]) -> tuple[list, list]:
    split = json.loads((ARTIFACTS_DIR / "dataset_split.json").read_text())
    records = load_episode_records(config["data"]["root"], config["data"]["camera"])
    train_ids = set(split["train_episode_indices"])
    validation_ids = set(split["validation_episode_indices"])
    train = [row for row in records if row.episode_index in train_ids]
    validation = [row for row in records if row.episode_index in validation_ids]
    if len(train) != len(train_ids) or len(validation) != len(validation_ids):
        raise AssertionError("split manifest does not match dataset")
    return train, validation


@torch.no_grad()
def evaluate(model, loader, device: torch.device, max_samples: int) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    absolute_errors: list[float] = []
    correct = 0
    seen = 0
    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        targets = batch["target_progress"].to(device, non_blocking=True)
        inputs = move_inputs(batch["inputs"], device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(inputs, labels)
        batch_size = labels.shape[0]
        losses.extend([float(output["loss"].item())] * batch_size)
        absolute_errors.extend(torch.abs(output["progress"] - targets).cpu().tolist())
        correct += int((output["logits"].argmax(dim=-1) == labels).sum().item())
        seen += batch_size
        if seen >= max_samples:
            break
    model.train()
    return {
        "loss": float(statistics.fmean(losses)),
        "mae": float(statistics.fmean(absolute_errors)),
        "bin_accuracy": correct / seen,
        "samples": seen,
    }


def plot_metrics(metrics: list[dict[str, Any]], output: Path) -> None:
    train = [row for row in metrics if row["kind"] == "train"]
    validation = [row for row in metrics if row["kind"] == "validation"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if train:
        axis.plot([row["step"] for row in train], [row["loss"] for row in train], label="train loss", alpha=0.8)
    if validation:
        axis.plot(
            [row["step"] for row in validation],
            [row["loss"] for row in validation],
            marker="o",
            label="validation loss",
        )
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("32-bin cross-entropy")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def maybe_init_trackio(config: dict[str, Any], mode: str, disabled: bool):
    if disabled:
        return None
    import trackio

    os.environ.setdefault("TRACKIO_DIR", str(ARTIFACTS_DIR / "trackio"))
    trackio.init(
        project=config["training"]["trackio_project"],
        name=(
            f"benchmark-full-schedule-seed-{config['experiment']['seed']}"
            if mode == "benchmark"
            else f"train-seed-{config['experiment']['seed']}"
        ),
        config=config,
        resume="never",
        auto_log_gpu=True,
        gpu_log_interval=10,
        auto_log_cpu=True,
        cpu_log_interval=10,
    )
    return trackio


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    prepare(args.config)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    training_config = config["training"]
    model_config = config["model"]
    data_config = config["data"]
    max_steps = args.max_steps or (
        int(training_config["benchmark_steps"]) if args.mode == "benchmark" else int(training_config["full_steps"])
    )
    validation_interval = (
        int(training_config["validation_interval_benchmark"])
        if args.mode == "benchmark"
        else int(training_config["validation_interval_full"])
    )
    validation_samples = (
        int(training_config["validation_samples_benchmark"])
        if args.mode == "benchmark"
        else int(data_config["validation_samples"])
    )
    output_dir = Path(args.output) if args.output else RESULTS_DIR / ("benchmark_50" if args.mode == "benchmark" else "training")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {metrics_path}")
    status_path = output_dir / "status.json"
    status = {
        "state": "starting",
        "mode": args.mode,
        "started_at": now_iso(),
        "max_steps": max_steps,
        "gradient_checkpointing": False,
        "device": str(device),
        "full_training_launched": args.mode == "train",
    }
    atomic_json(status_path, status)
    atomic_json(RESULTS_DIR / "status.json", status)
    protocol_manifest = json.loads((ARTIFACTS_DIR / "protocol_manifest.json").read_text())
    protocol_manifest.update(
        {
            "launch_updated_at": now_iso(),
            "launch_mode": args.mode,
            "full_training_launched": args.mode == "train",
        }
    )
    atomic_json(ARTIFACTS_DIR / "protocol_manifest.json", protocol_manifest)
    total_start = time.perf_counter()
    train_records, validation_records = load_split_records(config)

    processor_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        cache_dir=os.environ.get("HF_HUB_CACHE", "/var/tmp/vla_hf/hub"),
    )
    collator = ProgressCollator(processor, config["prompt"]["prefix"], config["prompt"]["suffix"])
    train_dataset = VideoProgressDataset(
        train_records,
        fps=int(data_config["fps"]),
        num_bins=int(model_config["num_progress_bins"]),
        max_frames=int(model_config["max_frames"]),
        backend=data_config["video_backend"],
        seed=seed,
        samples_per_epoch=int(data_config["train_samples_per_epoch"]),
    )
    validation_specs = build_validation_specs(
        validation_records,
        list(data_config["validation_bins"]),
        validation_samples,
        seed,
    )
    validation_dataset = VideoProgressDataset(
        validation_records,
        fps=int(data_config["fps"]),
        num_bins=int(model_config["num_progress_bins"]),
        max_frames=int(model_config["max_frames"]),
        backend=data_config["video_backend"],
        seed=seed,
        fixed_specs=validation_specs,
    )
    workers = int(data_config["dataloader_workers"])
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": int(training_config["per_device_batch_size"]),
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
        "collate_fn": collator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, drop_last=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    data_setup_seconds = time.perf_counter() - processor_start

    model_start = time.perf_counter()
    model = build_model(config, cache_dir=os.environ.get("HF_HUB_CACHE", "/var/tmp/vla_hf/hub"))
    audit = audit_trainable_parameters(model)
    atomic_json(ARTIFACTS_DIR / "trainable_parameters.json", audit_to_dict(audit))
    model.to(device)
    model.train()
    model_load_seconds = time.perf_counter() - model_start
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler_horizon_steps = int(training_config["full_steps"])
    warmup_steps = max(1, int(scheduler_horizon_steps * float(training_config["warmup_ratio"])))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        scheduler_horizon_steps,
    )
    trackio = maybe_init_trackio(config, args.mode, args.no_trackio)
    metrics: list[dict[str, Any]] = []
    validation_seconds_total = 0.0

    try:
        validation_start = time.perf_counter()
        initial_validation = evaluate(model, validation_loader, device, validation_samples)
        validation_seconds_total += time.perf_counter() - validation_start
        row = {"kind": "validation", "step": 0, **initial_validation}
        metrics.append(row)
        append_jsonl(metrics_path, row)
        if trackio:
            trackio.log({f"validation/{key}": value for key, value in initial_validation.items()}, step=0)

        iterator = iter(train_loader)
        step_seconds: list[float] = []
        accumulation = int(training_config["gradient_accumulation_steps"])
        for step in range(1, max_steps + 1):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                labels = batch["labels"].to(device, non_blocking=True)
                inputs = move_inputs(batch["inputs"], device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(inputs, labels)
                    loss = output["loss"] / accumulation
                loss.backward()
                accumulated_loss += float(output["loss"].item())
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, float(training_config["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - step_start
            step_seconds.append(elapsed)
            row = {
                "kind": "train",
                "step": step,
                "loss": accumulated_loss / accumulation,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": float(gradient_norm),
                "step_seconds": elapsed,
            }
            metrics.append(row)
            append_jsonl(metrics_path, row)
            if trackio:
                trackio.log({f"train/{key}": value for key, value in row.items() if key not in {"kind", "step"}}, step=step)
            if step % validation_interval == 0 or step == max_steps:
                validation_start = time.perf_counter()
                validation_metrics = evaluate(model, validation_loader, device, validation_samples)
                validation_elapsed = time.perf_counter() - validation_start
                validation_seconds_total += validation_elapsed
                validation_row = {
                    "kind": "validation",
                    "step": step,
                    "eval_seconds": validation_elapsed,
                    **validation_metrics,
                }
                metrics.append(validation_row)
                append_jsonl(metrics_path, validation_row)
                plot_metrics(metrics, output_dir / "loss_curve.png")
                if trackio:
                    trackio.log(
                        {f"validation/{key}": value for key, value in validation_metrics.items()}, step=step
                    )
            if args.mode == "train" and step % int(training_config["save_interval"]) == 0:
                model.save_adapter_checkpoint(
                    output_dir / f"checkpoints/{step:06d}",
                    {"step": step, "base_revision": model_config["revision"]},
                )
        train_loop_seconds = sum(step_seconds)
        warm_step_seconds = step_seconds[min(5, len(step_seconds)) :] or step_seconds
        median_step_seconds = statistics.median(warm_step_seconds)
        full_steps = int(training_config["full_steps"])
        full_validation_runs = math.ceil(full_steps / int(training_config["validation_interval_full"])) + 1
        validation_runs = max(1, sum(row["kind"] == "validation" for row in metrics))
        full_validation_samples = int(data_config["validation_samples"])
        runtime_estimate = estimate_full_runtime(
            median_step_seconds=median_step_seconds,
            full_steps=full_steps,
            validation_seconds_total=validation_seconds_total,
            observed_validation_runs=validation_runs,
            observed_validation_samples=validation_samples,
            full_validation_runs=full_validation_runs,
            full_validation_samples=full_validation_samples,
        )
        summary = {
            "state": "benchmark_complete" if args.mode == "benchmark" else "training_complete",
            "completed_at": now_iso(),
            "mode": args.mode,
            "steps": max_steps,
            "batch_size": int(training_config["per_device_batch_size"]),
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": int(training_config["per_device_batch_size"]) * accumulation,
            "scheduler_horizon_steps": scheduler_horizon_steps,
            "warmup_steps": warmup_steps,
            "gradient_checkpointing": False,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "gpu_free_at_training_start_gib": free_memory / 2**30,
            "gpu_total_memory_gib": total_memory / 2**30,
            "data_setup_seconds": data_setup_seconds,
            "model_load_seconds": model_load_seconds,
            "train_loop_seconds_excluding_validation": train_loop_seconds,
            "validation_seconds_total": validation_seconds_total,
            "total_wall_seconds": time.perf_counter() - total_start,
            "median_optimizer_step_seconds_after_warmup": median_step_seconds,
            "optimizer_steps_per_hour": 3600.0 / median_step_seconds,
            "estimated_full_steps": full_steps,
            "benchmark_validation_samples": validation_samples,
            "estimated_full_validation_samples": full_validation_samples,
            "estimated_full_validation_runs": full_validation_runs,
            **runtime_estimate,
            "peak_allocated_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_vram_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "trainable_parameters": audit.trainable_parameters,
            "total_parameters": audit.total_parameters,
            "last_train_loss": next(row["loss"] for row in reversed(metrics) if row["kind"] == "train"),
            "last_validation": next(row for row in reversed(metrics) if row["kind"] == "validation"),
            "loss_curve": str(output_dir / "loss_curve.png"),
            "full_training_launched": args.mode == "train",
        }
        atomic_json(output_dir / "summary.json", summary)
        atomic_json(status_path, summary)
        atomic_json(RESULTS_DIR / "status.json", summary)
        plot_metrics(metrics, output_dir / "loss_curve.png")
        return summary
    except torch.cuda.OutOfMemoryError as error:
        failure = {
            **status,
            "state": "oom",
            "failed_at": now_iso(),
            "error": str(error),
            "peak_allocated_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "next_action": "reduce per_device_batch_size before enabling gradient checkpointing",
        }
        atomic_json(status_path, failure)
        atomic_json(RESULTS_DIR / "status.json", failure)
        raise
    except Exception as error:
        failure = {**status, "state": "failed", "failed_at": now_iso(), "error": repr(error)}
        atomic_json(status_path, failure)
        atomic_json(RESULTS_DIR / "status.json", failure)
        raise
    finally:
        if trackio:
            trackio.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--mode", choices=["benchmark", "train"], default="benchmark")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output")
    parser.add_argument("--no-trackio", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
