from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from .constants import ARTIFACTS_DIR, MANIFEST_PATH, OWN_SCORES_PATH, ROBOMETER_SCORES_PATH
from .own_critic import encode_own_batch, load_own_critic
from .robometer import build_robometer, score_robometer_batch
from .utils import atomic_json, atomic_jsonl, load_config, read_jsonl, set_status, sha256_file
from .video import decode_uniform_frames


def _valid_existing(path: Path, expected: int, backend: str) -> bool:
    if not path.exists():
        return False
    rows = read_jsonl(path)
    return len(rows) == expected and len({row["video_id"] for row in rows}) == expected and all(
        row["critic"] == backend for row in rows
    )


def score(backend: str) -> None:
    config = load_config()
    manifest = read_jsonl(MANIFEST_PATH)
    expected = int(config["scope"]["expected_videos"])
    output = OWN_SCORES_PATH if backend == "own" else ROBOMETER_SCORES_PATH
    if _valid_existing(output, expected, backend):
        print(f"{backend}: complete output exists, skipping")
        return
    batch_size = int(config["own_critic" if backend == "own" else "robometer"]["batch_size"])
    started = time.perf_counter()
    set_status(f"scoring_{backend}", active_backend=backend, completed_predictions=0, expected_predictions=expected)
    if backend == "own":
        model, processor = load_own_critic(config, "/var/tmp/vla_hf/hub")
        checkpoint = Path(config["own_critic"]["checkpoint"])
        audit = {
            "base_repo": config["own_critic"]["base_repo"],
            "base_revision": config["own_critic"]["base_revision"],
            "checkpoint": str(checkpoint),
            "num_progress_bins": 32,
            "adapter_sha256": sha256_file(checkpoint / "adapter" / "adapter_model.safetensors"),
            "progress_head_sha256": sha256_file(checkpoint / "progress_head.safetensors"),
        }
    else:
        model, processor, audit = build_robometer(config)
    atomic_json(ARTIFACTS_DIR / f"{backend}_load_audit.json", audit)

    predictions: list[dict] = []
    for offset in range(0, len(manifest), batch_size):
        rows = manifest[offset : offset + batch_size]
        decoded = [decode_uniform_frames(row["video_path"], sample_count=4) for row in rows]
        frames = [item[0] for item in decoded]
        if backend == "own":
            encoded = encode_own_batch(processor, rows, frames, config)
            with torch.inference_mode():
                scores = model(encoded)["progress"].detach().float().cpu().tolist()
            sequences = None
        else:
            scores, sequences = score_robometer_batch(model, processor, rows, frames)
        for index, (row, score_value, video_info) in enumerate(zip(rows, scores, decoded, strict=True)):
            record = {
                "video_id": row["video_id"],
                "critic": backend,
                "score": float(score_value),
                "task_id": row["task_id"],
                "candidate": row["candidate"],
                "episode_ix": row["episode_ix"],
                "decoded_frame_count": video_info[1],
                "sampled_frame_indices": video_info[2],
            }
            if sequences is not None:
                record["frame_progress"] = [float(value) for value in sequences[index]]
            predictions.append(record)
        done = len(predictions)
        if done == expected or done % 40 == 0:
            elapsed = time.perf_counter() - started
            set_status(
                f"scoring_{backend}",
                active_backend=backend,
                completed_predictions=done,
                expected_predictions=expected,
                elapsed_seconds=elapsed,
                predictions_per_second=done / elapsed,
            )
            print(f"{backend}: {done}/{expected} ({done / elapsed:.2f} videos/s)", flush=True)
    if len(predictions) != expected:
        raise AssertionError(f"{backend}: expected {expected} scores, got {len(predictions)}")
    atomic_jsonl(output, predictions)
    set_status(
        f"{backend}_complete",
        active_backend=None,
        completed_predictions=expected,
        expected_predictions=expected,
        backend_seconds=time.perf_counter() - started,
    )
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["own", "robometer"])
    args = parser.parse_args()
    score(args.backend)


if __name__ == "__main__":
    main()
