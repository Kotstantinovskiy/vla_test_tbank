from __future__ import annotations

"""Round-trip verification of the official-HDF5 conversion.

For sampled episodes of a converted suite this script asserts, against the
original official files named in the conversion manifest:

1. actions and state are bit-exact float32 copies (state additionally checked
   against the ee_pos/ee_ori/gripper recipe),
2. decoded video frames, rotated back by 180 degrees, match the official raw
   frames within video-encoding noise only (same 128x128 resolution, no
   resize), for first/middle/last frames of both cameras,
3. episode order equals official order (file name sort, then demo index), so
   "first k episodes" selects official demo_0..demo_{k-1},
4. dataset metadata declares fps=20 and the expected task/episode/frame
   counts, and every task string matches its source file name.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pads

from .constants import (
    CAMERA_KEYS,
    CONVERTED_ROOT,
    FPS,
    IMAGE_SIZE,
    SEEN_SUITE,
    TARGET_SUITE,
    experiment_root,
)

MAX_FRAME_MAE = 4.0  # crf-18 AV1 on 128x128 frames; measured ~1 on samples
SAMPLE_STRIDE_DEFAULT = 25  # verify every 25th episode plus first/last


def episode_table(root: Path, episode_index: int):
    files = sorted(str(path) for path in (root / "data").rglob("*.parquet"))
    table = pads.dataset(files, format="parquet").to_table(
        columns=["frame_index", "action", "observation.state"],
        filter=pc.field("episode_index") == episode_index,
    )
    return table.sort_by("frame_index")


def stacked(column) -> np.ndarray:
    return np.stack([np.asarray(row, dtype=np.float32) for row in column.to_pylist()])


def decode_frames(root: Path, episode_index: int, frame_indices: list[int]):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = json.loads((root / "conversion_manifest.json").read_text())
    dataset = LeRobotDataset(
        manifest["repo_id"], root=root, episodes=[episode_index], video_backend="pyav"
    )
    frames: dict[int, dict[str, np.ndarray]] = {}
    for frame_index in frame_indices:
        item = dataset[frame_index]
        per_camera = {}
        for camera_key in CAMERA_KEYS:
            frame = item[camera_key].numpy()
            if frame.shape[0] != 3:
                raise AssertionError(f"Expected CHW frame, got {frame.shape}")
            frame = np.transpose(frame, (1, 2, 0))
            if frame.dtype != np.uint8:
                frame = np.clip(np.round(frame * 255.0), 0, 255).astype(np.uint8)
            per_camera[camera_key] = frame
        frames[frame_index] = per_camera
    return frames


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def verify_suite(suite: str, root: Path, stride: int) -> dict[str, Any]:
    suite_root = root / suite
    manifest = json.loads((suite_root / "conversion_manifest.json").read_text())
    episodes = manifest["episodes"]
    info = json.loads((suite_root / "meta/info.json").read_text())
    checks: list[str] = []

    if info["fps"] != FPS:
        raise AssertionError(f"info.json fps={info['fps']} != {FPS}")
    if info["total_episodes"] != manifest["total_episodes"] or info[
        "total_frames"
    ] != manifest["total_frames"]:
        raise AssertionError("info.json totals disagree with conversion manifest")
    checks.append(f"metadata: fps={FPS}, episodes={info['total_episodes']}, frames={info['total_frames']}")

    # Official order: within each file demo indices strictly increase from 0,
    # and files appear in sorted order.
    file_sequence = [item["official_file"] for item in episodes]
    if file_sequence != sorted(file_sequence, key=file_sequence.index):
        raise AssertionError("episode file grouping broken")
    seen_files = []
    for item in episodes:
        if not seen_files or seen_files[-1] != item["official_file"]:
            seen_files.append(item["official_file"])
            if item["official_demo_index"] != 0:
                raise AssertionError(f"first episode of {item['official_file']} is not demo_0")
            previous = 0
        else:
            if item["official_demo_index"] != previous + 1:
                raise AssertionError(
                    f"demo order broken in {item['official_file']}: "
                    f"{previous} -> {item['official_demo_index']}"
                )
            previous = item["official_demo_index"]
    if seen_files != sorted(seen_files):
        raise AssertionError("official files are not in sorted order")
    checks.append(f"episode order: official (files sorted, demo_0..demo_N contiguous) over {len(seen_files)} files")

    sample = sorted({0, len(episodes) - 1, *range(0, len(episodes), stride)})
    frame_records: list[dict[str, Any]] = []
    for episode_index in sample:
        item = episodes[episode_index]
        table = episode_table(suite_root, episode_index)
        actions = stacked(table["action"])
        state = stacked(table["observation.state"])
        with h5py.File(Path(manifest["source_root"]) / item["official_file"], "r") as handle:
            demo = handle[f"data/{item['official_demo']}"]
            official_actions = np.asarray(demo["actions"], dtype=np.float32)
            obs = demo["obs"]
            official_state = np.concatenate(
                [
                    np.asarray(obs["ee_pos"], dtype=np.float32),
                    np.asarray(obs["ee_ori"], dtype=np.float32),
                    np.asarray(obs["gripper_states"], dtype=np.float32),
                ],
                axis=-1,
            )
            if not np.array_equal(actions, official_actions):
                raise AssertionError(f"actions not bit-exact for episode {episode_index}")
            if not np.array_equal(state, official_state):
                raise AssertionError(f"state not bit-exact for episode {episode_index}")
            frame_indices = sorted({0, len(actions) // 2, len(actions) - 1})
            decoded = decode_frames(suite_root, episode_index, frame_indices)
            for frame_index in frame_indices:
                for camera_key, official_name in CAMERA_KEYS.items():
                    stored = decoded[frame_index][camera_key]
                    if stored.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                        raise AssertionError(f"unexpected stored frame shape {stored.shape}")
                    official = np.asarray(obs[official_name][frame_index])
                    value = mae(stored[::-1, ::-1], official)
                    frame_records.append(
                        {
                            "episode_index": episode_index,
                            "camera": camera_key,
                            "frame_index": frame_index,
                            "rot180back_vs_official_mae": value,
                        }
                    )
                    if value > MAX_FRAME_MAE:
                        raise AssertionError(
                            f"frame mismatch: episode {episode_index} {camera_key} "
                            f"frame {frame_index} MAE {value:.2f} > {MAX_FRAME_MAE}"
                        )
    checks.append(
        f"bit-exact actions+state and frames within MAE {MAX_FRAME_MAE} on "
        f"{len(sample)} episodes / {len(frame_records)} frame pairs"
    )

    # Task strings must match their source file names.
    for item in episodes:
        from .convert import task_from_file

        expected = task_from_file(Path(item["official_file"]), suite)
        if item["task"] != expected:
            raise AssertionError(
                f"task mismatch for {item['official_file']}: {item['task']!r} != {expected!r}"
            )
    checks.append("task strings match official file names")

    return {
        "suite": suite,
        "sampled_episodes": sample,
        "frame_pairs": len(frame_records),
        "mean_frame_mae": float(
            np.mean([record["rot180back_vs_official_mae"] for record in frame_records])
        ),
        "max_frame_mae": float(
            np.max([record["rot180back_vs_official_mae"] for record in frame_records])
        ),
        "checks": checks,
        "frames": frame_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suites", nargs="+", choices=(SEEN_SUITE, TARGET_SUITE), default=[TARGET_SUITE]
    )
    parser.add_argument("--root", type=Path, default=CONVERTED_ROOT)
    parser.add_argument("--stride", type=int, default=SAMPLE_STRIDE_DEFAULT)
    args = parser.parse_args()
    results = {}
    for suite in args.suites:
        result = verify_suite(suite, args.root, args.stride)
        results[suite] = result
        output = experiment_root() / "artifacts" / f"conversion_verification_{suite}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                suite: {k: v for k, v in result.items() if k != "frames"}
                for suite, result in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
