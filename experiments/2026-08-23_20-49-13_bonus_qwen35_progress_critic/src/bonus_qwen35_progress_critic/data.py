from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from lerobot.datasets.video_utils import decode_video_frames


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    task: str
    length: int
    video_path: str
    from_timestamp: float
    to_timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProgressSpec:
    episode_index: int
    target_bin: int


def load_episode_records(dataset_root: str | Path, camera: str) -> list[EpisodeRecord]:
    root = Path(dataset_root)
    columns = [
        "episode_index",
        "tasks",
        "length",
        f"videos/{camera}/chunk_index",
        f"videos/{camera}/file_index",
        f"videos/{camera}/from_timestamp",
        f"videos/{camera}/to_timestamp",
    ]
    rows: list[EpisodeRecord] = []
    for parquet_path in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        for item in pq.read_table(parquet_path, columns=columns).to_pylist():
            tasks = item["tasks"]
            if len(tasks) != 1:
                raise ValueError(f"episode {item['episode_index']} has {len(tasks)} tasks; expected one")
            chunk_index = int(item[f"videos/{camera}/chunk_index"])
            file_index = int(item[f"videos/{camera}/file_index"])
            video_path = root / "videos" / camera / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            rows.append(
                EpisodeRecord(
                    episode_index=int(item["episode_index"]),
                    task=str(tasks[0]),
                    length=int(item["length"]),
                    video_path=str(video_path),
                    from_timestamp=float(item[f"videos/{camera}/from_timestamp"]),
                    to_timestamp=float(item[f"videos/{camera}/to_timestamp"]),
                )
            )
    rows.sort(key=lambda row: row.episode_index)
    if not rows:
        raise ValueError(f"no episodes found under {root}")
    if len({row.episode_index for row in rows}) != len(rows):
        raise ValueError("episode indices are not unique")
    return rows


def split_records_by_task(
    records: Iterable[EpisodeRecord], validation_fraction: float, seed: int
) -> tuple[list[EpisodeRecord], list[EpisodeRecord], list[str]]:
    records = list(records)
    tasks = sorted({row.task for row in records})
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0,1)")
    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, math.ceil(len(tasks) * validation_fraction))
    validation_tasks = sorted(shuffled[:validation_count])
    validation_set = set(validation_tasks)
    train = [row for row in records if row.task not in validation_set]
    validation = [row for row in records if row.task in validation_set]
    if {row.task for row in train} & {row.task for row in validation}:
        raise AssertionError("task leakage between train and validation")
    return train, validation, validation_tasks


def frame_indices_for_endpoint(endpoint: int, max_frames: int = 4) -> list[int]:
    if endpoint < 0:
        raise ValueError("endpoint must be non-negative")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    return np.linspace(0, endpoint, max_frames, dtype=np.int64).tolist()


def endpoint_and_bin(length: int, requested_bin: int, num_bins: int) -> tuple[int, int]:
    if length < 2:
        raise ValueError("trajectory must contain at least two frames")
    if not 0 <= requested_bin < num_bins:
        raise ValueError("requested_bin outside valid range")
    endpoint = int(round((length - 1) * requested_bin / (num_bins - 1)))
    target_bin = int(round((num_bins - 1) * endpoint / (length - 1)))
    return endpoint, target_bin


def build_validation_specs(
    records: Iterable[EpisodeRecord], bins: list[int], max_samples: int, seed: int
) -> list[ProgressSpec]:
    records = list(records)
    candidates = [ProgressSpec(row.episode_index, target_bin) for row in records for target_bin in bins]
    random.Random(seed).shuffle(candidates)
    return candidates[: min(max_samples, len(candidates))]


class VideoProgressDataset(Dataset):
    def __init__(
        self,
        records: list[EpisodeRecord],
        *,
        fps: int,
        num_bins: int,
        max_frames: int,
        backend: str,
        seed: int,
        samples_per_epoch: int | None = None,
        fixed_specs: list[ProgressSpec] | None = None,
    ) -> None:
        if not records:
            raise ValueError("records cannot be empty")
        self.records = records
        self.by_episode = {row.episode_index: row for row in records}
        self.fps = fps
        self.num_bins = num_bins
        self.max_frames = max_frames
        self.backend = backend
        self.seed = seed
        self.samples_per_epoch = samples_per_epoch
        self.fixed_specs = fixed_specs
        if fixed_specs is None and samples_per_epoch is None:
            raise ValueError("samples_per_epoch is required for a stochastic dataset")
        if fixed_specs is not None:
            missing = {spec.episode_index for spec in fixed_specs} - set(self.by_episode)
            if missing:
                raise ValueError(f"validation specs reference missing episodes: {sorted(missing)[:5]}")

    def __len__(self) -> int:
        return len(self.fixed_specs) if self.fixed_specs is not None else int(self.samples_per_epoch)

    def _spec(self, index: int) -> tuple[EpisodeRecord, int]:
        if self.fixed_specs is not None:
            spec = self.fixed_specs[index]
            return self.by_episode[spec.episode_index], spec.target_bin
        rng = np.random.default_rng(self.seed + index * 1_000_003)
        row = self.records[int(rng.integers(0, len(self.records)))]
        target_bin = int(rng.integers(0, self.num_bins))
        return row, target_bin

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, requested_bin = self._spec(index)
        endpoint, target_bin = endpoint_and_bin(row.length, requested_bin, self.num_bins)
        frame_indices = frame_indices_for_endpoint(endpoint, self.max_frames)
        timestamps = [row.from_timestamp + frame_index / self.fps for frame_index in frame_indices]
        frames = decode_video_frames(
            row.video_path,
            timestamps=timestamps,
            tolerance_s=(0.5 / self.fps) + 1e-4,
            backend=self.backend,
            return_uint8=True,
        )
        images = [Image.fromarray(frame.permute(1, 2, 0).numpy()) for frame in frames]
        return {
            "images": images,
            "task": row.task,
            "target_bin": target_bin,
            "target_progress": target_bin / (self.num_bins - 1),
            "episode_index": row.episode_index,
            "endpoint": endpoint,
            "frame_indices": frame_indices,
        }


class ProgressCollator:
    def __init__(self, processor: Any, prompt_prefix: str, prompt_suffix: str) -> None:
        self.processor = processor
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix

    def _conversation(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": self.prompt_prefix.format(task=item["task"])},
        ]
        for frame_number, image in enumerate(item["images"], start=1):
            content.append({"type": "text", "text": f"\nFrame {frame_number}:"})
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": f"\n{self.prompt_suffix}"})
        return [{"role": "user", "content": content}]

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        conversations = [self._conversation(item) for item in items]
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        return {
            "inputs": dict(inputs),
            "labels": torch.tensor([item["target_bin"] for item in items], dtype=torch.long),
            "target_progress": torch.tensor(
                [item["target_progress"] for item in items], dtype=torch.float32
            ),
            "episode_indices": [item["episode_index"] for item in items],
        }


def task_counts(records: Iterable[EpisodeRecord]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in records:
        counts[row.task] += 1
    return dict(sorted(counts.items()))
