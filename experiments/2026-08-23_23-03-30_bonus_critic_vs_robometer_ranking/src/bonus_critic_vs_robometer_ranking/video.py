from __future__ import annotations

from pathlib import Path

import av
import numpy as np


def uniform_frame_indices(frame_count: int, sample_count: int = 4) -> list[int]:
    if frame_count < 1:
        raise ValueError("video contains no decodable frames")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return np.linspace(0, frame_count - 1, sample_count).round().astype(int).tolist()


def decode_uniform_frames(path: str | Path, sample_count: int = 4) -> tuple[np.ndarray, int, list[int]]:
    decoded: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            decoded.append(frame.to_ndarray(format="rgb24"))
    indices = uniform_frame_indices(len(decoded), sample_count)
    return np.stack([decoded[index] for index in indices]), len(decoded), indices
