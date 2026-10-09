"""Read a clip into memory (downscaled, frame-rate capped) and write debug video."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoClip:
    path: str
    frames: list[np.ndarray]          # BGR uint8 at processing resolution
    times: np.ndarray                 # seconds from the start of the source video
    source_fps: float
    source_size: tuple[int, int]      # (width, height) of the source video
    scale: float                      # processing px = source px * scale
    source_frame_count: int
    source_indices: list[int] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.frames[0].shape[:2]
        return w, h

    @property
    def processed_fps(self) -> float:
        if len(self.times) < 2:
            return self.source_fps
        return float(1.0 / np.median(np.diff(self.times)))


def load_video(path: str | Path, max_side: int = 960, max_fps: float = 15.0,
               max_seconds: float | None = None) -> VideoClip:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(source_fps / max_fps))) if max_fps > 0 else 1

    frames, times, indices = [], [], []
    source_size = None
    scale = 1.0
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if source_size is None:
            source_size = (frame.shape[1], frame.shape[0])
            scale = min(1.0, max_side / max(source_size))
        if max_seconds is not None and index / source_fps > max_seconds:
            break
        if index % stride == 0:
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(frame)
            times.append(index / source_fps)
            indices.append(index)
        index += 1
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return VideoClip(path=str(path), frames=frames, times=np.array(times), source_fps=float(source_fps),
                     source_size=source_size, scale=scale, source_frame_count=n_total or index,
                     source_indices=indices)


class VideoWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()
