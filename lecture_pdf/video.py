"""Video access helpers: metadata, seeking, and temporal clean plates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import median_plate


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if fps <= 0 or not np.isfinite(fps):
        fps = 25.0
    if frame_count <= 0 or width <= 0 or height <= 0:
        # Some containers lie about their length; count by walking the stream.
        capture = cv2.VideoCapture(str(path))
        counted = 0
        while capture.grab():
            counted += 1
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read any frame from: {path}")
        frame_count = max(counted, 1)
        height, width = frame.shape[:2]
    return VideoInfo(path=path, fps=fps, frame_count=frame_count, width=width, height=height)


class FrameReader:
    """Random access to frames, reusing one capture handle."""

    def __init__(self, info: VideoInfo):
        self.info = info
        self._capture = cv2.VideoCapture(str(info.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open video: {info.path}")

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def frame_at(self, timestamp: float):
        target = int(round(timestamp * self.info.fps))
        target = max(0, min(self.info.frame_count - 1, target))
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = self._capture.read()
        return frame if ok else None

    def window_frames(self, centre: float, span: float, count: int):
        """Read ``count`` frames spread over ``span`` seconds around ``center``.

        One seek followed by sequential grabs; seeking per frame is far slower and
        less reliable on long files.
        """
        if count <= 1:
            frame = self.frame_at(centre)
            return [frame] if frame is not None else []

        duration = self.info.duration
        half = span / 2.0
        start = max(0.0, min(centre - half, max(0.0, duration - span)))
        step_frames = max(1, int(round(self.info.fps * span / count)))
        start_frame = max(0, int(round(start * self.info.fps)))
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        index = 0
        while len(frames) < count:
            if index % step_frames == 0:
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
            else:
                if not self._capture.grab():
                    break
            index += 1
        return frames

    def plate_at(self, centre: float, span: float = 8.0, count: int = 9):
        """Clean plate around a timestamp: the moving lecturer medians away."""
        frames = self.window_frames(centre, span, count)
        if not frames:
            frame = self.frame_at(centre)
            return frame
        return median_plate(frames)


