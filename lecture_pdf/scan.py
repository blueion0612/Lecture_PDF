"""First pass: cheap whole-video scan that yields the cues geometry needs.

Three maps separate a slide region from everything else in an unknown room:

* ``motion``      - fraction of consecutive sample pairs in which a pixel moved
                    perceptibly.  A *frequency* rather than a mean is what makes
                    this work on any recording: a slide change is rare, so even a
                    fast-cutting screencast stays low, while a lecturer - who
                    moves nearly all the time - stays high.  A mean would confuse
                    the two whenever slides change often.
* ``variability`` - spread of each pixel over the whole recording.  A screen is
                    high (slides change, annotations accumulate), a wall is zero,
                    and the lecturer is also high - which is why the two maps are
                    used together rather than alone.
* ``plates``      - per-pixel medians of frames spread far apart in time, so the
                    lecturer averages out and the static furniture does not.

A screen is the region that *changes but does not move*: high variability, low
motion.  A wall fails both, a logo or podium fails variability, and the lecturer
fails motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import median_plate
from .video import VideoInfo


def iter_samples(info: VideoInfo, interval: float, max_width: int = 0):
    """Yield ``(timestamp, frame)`` by decoding only the frames actually needed."""
    capture = cv2.VideoCapture(str(info.path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {info.path}")
    stride = max(1, int(round(info.fps * interval)))
    index = 0
    try:
        while True:
            if not capture.grab():
                break
            if index % stride == 0:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    break
                if max_width and frame.shape[1] > max_width:
                    height = max(1, int(round(frame.shape[0] * max_width / frame.shape[1])))
                    frame = cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)
                yield index / info.fps, frame
            index += 1
    finally:
        capture.release()


@dataclass
class GeometryScan:
    """Cue maps plus a reservoir of widely spaced frames, at scan resolution."""

    width: int
    height: int
    motion: np.ndarray
    variability: np.ndarray
    plate_times: list = field(default_factory=list)
    plate_frames: list = field(default_factory=list)
    sample_count: int = 0
    _slide_variability: object = None

    @property
    def slide_variability(self) -> np.ndarray:
        """The variability cue detection should use: plate-based when possible."""
        if self._slide_variability is None:
            computed = self.plate_variability()
            self._slide_variability = self.variability if computed is None else computed
        return self._slide_variability

    def plate_variability(self, groups: int = 8) -> np.ndarray | None:
        """How much the *slide* changes, measured between person-free plates.

        Raw per-pixel variability cannot tell a screen from the desk in front of
        it: the desk carries real edges, and a lecturer moving past it makes those
        pixels vary just as much as changing slides do.  Taking the median within
        each time window first removes the lecturer, so what is left varying
        between windows is only content that genuinely changed on screen.  Static
        furniture, walls, bezels and logos fall to zero.
        """
        if len(self.plate_frames) < 6:
            return None
        groups = max(3, min(groups, len(self.plate_frames) // 2))
        chunks = [chunk for chunk in np.array_split(np.arange(len(self.plate_frames)), groups) if len(chunk)]
        if len(chunks) < 3:
            return None
        grays = []
        for chunk in chunks:
            plate = median_plate([self.plate_frames[index] for index in chunk])
            grays.append(cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32))
        stack = np.stack(grays, axis=0)
        median = np.median(stack, axis=0)
        deviation = np.mean(np.abs(stack - median), axis=0)
        return cv2.GaussianBlur(deviation, (0, 0), 2.0)

    def plate(self, start: float | None = None, end: float | None = None):
        if not self.plate_frames:
            return None
        if start is None and end is None:
            chosen = self.plate_frames
        else:
            low = -np.inf if start is None else start
            high = np.inf if end is None else end
            chosen = [
                frame
                for timestamp, frame in zip(self.plate_times, self.plate_frames)
                if low <= timestamp < high
            ]
            if len(chosen) < 3:
                chosen = self.plate_frames
        return median_plate(chosen)


def scan_geometry(
    info: VideoInfo,
    interval: float = 2.0,
    work_width: int = 480,
    reservoir: int = 64,
    move_threshold: float = 12.0,
    global_change: float = 0.25,
    progress=None,
) -> GeometryScan:
    expected = max(1, int(info.duration / max(0.05, interval)))
    keep_every = max(1, expected // max(1, reservoir))

    moved_count = None
    total = None
    total_square = None
    previous = None
    count = 0
    pair_count = 0
    plate_times: list = []
    plate_frames: list = []
    width = height = 0

    for index, (timestamp, frame) in enumerate(iter_samples(info, interval, work_width)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if total is None:
            height, width = gray.shape
            moved_count = np.zeros_like(gray)
            total = np.zeros_like(gray)
            total_square = np.zeros_like(gray)
        elif gray.shape != total.shape:
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        if previous is not None:
            moved = (cv2.absdiff(gray, previous) > move_threshold).astype(np.float32)
            # A slide change or a camera cut moves most of the picture at once.
            # Counting those would make a fast-moving deck look exactly like a
            # person, so whole-frame events are excluded and only local movement
            # - which is what a person is - accumulates here.
            if float(moved.mean()) <= global_change:
                moved_count += moved
                pair_count += 1
        previous = gray
        total += gray
        total_square += gray * gray
        count += 1

        if index % keep_every == 0 and len(plate_frames) < reservoir * 2:
            plate_times.append(timestamp)
            plate_frames.append(frame.copy())
        if progress is not None and count % 200 == 0:
            progress(timestamp)

    if total is None or count == 0:
        raise RuntimeError(f"No frames decoded from {info.path}")

    mean = total / count
    variance = np.maximum(0.0, total_square / count - mean * mean)
    variability = cv2.GaussianBlur(np.sqrt(variance), (0, 0), 2.0)
    motion = cv2.GaussianBlur(moved_count / max(1, pair_count), (0, 0), 2.0)

    return GeometryScan(
        width=width,
        height=height,
        motion=motion,
        variability=variability,
        plate_times=plate_times,
        plate_frames=plate_frames,
        sample_count=count,
    )
