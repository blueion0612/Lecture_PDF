"""Second pass: find slide changes, working in rectified slide space.

Doing the comparison *after* rectification is what makes the thresholds portable.
In raw camera coordinates, "the top 22% of the frame" means something different
for every recording; in slide space it always means the title band.  A screencast
and an off-axis camera shot of a projector become the same problem.

Three things must be told apart, and each has a distinct signature:

* a slide change    - a large fraction of pixels change at once, and stay changed,
* an annotation     - a small fraction change, and accumulate gradually,
* a passing person  - a large fraction change, then change back.

Persistence is therefore checked explicitly rather than inferred from magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import quad_for
from .scan import iter_samples
from .video import VideoInfo

ANALYSIS_WIDTH = 160
ANALYSIS_HEIGHT = 90
TINT_WIDTH = 80
TINT_HEIGHT = 45
TITLE_BAND = 0.22


@dataclass
class Sample:
    """One analyzed moment, already mapped into slide space.

    Only the small grayscale view is kept.  Scene detection runs over the whole
    recording, so per-sample cost decides whether a two-hour lecture is analysable
    at all; edges are recomputed on demand, which is far cheaper than storing them.
    """

    timestamp: float
    gray: np.ndarray
    tint: np.ndarray
    sharpness: float
    _edges: object = None

    @property
    def edges(self) -> np.ndarray:
        if self._edges is None:
            self._edges = cv2.Canny(self.gray, 40, 110)
        return self._edges


@dataclass
class Scene:
    """A stretch of video showing one underlying slide."""

    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def rectified_views(info: VideoInfo, segments, interval: float, width: int):
    """Stream ``(timestamp, view)`` rectified into slide space at ``width``."""
    for timestamp, frame in iter_samples(info, interval, 0):
        quad = quad_for(segments, timestamp)
        yield timestamp, quad.rectify(frame, width, width, sharpen=0.0)


def rectified_samples(info: VideoInfo, segments, interval: float, progress=None):
    """Stream lightweight samples for scene detection."""
    for timestamp, view in rectified_views(info, segments, interval, ANALYSIS_WIDTH * 2):
        # A fixed canvas, not the slide's own aspect: when the framing changes
        # part-way through, samples from either side still have to be comparable,
        # and a little distortion costs nothing to a change detector.
        small = cv2.resize(view, (ANALYSIS_WIDTH, ANALYSIS_HEIGHT), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        yield Sample(
            timestamp=timestamp,
            gray=cv2.GaussianBlur(gray, (5, 5), 0),
            tint=cv2.resize(small, (TINT_WIDTH, TINT_HEIGHT), interpolation=cv2.INTER_AREA),
            sharpness=sharpness,
        )
        if progress is not None:
            progress(timestamp)


def _band(image: np.ndarray, fraction: float) -> np.ndarray:
    height = image.shape[0]
    return image[: max(1, int(round(height * fraction)))]


def change_metrics(previous: Sample, current: Sample, fraction: float | None = None):
    """(changed-pixel share, edge disagreement, share of old content kept)."""
    if fraction is None:
        first_gray, second_gray = previous.gray, current.gray
        first_edges, second_edges = previous.edges, current.edges
    else:
        first_gray, second_gray = _band(previous.gray, fraction), _band(current.gray, fraction)
        first_edges, second_edges = _band(previous.edges, fraction), _band(current.edges, fraction)

    difference = cv2.absdiff(first_gray, second_gray)
    changed = float(np.count_nonzero(difference > 18)) / max(1, difference.size)

    # Two slides can share a luminance and differ only in hue - a recoloured
    # template, a different section background.  Grayscale would call those
    # identical, so color is compared too, on a thumbnail because the difference
    # that matters here is a broad wash rather than fine detail.
    first_tint = previous.tint if fraction is None else _band(previous.tint, fraction)
    second_tint = current.tint if fraction is None else _band(current.tint, fraction)
    tint_difference = cv2.absdiff(first_tint, second_tint).max(axis=2)
    changed = max(changed, float(np.count_nonzero(tint_difference > 18)) / max(1, tint_difference.size))

    first_set = first_edges > 0
    second_set = second_edges > 0
    union = int(np.count_nonzero(first_set | second_set))
    edge_xor = float(np.count_nonzero(first_set ^ second_set)) / max(1, union)

    # How much of what was on screen is still on screen.  This is the difference
    # between replacing a slide and writing on one: a new slide destroys the old
    # text, while an annotation leaves every bit of it in place and simply adds
    # more.  Both raise the changed-pixel count, so magnitude alone cannot tell
    # them apart - but only one of them takes the old content away.
    previous_ink = int(np.count_nonzero(first_set))
    retained = float(np.count_nonzero(first_set & second_set)) / max(1, previous_ink)
    return changed, edge_xor, retained


def _looks_like_change(previous: Sample, current: Sample, sensitivity: float, keeps: float = 0.80) -> bool:
    body_changed, body_xor, body_retained = change_metrics(previous, current)
    if body_retained > keeps and body_changed < 0.45:
        # Everything that was there is still there: this is annotation, not a new
        # slide.  A wholesale change is still allowed through, because a new slide
        # can happen to reuse the previous one's layout.
        return False
    if body_changed >= sensitivity or (body_xor >= 0.45 and body_changed >= sensitivity * 0.45):
        return True
    # A deck can reuse one body layout under different titles, and vice versa, so
    # the title band gets its own test rather than being averaged into the whole.
    title_changed, title_xor, title_retained = change_metrics(previous, current, TITLE_BAND)
    if title_retained > keeps:
        return False
    return title_changed >= sensitivity * 1.15 or (
        title_xor >= 0.5 and title_changed >= sensitivity * 0.6
    )


def detect_scenes(
    samples,
    duration: float,
    sensitivity: float = 0.12,
    minimum_seconds: float = 3.0,
    confirm_seconds: float = 2.0,
    interval: float = 0.5,
):
    """Split the sample stream into scenes, one per underlying slide.

    A candidate change is only accepted if the picture is *still* different a
    couple of seconds later.  That single test is what keeps a lecturer walking
    across the screen - which changes far more pixels than a slide transition -
    from being read as a slide change.
    """
    samples = list(samples)
    if len(samples) < 2:
        return [Scene(index=0, start=0.0, end=duration)], []

    confirm = max(1, int(round(confirm_seconds / max(0.05, interval))))
    boundaries: list[float] = []
    changes: list[float] = []
    for index in range(1, len(samples)):
        previous, current = samples[index - 1], samples[index]
        if not _looks_like_change(previous, current, sensitivity):
            continue
        changes.append(current.timestamp)
        later = samples[min(len(samples) - 1, index + confirm)]
        if not _looks_like_change(previous, later, sensitivity * 0.8):
            continue  # the picture came back: something passed in front
        if boundaries and current.timestamp - boundaries[-1] < minimum_seconds:
            continue
        boundaries.append(current.timestamp)

    scenes: list[Scene] = []
    start = samples[0].timestamp
    for boundary in boundaries:
        if boundary - start >= minimum_seconds:
            scenes.append(Scene(index=len(scenes), start=start, end=boundary))
        start = boundary
    if duration - start >= minimum_seconds or not scenes:
        scenes.append(Scene(index=len(scenes), start=start, end=duration))
    return scenes, changes


def samples_in(samples, scene: Scene, margin: float = 0.0):
    low = scene.start + margin
    high = scene.end - margin
    chosen = [sample for sample in samples if low <= sample.timestamp < high]
    if not chosen:
        chosen = [sample for sample in samples if scene.start <= sample.timestamp < scene.end]
    return chosen


def perceptual_hash(gray: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:8, :8].copy()
    median = float(np.median(low.flat[1:]))
    return (low > median).reshape(-1)


def hamming(first: np.ndarray, second: np.ndarray) -> int:
    return int(np.count_nonzero(first != second))
