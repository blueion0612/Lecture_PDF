"""Annotation tracking: what the lecturer wrote, and when they wiped it off.

Ink is measured as *difference from the slide as it first appeared*, not by
looking for particular pen colors.  A color rule only ever fits the recording
it was tuned on - it misses a blue pen, chalk on a blackboard, a dark marker on a
white slide, or a highlighter - whereas "changed since the slide arrived" is
true of annotation everywhere, and of nothing else once the lecturer has been
medianed out of the picture.

The interesting case is a lecturer who fills a slide, wipes it, and fills it
again.  Keeping only the final frame would lose the first board entirely, and
keeping every frame would bury the reader in near-duplicates.  So a slide is cut
into *epochs*: an epoch runs while ink accumulates and ends when most of what was
written disappears at once.  Each epoch contributes exactly one page - its
fullest moment - which is what someone reading the notes afterwards wants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class InkEpoch:
    """One write-and-wipe cycle over a single slide."""

    scene_index: int
    index: int
    start: float
    end: float
    peak_time: float
    peak_ratio: float
    window: tuple


@dataclass
class InkTrace:
    """Per-sample ink measurements for one scene, plus its epochs."""

    scene_index: int
    times: list = field(default_factory=list)
    ratios: list = field(default_factory=list)
    epochs: list = field(default_factory=list)
    cleanest_time: float = 0.0
    cleanest_ratio: float = 0.0


def changed_mask(view: np.ndarray, base: np.ndarray, threshold: int = 26) -> np.ndarray:
    """Pixels that differ from the pristine slide by more than sensor noise."""
    difference = cv2.absdiff(view, base)
    strength = difference.max(axis=2) if difference.ndim == 3 else difference
    mask = (strength > threshold).astype(np.uint8)
    # Opening drops compression speckle; a real stroke is several pixels wide
    # even at analysis resolution.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


# Backwards-compatible alias for the single-frame test.
ink_mask = changed_mask


def drop_solid_blobs(mask: np.ndarray, kernel_size: int, minimum_area: float) -> np.ndarray:
    """Remove large filled shapes, keeping thin ones.

    Returns the thin part and the solid part separately: the solid part is where
    a body is standing, and knowing that is as useful as removing it.

    Handwriting is thin almost by definition - it is made with a pen - while a
    person is a big solid silhouette.  Opening the mask therefore erases the
    writing and keeps the body, so subtracting what survives leaves the writing
    alone.  It is the one property that separates them regardless of color,
    position or how long anybody stands still, which a purely temporal test
    cannot manage for a lecturer who pauses.
    """
    if kernel_size < 3:
        return mask, np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    solid = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if count <= 1:
        return mask, np.zeros_like(mask)
    keep = np.zeros_like(mask)
    for index in range(1, count):
        # Small thick marks - a filled arrow, a blob of highlighter - are real
        # annotation and must survive; only body-sized shapes are discarded.
        if stats[index, cv2.CC_STAT_AREA] >= minimum_area:
            keep[labels == index] = 1
    if not keep.any():
        return mask, np.zeros_like(mask)
    keep = cv2.dilate(keep, kernel)
    return cv2.bitwise_and(mask, cv2.bitwise_not(keep * 255) // 255), keep


class InkTracker:
    """Streaming ink measurement and epoch cutting for one scene.

    Samples are fed in as they are decoded and only a short rolling window is
    retained, so a two-hour recording costs no more memory than a two-minute one.
    """

    def __init__(
        self,
        scene_index: int,
        erase_ratio: float = 0.55,
        persist_samples: int = 3,
        minimum_ink: float = 0.0015,
        threshold: int = 26,
        window_seconds: float = 6.0,
        settle_samples: int = 8,
        base_samples: int = 12,
        blob_fraction: float = 0.035,
        erase_drop: float = 0.55,
    ):
        self.trace = InkTrace(scene_index=scene_index)
        self.erase_ratio = erase_ratio
        self.persist_samples = max(1, persist_samples)
        self.minimum_ink = minimum_ink
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.settle_samples = max(1, settle_samples)
        self.base_samples = max(2, base_samples)
        self.blob_fraction = blob_fraction
        self.erase_drop = erase_drop
        self.hold_samples = max(2, settle_samples)
        self._blob_kernel = 0
        self._blob_area = 0.0

        self._warmup: list = []
        self._base = None
        self._hits = None
        self._occluded = None
        self._area = 1.0
        self._peak_mask = None
        self._peak_ratio = 0.0
        self._peak_time = 0.0
        self._epoch_start = None
        self._epoch_index = 0
        self._erase_streak = 0
        self._erase_time = 0.0
        self._cleanest = None
        self._last_time = 0.0

    def add(self, timestamp: float, view: np.ndarray) -> None:
        self._last_time = timestamp
        if self._epoch_start is None:
            self._epoch_start = timestamp
            self._peak_time = timestamp
            self._area = float(view.shape[0] * view.shape[1])

        if self._base is None:
            # Establish the pristine slide from the opening moments.  The window
            # has to be long enough for the median to have walked the lecturer out
            # of it, or her starting position becomes part of "the slide" and
            # everywhere she later isn't reads as fresh ink.
            self._warmup.append(view)
            if len(self._warmup) < self.base_samples:
                return
            self._base = np.median(np.stack(self._warmup, axis=0), axis=0).astype(np.uint8)
            self._warmup = []
            self._hits = np.zeros(self._base.shape[:2], np.int32)
            self._occluded = np.zeros(self._base.shape[:2], np.int32)
            self._blob_kernel = max(3, (int(round(self._base.shape[1] * self.blob_fraction)) | 1))
            self._blob_area = float(self._base.shape[0] * self._base.shape[1]) * 0.004

        if view.shape != self._base.shape:
            # The framing changed mid-scene; keep measuring on the base's grid.
            view = cv2.resize(view, (self._base.shape[1], self._base.shape[0]), interpolation=cv2.INTER_AREA)

        # Ink is what *stays* changed.  A lecturer crossing the board changes the
        # pixels she covers for a moment and then gives them back; a written
        # stroke never does.  Counting consecutive changed observations separates
        # the two without needing to detect or model a person at all.
        changed, occluded = drop_solid_blobs(
            changed_mask(view, self._base, self.threshold), self._blob_kernel, self._blob_area
        )
        # Count up while a pixel stays changed and decay when it does not, rather
        # than resetting.  A hand sweeping over finished writing hides it for a
        # sample or two; resetting would make that look like an erase, while a
        # decay lets the writing survive the pass and still lets a real wipe fall
        # away quickly.  The cap is kept just above the threshold so that a wipe
        # clears in a few samples instead of having to unwind a long history;
        # brief occlusion is handled by the erase *ratio*, not by this timing.
        cap = self.settle_samples + 4
        updated = np.where(
            changed > 0, np.minimum(self._hits + 1, cap), np.maximum(self._hits - 2, 0)
        )
        # Behind the lecturer there is no evidence either way, so hold the
        # previous state instead of decaying it: treating "hidden" as "erased" is
        # what made someone standing in front of their own writing look like they
        # had wiped the board.  The hold is time-limited, because a board really
        # can be wiped by someone who then stays in front of it, and the state has
        # to be allowed to go stale rather than being trusted forever.
        self._occluded = np.where(occluded > 0, self._occluded + 1, 0)
        hold = (occluded > 0) & (self._occluded <= self.hold_samples)
        self._hits = np.where(hold, self._hits, updated)
        mask = (self._hits >= self.settle_samples).astype(np.uint8)
        ratio = float(np.count_nonzero(mask)) / max(1.0, self._area)
        self.trace.times.append(timestamp)
        self.trace.ratios.append(ratio)
        if self._cleanest is None or ratio < self._cleanest:
            self._cleanest = ratio
            self.trace.cleanest_time = timestamp

        if self._peak_mask is not None and self._peak_ratio >= self.minimum_ink:
            kept = float(np.count_nonzero(cv2.bitwise_and(mask, self._peak_mask)))
            retained = kept / max(1.0, float(np.count_nonzero(self._peak_mask)))
            # Both tests have to fail before this counts as an erase: most of the
            # earlier writing must be gone *and* there must be much less writing
            # overall.  Overlap alone drops whenever the lecturer reworks a
            # diagram in place, which is still the same board and belongs on the
            # same page; only wiping actually leaves the slide emptier.
            shrank = ratio <= self._peak_ratio * self.erase_drop
            if retained < self.erase_ratio and shrank:
                if self._erase_streak == 0:
                    self._erase_time = timestamp
                self._erase_streak += 1
                # Demand persistence: the lecturer stepping in front of her own
                # writing hides it for a moment and would otherwise read as an
                # erase.  A wipe stays wiped.
                if self._erase_streak >= self.persist_samples:
                    self._close(self._erase_time)
                    self._epoch_start = timestamp
                    self._base = view.copy()  # whatever survived is the new slate
                    self._hits = np.zeros(self._base.shape[:2], np.int32)
                    self._occluded = np.zeros(self._base.shape[:2], np.int32)
                    self._peak_mask = None
                    self._peak_ratio = 0.0
                    self._peak_time = timestamp
                    self._erase_streak = 0
                    return
            else:
                self._erase_streak = 0

        if self._peak_mask is None or ratio > self._peak_ratio:
            self._peak_mask = mask
            self._peak_ratio = ratio
            self._peak_time = timestamp

    def _close(self, end: float) -> None:
        if self._peak_mask is None or self._peak_ratio < self.minimum_ink:
            return
        low = max(self._epoch_start or 0.0, self._peak_time - self.window_seconds)
        high = min(end, self._peak_time + self.window_seconds)
        if high <= low:
            high = end
        self.trace.epochs.append(
            InkEpoch(
                scene_index=self.trace.scene_index,
                index=self._epoch_index,
                start=self._epoch_start or 0.0,
                end=end,
                peak_time=self._peak_time,
                peak_ratio=self._peak_ratio,
                window=(low, high),
            )
        )
        self._epoch_index += 1

    def finish(self, end: float | None = None) -> InkTrace:
        self._close(end if end is not None else self._last_time)
        self.trace.cleanest_ratio = float(self._cleanest or 0.0)
        return self.trace
