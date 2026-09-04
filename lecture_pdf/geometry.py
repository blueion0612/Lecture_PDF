"""Automatic slide-region detection and rectification.

The goal is that any lecture recording works without hand calibration:

* a full-screen capture where the slide already fills the frame,
* a fixed camera pointed at a display or projection screen,
* a tilted/off-axis camera that leaves the screen visibly keystoned,
* recordings whose framing changes part-way through (cuts, zooms, layout swaps).

Detection runs on *clean plates* - per-pixel temporal medians that remove the
moving lecturer - and combines three cues that all survive an unknown room:
local edge density, local contrast, and how much a pixel changes when the slide
changes.  Walls, podiums and logos score low on all three.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

# Aspect ratios worth snapping to once the measurement is close enough that the
# residual is certainly camera noise rather than a genuinely odd screen.
COMMON_ASPECTS = (16 / 9, 16 / 10, 4 / 3, 3 / 2, 5 / 4, 21 / 9, 2 / 1, 1 / 1)
ASPECT_SNAP_TOLERANCE = 0.015

# A detected region this large is treated as "the slide already fills the frame".
FULL_FRAME_AREA = 0.90
AXIS_ALIGNED_TOLERANCE = 0.006  # normalized corner deviation from a plain rectangle


def _normalize(values: np.ndarray) -> np.ndarray:
    low = float(np.percentile(values, 2))
    high = float(np.percentile(values, 98))
    if high - low < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def median_plate(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Per-pixel median of equally sized frames; suppresses anything that moves."""
    if not frames:
        raise ValueError("median_plate needs at least one frame")
    if len(frames) == 1:
        return frames[0].copy()
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points clockwise starting at the top-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    clockwise = points[np.argsort(angles)]
    # Start at the corner nearest the top-left of the bounding box, which keeps
    # the ordering stable for rotated as well as axis-aligned quads.
    reference = np.array([points[:, 0].min(), points[:, 1].min()], dtype=np.float32)
    start = int(np.argmin(np.linalg.norm(clockwise - reference, axis=1)))
    return np.roll(clockwise, -start, axis=0)


def _intersect(first, second):
    vx1, vy1, x1, y1 = first
    vx2, vy2, x2, y2 = second
    denominator = vx1 * vy2 - vy1 * vx2
    if abs(denominator) < 1e-9:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denominator
    return (x1 + t * vx1, y1 + t * vy1)


def _line_segments(gray: np.ndarray):
    """Straight edge segments in the plate, long enough to be a screen border."""
    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    low = int(max(10, 0.66 * median))
    high = int(min(255, max(low + 20, 1.33 * median)))
    edges = cv2.Canny(blurred, low, high)
    minimum = max(12, int(0.10 * min(width, height)))
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360.0,
        threshold=max(20, minimum // 2),
        minLineLength=minimum,
        maxLineGap=max(4, minimum // 3),
    )
    if segments is None:
        return np.zeros((0, 4), np.float32)
    return segments.reshape(-1, 4).astype(np.float32)


def snap_quad_to_lines(gray: np.ndarray, quad: np.ndarray, band_ratio: float = 0.16):
    """Move each edge of a roughly-right quad onto the real border beside it.

    Temporal cues place the screen but cannot place its *edge* precisely: they
    only know where content changed, which stops short of the slide's margins and
    is blocked wherever the lecturer stands.  The physical border - a bezel
    against a wall, or the lit rectangle of a projection - is by contrast one of
    the strongest straight edges in the frame.  Voting for it among the detected
    segments, per side, gives a border that is exact where the temporal cue was
    merely approximate, and each side is fitted independently so keystoning
    survives.
    """
    segments = _line_segments(gray)
    if len(segments) == 0:
        return quad, 0

    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    centre = quad.mean(axis=0)
    extent = min(
        float(np.linalg.norm(quad[1] - quad[0])),
        float(np.linalg.norm(quad[3] - quad[0])),
    )
    if extent < 10.0:
        return quad, 0
    band = max(6.0, extent * band_ratio)

    first = segments[:, 0:2]
    second = segments[:, 2:4]
    middles = (first + second) / 2.0
    directions = second - first
    lengths = np.linalg.norm(directions, axis=1)
    valid = lengths > 1e-6
    unit = np.zeros_like(directions)
    unit[valid] = directions[valid] / lengths[valid, None]

    lines = []
    snapped = 0
    for index in range(4):
        start = quad[index]
        end = quad[(index + 1) % 4]
        edge = end - start
        edge_length = float(np.linalg.norm(edge))
        if edge_length < 1e-6:
            return quad, snapped
        along = edge / edge_length
        normal = np.array([-along[1], along[0]], dtype=np.float32)
        if float(normal @ (centre - start)) < 0:
            normal = -normal  # point the normal inward

        parallel = np.abs(unit @ along) > math.cos(math.radians(14.0))
        offsets = (middles - start) @ normal
        projections = (middles - start) @ along
        usable = (
            parallel
            & valid
            & (np.abs(offsets) <= band)
            & (projections > -0.15 * edge_length)
            & (projections < 1.15 * edge_length)
        )
        if int(usable.sum()) == 0:
            lines.append(None)
            continue

        # Vote on the offset; a border wins because it is long and continuous.
        # Several offsets can win at once - a bezel shows both its outer and its
        # inner edge - so keep the strongest few and decide between them later,
        # once all four sides are known and the shape can be judged as a whole.
        chosen_offsets = offsets[usable]
        # Weight support by nearness to the seed edge.  The seed comes from where
        # slide content actually is, so it brackets the screen even when it does
        # not locate it; without this bias a longer, stronger line further out -
        # the edge of a desk, a monitor stand, a window frame - would outvote the
        # screen's own border simply for being more prominent.
        weights = lengths[usable] * np.exp(-np.abs(chosen_offsets) / (0.45 * band))
        bins = np.arange(-band, band + 1.0, 1.0)
        histogram, _ = np.histogram(chosen_offsets, bins=bins, weights=weights)
        histogram = cv2.GaussianBlur(histogram.reshape(-1, 1).astype(np.float32), (1, 5), 0).ravel()
        if histogram.max() <= 0:
            lines.append([])
            continue

        separation = max(3, int(round(extent * 0.02)))
        candidates = []
        remaining = histogram.copy()
        for _ in range(3):
            position = int(np.argmax(remaining))
            if remaining[position] <= 0.25 * histogram.max():
                break
            peak = float(bins[position]) + 0.5
            near = usable.copy()
            near[usable] = np.abs(chosen_offsets - peak) <= max(2.0, extent * 0.012)
            support = float(lengths[near].sum())
            if support >= 0.35 * edge_length:
                points = np.vstack([first[near], second[near]])
                vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
                candidates.append(((float(vx), float(vy), float(x0), float(y0)), support))
            remaining[max(0, position - separation) : position + separation + 1] = 0.0
        lines.append(candidates)

    if sum(1 for options in lines if options) < 2:
        return quad, 0

    # Fall back to the seed edge wherever no border was found.
    for index in range(4):
        if not lines[index]:
            start = quad[index]
            end = quad[(index + 1) % 4]
            direction = end - start
            lines[index] = [((float(direction[0]), float(direction[1]), float(start[0]), float(start[1])), 0.0)]

    snapped = sum(1 for options in lines if options and options[0][1] > 0)
    best_quad = quad
    best_score = -1.0
    total_support = sum(max(support for _, support in options) for options in lines) or 1.0

    for a in lines[0]:
        for b in lines[1]:
            for c in lines[2]:
                for d in lines[3]:
                    combination = (a, b, c, d)
                    corners = []
                    for index in range(4):
                        point = _intersect(combination[index - 1][0], combination[index][0])
                        if point is None:
                            corners = None
                            break
                        corners.append(point)
                    if corners is None:
                        continue
                    candidate = np.array(corners, dtype=np.float32)
                    if float(np.max(np.linalg.norm(candidate - quad, axis=1))) > band * 2.0:
                        continue
                    top = float(np.linalg.norm(candidate[1] - candidate[0]))
                    bottom = float(np.linalg.norm(candidate[2] - candidate[3]))
                    left = float(np.linalg.norm(candidate[3] - candidate[0]))
                    right = float(np.linalg.norm(candidate[2] - candidate[1]))
                    if min(top, bottom, left, right) < extent * 0.4:
                        continue
                    aspect = ((top + bottom) / 2.0) / max(1e-6, (left + right) / 2.0)
                    # A display's *active* area has a standard aspect ratio; the
                    # same border plus an asymmetric bezel chin does not.  That
                    # makes the ratio the tie-breaker between the candidates.
                    error = min(abs(aspect - target) / target for target in COMMON_ASPECTS)
                    support = sum(item[1] for item in combination) / total_support
                    score = support * (1.0 / (1.0 + 9.0 * error))
                    if score > best_score:
                        best_score = score
                        best_quad = candidate

    if best_score < 0:
        return quad, snapped
    return order_corners(best_quad), snapped


def _quad_from_mask(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None

    # A rotated bounding rectangle is the right seed: it is set by the blob's
    # extremes, so a bite taken out of the middle or a corner leaves it intact.
    # Precision comes afterwards from snapping to the screen's real border.
    quad = order_corners(np.asarray(cv2.boxPoints(cv2.minAreaRect(contour)), dtype=np.float32))

    # How much of the detected blob the quad actually explains.  A screen fills
    # its own quad; a ragged patch of wall texture does not.
    quad_area = float(abs(cv2.contourArea(quad.astype(np.float32))))
    fill = area / quad_area if quad_area > 0 else 0.0
    return quad, min(1.0, fill)


def _best_quad_over_thresholds(score_u8: np.ndarray, minimum_area: float):
    """Pick the threshold whose largest region is most convincingly a screen.

    A single threshold cannot serve every room.  Too low and the screen merges
    with whatever sits beside it - a desk, a podium, a lit wall; too high and the
    slide's darker areas fall away.  Sweeping instead and keeping the level whose
    region best *fills its own quadrilateral* is self-tuning: only a real screen
    stays rectangular as the level moves, so the objective peaks there.
    """
    height, width = score_u8.shape[:2]
    frame_area = float(width * height)
    close_size = max(5, (min(width, height) // 14) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    positive = score_u8[score_u8 > 4]
    if positive.size < 64:
        return None
    levels = np.unique(np.percentile(positive, np.linspace(20, 88, 16)).astype(np.int32))

    best = None
    best_objective = 0.0
    for level in levels:
        _, mask = cv2.threshold(score_u8, float(level), 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            continue
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        blob = (labels == largest).astype(np.uint8) * 255
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

        result = _quad_from_mask(blob)
        if result is None:
            continue
        quad, fill = result
        quad_area = float(abs(cv2.contourArea(quad.astype(np.float32))))
        if quad_area / frame_area < minimum_area or quad_area / frame_area > 0.995:
            continue

        # Does the quad also *contain* the region it was fitted to?  A quad that
        # explains only part of a sprawling blob scores badly here.
        painted = np.zeros_like(blob)
        cv2.fillConvexPoly(painted, quad.astype(np.int32), 255)
        blob_pixels = float(np.count_nonzero(blob))
        if blob_pixels < 1.0:
            continue
        coverage = float(np.count_nonzero(cv2.bitwise_and(blob, painted))) / blob_pixels

        objective = fill * coverage * (quad_area / frame_area) ** 0.30
        if objective > best_objective:
            best_objective = objective
            best = (quad, fill, float(level))
    return best


def _content_box(plate: np.ndarray):
    """Bounding box of non-uniform content, used to drop letterbox bars."""
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    row_spread = gray.max(axis=1).astype(np.int16) - gray.min(axis=1).astype(np.int16)
    column_spread = gray.max(axis=0).astype(np.int16) - gray.min(axis=0).astype(np.int16)
    rows = np.flatnonzero(row_spread > 12)
    columns = np.flatnonzero(column_spread > 12)
    if rows.size == 0 or columns.size == 0:
        return 0.0, 0.0, 1.0, 1.0
    return (
        float(columns[0]) / width,
        float(rows[0]) / height,
        float(columns[-1] + 1) / width,
        float(rows[-1] + 1) / height,
    )


@dataclass(frozen=True)
class ScreenQuad:
    """Normalized slide region within the source frame."""

    corners: tuple
    confidence: float
    full_frame: bool
    source: str = "auto"

    @staticmethod
    def whole_frame(source: str = "full-frame") -> "ScreenQuad":
        return ScreenQuad(
            corners=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            confidence=1.0,
            full_frame=True,
            source=source,
        )

    @staticmethod
    def from_box(x1: float, y1: float, x2: float, y2: float, source: str = "content-box") -> "ScreenQuad":
        return ScreenQuad(
            corners=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
            confidence=1.0,
            full_frame=True,
            source=source,
        )

    @staticmethod
    def from_corners(corners, source: str = "manual") -> "ScreenQuad":
        ordered = order_corners(np.array(corners, dtype=np.float32))
        return ScreenQuad(
            corners=tuple((float(x), float(y)) for x, y in ordered),
            confidence=1.0,
            full_frame=False,
            source=source,
        )

    def pixel_corners(self, width: int, height: int) -> np.ndarray:
        return np.float32([(x * width, y * height) for x, y in self.corners])

    @property
    def area_fraction(self) -> float:
        polygon = np.array(self.corners, dtype=np.float32)
        return float(abs(cv2.contourArea(polygon)))

    @property
    def is_axis_aligned(self) -> bool:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = self.corners
        return (
            abs(y0 - y1) < AXIS_ALIGNED_TOLERANCE
            and abs(y2 - y3) < AXIS_ALIGNED_TOLERANCE
            and abs(x0 - x3) < AXIS_ALIGNED_TOLERANCE
            and abs(x1 - x2) < AXIS_ALIGNED_TOLERANCE
        )

    def source_aspect(self, width: int, height: int, snap: bool = True) -> float:
        source = self.pixel_corners(width, height)
        top = float(np.linalg.norm(source[1] - source[0]))
        bottom = float(np.linalg.norm(source[2] - source[3]))
        left = float(np.linalg.norm(source[3] - source[0]))
        right = float(np.linalg.norm(source[2] - source[1]))
        aspect = ((top + bottom) / 2.0) / max(1.0, (left + right) / 2.0)
        if snap:
            for candidate in COMMON_ASPECTS:
                if abs(aspect - candidate) / candidate <= ASPECT_SNAP_TOLERANCE:
                    return candidate
        return aspect

    def native_width(self, width: int, height: int) -> int:
        source = self.pixel_corners(width, height)
        top = float(np.linalg.norm(source[1] - source[0]))
        bottom = float(np.linalg.norm(source[2] - source[3]))
        return max(16, int(round(max(top, bottom))))

    def output_size(self, width: int, height: int, target_width: int, max_width: int):
        chosen = target_width if target_width > 0 else self.native_width(width, height)
        chosen = max(16, min(int(chosen), int(max_width)))
        aspect = self.source_aspect(width, height)
        return chosen, max(16, int(round(chosen / aspect)))

    def rectify(self, frame: np.ndarray, target_width: int, max_width: int, sharpen: float = 0.0) -> np.ndarray:
        height, width = frame.shape[:2]
        out_width, out_height = self.output_size(width, height, target_width, max_width)
        source = self.pixel_corners(width, height)

        if self.is_axis_aligned:
            # A plain crop and resize avoids the resampling warpPerspective would
            # apply to an already rectangular region.
            left = int(round(max(0.0, float(source[:, 0].min()))))
            top = int(round(max(0.0, float(source[:, 1].min()))))
            right = int(round(min(float(width), float(source[:, 0].max()))))
            bottom = int(round(min(float(height), float(source[:, 1].max()))))
            region = frame[top : max(top + 1, bottom), left : max(left + 1, right)]
            if (region.shape[1], region.shape[0]) != (out_width, out_height):
                interpolation = cv2.INTER_AREA if region.shape[1] > out_width else cv2.INTER_LANCZOS4
                region = cv2.resize(region, (out_width, out_height), interpolation=interpolation)
            result = region
        else:
            destination = np.float32(
                [(0, 0), (out_width - 1, 0), (out_width - 1, out_height - 1), (0, out_height - 1)]
            )
            transform = cv2.getPerspectiveTransform(source, destination)
            result = cv2.warpPerspective(
                frame,
                transform,
                (out_width, out_height),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
        if sharpen > 0:
            blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
            result = cv2.addWeighted(result, 1.0 + sharpen, blurred, -sharpen, 0)
        return result

    def corner_distance(self, other: "ScreenQuad") -> float:
        first = np.array(self.corners, dtype=np.float32)
        second = np.array(other.corners, dtype=np.float32)
        return float(np.max(np.linalg.norm(first - second, axis=1)))

    def as_list(self):
        return [[round(float(x), 5), round(float(y), 5)] for x, y in self.corners]


ACTIVITY_SUPPORT_FLOOR = 0.02  # fraction of frame that must look "alive"


def _structure_map(plate: np.ndarray, window: int) -> np.ndarray:
    """Edge density and local contrast: where the plate carries detail at all."""
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blur = 5 if window >= 9 else 3
    edges = cv2.Canny(cv2.GaussianBlur(gray, (blur, blur), 0), 40, 120)
    density = cv2.boxFilter(edges.astype(np.float32) / 255.0, -1, (window, window))

    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, -1, (window, window))
    mean_square = cv2.boxFilter(gray_f * gray_f, -1, (window, window))
    contrast = np.sqrt(np.maximum(0.0, mean_square - mean * mean))
    return 0.5 * _normalize(density) + 0.5 * _normalize(contrast)


# ``motion`` is a frequency: the share of sampled pairs in which a pixel moved.
# Slides that change every few seconds still sit near the floor, while a person
# sits far above the knee.  Absolute levels beat normalization here - normalizing
# would rescale a person-free screencast until its slide changes look like one.
MOTION_FLOOR = 0.05
MOTION_KNEE = 0.18


def _moving_share(motion: np.ndarray, size, window: int) -> np.ndarray:
    width, height = size
    resized = cv2.resize(motion.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    if window > 1:
        resized = cv2.boxFilter(resized, -1, (window, window))
    return np.clip((resized - MOTION_FLOOR) / max(1e-6, MOTION_KNEE - MOTION_FLOOR), 0.0, 1.0)


def activity_map(
    motion: np.ndarray | None,
    variability: np.ndarray | None,
    size,
    motion_weight: float = 0.95,
):
    """"Content changes here, and it is not a person moving."

    This is the cue that actually separates a screen from the room.  A wall, a
    podium, a bezel and a logo are all perfectly still, so they score zero no
    matter how much visual detail they carry; the lecturer changes constantly and
    is removed by the motion term.  Only a display survives both tests.
    """
    if variability is None or float(variability.max()) <= 1e-6:
        return None
    width, height = size
    resized = cv2.resize(variability.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    activity = _normalize(resized)
    if motion is not None and float(motion.max()) > 1e-6:
        activity = activity * np.clip(1.0 - motion_weight * _moving_share(motion, size, 1), 0.0, 1.0)
    return np.clip(activity, 0.0, 1.0).astype(np.float32)


def _motion_penalty(motion: np.ndarray | None, size, weight: float, window: int):
    if motion is None or float(motion.max()) <= 1e-6:
        return None
    return np.clip(1.0 - weight * _moving_share(motion, size, window), 0.0, 1.0)


def screen_score_map(
    plate: np.ndarray,
    motion: np.ndarray | None = None,
    variability: np.ndarray | None = None,
    motion_weight: float = 0.85,
) -> np.ndarray:
    """Per-pixel "this is slide surface" score in [0, 1] at the plate's size.

    Activity leads because it is the only cue that rejects static furniture;
    structure fills in slide areas that happen never to change, and becomes the
    sole cue for the rare recording whose single slide never changes at all.
    """
    height, width = plate.shape[:2]
    window = max(9, (min(width, height) // 12) | 1)
    structure = _structure_map(plate, window)

    activity = activity_map(motion, variability, (width, height))
    if activity is not None and float((activity > 0.15).mean()) >= ACTIVITY_SUPPORT_FLOOR:
        # The *product* is what makes this discriminative.  Requiring both cues
        # rejects the two things a sum lets through: a spot the lecturer visits
        # occasionally (it changes, but the clean plate has no detail there) and
        # a podium, bezel or logo (plenty of detail, but it never changes).
        smoothed = _normalize(cv2.boxFilter(activity, -1, (window, window)))
        score = np.sqrt(np.clip(smoothed, 0.0, 1.0) * np.clip(structure, 0.0, 1.0))
    else:
        score = structure

    penalty = _motion_penalty(motion, (width, height), motion_weight, window)
    if penalty is not None:
        score = score * penalty
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _expand_quad(corners: np.ndarray, factor: float) -> np.ndarray:
    centre = corners.mean(axis=0)
    return centre + (corners - centre) * factor


def _uniform_border_extent(lines: np.ndarray, tolerance: float, limit: int) -> int:
    """How many leading rows/columns are a flat band matching the outermost one.

    Standard auto-crop logic, and deliberately conservative: it advances only
    while a line is both internally flat *and* the same shade as the border it
    started from.  A black bezel or a letterbox bar satisfies that; a slide's own
    background does not, because its shade differs from the bar it sits inside.
    """
    if len(lines) == 0:
        return 0
    reference = np.median(lines[0], axis=0)
    index = 0
    while index < limit and index < len(lines):
        line = lines[index]
        spread = float(np.percentile(np.abs(line - np.median(line, axis=0)), 92))
        difference = float(np.max(np.abs(np.median(line, axis=0) - reference)))
        if spread > tolerance or difference > tolerance * 1.5:
            break
        index += 1
    return index


def refine_quad_by_content(
    quad: "ScreenQuad",
    plate: np.ndarray,
    motion: np.ndarray | None = None,
    variability: np.ndarray | None = None,
    expand: float = 1.0,
    work_width: int = 640,
    tolerance: float = 11.0,
):
    """Trim flat bezel or letterbox bands from the edges of a located screen.

    The border snap lands on the display's physical outline, which on a monitor
    includes the bezel and on a video file can include letterbox bars.  Both are
    featureless bands, so removing them is a safe, purely local decision - unlike
    trimming by where content *changed*, which would also eat a slide's blank
    margins and everything the lecturer stands in front of.
    """
    height, width = plate.shape[:2]

    source = _expand_quad(quad.pixel_corners(width, height), expand)
    aspect = max(0.2, quad.source_aspect(width, height, snap=False))
    out_width = work_width
    out_height = max(16, int(round(out_width / aspect)))
    destination = np.float32(
        [(0, 0), (out_width - 1, 0), (out_width - 1, out_height - 1), (0, out_height - 1)]
    )
    transform = cv2.getPerspectiveTransform(source.astype(np.float32), destination)
    rectified = cv2.warpPerspective(
        plate,
        transform,
        (out_width, out_height),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)

    vertical_limit = int(out_height * 0.30)
    horizontal_limit = int(out_width * 0.30)
    top = _uniform_border_extent(rectified, tolerance, vertical_limit)
    bottom = out_height - _uniform_border_extent(rectified[::-1], tolerance, vertical_limit)
    columns = np.transpose(rectified, (1, 0, 2))
    left = _uniform_border_extent(columns, tolerance, horizontal_limit)
    right = out_width - _uniform_border_extent(columns[::-1], tolerance, horizontal_limit)

    if right - left < out_width * 0.4 or bottom - top < out_height * 0.4:
        return quad

    box = np.float32(
        [(left, top), (right - 1, top), (right - 1, bottom - 1), (left, bottom - 1)]
    ).reshape(-1, 1, 2)
    inverse = np.linalg.inv(transform)
    mapped = cv2.perspectiveTransform(box, inverse).reshape(4, 2)
    corners = tuple(
        (float(np.clip(x / width, 0.0, 1.0)), float(np.clip(y / height, 0.0, 1.0))) for x, y in mapped
    )
    refined = ScreenQuad(corners=corners, confidence=quad.confidence, full_frame=quad.full_frame)

    # A refinement that throws away most of the region, or produces an absurd
    # shape, means the content profile was misleading - keep the coarse quad.
    if refined.area_fraction < quad.area_fraction * 0.45:
        return quad
    refined_aspect = refined.source_aspect(width, height, snap=False)
    if not 0.55 <= refined_aspect <= 4.0:
        return quad
    return refined


def _full_frame_quad(plate: np.ndarray, motion, variability, minimum_area: float = 0.5):
    """Recognise a source that is already all slide, and only trim its bars.

    A screen recording has no room in it: the changing content runs edge to edge,
    or reaches two opposite edges with nothing but flat letterbox bars beside it.
    A camera pointed at a screen never does that - the screen is inset on all four
    sides.  Checking which edges the activity reaches separates the two cases
    cleanly, and matters because searching a screencast for a "screen" inside
    itself would happily crop away most of the slide.
    """
    height, width = plate.shape[:2]
    activity = activity_map(motion, variability, (width, height))
    if activity is None:
        return None
    live = activity > 0.15
    if float(live.mean()) < 0.25:
        return None

    rows = np.flatnonzero(live.any(axis=1))
    columns = np.flatnonzero(live.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None
    margin_y = max(1, int(round(height * 0.02)))
    margin_x = max(1, int(round(width * 0.02)))
    spans_vertically = rows[0] <= margin_y and rows[-1] >= height - 1 - margin_y
    spans_horizontally = columns[0] <= margin_x and columns[-1] >= width - 1 - margin_x
    if not (spans_vertically or spans_horizontally):
        return None

    box_area = ((rows[-1] - rows[0] + 1) * (columns[-1] - columns[0] + 1)) / float(width * height)
    if box_area < minimum_area:
        return None

    x1, y1, x2, y2 = _content_box(plate)
    if (x2 - x1) * (y2 - y1) >= FULL_FRAME_AREA:
        return ScreenQuad.whole_frame()
    return ScreenQuad.from_box(x1, y1, x2, y2, source="letterbox-trimmed")


def detect_screen_quad(
    plate: np.ndarray,
    motion: np.ndarray | None = None,
    variability: np.ndarray | None = None,
    minimum_area: float = 0.05,
    minimum_fill: float = 0.62,
    refine: bool = True,
):
    """Locate the slide region in a clean plate, or None when nothing looks like one."""
    source_height, source_width = plate.shape[:2]
    work_width = 480
    work_height = max(16, int(round(work_width * source_height / source_width)))
    small = cv2.resize(plate, (work_width, work_height), interpolation=cv2.INTER_AREA)

    full = _full_frame_quad(plate, motion, variability)
    if full is not None:
        return full

    score = screen_score_map(small, motion, variability)
    score_u8 = np.clip(score * 255.0, 0, 255).astype(np.uint8)

    best = _best_quad_over_thresholds(score_u8, minimum_area)
    if best is None:
        return None
    quad_pixels, fill, _ = best

    # Snap to the screen's physical border, which the temporal cues only bracket.
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    quad_pixels, snapped = snap_quad_to_lines(gray, quad_pixels)
    if snapped >= 2:
        fill = max(fill, 0.75)

    normalized = tuple(
        (float(np.clip(x / work_width, -0.02, 1.02)), float(np.clip(y / work_height, -0.02, 1.02)))
        for x, y in quad_pixels
    )
    candidate = ScreenQuad(corners=normalized, confidence=fill, full_frame=False)

    area = candidate.area_fraction
    if area < minimum_area or fill < minimum_fill:
        return None

    if area >= FULL_FRAME_AREA:
        x1, y1, x2, y2 = _content_box(plate)
        if (x2 - x1) * (y2 - y1) >= FULL_FRAME_AREA:
            return ScreenQuad.whole_frame()
        return ScreenQuad.from_box(x1, y1, x2, y2)

    aspect = candidate.source_aspect(source_width, source_height, snap=False)
    if not 0.55 <= aspect <= 4.0:
        return None
    clipped = tuple((float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))) for x, y in candidate.corners)
    coarse = ScreenQuad(corners=clipped, confidence=fill, full_frame=False)
    if not refine:
        return coarse

    refined = refine_quad_by_content(coarse, plate, motion, variability)
    if refined.area_fraction >= FULL_FRAME_AREA:
        return ScreenQuad.whole_frame()
    return refined


@dataclass
class LayoutSegment:
    """A time range that shares one slide-region geometry."""

    start: float
    end: float
    quad: ScreenQuad
    probe_count: int = 1

    def contains(self, timestamp: float) -> bool:
        return self.start <= timestamp < self.end


def segment_layouts(
    probes,
    duration: float,
    tolerance: float = 0.035,
    minimum_seconds: float = 20.0,
):
    """Group timed detections into contiguous runs that share one geometry.

    A single fixed framing collapses to one segment.  A camera cut or zoom that
    persists produces a second segment; a one-off misdetection does not, because
    runs shorter than ``minimum_seconds`` are absorbed by their neighbour.
    """
    valid = [(timestamp, quad) for timestamp, quad in probes if quad is not None]
    if not valid:
        return []

    runs = []
    for timestamp, quad in valid:
        if runs and runs[-1][-1][1].corner_distance(quad) <= tolerance:
            runs[-1].append((timestamp, quad))
        else:
            runs.append([(timestamp, quad)])

    merged = []
    for run in runs:
        span = run[-1][0] - run[0][0]
        if merged and (span < minimum_seconds or len(run) < 2):
            merged[-1].extend(run)
        else:
            merged.append(run)
    if len(merged) > 1 and len(merged[0]) < 2:
        head = merged.pop(0)
        merged[0] = head + merged[0]

    segments = []
    for index, run in enumerate(merged):
        quads = [quad for _, quad in run]
        corners = np.array([quad.corners for quad in quads], dtype=np.float32)
        median = np.median(corners, axis=0)
        representative = ScreenQuad(
            corners=tuple((float(x), float(y)) for x, y in median),
            confidence=float(np.median([quad.confidence for quad in quads])),
            full_frame=bool(np.mean([1.0 if quad.full_frame else 0.0 for quad in quads]) >= 0.5),
            source="auto",
        )
        start = 0.0 if index == 0 else (merged[index - 1][-1][0] + run[0][0]) / 2.0
        end = duration if index == len(merged) - 1 else (run[-1][0] + merged[index + 1][0][0]) / 2.0
        segments.append(LayoutSegment(start=start, end=end, quad=representative, probe_count=len(run)))
    return segments


def quad_for(segments, timestamp: float) -> ScreenQuad:
    for segment in segments:
        if segment.contains(timestamp):
            return segment.quad
    if segments:
        return segments[-1].quad if timestamp >= segments[-1].start else segments[0].quad
    return ScreenQuad.whole_frame()
