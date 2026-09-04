"""Orchestration: video in, selected pages out.

Four passes, each streaming, so memory does not grow with the length of the
recording:

1. geometry - locate the slide region and notice if the framing ever changes,
2. scenes   - find slide transitions in rectified slide space,
3. ink      - measure annotation per scene and cut write-and-wipe epochs,
4. render   - re-read only the chosen moments, at full resolution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import analysis, geometry, ink
from .geometry import LayoutSegment, ScreenQuad, quad_for, segment_layouts
from .scan import scan_geometry
from .util import format_time, timestamp_slug
from .video import FrameReader, VideoInfo, probe_video

INK_MODES = ("clean", "final", "epochs", "all")


@dataclass
class Page:
    """One page of the finished PDF, and where it came from."""

    video: Path
    video_index: int
    scene_index: int
    kind: str
    timestamp: float
    window: tuple
    scene_start: float
    scene_end: float
    ink_ratio: float
    order: int
    epoch_index: int = -1
    epoch_count: int = 1
    sharpness: float = 0.0
    fingerprint: object = None
    image_path: object = None
    quad: object = None

    @property
    def label(self) -> str:
        name = f"{self.video.stem} {format_time(self.timestamp)}"
        if self.kind == "epoch" and self.epoch_count > 1:
            return f"{name} (note {self.epoch_index + 1}/{self.epoch_count})"
        if self.kind == "epoch":
            return f"{name} (notes)"
        return name

    def filename(self, extension: str) -> str:
        parts = [f"scene_{self.scene_index + 1:03d}"]
        if self.kind == "epoch":
            parts.append(f"note_{self.epoch_index + 1:02d}")
        elif self.kind == "clean":
            parts.append("clean")
        parts.append(timestamp_slug(self.timestamp))
        return "_".join(parts) + extension


@dataclass
class VideoResult:
    video: Path
    info: VideoInfo
    segments: list = field(default_factory=list)
    scenes: list = field(default_factory=list)
    changes: list = field(default_factory=list)
    traces: list = field(default_factory=list)
    pages: list = field(default_factory=list)
    seconds: float = 0.0


def resolve_geometry(info: VideoInfo, options, log=print):
    """Decide the slide region: manual, disabled, or detected automatically."""
    if options.no_crop:
        return [LayoutSegment(0.0, info.duration, ScreenQuad.whole_frame("--no-crop"), 1)], None
    if options.screen_corners is not None:
        quad = ScreenQuad.from_corners(options.screen_corners, source="--screen-corners")
        return [LayoutSegment(0.0, info.duration, quad, 1)], None

    scan = scan_geometry(
        info,
        interval=options.geometry_interval,
        work_width=480,
        reservoir=options.geometry_plates,
    )
    variability = scan.slide_variability
    plate = scan.plate()
    if plate is None:
        return [LayoutSegment(0.0, info.duration, ScreenQuad.whole_frame("fallback"), 1)], scan

    overall = geometry.detect_screen_quad(plate, scan.motion, variability)
    if overall is None:
        log("    no slide region found; keeping the whole frame")
        return [LayoutSegment(0.0, info.duration, ScreenQuad.whole_frame("fallback"), 1)], scan

    # Probe windows separately so a mid-recording cut or zoom is noticed.  A
    # single stable framing collapses back to one segment.
    windows = max(1, options.geometry_windows)
    step = info.duration / windows
    probes = []
    for index in range(windows):
        window_plate = scan.plate(index * step, (index + 1) * step)
        found = (
            geometry.detect_screen_quad(window_plate, scan.motion, variability)
            if window_plate is not None
            else None
        )
        probes.append(((index + 0.5) * step, found))

    agree = [
        quad for _, quad in probes if quad is not None and quad.corner_distance(overall) <= options.layout_tolerance
    ]
    if len(agree) >= max(1, int(0.7 * sum(1 for _, quad in probes if quad is not None))):
        return [LayoutSegment(0.0, info.duration, overall, len(agree))], scan

    segments = segment_layouts(probes, info.duration, tolerance=options.layout_tolerance)
    if not segments:
        return [LayoutSegment(0.0, info.duration, overall, 1)], scan
    if len(segments) > 1:
        log(f"    framing changes {len(segments)} time(s); using per-segment geometry")
    return segments, scan


def _fingerprint(gray: np.ndarray) -> np.ndarray:
    return analysis.perceptual_hash(gray)


def _sample_near(samples, timestamp: float):
    if not samples:
        return None
    return min(samples, key=lambda sample: abs(sample.timestamp - timestamp))


def analyse_video(video: Path, video_index: int, options, log=print) -> VideoResult:
    started = time.monotonic()
    info = probe_video(video)
    log(f"  {video.name}: {info.width}x{info.height}, {format_time(info.duration)}")

    segments, _scan = resolve_geometry(info, options, log)
    quad = segments[0].quad
    log(
        f"    slide region: {quad.source} "
        f"{'full frame' if quad.full_frame else f'{quad.area_fraction * 100:.0f}% of frame'}, "
        f"aspect {quad.source_aspect(info.width, info.height):.3f}"
    )

    samples = list(analysis.rectified_samples(info, segments, options.interval))
    scenes, changes = analysis.detect_scenes(
        samples,
        info.duration,
        sensitivity=options.sensitivity,
        minimum_seconds=options.minimum_scene_seconds,
        confirm_seconds=options.confirm_seconds,
        interval=options.interval,
    )
    scenes = [scene for scene in scenes if scene.end > options.skip_opening_seconds]
    for scene in scenes:
        scene.start = max(scene.start, options.skip_opening_seconds)
    log(f"    {len(scenes)} scene(s) from {len(changes)} candidate change(s)")

    traces = _trace_ink(info, segments, scenes, options) if options.ink_mode != "clean" else []
    pages = _select_pages(video, video_index, samples, scenes, traces, segments, options)

    result = VideoResult(
        video=video,
        info=info,
        segments=segments,
        scenes=scenes,
        changes=changes,
        traces=traces,
        pages=pages,
        seconds=time.monotonic() - started,
    )
    return result


def _trace_ink(info: VideoInfo, segments, scenes, options):
    """Third pass: stream annotation measurements scene by scene."""
    if not scenes:
        return []
    traces = []
    tracker = None
    current = 0
    boundaries = list(scenes)

    for timestamp, view in analysis.rectified_views(
        info, segments, options.ink_interval, options.ink_width
    ):
        while current < len(boundaries) and timestamp >= boundaries[current].end:
            if tracker is not None:
                traces.append(tracker.finish(boundaries[current].end))
                tracker = None
            current += 1
        if current >= len(boundaries):
            break
        scene = boundaries[current]
        if timestamp < scene.start:
            continue
        if tracker is None:
            tracker = ink.InkTracker(
                scene_index=scene.index,
                erase_ratio=options.erase_ratio,
                persist_samples=max(1, int(round(options.erase_persist_seconds / options.ink_interval))),
                minimum_ink=options.minimum_ink,
                threshold=options.ink_threshold,
                settle_samples=max(2, int(round(options.ink_settle_seconds / options.ink_interval))),
                erase_drop=options.erase_drop,
            )
        tracker.add(timestamp, view)

    if tracker is not None:
        traces.append(tracker.finish(boundaries[min(current, len(boundaries) - 1)].end))
    return traces


def _select_pages(video: Path, video_index: int, samples, scenes, traces, segments, options):
    """Turn scenes and epochs into the page list the chosen mode calls for."""
    by_scene = {trace.scene_index: trace for trace in traces}
    pages: list[Page] = []
    order = 0

    for scene in scenes:
        margin = min(2.0, max(0.3, scene.duration * 0.12))
        in_scene = analysis.samples_in(samples, scene, margin)
        if not in_scene:
            continue
        trace = by_scene.get(scene.index)

        def make(kind, timestamp, window, ink_ratio, epoch_index=-1, epoch_count=1):
            nonlocal order
            reference = _sample_near(in_scene, timestamp)
            page = Page(
                video=video,
                video_index=video_index,
                scene_index=scene.index,
                kind=kind,
                timestamp=timestamp,
                window=window,
                scene_start=scene.start,
                scene_end=scene.end,
                ink_ratio=ink_ratio,
                order=order,
                epoch_index=epoch_index,
                epoch_count=epoch_count,
                sharpness=reference.sharpness if reference else 0.0,
                fingerprint=_fingerprint(reference.gray) if reference else None,
                quad=quad_for(segments, timestamp),
            )
            order += 1
            pages.append(page)

        clean_time = trace.cleanest_time if trace and trace.times else in_scene[0].timestamp
        clean_ratio = trace.cleanest_ratio if trace else 0.0
        clean_window = (max(scene.start, clean_time - 4.0), min(scene.end, clean_time + 4.0))

        if options.ink_mode in ("clean", "all"):
            make("clean", clean_time, clean_window, clean_ratio)

        if options.ink_mode == "final":
            if trace and trace.epochs:
                last = trace.epochs[-1]
                make("epoch", last.peak_time, last.window, last.peak_ratio)
            else:
                make("clean", clean_time, clean_window, clean_ratio)
        elif options.ink_mode in ("epochs", "all"):
            epochs = trace.epochs if trace else []
            for epoch in epochs:
                make("epoch", epoch.peak_time, epoch.window, epoch.peak_ratio, epoch.index, len(epochs))
            if not epochs and options.ink_mode == "epochs":
                make("clean", clean_time, clean_window, clean_ratio)
    return pages


def deduplicate(pages, distance: int = 10, ink_tolerance: float = 0.004):
    """Drop pages that repeat one already kept.

    Two pages match only when the picture *and* the amount of annotation agree:
    a slide revisited after more was written on it is a different page, and in
    annotation modes that difference is the whole point.
    """
    kept: list[Page] = []
    for page in sorted(pages, key=lambda item: item.order):
        duplicate = False
        for existing in kept:
            # Successive annotation cycles on one slide are, by construction, the
            # different boards the lecturer wrote - which is the entire reason for
            # capturing them.  They are near-identical to any whole-page
            # comparison, so they are exempted rather than compared.
            same_scene = (
                page.video == existing.video
                and page.scene_index == existing.scene_index
                and page.kind == "epoch"
                and existing.kind == "epoch"
                and page.epoch_index != existing.epoch_index
            )
            if same_scene:
                continue
            if page.fingerprint is None or existing.fingerprint is None:
                continue
            if analysis.hamming(page.fingerprint, existing.fingerprint) > distance:
                continue
            if abs(page.ink_ratio - existing.ink_ratio) > ink_tolerance:
                continue
            duplicate = True
            break
        if not duplicate:
            kept.append(page)
    return kept


def render_pages(pages, frames_root: Path, options, log=print):
    """Fourth pass: re-read the chosen moments and write page images."""
    extension = ".jpg" if options.jpeg else ".png"
    by_video: dict[Path, list[Page]] = {}
    for page in pages:
        by_video.setdefault(page.video, []).append(page)

    for video, group in by_video.items():
        info = probe_video(video)
        target = frames_root / video.stem
        target.mkdir(parents=True, exist_ok=True)
        log(f"  rendering {len(group)} page image(s) from {video.name}")
        with FrameReader(info) as reader:
            for page in sorted(group, key=lambda item: item.timestamp):
                quad = page.quad or ScreenQuad.whole_frame()
                frame = _page_frame(reader, page, options)
                image = quad.rectify(frame, options.output_width, options.max_output_width, options.sharpen)
                destination = target / page.filename(extension)
                if options.jpeg:
                    ok = cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, options.jpeg_quality])
                else:
                    ok = cv2.imwrite(str(destination), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if not ok:
                    raise RuntimeError(f"Cannot write image: {destination}")
                page.image_path = destination


def _page_frame(reader: FrameReader, page: Page, options):
    """The frame to render: a single instant, or a lecturer-free composite."""
    if not options.clean_plate:
        frame = reader.frame_at(page.timestamp)
        if frame is not None:
            return frame
    low, high = page.window
    span = max(1.0, high - low)
    centre = (low + high) / 2.0
    plate = reader.plate_at(centre, span=span, count=options.plate_frames)
    if plate is not None:
        return plate
    frame = reader.frame_at(page.timestamp)
    if frame is None:
        raise RuntimeError(f"Cannot read {page.video.name} at {format_time(page.timestamp)}")
    return frame
