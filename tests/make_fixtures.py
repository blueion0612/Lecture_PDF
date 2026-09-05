"""Generate synthetic lecture videos covering the shapes real recordings take.

Each fixture isolates one thing the extractor is supposed to survive: a slide
that fills the frame, a slide seen at an angle across a room with someone
standing in front of it, a camera that changes framing part-way through,
letterbox bars, and a lecturer who fills a board, wipes it, and fills it again.
Because the ground truth is constructed rather than guessed, the tests can
assert exact page counts and crop corners.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

FPS = 10
PALETTE = [
    ((28, 24, 22), (240, 240, 235), (60, 170, 240)),
    ((250, 248, 244), (35, 35, 40), (30, 90, 200)),
    ((45, 30, 60), (245, 240, 250), (120, 220, 160)),
    ((18, 42, 30), (235, 245, 238), (80, 200, 250)),
    ((240, 235, 225), (40, 45, 55), (200, 90, 40)),
]


def make_slide(width: int, height: int, index: int, lines: int = 6) -> np.ndarray:
    """A slide with a title band and body text, distinct from its neighbors."""
    background, foreground, accent = PALETTE[index % len(PALETTE)]
    slide = np.full((height, width, 3), background, np.uint8)
    rng = np.random.default_rng(1000 + index)

    cv2.putText(
        slide,
        f"Chapter {index + 1}: Topic {chr(65 + index % 26)}",
        (int(width * 0.06), int(height * 0.16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        width / 900.0,
        foreground,
        max(1, width // 500),
        cv2.LINE_AA,
    )
    cv2.line(
        slide,
        (int(width * 0.06), int(height * 0.22)),
        (int(width * 0.94), int(height * 0.22)),
        accent,
        max(1, width // 400),
    )
    for row in range(lines):
        y = int(height * (0.34 + row * 0.095))
        if y > height * 0.92:
            break
        length = float(rng.uniform(0.35, 0.86))
        cv2.putText(
            slide,
            "-",
            (int(width * 0.08), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            width / 1400.0,
            accent,
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            slide,
            (int(width * 0.12), y - int(height * 0.032)),
            (int(width * (0.12 + length)), y),
            foreground,
            -1,
        )
    return slide


def make_room(width: int, height: int) -> np.ndarray:
    """A textured wall with a podium, so the detector has real distractors."""
    room = np.zeros((height, width, 3), np.uint8)
    top = np.array([196, 205, 188], np.float32)
    bottom = np.array([150, 158, 146], np.float32)
    for y in range(height):
        room[y, :] = top + (bottom - top) * (y / max(1, height - 1))
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 3.0, (height, width, 3))
    room = np.clip(room.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # A podium below the screen and a logo in a corner: both are detailed and
    # completely static, which is exactly the trap the detector has to avoid.
    cv2.rectangle(room, (int(width * 0.30), int(height * 0.86)), (int(width * 0.70), height), (70, 72, 78), -1)
    cv2.rectangle(room, (int(width * 0.34), int(height * 0.88)), (int(width * 0.66), int(height * 0.93)), (110, 112, 118), 2)
    cv2.putText(room, "UNIVERSITY", (int(width * 0.74), int(height * 0.07)),
                cv2.FONT_HERSHEY_SIMPLEX, width / 1600.0, (90, 95, 110), 1, cv2.LINE_AA)
    return room


def paste_screen(room: np.ndarray, slide: np.ndarray, corners: np.ndarray, bezel: int = 6) -> np.ndarray:
    """Composite a slide onto the room at the given quad, with a dark bezel."""
    height, width = room.shape[:2]
    frame = room.copy()
    outer = corners.astype(np.float32)
    centre = outer.mean(axis=0)
    grown = centre + (outer - centre) * (1.0 + bezel / 100.0)
    cv2.fillConvexPoly(frame, grown.astype(np.int32), (18, 18, 20))

    source = np.float32(
        [(0, 0), (slide.shape[1] - 1, 0), (slide.shape[1] - 1, slide.shape[0] - 1), (0, slide.shape[0] - 1)]
    )
    transform = cv2.getPerspectiveTransform(source, outer)
    warped = cv2.warpPerspective(slide, transform, (width, height))
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, outer.astype(np.int32), 255)
    frame[mask > 0] = warped[mask > 0]
    return frame


def draw_person(frame: np.ndarray, centre_x: float, scale: float = 1.0) -> None:
    """A simple opaque figure, to occlude the screen the way a lecturer does."""
    height, width = frame.shape[:2]
    x = int(centre_x * width)
    body_top = int(height * (1.0 - 0.52 * scale))
    cv2.ellipse(frame, (x, body_top), (int(width * 0.030 * scale), int(height * 0.062 * scale)),
                0, 0, 360, (96, 108, 130), -1)
    cv2.rectangle(frame, (x - int(width * 0.045 * scale), body_top + int(height * 0.055 * scale)),
                  (x + int(width * 0.045 * scale), height), (120, 126, 140), -1)


def annotate(slide: np.ndarray, strokes, upto: int) -> np.ndarray:
    """Draw the first ``upto`` handwriting strokes onto a copy of the slide."""
    marked = slide.copy()
    for stroke in strokes[:upto]:
        points, colour, thickness = stroke
        cv2.polylines(marked, [points], False, colour, thickness, cv2.LINE_AA)
    return marked


def make_strokes(width: int, height: int, count: int, seed: int):
    rng = np.random.default_rng(seed)
    colours = [(60, 240, 255), (70, 90, 250), (110, 240, 130)]
    strokes = []
    for index in range(count):
        start = np.array([rng.uniform(0.12, 0.8) * width, rng.uniform(0.3, 0.85) * height])
        points = [start]
        for _ in range(5):
            points.append(points[-1] + rng.normal(0, [width * 0.05, height * 0.05]))
        strokes.append(
            (
                np.array(points, np.int32).reshape(-1, 1, 2),
                colours[index % len(colours)],
                max(2, width // 260),
            )
        )
    return strokes


class Writer:
    def __init__(self, path: Path, width: int, height: int):
        self.path = path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, FPS, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open writer for {path}")

    def hold(self, frame: np.ndarray, seconds: float) -> None:
        for _ in range(max(1, int(round(seconds * FPS)))):
            self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


def build_screencast(path: Path, slides: int = 4, seconds_each: float = 6.0, size=(640, 360)):
    """The slide fills the frame; nothing else is present."""
    width, height = size
    writer = Writer(path, width, height)
    for index in range(slides):
        writer.hold(make_slide(width, height, index), seconds_each)
    writer.close()
    return {"kind": "screencast", "slides": slides, "corners": [[0, 0], [1, 0], [1, 1], [0, 1]]}


def build_letterboxed(path: Path, slides: int = 3, seconds_each: float = 6.0, size=(640, 360), bar=0.14):
    """A 16:9 frame carrying a 4:3 slide between black bars."""
    width, height = size
    writer = Writer(path, width, height)
    inner = int(width * (1 - 2 * bar))
    left = (width - inner) // 2
    for index in range(slides):
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:, left : left + inner] = make_slide(inner, height, index)
        writer.hold(frame, seconds_each)
    writer.close()
    return {
        "kind": "letterboxed",
        "slides": slides,
        "corners": [
            [left / width, 0.0],
            [(left + inner) / width, 0.0],
            [(left + inner) / width, 1.0],
            [left / width, 1.0],
        ],
    }


ROOM_CORNERS = np.float32([(0.10, 0.12), (0.72, 0.09), (0.74, 0.72), (0.11, 0.75)])


def build_room(path: Path, slides: int = 4, seconds_each: float = 8.0, size=(720, 405), person=True):
    """A keystoned screen across a room, with someone walking in front of it."""
    width, height = size
    room = make_room(width, height)
    corners = ROOM_CORNERS * np.float32([width, height])
    writer = Writer(path, width, height)
    slide_size = (480, 270)
    step = 0
    for index in range(slides):
        slide = make_slide(slide_size[0], slide_size[1], index)
        base = paste_screen(room, slide, corners)
        for _ in range(int(round(seconds_each * FPS))):
            frame = base.copy()
            if person:
                # A slow sweep: present in every frame, never in one place long.
                draw_person(frame, 0.30 + 0.35 * (0.5 + 0.5 * np.sin(step / 26.0)))
            writer.write_frame = None
            writer.writer.write(frame)
            step += 1
    writer.close()
    return {"kind": "room", "slides": slides, "corners": ROOM_CORNERS.tolist()}


def build_reframe(path: Path, slides: int = 4, seconds_each: float = 8.0, size=(720, 405)):
    """The camera cuts to a tighter framing half-way through."""
    width, height = size
    room = make_room(width, height)
    wide = ROOM_CORNERS * np.float32([width, height])
    tight = (ROOM_CORNERS * np.float32([1.18, 1.12]) + np.float32([-0.02, -0.02])) * np.float32([width, height])
    writer = Writer(path, width, height)
    step = 0
    for index in range(slides):
        corners = wide if index < slides // 2 else tight
        slide = make_slide(480, 270, index)
        base = paste_screen(room, slide, corners)
        for _ in range(int(round(seconds_each * FPS))):
            frame = base.copy()
            draw_person(frame, 0.32 + 0.30 * (0.5 + 0.5 * np.sin(step / 22.0)))
            writer.writer.write(frame)
            step += 1
    writer.close()
    return {"kind": "reframe", "slides": slides, "corners": ROOM_CORNERS.tolist()}


def build_write_erase(
    path: Path,
    size=(720, 405),
    cycles: int = 2,
    strokes_each: int = 7,
    seconds_per_stroke: float = 2.2,
    blank_seconds: float = 4.0,
):
    """One slide, annotated to completion, wiped clean, and annotated again."""
    width, height = size
    room = make_room(width, height)
    corners = ROOM_CORNERS * np.float32([width, height])
    slide = make_slide(480, 270, 1)
    writer = Writer(path, width, height)
    step = 0

    def emit(image, seconds):
        nonlocal step
        base = paste_screen(room, image, corners)
        for _ in range(int(round(seconds * FPS))):
            frame = base.copy()
            draw_person(frame, 0.30 + 0.34 * (0.5 + 0.5 * np.sin(step / 24.0)))
            writer.writer.write(frame)
            step += 1

    emit(slide, blank_seconds)
    for cycle in range(cycles):
        strokes = make_strokes(480, 270, strokes_each, seed=40 + cycle)
        for count in range(1, strokes_each + 1):
            emit(annotate(slide, strokes, count), seconds_per_stroke)
        emit(annotate(slide, strokes, strokes_each), 6.0)  # dwell on the finished board
        if cycle < cycles - 1:
            emit(slide, blank_seconds + 10.0)  # wiped clean, then a pause
    writer.close()
    return {"kind": "write_erase", "slides": 1, "cycles": cycles, "corners": ROOM_CORNERS.tolist()}


def build_all(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "screencast.mp4": build_screencast(root / "screencast.mp4"),
        "letterboxed.mp4": build_letterboxed(root / "letterboxed.mp4"),
        "room.mp4": build_room(root / "room.mp4"),
        "reframe.mp4": build_reframe(root / "reframe.mp4"),
        "write_erase.mp4": build_write_erase(root / "write_erase.mp4"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "fixtures"
    for name, spec in build_all(target).items():
        print(f"{name}: {spec['kind']}, {spec['slides']} slide(s)")
    print(f"written to {target}")
