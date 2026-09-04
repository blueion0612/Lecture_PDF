"""Small shared helpers."""

from __future__ import annotations

import re
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".wmv", ".flv", ".mpg", ".mpeg", ".ts"}


def natural_key(path: Path) -> list[object]:
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", path.name)]


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def timestamp_slug(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def discover_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in VIDEO_EXTENSIONS else []
    if not input_path.is_dir():
        return []
    videos = [
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos, key=natural_key)


