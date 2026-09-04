"""Writing the results: PDF, review contact sheet, and the analysis report."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from .util import format_time


def create_pdf(pages, destination: Path, title: str) -> None:
    document = pymupdf.open()
    contents: list = []
    try:
        for number, page in enumerate(pages, start=1):
            if page.image_path is None:
                raise RuntimeError("A selected page was never rendered")
            with Image.open(page.image_path) as image:
                width, height = image.size
            # Half a pixel per point puts the page at 144 dpi, which keeps every
            # source pixel while giving the document a sensible physical size.
            sheet = document.new_page(width=width * 0.5, height=height * 0.5)
            sheet.insert_image(sheet.rect, filename=str(page.image_path), keep_proportion=False)
            contents.append([1, page.label, number])
        if contents:
            document.set_toc(contents)
        document.set_metadata(
            {
                "title": title,
                "author": "Lecture Video to PDF",
                "subject": "Slides and annotations extracted from lecture video",
                "creator": "lecture_video_to_pdf.py",
                "creationDate": datetime.now().astimezone().strftime("D:%Y%m%d%H%M%S%z"),
            }
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        document.save(str(destination), garbage=4, deflate=True)
    finally:
        document.close()


def create_review_sheet(pages, destination: Path, columns: int = 3) -> None:
    """One image showing every page, for checking a run at a glance."""
    usable = [page for page in pages if page.image_path is not None]
    if not usable:
        return
    thumb_width, thumb_height = 480, 270
    rows = math.ceil(len(usable) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), "white")
    font = ImageFont.load_default()
    for index, page in enumerate(usable):
        with Image.open(page.image_path) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_width, thumb_height), "white")
        tile.paste(thumb, ((thumb_width - thumb.width) // 2, (thumb_height - thumb.height) // 2))
        draw = ImageDraw.Draw(tile)
        label = f"{index + 1:02d}  {page.label}"
        draw.rectangle((0, 0, min(thumb_width, 8 * len(label) + 14), 20), fill=(0, 0, 0))
        draw.text((6, 5), label, fill=(255, 255, 255), font=font)
        sheet.paste(tile, ((index % columns) * thumb_width, (index // columns) * thumb_height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)


def page_record(page) -> dict:
    return {
        "video": page.video.name,
        "kind": page.kind,
        "scene": page.scene_index + 1,
        "epoch": page.epoch_index + 1 if page.epoch_index >= 0 else None,
        "epochs_in_scene": page.epoch_count if page.epoch_index >= 0 else None,
        "timestamp": round(page.timestamp, 3),
        "timestamp_text": format_time(page.timestamp),
        "scene_start": round(page.scene_start, 3),
        "scene_end": round(page.scene_end, 3),
        "composite_window": [round(page.window[0], 3), round(page.window[1], 3)],
        "ink_ratio": round(page.ink_ratio, 6),
        "sharpness": round(page.sharpness, 2),
        "image": str(page.image_path) if page.image_path else None,
    }


def video_record(result) -> dict:
    return {
        "video": result.video.name,
        "duration_seconds": round(result.info.duration, 3),
        "width": result.info.width,
        "height": result.info.height,
        "analysis_seconds": round(result.seconds, 2),
        "layout_segments": [
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "source": segment.quad.source,
                "full_frame": segment.quad.full_frame,
                "confidence": round(segment.quad.confidence, 3),
                "corners": segment.quad.as_list(),
            }
            for segment in result.segments
        ],
        "scenes": [
            {
                "index": scene.index + 1,
                "start": round(scene.start, 3),
                "end": round(scene.end, 3),
                "start_text": format_time(scene.start),
            }
            for scene in result.scenes
        ],
        "annotation_epochs": [
            {
                "scene": trace.scene_index + 1,
                "cleanest_time": round(trace.cleanest_time, 3),
                "cleanest_ink": round(trace.cleanest_ratio, 6),
                "epochs": [
                    {
                        "index": epoch.index + 1,
                        "start": round(epoch.start, 3),
                        "end": round(epoch.end, 3),
                        "peak_time": round(epoch.peak_time, 3),
                        "peak_text": format_time(epoch.peak_time),
                        "peak_ink": round(epoch.peak_ratio, 6),
                    }
                    for epoch in trace.epochs
                ],
            }
            for trace in result.traces
        ],
        "candidate_changes": [round(value, 3) for value in result.changes],
    }


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
