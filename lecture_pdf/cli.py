"""Command line entry point."""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from . import output, pipeline
from .util import discover_videos


def parse_screen_corners(value: str):
    """Parse normalized corners given as x,y;x,y;x,y;x,y."""
    try:
        points = []
        for pair in value.split(";"):
            x_text, y_text = pair.split(",", maxsplit=1)
            point = (float(x_text.strip()), float(y_text.strip()))
            if not (0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0):
                raise ValueError
            points.append(point)
        if len(points) != 4:
            raise ValueError
        return tuple(points)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use four normalized points: x,y;x,y;x,y;x,y (top-left, top-right, bottom-right, bottom-left)"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    default_input = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(
        prog="lecture_video_to_pdf",
        description=(
            "Turn lecture video into PDF handouts.  The slide region is located "
            "automatically, so no per-recording calibration is needed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="?", type=Path, default=default_input, help="Video file or folder")
    parser.add_argument("--output", type=Path, help="Output folder (default: INPUT/pdf_output)")

    group = parser.add_argument_group("what to keep")
    group.add_argument(
        "--ink-mode",
        choices=pipeline.INK_MODES,
        default="clean",
        help=(
            "clean: the pristine slide only.  final: the slide as it was last "
            "annotated.  epochs: one page per write-and-wipe cycle, so a board "
            "that is filled, erased and filled again yields both boards.  "
            "all: the clean slide plus every annotation cycle"
        ),
    )
    group.add_argument("--combine", action="store_true", help="Also write one PDF covering all videos")
    group.add_argument(
        "--cross-video-dedup",
        action="store_true",
        help="Treat repeats across different videos as one page (off by default: each video stands alone)",
    )
    group.add_argument(
        "--skip-opening-seconds",
        type=float,
        default=0.0,
        help="Ignore everything before this time, for title animations",
    )

    group = parser.add_argument_group("slide region")
    group.add_argument("--no-crop", action="store_true", help="Keep the whole camera frame")
    group.add_argument(
        "--screen-corners",
        type=parse_screen_corners,
        default=None,
        metavar="POINTS",
        help="Override detection with normalized TL;TR;BR;BL points as x,y;x,y;x,y;x,y",
    )
    group.add_argument("--geometry-interval", type=float, default=2.0, help="Seconds between geometry-scan samples")
    group.add_argument("--geometry-plates", type=int, default=64, help="Frames kept for lecturer-free plates")
    group.add_argument("--geometry-windows", type=int, default=8, help="Windows checked for a change of framing")
    group.add_argument("--layout-tolerance", type=float, default=0.035, help="Corner movement treated as the same framing")

    group = parser.add_argument_group("slide changes")
    group.add_argument("--interval", type=float, default=0.5, help="Seconds between analysis samples")
    group.add_argument(
        "--sensitivity",
        type=float,
        default=0.12,
        help="Share of the slide that must change to call it a new slide; lower finds more",
    )
    group.add_argument("--minimum-scene-seconds", type=float, default=3.0, help="Ignore scenes shorter than this")
    group.add_argument(
        "--confirm-seconds",
        type=float,
        default=2.0,
        help="How long a change must persist; this is what ignores someone walking past",
    )

    group = parser.add_argument_group("annotations")
    group.add_argument("--ink-interval", type=float, default=1.0, help="Seconds between annotation samples")
    group.add_argument("--ink-width", type=int, default=256, help="Analysis width for annotation tracking")
    group.add_argument(
        "--ink-threshold",
        type=int,
        default=26,
        help="Brightness change counted as a pen stroke, 0-255",
    )
    group.add_argument(
        "--erase-ratio",
        type=float,
        default=0.55,
        help="Fraction of writing that must vanish to count as an erase",
    )
    group.add_argument(
        "--erase-persist-seconds",
        type=float,
        default=3.0,
        help="How long writing must stay gone; this is what ignores the lecturer standing in front of it",
    )
    group.add_argument(
        "--ink-settle-seconds",
        type=float,
        default=8.0,
        help="How long a mark must stay put to count as writing rather than a passing hand",
    )
    group.add_argument(
        "--erase-drop",
        type=float,
        default=0.55,
        help="How far the total amount of writing must fall for a wipe to count",
    )
    group.add_argument("--minimum-ink", type=float, default=0.0015, help="Ignore annotation cycles smaller than this")

    group = parser.add_argument_group("page images")
    group.add_argument(
        "--output-width",
        type=int,
        default=0,
        help="Page width in pixels; 0 keeps the slide region's own resolution",
    )
    group.add_argument("--max-output-width", type=int, default=3840, help="Upper bound on page width")
    group.add_argument("--sharpen", type=float, default=0.18, help="Unsharp-mask amount after cropping; 0 disables")
    group.add_argument("--jpeg", action="store_true", help="Write JPEG pages instead of lossless PNG")
    group.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality when --jpeg is used")
    group.add_argument(
        "--no-clean-plate",
        dest="clean_plate",
        action="store_false",
        help="Render one instant instead of compositing nearby frames to remove the lecturer",
    )
    group.add_argument("--plate-frames", type=int, default=9, help="Frames composited for each page")

    parser.add_argument("--workers", type=int, default=0, help="Videos analysed in parallel; 0 chooses automatically")
    parser.add_argument("--analyze-only", action="store_true", help="Report the analysis without writing pages")
    parser.add_argument("--quiet", action="store_true", help="Only print warnings and the final summary")
    return parser


def validate(options) -> None:
    if options.interval <= 0 or options.ink_interval <= 0 or options.geometry_interval <= 0:
        raise SystemExit("sampling intervals must be greater than zero")
    if options.output_width < 0 or options.max_output_width <= 0:
        raise SystemExit("output widths cannot be negative")
    if not 0.0 < options.erase_ratio < 1.0:
        raise SystemExit("--erase-ratio must be between 0 and 1")
    if options.workers < 0:
        raise SystemExit("--workers cannot be negative")
    if options.plate_frames < 1:
        raise SystemExit("--plate-frames must be at least 1")


def main(argv=None) -> int:
    options = build_parser().parse_args(argv)
    validate(options)

    def log(message=""):
        if not options.quiet:
            print(message, flush=True)

    input_path = options.input.resolve()
    output_dir = (
        options.output.resolve()
        if options.output
        else (input_path if input_path.is_dir() else input_path.parent) / "pdf_output"
    )
    videos = discover_videos(input_path)
    if not videos:
        raise SystemExit(f"No supported video files found: {input_path}")

    workers = options.workers or min(len(videos), max(1, (os.cpu_count() or 2) // 2))
    workers = max(1, min(workers, len(videos)))

    log("Lecture Video to PDF")
    log(f"Input : {input_path}")
    log(f"Output: {output_dir}")
    log(f"Videos: {len(videos)}   mode: {options.ink_mode}   workers: {workers}")
    started = time.monotonic()

    # One unreadable file must not cost the whole batch its output, so each
    # video is isolated: a failure is reported and the rest still finish.
    results = []
    failures = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="analyse") as executor:
            futures = {
                executor.submit(pipeline.analyse_video, video, index, options, log): video
                for index, video in enumerate(videos)
            }
            for future, video in futures.items():
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures.append((video, exc))
    else:
        for index, video in enumerate(videos):
            try:
                results.append(pipeline.analyse_video(video, index, options, log))
            except Exception as exc:
                failures.append((video, exc))

    for video, exc in failures:
        print(f"  SKIPPED {video.name}: {_reason(exc)}", file=sys.stderr, flush=True)
    if not results:
        raise SystemExit("No video could be read.")

    for result in results:
        kept = pipeline.deduplicate(result.pages)
        dropped = len(result.pages) - len(kept)
        result.pages = kept
        log(f"  {result.video.name}: {len(kept)} page(s)" + (f" ({dropped} duplicate(s) dropped)" if dropped else ""))

    combined = [page for result in results for page in result.pages]
    if options.cross_video_dedup:
        before = len(combined)
        combined = pipeline.deduplicate(combined)
        log(f"  across videos: {len(combined)} page(s) ({before - len(combined)} duplicate(s) dropped)")

    report = {
        "tool": "Lecture Video to PDF",
        "generated_at": datetime.now().astimezone().isoformat(),
        "input": str(input_path),
        "output": str(output_dir),
        "settings": {key: _plain(value) for key, value in sorted(vars(options).items())},
        "videos": [output.video_record(result) for result in results],
        "skipped": [{"video": video.name, "reason": _reason(exc)} for video, exc in failures],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if not options.analyze_only:
        renderable = list(combined) if options.cross_video_dedup else [
            page for result in results for page in result.pages
        ]
        pipeline.render_pages(renderable, output_dir / "frames", options, log)

        for result in results:
            pages = [page for page in result.pages if page.image_path is not None]
            if not pages:
                continue
            pdf_path = output_dir / f"{result.video.stem}_lecture_material.pdf"
            output.create_pdf(pages, pdf_path, f"{result.video.stem} lecture material")
            output.create_review_sheet(pages, output_dir / f"{result.video.stem}_review.jpg")
            log(f"  PDF: {pdf_path} ({len(pages)} page(s))")

        if options.combine and combined:
            pages = [page for page in combined if page.image_path is not None]
            if pages:
                combined_pdf = output_dir / "all_lecture_material.pdf"
                output.create_pdf(pages, combined_pdf, "Combined lecture material")
                output.create_review_sheet(pages, output_dir / "selection_review.jpg")
                log(f"  PDF: {combined_pdf} ({len(pages)} page(s))")

        report["pages"] = {
            result.video.name: [output.page_record(page) for page in result.pages] for result in results
        }

    report_path = output_dir / "extraction_report.json"
    output.write_report(report_path, report)
    total = sum(len(result.pages) for result in results)
    log(f"Done in {time.monotonic() - started:.1f}s   {total} page(s)")
    log(f"Report: {report_path}")
    if failures:
        log(f"Skipped {len(failures)} unreadable file(s); see the messages above.")
    return 1 if failures else 0


def _reason(exc: Exception) -> str:
    """A short explanation, since OpenCV mostly fails by returning nothing."""
    text = str(exc).strip().splitlines()
    return text[0] if text else f"{type(exc).__name__}"


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


if __name__ == "__main__":
    sys.exit(main())
