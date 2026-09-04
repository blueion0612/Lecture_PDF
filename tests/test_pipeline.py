"""Check the extractor against fixtures whose correct answer is known exactly.

Runs either way::

    pytest                                   # from the project root
    python tests/test_pipeline.py [dir]      # no pytest needed

Fixtures are rendered on first use and cached, so only the first run is slow.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lecture_pdf import pipeline  # noqa: E402
from lecture_pdf.cli import build_parser  # noqa: E402
from lecture_pdf.geometry import ScreenQuad  # noqa: E402

import make_fixtures  # noqa: E402


class Result:
    """Collects every failed expectation, so one run reports all of them.

    Stopping at the first failure would hide the rest, and when a detector
    changes it is the *pattern* of what broke that says what went wrong.
    """

    def __init__(self, name):
        self.name = name
        self.failures = []

    def check(self, condition, message):
        if not condition:
            self.failures.append(message)
        return condition

    @property
    def passed(self) -> bool:
        return not self.failures

    def finish(self) -> None:
        if self.failures:
            raise AssertionError("; ".join(self.failures))


def options_for(**overrides):
    parser = build_parser()
    options = parser.parse_args([str(Path.cwd())])
    for key, value in overrides.items():
        setattr(options, key, value)
    return options


def analyse(video: Path, **overrides):
    options = options_for(**overrides)
    return pipeline.analyse_video(video, 0, options, log=lambda *_: None), options


def corner_error(found: ScreenQuad, expected) -> float:
    return found.corner_distance(ScreenQuad.from_corners(expected))


def test_screencast(root: Path, manifest) -> Result:
    """A full-frame slide deck: every slide, no crop, nothing invented."""
    result = Result("screencast")
    spec = manifest["screencast.mp4"]
    analysis, _ = analyse(root / "screencast.mp4")
    pages = pipeline.deduplicate(analysis.pages)
    result.check(
        len(pages) == spec["slides"],
        f"expected {spec['slides']} pages, got {len(pages)}",
    )
    quad = analysis.segments[0].quad
    result.check(quad.full_frame or quad.area_fraction > 0.95,
                 f"expected the whole frame to be used, got area {quad.area_fraction:.3f}")
    result.finish()


def test_letterboxed(root: Path, manifest) -> Result:
    """Black bars must be cropped away, and every slide still found."""
    result = Result("letterboxed")
    spec = manifest["letterboxed.mp4"]
    analysis, _ = analyse(root / "letterboxed.mp4")
    pages = pipeline.deduplicate(analysis.pages)
    result.check(len(pages) == spec["slides"], f"expected {spec['slides']} pages, got {len(pages)}")
    error = corner_error(analysis.segments[0].quad, spec["corners"])
    result.check(error < 0.06, f"letterbox crop off by {error:.3f}")
    result.finish()


def test_room(root: Path, manifest) -> Result:
    """A keystoned screen with someone in front of it, located without help."""
    result = Result("room")
    spec = manifest["room.mp4"]
    analysis, _ = analyse(root / "room.mp4")
    pages = pipeline.deduplicate(analysis.pages)
    result.check(len(pages) == spec["slides"], f"expected {spec['slides']} pages, got {len(pages)}")
    quad = analysis.segments[0].quad
    error = corner_error(quad, spec["corners"])
    result.check(error < 0.08, f"screen crop off by {error:.3f}: {quad.as_list()}")
    result.check(not quad.full_frame, "the screen should not be read as the whole frame")
    result.finish()


def test_reframe(root: Path, manifest) -> Result:
    """A mid-recording framing change must not cost any slides."""
    result = Result("reframe")
    spec = manifest["reframe.mp4"]
    analysis, _ = analyse(root / "reframe.mp4")
    pages = pipeline.deduplicate(analysis.pages)
    result.check(
        len(pages) >= spec["slides"],
        f"expected at least {spec['slides']} pages, got {len(pages)}",
    )
    result.check(len(analysis.segments) >= 1, "no layout segments produced")
    result.finish()


def test_write_erase(root: Path, manifest) -> Result:
    """The headline case: fill a board, wipe it, fill it again."""
    result = Result("write_erase")
    spec = manifest["write_erase.mp4"]
    cycles = spec["cycles"]
    video = root / "write_erase.mp4"

    clean, _ = analyse(video, ink_mode="clean")
    clean_pages = pipeline.deduplicate(clean.pages)
    result.check(
        len(clean_pages) == 1,
        f"clean mode should collapse annotation into 1 page, got {len(clean_pages)}",
    )

    epochs, _ = analyse(video, ink_mode="epochs")
    epoch_pages = pipeline.deduplicate(epochs.pages)
    result.check(
        len(epoch_pages) == cycles,
        f"epochs mode should give one page per write/erase cycle ({cycles}), got {len(epoch_pages)}",
    )
    ratios = [round(page.ink_ratio, 4) for page in epoch_pages]
    result.check(all(ratio > 0 for ratio in ratios), f"epoch pages carry no annotation: {ratios}")

    final, _ = analyse(video, ink_mode="final")
    final_pages = pipeline.deduplicate(final.pages)
    result.check(len(final_pages) == 1, f"final mode should give 1 page, got {len(final_pages)}")
    if final_pages and epoch_pages:
        result.check(
            final_pages[0].timestamp >= epoch_pages[-1].timestamp - 1.0,
            "final mode should land on the last annotation cycle",
        )

    traces = epochs.traces
    found = sum(len(trace.epochs) for trace in traces)
    result.check(found == cycles, f"expected {cycles} annotation epochs, tracer found {found}")
    result.finish()


def test_render(root: Path, manifest) -> Result:
    """End to end: a PDF with the expected page count actually appears."""
    result = Result("render_pdf")
    workspace = Path(tempfile.mkdtemp(prefix="lecture_pdf_test_"))
    try:
        from lecture_pdf.cli import main

        code = main([
            str(root / "room.mp4"),
            "--output", str(workspace),
            "--jpeg", "--jpeg-quality", "88",
            "--quiet",
        ])
        result.check(code == 0, f"cli returned {code}")
        pdfs = list(workspace.glob("*.pdf"))
        result.check(len(pdfs) == 1, f"expected one PDF, got {[p.name for p in pdfs]}")
        if pdfs:
            import pymupdf

            with pymupdf.open(pdfs[0]) as document:
                count = document.page_count
            result.check(
                count == manifest["room.mp4"]["slides"],
                f"PDF has {count} pages, expected {manifest['room.mp4']['slides']}",
            )
        images = list((workspace / "frames").rglob("*.jpg"))
        result.check(bool(images), "no page images were written")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    result.finish()


TESTS = [test_screencast, test_letterboxed, test_room, test_reframe, test_write_erase, test_render]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[0]) if argv else Path(__file__).parent / "fixtures"
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        print(f"building fixtures in {root} ...", flush=True)
        make_fixtures.build_all(root)
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures = 0
    for test in TESTS:
        name = test.__name__.removeprefix("test_")
        try:
            test(root, manifest)
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {name}")
            for message in str(exc).split("; "):
                print(f"         {message}")
            continue
        except Exception as exc:  # a crash is a failure, not an interruption
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        print(f"[ok]   {name}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
