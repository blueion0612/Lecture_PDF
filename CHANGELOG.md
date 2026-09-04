# Changelog

## 2.0.0

Rewritten so that any lecture recording works without per-video calibration,
and so that the lecturer's handwriting can be kept rather than avoided.

### Added
- Automatic slide-region detection.  A screen is found as the region that
  *changes but does not move*, then snapped to its real physical border.
  Handles full-screen captures, letterboxed video, off-axis cameras that
  leave the screen keystoned, and recordings whose framing changes part-way.
- `--ink-mode` with four settings.  `epochs` cuts each slide into
  write-and-wipe cycles and keeps one page per cycle, so a board that is
  filled, erased and filled again yields both boards and nothing in between.
- Lecturer-free page rendering: nearby frames are composited so content the
  lecturer was standing in front of is recovered.
- Synthetic test suite with known-correct answers (`tests/`).
- Per-video error isolation: one unreadable file no longer costs the batch
  its output.

### Changed
- Analysis runs in rectified slide space, so thresholds no longer depend on
  camera framing.
- Annotation is detected as "changed since the slide arrived" rather than by
  pen colour, so any pen, chalk or marker works.
- Slide-change detection checks how much of the previous slide *survives*,
  which is what separates a new slide from an annotation.
- Deduplication defaults to per-video; `--cross-video-dedup` restores the old
  behaviour.
- `--output-width 0` (the default) keeps the slide region's own resolution
  instead of upscaling to 1920.
- `setup.cmd` detects and rebuilds a `.venv` copied from another machine.

### Known limitations
- The detected region can include a monitor bezel; `--screen-corners` overrides it.
- Long uninterrupted annotation may split into more epochs than necessary.
- Validated on one real recording setup and five synthetic fixtures; broader
  footage is likely to shift the annotation defaults.
