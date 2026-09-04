# Lecture Video to PDF

Turn a recording of a lecture into a PDF of the slides, including what the lecturer wrote on them.

Point it at any lecture video. It finds the projected screen or whiteboard on its own, so there is
**no per-recording calibration**: off-axis cameras that leave the screen keystoned, screen captures
where the slide fills the frame, letterboxed video, and recordings whose framing changes part-way
are all handled automatically.

It can also keep the handwriting. When a lecturer writes on one slide, wipes it and writes again,
the tool emits one page per write-and-wipe cycle instead of one page for the whole slide.

---

## Quick start (Windows)

Put this folder inside the folder that holds your videos:

```
Lectures/
├── week01.mp4
├── week02.mp4
└── Lecture_Video_to_PDF/     <- this repository
    ├── run.cmd
    └── run_with_notes.cmd
```

| Double-click | Result |
|---|---|
| `run.cmd` | Clean slides only |
| `run_with_notes.cmd` | Clean slides **and** the handwritten versions |

The first run creates a virtual environment and installs the dependencies, which takes a minute.
Results appear in `Lectures/pdf_output`:

| File | Contents |
|---|---|
| `<video>_lecture_material.pdf` | The handout |
| `<video>_review.jpg` | Every page as one contact sheet, for a quick check |
| `frames/` | The page images that went into the PDF |
| `extraction_report.json` | Detected screen regions, scenes and annotation cycles |

If the environment ever breaks, run `setup.cmd`.

## Command line

```bash
python lecture_video_to_pdf.py "path/to/lectures" --output "path/to/pdf_output"
```

The input may be a single video file or a folder of them. Any of the options below can be appended
to `run.cmd` as well, for example `run.cmd --ink-mode epochs`.

---

## What to keep — `--ink-mode`

This is the main choice.

| Mode | Pages produced | Use when |
|---|---|---|
| `clean` (default) | One per slide, at its **least** annotated | You want the original handout |
| `final` | One per slide, at its **most** annotated | The writing is the content |
| `epochs` | One per **write-and-wipe cycle** | The lecturer writes, erases and writes again |
| `all` | The clean slide plus every annotation cycle | You want both |

### Why `epochs` exists

Consider a lecturer who fills a slide with working, wipes it, and fills it again.

- Keeping only the last frame **loses the first board entirely**.
- Keeping every frame produces dozens of near-identical pages.

`epochs` cuts each slide into annotation cycles. A cycle runs while writing accumulates and closes
when **most of it disappears at once**; the fullest moment of each cycle becomes one page. Both
boards survive, and the intermediate states do not.

An erase has to satisfy two conditions, which is what makes this work: most of the previous writing
must be gone (`--erase-ratio`) **and** the total amount of writing must drop sharply (`--erase-drop`).
Redrawing part of a diagram satisfies only the first, so it stays on the same page.

---

## How it works

Each video is scanned four times. No stage holds more than a bounded number of frames in memory, so
multi-hour recordings are fine.

**1. Locate the slide region.** A screen is the part of the frame that *changes but does not move*.
A wall does neither, a lectern or logo does not change, and the lecturer never stops moving — taken
together those three facts isolate the screen without knowing anything about the room. The region is
then snapped to the real physical border by looking for straight edges, and rectified out of its
perspective. Screen captures and letterboxed video are recognised as such and left uncropped.

**2. Find slide changes.** Comparison happens in rectified slide coordinates, so the thresholds do
not depend on how the camera was placed.

- Someone walking past changes the frame a great deal and then **changes it back**, so a candidate
  is only accepted if the frame still differs some seconds later.
- Annotation **adds to** what is already there, so the check is how much of the previous slide
  survives, not how much changed.
- Colour is compared as well as brightness, so two slides that differ only in hue are not missed.

**3. Track annotation.** Ink is not defined by pen colour but as *what differs from the slide when it
first appeared*, so red, yellow and green markers, chalk, and black pen on white all behave the same.

- A mark must persist for a while before it counts, which separates writing from a hand sweeping past.
- People are large and continuous while writing is thin, so the lecturer's body is filtered out.
- Whatever the lecturer occludes is treated as *unknown* rather than *erased*, so the state survives
  them standing in front of the board.

**4. Render pages.** Frames around the chosen instant are composited by median, which **removes the
lecturer** and recovers the content they were standing in front of. Use `--no-clean-plate` to render
a single instant instead.

---

## Options

```text
--ink-mode clean|final|epochs|all   What to keep (see the table above)
--skip-opening-seconds 8            Ignore a title animation at the start
--combine                           Also write one PDF spanning every video
--cross-video-dedup                 Treat slides repeated across videos as one

--sensitivity 0.12                  Share of the slide that must change; lower finds more
--minimum-scene-seconds 3           Ignore scenes shorter than this
--confirm-seconds 2                 How long a change must persist (ignores passers-by)

--erase-ratio 0.55                  Fraction of writing that must vanish to count as an erase
--erase-drop 0.55                   How far total writing must fall for a wipe to count
--ink-settle-seconds 8              How long a mark must stay to count as writing
--minimum-ink 0.0015                Ignore annotation cycles smaller than this

--output-width 0                    0 keeps the slide region's own resolution
--jpeg --jpeg-quality 95            Smaller files at high quality
--no-crop                           Keep the whole camera frame
--screen-corners "..."              Give the four corners instead of detecting them
--no-clean-plate                    Render one instant instead of compositing
--workers 3                         Videos analysed in parallel
--analyze-only                      Report the analysis without writing anything
```

`--help` lists the full set, including the sampling intervals and rendering knobs.

`--screen-corners` takes top-left, top-right, bottom-right, bottom-left as fractions of the frame:

```text
--screen-corners "0.0844,0.0972;0.7812,0.0917;0.7820,0.7903;0.0852,0.7931"
```

---

## When the result is wrong

| Symptom | Try |
|---|---|
| Slides are missed | `--sensitivity 0.08` |
| One slide appears several times | `--sensitivity 0.18`, `--minimum-scene-seconds 5` |
| The title animation becomes pages | `--skip-opening-seconds 10` |
| The crop includes the lectern or wall | Set `--screen-corners` explicitly |
| Too many annotation pages | `--erase-drop 0.4`, `--minimum-ink 0.004` |
| Too few annotation pages | `--erase-ratio 0.7`, `--ink-settle-seconds 5` |
| The lecturer leaves ghosting | `--plate-frames 15` |
| You want the untouched frame | `--no-clean-plate` |

Running with `--analyze-only` first and reading `layout_segments`, `scenes` and `annotation_epochs`
in `extraction_report.json` usually shows which knob to reach for.

---

## Install as a package

```bash
pip install -e .
lecture-pdf "path/to/lectures" --ink-mode all
```

Not needed if you only use `run.cmd` and `setup.cmd`.

Requires Python 3.10+, NumPy, OpenCV, Pillow and PyMuPDF.

## Validation

The tests render synthetic lecture videos whose correct answers are known in advance, then check
what the pipeline recovers. Fixtures are built on the first run and cached afterwards, so only the
first run is slow (about a minute).

```bash
pytest                        # if pytest is installed
python tests/test_pipeline.py # works without it
```

Six checks over five synthetic recordings: full-screen capture, letterboxed video, a keystoned room
camera with the lecturer occluding the screen, a recording that is reframed part-way, a write-and-erase
sequence, and PDF generation.

## Known limitations

- The detected region can include some of the monitor bezel. `--screen-corners` overrides it.
- Content the lecturer stands in front of for the whole scene cannot be recovered. Nothing is
  invented that is not somewhere in the video.
- Long uninterrupted annotation on one slide may split into more epochs than necessary. Lower
  `--erase-drop` to reduce that.
- Validated on one real recording setup (one lecture hall and camera) plus the five synthetic
  fixtures. Other footage may need `--sensitivity` and `--erase-drop` adjusted from their defaults.

## License

MIT — see [LICENSE](LICENSE).
