"""Draw the README hero: one video frame beside the page recovered from it.

Builds the synthetic room fixture, runs the pipeline on it, then places one raw
frame next to the page the pipeline recovered. Nothing is staged: the right-hand
image is the tool's own output.

    python docs/figures/make_hero.py

Writes hero_before_after.png and hero_before_after-dark.png.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep
ROOT = Path(HERE).resolve().parent.parent
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def build_inputs(work: Path):
    """Make the fixture and run the tool on it, returning a frame and a page."""
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_fixtures.py"), str(work)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    out = work / "out"
    subprocess.run([sys.executable, str(ROOT / "lecture_video_to_pdf.py"),
                    str(work / "room.mp4"), "--output", str(out)],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)

    cap = cv2.VideoCapture(str(work / "room.mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.45))   # mid second slide, lecturer in shot
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read a frame from the fixture")

    pages = sorted((out / "frames" / "room").glob("*.png")) or \
        sorted((out / "frames" / "room").glob("*.jpg"))
    if not pages:
        raise SystemExit("the pipeline produced no page images")
    page = cv2.imread(str(pages[1 if len(pages) > 1 else 0]))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cv2.cvtColor(page, cv2.COLOR_BGR2RGB)


def make(frame, page):
    def draw(T):
        fig, axes = plt.subplots(1, 2, figsize=(figstyle.WIDTH, 3.4))
        for ax, img, title, sub in (
            (axes[0], frame, "One video frame", "off-axis camera, lecturer in front of the screen"),
            (axes[1], page, "Recovered page", "screen located and rectified, no calibration given"),
        ):
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(T["line"])
                spine.set_linewidth(1.4)
            ax.set_title(title, pad=12)
            ax.text(0.5, -0.07, sub, transform=ax.transAxes, ha="center", va="top",
                    fontsize=figstyle.SMALL, color=T["muted"])
        fig.tight_layout(pad=0.6)
        return fig
    return draw


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        frame, page = build_inputs(Path(tmp))
        figstyle.save_both(make(frame, page), HERE + "hero_before_after")
