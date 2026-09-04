"""Render the before-and-after figure used at the top of the README.

Builds the synthetic room fixture, runs the pipeline on it, then places one raw
frame next to the page the pipeline recovered from it. Nothing is staged: the
right-hand image is the tool's own output.

    python docs/figures/make_hero.py

Writes hero_before_after.png and hero_before_after-dark.png.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

THEMES = {
    "light": dict(bg="white", ink="#1c2530", muted="#5b6875", frame="#b9c3cf"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", muted="#9198a1", frame="#3d444d"),
}


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


def render(theme, out_path, frame, page):
    T = THEMES[theme]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=170)
    fig.patch.set_facecolor(T["bg"])
    for ax, img, title, sub in (
        (axes[0], frame, "One video frame", "off-axis camera, lecturer in front of the screen"),
        (axes[1], page, "Recovered page", "screen located and rectified, no calibration given"),
    ):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(T["frame"])
            spine.set_linewidth(1.4)
        ax.set_title(title, fontsize=12, color=T["ink"], fontweight="bold", pad=13)
        ax.text(0.5, -0.07, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=9.4, color=T["muted"])
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=T["bg"])
    plt.close(fig)
    print("wrote", out_path)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        frame, page = build_inputs(Path(tmp))
        render("light", HERE / "hero_before_after.png", frame, page)
        render("dark", HERE / "hero_before_after-dark.png", frame, page)
