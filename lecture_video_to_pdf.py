#!/usr/bin/env python3
"""Turn lecture video into PDF handouts.

Run ``lecture_video_to_pdf.py --help`` for options.  The implementation lives in
the ``lecture_pdf`` package next to this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import cv2  # noqa: F401
    import numpy  # noqa: F401
    import pymupdf  # noqa: F401
    from PIL import Image  # noqa: F401
except ImportError as exc:  # pragma: no cover - friendly command-line failure
    print(f"Missing dependency: {exc}", file=sys.stderr)
    print("Run setup.cmd first.", file=sys.stderr)
    raise SystemExit(2) from exc

from lecture_pdf.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
