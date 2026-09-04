"""pytest wiring, so the suite runs both under pytest and standalone.

The fixtures are rendered on first use and cached in ``tests/fixtures``; they
are deliberately not committed, because they are reproducible from the
generator and would otherwise be several megabytes of binary in the history.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import make_fixtures  # noqa: E402


@pytest.fixture(scope="session")
def root(pytestconfig) -> Path:
    target = pytestconfig.getoption("--fixtures-dir")
    directory = Path(target) if target else Path(__file__).parent / "fixtures"
    if not (directory / "manifest.json").exists():
        make_fixtures.build_all(directory)
    return directory


@pytest.fixture(scope="session")
def manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def pytest_addoption(parser):
    parser.addoption(
        "--fixtures-dir",
        action="store",
        default=None,
        help="Reuse pre-built fixtures from this directory instead of rendering them",
    )
