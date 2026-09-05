"""requirements.txt, which the Windows launchers install from, lists exactly the
dependencies pyproject.toml declares, so the two cannot drift apart.

pyproject.toml is read with a regular expression rather than tomllib, which
only exists from Python 3.11, and the test runs on 3.10 as well."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _norm(spec):
    return spec.strip().replace(" ", "").lower()


def test_requirements_match_pyproject():
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as fh:
        text = fh.read()
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M).group(1)
    declared = {_norm(m) for m in re.findall(r'"([^"]+)"', block)}
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as fh:
        listed = {_norm(l) for l in fh if l.strip() and not l.startswith("#")}
    assert declared, "no dependencies parsed from pyproject.toml"
    assert listed == declared, (sorted(listed), sorted(declared))
