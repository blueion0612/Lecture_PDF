"""requirements.txt, which the Windows launchers install from, lists exactly the
dependencies pyproject.toml declares, so the two cannot drift apart."""
import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _norm(spec):
    return spec.strip().replace(" ", "").lower()


def test_requirements_match_pyproject():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        declared = {_norm(d) for d in tomllib.load(fh)["project"]["dependencies"]}
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as fh:
        listed = {_norm(l) for l in fh if l.strip() and not l.startswith("#")}
    assert listed == declared, (sorted(listed), sorted(declared))
