"""Every documented import must actually import.

#287 catalogued ~40 documented symbols that raise on import: a half-finished module
rename, and in-package READMEs describing APIs that were never written. Prose cannot be
verified by review -- a reader only finds out by pasting it and watching it fail. So the
docs are executed here instead.

Scope is deliberately narrow: `from torchtrade... import X` lines, which are unambiguous
and mechanically checkable. Config kwargs and observation keys in prose are not covered
and stay a human problem.
"""

import importlib
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
IMPORT_LINE = re.compile(r"^from (torchtrade[\w.]*) import ([^\n(]+)$", re.M)


def _documented_imports():
    sources = sorted(REPO.glob("README.md"))
    sources += sorted(p for p in (REPO / "docs").rglob("*.md") if "superpowers" not in str(p))
    sources += sorted((REPO / "torchtrade").rglob("README.md"))
    for path in sources:
        for match in IMPORT_LINE.finditer(path.read_text()):
            for name in (n.strip() for n in match.group(2).split(",")):
                if name.isidentifier():
                    yield f"{path.relative_to(REPO)}::{match.group(1)}.{name}", match.group(1), name


CASES = list(_documented_imports())


@pytest.mark.parametrize("module,name", [(m, n) for _, m, n in CASES],
                         ids=[i for i, _, _ in CASES])
def test_a_documented_import_resolves(module, name):
    """A doc example that cannot run is worse than no example: it costs the reader the
    time to find out, and it looks authoritative while doing it."""
    try:
        resolved = importlib.import_module(module)
    except ImportError as exc:                      # pragma: no cover - reported as failure
        pytest.fail(f"documented module does not exist: {module} ({exc})")
    assert hasattr(resolved, name), f"{module} has no {name!r}"


def test_the_sweep_actually_found_something_to_guard():
    """Guards against the whole file passing because the regex stopped matching."""
    assert len(CASES) > 50, f"only {len(CASES)} documented imports discovered"
