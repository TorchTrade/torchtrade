"""The docs are executed here, because prose cannot be verified by review.

Scope: `from torchtrade... import X` (flat AND parenthesized), plus a syntax check on the
python blocks in the in-package READMEs. Config kwargs and observation keys live in prose
and are not covered.
"""

import ast
import importlib
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
# Parenthesized too: a flat-only pattern skipped 30 of 225 documented symbols, and the
# skipped set contained the exact phantoms the sweep that added this test had missed.
IMPORT_LINE = re.compile(r"^from (torchtrade[\w.]*) import (?:\(([^)]*)\)|([^\n(]+))$", re.M)
PY_BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def _safe_walk(block):
    """Blocks using the `Config(a=1, ...)` ellipsis idiom do not parse; skip them here."""
    try:
        return list(ast.walk(ast.parse(block)))
    except SyntaxError:
        return []


def _doc_sources():
    """Tracked markdown only. Sourcing from the index rather than the filesystem keeps
    anyone's untracked local notes from failing the suite, without hardcoding a
    directory name that only exists on one machine."""
    listing = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO,
                             capture_output=True, text=True, check=True)
    for line in listing.stdout.split("\n"):
        if line.strip():
            yield REPO / line.strip()


def _documented_imports():
    for path in _doc_sources():
        for match in IMPORT_LINE.finditer(path.read_text()):
            module = match.group(1)
            for name in (n.strip() for n in (match.group(2) or match.group(3)).split(",")):
                if name.isidentifier():
                    yield pytest.param(module, name,
                                       id=f"{path.relative_to(REPO)}::{module}.{name}")


CASES = list(_documented_imports())
README_BLOCKS = [
    pytest.param(block, id=f"{p.relative_to(REPO)}::block{i}")
    for p in _doc_sources() if p.name == "README.md" and p.is_relative_to(REPO / "torchtrade")
    for i, block in enumerate(PY_BLOCK.findall(p.read_text()))
]


@pytest.mark.parametrize("module,name", CASES)
def test_a_documented_import_resolves(module, name):
    assert hasattr(importlib.import_module(module), name), f"{module} has no {name!r}"


@pytest.mark.parametrize("block", README_BLOCKS)
def test_a_documented_code_block_parses(block):
    """Syntax only -- most blocks need credentials or data to run. Cheap, and it catches
    what review misses: deleting the line that opened a call, leaving its arguments."""
    ast.parse(block)


def test_the_sweep_still_covers_every_source():
    """Floors set just under the real counts, because a generous one hides exactly what it
    is for: >140 against 190 still passed with the largest source (49) removed, and >20
    against 66 tolerated losing two thirds. Raise these when the docs grow."""
    assert len(CASES) > 185, f"only {len(CASES)} documented imports discovered"
    assert len(README_BLOCKS) > 60, f"only {len(README_BLOCKS)} code blocks discovered"


# NOT ENABLED YET: an AST pass comparing documented kwargs against dataclasses.fields()
# turns up 31 in-package README call sites using fields that do not exist -- e.g.
# AlpacaTradingEnvConfig(api_key=..., timeframe=...), where api_key is an env constructor
# argument and the field is time_frames. That is #287's remaining half: the bodies, not
# the import lines. Enabling this guard is the first step of that pass, not this one.
