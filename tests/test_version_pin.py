"""The version lives in one file; keep it that way."""

import pathlib
import re

import torchtrade

REPO = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = (REPO / "pyproject.toml").read_text()


def test_pyproject_derives_the_version_from_the_package():
    """`torchtrade/__init__.py` is the single source and hatchling reads it.

    It used to be two numbers in two files, which is the shape that drifts, and it
    had drifted to three: a stale `torchtrade.egg-info/PKG-INFO` from an old editable
    install still said 0.0.1 and shadowed `importlib.metadata` in the repo.

    Kills the mutation that puts a literal `version = "..."` back under [project],
    which hatchling rejects only when `dynamic` also lists it, and the mutation that
    points hatch at a file that does not define `__version__`.
    """
    assert re.search(r'^dynamic = \["version"\]', PYPROJECT, re.M), (
        "[project] no longer declares the version dynamic"
    )
    assert not re.search(r'^version = "', PYPROJECT, re.M), (
        "a literal version came back to pyproject.toml; there is one source"
    )
    path = re.search(r'^\[tool\.hatch\.version\]\npath = "([^"]+)"', PYPROJECT, re.M)
    assert path, "pyproject.toml does not tell hatchling where the version lives"
    source = REPO / path.group(1)
    assert source.is_file(), f"hatch version path does not exist: {path.group(1)}"
    assert re.search(r'^__version__ = "', source.read_text(), re.M), (
        f"{path.group(1)} defines no __version__ for hatchling to read"
    )


def test_the_version_is_a_release_not_the_placeholder():
    """0.0.1 was the hatchling default and stayed for 1000+ commits."""
    assert torchtrade.__version__ != "0.0.1"
    assert re.fullmatch(r"\d+\.\d+\.\d+", torchtrade.__version__), (
        f"{torchtrade.__version__} is not a plain semver triple"
    )
