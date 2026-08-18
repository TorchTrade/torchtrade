"""The version lives in two files; keep them one number."""

import pathlib
import re

import torchtrade

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_the_package_version_matches_pyproject():
    """Two files holding one number is exactly the shape that drifts.

    `pyproject.toml` is what pip installs and `torchtrade.__version__` is what a user
    prints when reporting a bug -- so a mismatch means the version in a bug report is
    not the version that was installed, which is the one case the number exists for.
    """
    declared = re.search(
        r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(), re.M
    )
    assert declared, "pyproject.toml has no version"
    assert torchtrade.__version__ == declared.group(1), (
        f"torchtrade.__version__ is {torchtrade.__version__} but pyproject.toml says "
        f"{declared.group(1)}"
    )


def test_the_version_is_a_release_not_the_placeholder():
    """0.0.1 was the hatchling default and stayed for 1000+ commits."""
    assert torchtrade.__version__ != "0.0.1"
    assert re.fullmatch(r"\d+\.\d+\.\d+", torchtrade.__version__), (
        f"{torchtrade.__version__} is not a plain semver triple"
    )
