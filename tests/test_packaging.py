"""What ships, and whether the credentials we document are the ones the code reads."""

import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = (REPO / "pyproject.toml").read_text()

# `os.getenv("X")` AND `os.getenv("X", "")`. Not requiring the closing paren is the
# whole point: a pattern that did missed every call site with a default, which is how
# examples/online_rl/ppo/live.py sat reading a name .env.example did not provide.
READS = re.compile(r'os\.getenv\(\s*"([A-Z_]+)"|os\.environ\[\s*"([A-Z_]+)"\]')

# By shape, not by venue prefix. The names that actually shipped wrong were bare
# API_KEY and SECRET_KEY, which no venue prefix matches, so filtering on one would
# have made these guards blind to their own motivating case.
CREDENTIAL = re.compile(r"(KEY|SECRET|TOKEN|PASSPHRASE)$")

# Read out of the environment by a dependency rather than by this tree.
CONSUMED_BY_DEPENDENCIES = {"WANDB_API_KEY"}


def _tracked(pattern):
    out = subprocess.run(["git", "ls-files", pattern], cwd=REPO,
                         capture_output=True, text=True, check=True)
    return [REPO / line for line in out.stdout.splitlines() if line.strip()]


def _names_read_in(paths):
    found = set()
    for path in paths:
        for a, b in READS.findall(path.read_text()):
            found.add(a or b)
    return {n for n in found if CREDENTIAL.search(n)}


def _env_example_names():
    return {line.split("=", 1)[0] for line in (REPO / ".env.example").read_text().splitlines()
            if line.strip() and not line.startswith("#")}


def test_env_example_offers_no_credential_the_tree_never_reads():
    """.env.example is copied verbatim, so a name nothing reads is a variable the user
    sets and no code consumes. It offered API_KEY, SECRET_KEY and BINANCE_SECRET while
    the tree read ALPACA_API_KEY, ALPACA_SECRET_KEY and BINANCE_SECRET_KEY.
    """
    read = _names_read_in(_tracked("*.py") + _tracked("*.md"))
    offered = {n for n in _env_example_names()
               if CREDENTIAL.search(n)} - CONSUMED_BY_DEPENDENCIES
    assert not offered - read, f"nothing reads: {sorted(offered - read)}"


def test_every_credential_an_example_reads_is_in_env_example():
    """The other direction, scoped to examples/ because those are the scripts a user
    runs. A name read there and absent from .env.example is a script that starts with an
    empty credential and fails at the venue instead of at startup.

    Not scoped to the whole tree: bitget/order_executor.py reads the deprecated
    BITGET_* spellings on purpose, to warn that they are deprecated.
    """
    read = _names_read_in([p for p in _tracked("examples/*") if p.suffix == ".py"])
    offered = _env_example_names()
    assert not read - offered, f"read by an example, missing from .env.example: {sorted(read - offered)}"


def test_the_sdist_ships_an_allowlist():
    """hatchling defaults to everything not gitignored, which shipped .superpowers,
    local_verify and benchmarks: untracked working directories that are not gitignored.
    An exclude-only config would silently decay back to that default.
    """
    block = re.search(r"^\[tool\.hatch\.build\.targets\.sdist\]\n(.*?)(?=^\[|\Z)",
                      PYPROJECT, re.M | re.S)
    assert block, "pyproject.toml declares no sdist target; hatchling would ship everything"
    assert re.search(r"^include = \[", block.group(1), re.M), (
        "the sdist target has no include allowlist"
    )
    assert '"torchtrade/",' in block.group(1), "the sdist would not contain the package"
