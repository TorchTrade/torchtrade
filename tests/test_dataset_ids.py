"""Guard against HuggingFace dataset ids rotting in source, config and docs.

Five ids referenced from library code, configs, docs and tests were 404 for an
unknown length of time. Nothing caught it: the tests that touched them sat behind
`skipif(not available())`, so a dead dataset read as *skip*, not failure, and the
configs and docs were never checked at all.

What this checks, and what it does not:

* It resolves each id against the Hub with `HfApi().dataset_info` -- metadata only,
  one request per distinct id, no download. Existence is all that proves. A repo can
  resolve here and still break `datasets.load_dataset(...)`: no `train` split (which
  `load_torch_trade_dataset` hardcodes), a required `config_name`, or a schema the
  offline samplers cannot read.
* Ids assembled at runtime -- f-strings, concatenation -- are invisible to a text
  scan, as are file types outside {.py, .yaml, .yml, .md} and anything not yet
  git-tracked.
"""

import os
import pathlib
import re
import subprocess

import pytest
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

REPO_ROOT = pathlib.Path(__file__).parent.parent
SUFFIXES = {".py", ".yaml", ".yml", ".md"}

# The orgs this project publishes or has published under. Matching every `a/b` string
# instead would drown in `BTC/USD`, `America/New_York`, `train/actor_loss`. Third-party
# ids the repo depends on (amazon/chronos-*, Qwen/*) are models, so `dataset_info` is
# the wrong call for them -- they are out of scope, not overlooked.
# The tail cannot end in `.`, or a sentence-final period in prose is swallowed into the
# id and a live dataset reports as dead.
ID_PATTERN = re.compile(
    r"\b((?:Torch-Trade|Sebasdi)/[A-Za-z0-9_-](?:[A-Za-z0-9._-]*[A-Za-z0-9_-])?)"
)

# Skipping when the Hub is unreachable is the exact mechanism that hid the original
# rot, so CI sets this to turn the skip into a failure.
REQUIRE_HF = os.environ.get("TORCHTRADE_REQUIRE_HF") == "1"

# A floor, not an inventory: adding an id should not need a test edit, but silently
# *losing* one should fail. `assert ids` alone cannot do that -- narrowing the pattern
# or shrinking SUFFIXES leaves it green.
EXPECTED_IDS = {
    "Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025",
    "Torch-Trade/ethusdt_spot_1m_05_2021_to_03_2026",
}


def _referenced_dataset_ids():
    """Map of dataset id -> sorted list of "path:line" references."""
    # git-tracked only: hydra output dirs and gitignored scratch both carry stale ids
    # that are history rather than bugs, and neither is repo content.
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\0")

    this_file = pathlib.Path(__file__).resolve()
    found = {}
    for name in tracked:
        if not name or pathlib.Path(name).suffix not in SUFFIXES:
            continue
        path = REPO_ROOT / name
        # Skip self: the ids pinned in EXPECTED_IDS would otherwise satisfy the scan on
        # their own, so deleting every real reference would still look healthy.
        if path.resolve() == this_file:
            continue
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            for match in ID_PATTERN.findall(line):
                found.setdefault(match, set()).add(f"{name}:{lineno}")
    return {k: sorted(v) for k, v in sorted(found.items())}


@pytest.mark.parametrize("text,expected", [
    ("df = load_dataset('Torch-Trade/foo_bar')", ["Torch-Trade/foo_bar"]),
    ("data_path: Torch-Trade/foo_bar", ["Torch-Trade/foo_bar"]),
    # A live id at the end of a sentence must not absorb the period: dataset_info
    # raises HFValidationError on the mangled form, i.e. red CI for a healthy dataset.
    ("The default is Torch-Trade/foo_bar.", ["Torch-Trade/foo_bar"]),
    ("See Torch-Trade/foo_bar, which is live.", ["Torch-Trade/foo_bar"]),
    ("versioned Torch-Trade/foo.v2 here", ["Torch-Trade/foo.v2"]),
    ("unrelated BTC/USD and America/New_York", []),
    # Sebasdi has no live references, so only this pins the alternation: one of the five
    # ids that died was Sebasdi/..., and a re-introduction must still be reported.
    ("legacy Sebasdi/TorchTrade_btcusd_spot_1m", ["Sebasdi/TorchTrade_btcusd_spot_1m"]),
], ids=["code", "yaml", "trailing-period", "trailing-comma", "internal-dot",
        "no-false-hits", "legacy-org"])
def test_pattern_extracts_ids_without_mangling_them(text, expected):
    """A mangled id resolves to nothing on the Hub, so an over-greedy pattern reports a
    healthy dataset as dead -- a false alarm is as corrosive here as a miss."""
    assert ID_PATTERN.findall(text) == expected


def test_extraction_still_finds_the_known_references():
    """Catches a narrowed pattern or a shrunk SUFFIXES, which would quietly reduce what
    the Hub check covers while leaving it green."""
    ids = _referenced_dataset_ids()

    missing = EXPECTED_IDS - set(ids)
    assert not missing, f"stopped finding known ids: {sorted(missing)}"

    suffixes = {
        pathlib.Path(ref.rsplit(":", 1)[0]).suffix for refs in ids.values() for ref in refs
    }
    assert {".py", ".yaml"} <= suffixes, f"stopped scanning a file type: found {sorted(suffixes)}"


def test_referenced_dataset_ids_resolve_on_the_hub():
    """A dataset id in source, a config or the docs must still exist.

    There is no reachability pre-check: probing with one of the ids under test means the
    most-referenced dataset dying would read as "Hub down" and skip. Instead each failure
    is classified -- a missing repo fails, a transport error skips unless CI demands it.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    dead, unreachable = [], []
    for dataset_id, refs in _referenced_dataset_ids().items():
        try:
            api.dataset_info(dataset_id)
        except (RepositoryNotFoundError, HFValidationError) as exc:
            dead.append(f"{dataset_id} ({type(exc).__name__})\n      " + "\n      ".join(refs))
        except Exception as exc:
            unreachable.append(f"{dataset_id} ({type(exc).__name__})")

    if unreachable and not dead and not REQUIRE_HF:
        pytest.skip(
            f"HuggingFace Hub unreachable ({len(unreachable)} ids); "
            "set TORCHTRADE_REQUIRE_HF=1 to fail instead of skip"
        )
    assert not dead, "dataset ids no longer resolve:\n  " + "\n  ".join(dead)
    assert not unreachable, "Hub unreachable:\n  " + "\n  ".join(unreachable)
