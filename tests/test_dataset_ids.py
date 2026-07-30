"""Guard against HuggingFace dataset ids rotting in source, config and docs.

Five ids referenced from library code, configs, docs and tests were 404 for an
unknown length of time. Nothing caught it: the tests that touched them sat behind
`skipif(not available())`, so a dead dataset read as *skip*, not failure, and the
docs and configs were never checked at all.

Resolution is metadata-only (`HfApi().dataset_info`), so this costs one HTTP call
per distinct id, not a download.
"""

import os
import pathlib
import re
import subprocess

import pytest

import torchtrade

REPO_ROOT = pathlib.Path(torchtrade.__file__).parent.parent
SUFFIXES = {".py", ".yaml", ".yml", ".md"}

# The orgs this project publishes under. A dataset from anywhere else is not covered
# -- narrow on purpose, since matching every `a/b` string would drown in false hits.
ID_PATTERN = re.compile(r"\b((?:Torch-Trade|Sebasdi)/[A-Za-z0-9._-]+)")

# Skipping when the Hub is unreachable is the exact mechanism that hid the original
# rot, so CI sets this to turn the skip into a failure.
REQUIRE_HF = os.environ.get("TORCHTRADE_REQUIRE_HF") == "1"


def _referenced_dataset_ids():
    """Map of dataset id -> sorted list of "path:line" references."""
    # git-tracked only: hydra output dirs and gitignored scratch both carry stale ids
    # that are history rather than bugs, and neither is repo content.
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\0")
    paths = [REPO_ROOT / f for f in tracked if f and pathlib.Path(f).suffix in SUFFIXES]

    found = {}
    for path in paths:
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            for match in ID_PATTERN.findall(line):
                found.setdefault(match, set()).add(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}"
                )
    return {k: sorted(v) for k, v in sorted(found.items())}


def _hub_reachable():
    from huggingface_hub import HfApi

    try:
        HfApi().dataset_info("Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025", token=_token())
        return True
    except Exception:
        return False


def _token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def test_referenced_dataset_ids_are_discoverable():
    """Every id the repo names must actually parse as an id we can look for.

    Runs offline, so it always guards the extraction itself -- if the scan silently
    stopped matching, the network test below would pass by finding nothing.
    """
    ids = _referenced_dataset_ids()
    assert ids, "extracted no dataset ids at all -- the scan pattern has stopped matching"
    for dataset_id in ids:
        assert dataset_id.count("/") == 1, f"not an org/name id: {dataset_id}"


@pytest.mark.skipif(
    not REQUIRE_HF and not _hub_reachable(),
    reason="HuggingFace Hub unreachable (set TORCHTRADE_REQUIRE_HF=1 to fail instead of skip)",
)
def test_referenced_dataset_ids_resolve_on_the_hub():
    """A dataset id in source, a config or the docs must still exist."""
    from huggingface_hub import HfApi

    api, token = HfApi(), _token()
    dead = []
    for dataset_id, refs in _referenced_dataset_ids().items():
        try:
            api.dataset_info(dataset_id, token=token)
        except Exception as exc:
            dead.append(f"{dataset_id} ({type(exc).__name__})\n      " + "\n      ".join(refs))

    assert not dead, "dataset ids no longer resolve:\n  " + "\n  ".join(dead)
