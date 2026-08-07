"""Guard against HuggingFace dataset ids rotting in source, config and docs.

Five ids referenced from library code, configs, docs and tests were 404 for an
unknown length of time. Nothing caught it: the tests that touched them sat behind
`skipif(not available())`, so a dead dataset read as *skip*, not failure, and the
configs and docs were never checked at all.

What this checks, and what it does not:

* It resolves each id against the Hub with `HfApi().dataset_info` -- metadata only,
  one request per distinct id, no download. Existence is all that proves. A repo can
  resolve here and still break `datasets.load_dataset(...)`: no `train` split (which
  `load_torch_trade_dataset` defaults to), a required `config_name`, or a schema the
  offline samplers cannot read.
* A literal prefix before an interpolation (`f"...{tf}"`, `...${tf}`) is discarded
  rather than reported, because the prefix alone is a well-formed id that would 404
  and read as rot. Ids built by concatenation are not recognised as such, and forms
  the Hub rejects outright (`a--b`, `a.git`) are dropped rather than reported.
* File types outside {.py, .yaml, .yml, .md}, and anything not yet git-tracked, are
  not scanned.
"""

import os
import pathlib
import re
import subprocess

import pytest
import requests
from huggingface_hub import HfApi
from _pytest.outcomes import Skipped
from huggingface_hub.errors import (
    DisabledRepoError,
    HFValidationError,
    HfHubHTTPError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
)
from huggingface_hub.utils import validate_repo_id

REPO_ROOT = pathlib.Path(__file__).parent.parent
SUFFIXES = {".py", ".yaml", ".yml", ".md"}

# The orgs this project publishes or has published under. Matching every `a/b` string
# instead would drown in `BTC/USD`, `America/New_York`, `train/actor_loss`. Third-party
# ids the repo depends on (amazon/chronos-*, Qwen/*) are models, so `dataset_info` is
# the wrong call for them -- they are out of scope, not overlooked.
ID_PATTERN = re.compile(r"\b((?:Torch-Trade|Sebasdi)/[A-Za-z0-9._-]+)")

# Only ConnectionError/Timeout mean "no answer from the Hub". Everything the Hub itself
# raises is a fact about the repo, and anything unexpected must surface as an error
# rather than a skip -- a guard that skips when it breaks is this file's own bug, one
# layer down.
NO_ANSWER = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    OfflineModeIsEnabled,  # subclasses the builtin ConnectionError, not requests'
)
TRANSIENT_STATUS = {429, 500, 502, 503, 504}

# Skipping when the Hub is unreachable is the exact mechanism that hid the original
# rot, so CI sets this to turn the skip into a failure.
REQUIRE_HF = os.environ.get("TORCHTRADE_REQUIRE_HF") == "1"

# A floor, not an inventory: adding an id needs no edit here, but silently losing one
# must fail. `assert ids` alone cannot do that -- narrowing the pattern or shrinking
# SUFFIXES leaves it green.
EXPECTED_IDS = {
    "Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025",
    "Torch-Trade/ethusdt_spot_1m_05_2021_to_03_2026",
}


def _ids_in(line):
    """Well-formed ids on one line, minus runtime-assembled ones.

    `validate_repo_id` is the Hub's own rule, so prose that runs an id into surrounding
    punctuation (`foo--bar`, `foo-`, a `.git` clone URL) is dropped here rather than
    reported dead -- a false alarm costs as much as a miss.
    """
    for match in ID_PATTERN.finditer(line):
        if line[match.end():match.end() + 1] in ("{", "$"):
            continue  # f"...{var}" or hydra "...${var}" -- the prefix is not the id
        # Trim trailing punctuation the Hub forbids as a final character, so an id at
        # the end of a sentence still gets checked rather than silently dropped.
        candidate = match.group(1).rstrip(".-")
        try:
            validate_repo_id(candidate)
        except HFValidationError:
            continue
        yield candidate


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
        # Skip self: this file carries deliberately fake ids as pattern fixtures, and
        # scanning it would report them dead.
        if path.resolve() == this_file:
            continue
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            for dataset_id in _ids_in(line):
                found.setdefault(dataset_id, set()).add(f"{name}:{lineno}")
    return {k: sorted(v) for k, v in sorted(found.items())}


@pytest.mark.parametrize("text,expected", [
    ("versioned Torch-Trade/foo.v2 here", ["Torch-Trade/foo.v2"]),
    ("unrelated BTC/USD and America/New_York", []),
    # Sebasdi has no live reference, so only this pins the alternation: one of the five
    # ids that died was Sebasdi/..., and a re-introduction must still be reported.
    ("legacy Sebasdi/TorchTrade_btcusd_spot_1m", ["Sebasdi/TorchTrade_btcusd_spot_1m"]),
    # Trailing punctuation is trimmed so the id is still checked; genuinely malformed
    # forms are dropped. Either way a healthy dataset must never be reported dead.
    ("The default is Torch-Trade/foo_bar.", ["Torch-Trade/foo_bar"]),
    ("use Torch-Trade/foo_bar--the fast one", []),
    ("the Torch-Trade/foo_bar- dataset", ["Torch-Trade/foo_bar"]),
    ('load_dataset(f"Torch-Trade/btcusdt_{tf}")', []),
    ('data_path: Torch-Trade/btcusdt_${asset}_1m', []),
], ids=["internal-dot", "no-false-hits", "legacy-org", "trailing-period",
        "double-dash", "trailing-dash", "f-string", "hydra-interpolation"])
def test_extraction_does_not_fabricate_ids(text, expected):
    """A mangled id 404s, so an over-greedy scan reports a live dataset as rotted."""
    assert list(_ids_in(text)) == expected


def test_extraction_still_finds_the_known_references():
    """Catches a narrowed pattern or a shrunk SUFFIXES, which would quietly reduce what
    the Hub check covers while leaving it green."""
    ids = _referenced_dataset_ids()

    missing = EXPECTED_IDS - set(ids)
    assert not missing, (
        f"stopped finding known ids: {sorted(missing)} -- either the scan regressed, or "
        "these references were intentionally removed and EXPECTED_IDS needs updating"
    )

    suffixes = {
        pathlib.Path(ref.rsplit(":", 1)[0]).suffix for refs in ids.values() for ref in refs
    }
    # .md included deliberately: "docs were never checked" is half the original bug.
    assert {".py", ".yaml", ".md"} <= suffixes, f"stopped scanning a file type: {sorted(suffixes)}"


def test_referenced_dataset_ids_resolve_on_the_hub():
    """A dataset id in source, a config or the docs must still exist.

    There is no reachability pre-check: probing with one of the ids under test means the
    most-referenced dataset dying would read as "Hub down" and skip.
    """
    api = HfApi()
    dead, unreachable = [], []
    for dataset_id, refs in _referenced_dataset_ids().items():
        try:
            api.dataset_info(dataset_id)
        except HfHubHTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            entry = f"{dataset_id} ({type(exc).__name__} {status})"
            if status in TRANSIENT_STATUS:
                unreachable.append(entry)
            else:
                dead.append(entry + "\n      " + "\n      ".join(refs))
        except NO_ANSWER as exc:
            unreachable.append(f"{dataset_id} ({type(exc).__name__})")

    # `not dead` is load-bearing: a genuine 404 must not be masked by a transport blip
    # on some other id.
    if unreachable and not dead and not REQUIRE_HF:
        pytest.skip(
            f"HuggingFace Hub unreachable ({len(unreachable)} ids); "
            "set TORCHTRADE_REQUIRE_HF=1 to fail instead of skip"
        )
    assert not dead, "dataset ids no longer resolve:\n  " + "\n  ".join(dead)
    assert not unreachable, "Hub unreachable:\n  " + "\n  ".join(unreachable)


def _resolve_outcome():
    """Run the resolve test and name what happened. Never let Skipped escape: it would
    turn a failed expectation into a green skip, which is the bug under test."""
    try:
        test_referenced_dataset_ids_resolve_on_the_hub()
        return "pass"
    except Skipped:
        return "skip"
    except AssertionError:
        return "fail"
    except Exception:
        return "raise"


def _response(status):
    response = requests.Response()
    response.status_code = status
    return response


@pytest.mark.parametrize("exc,outcome", [
    (RepositoryNotFoundError("gone", response=_response(404)), "fail"),
    (DisabledRepoError("disabled", response=_response(403)), "fail"),
    (HfHubHTTPError("boom", response=_response(500)), "skip"),
    (HfHubHTTPError("slow down", response=_response(429)), "skip"),
    (requests.exceptions.ConnectionError("no route"), "skip"),
    (AttributeError("the guard itself is broken"), "raise"),
], ids=["404-dead", "disabled-dead", "5xx-transient", "429-transient",
        "no-answer", "broken-guard"])
def test_hub_failures_are_classified(monkeypatch, exc, outcome):
    """Both fail-open defects this file has shipped lived in these twelve lines: one
    bucketed a broken guard as a network outage, the other bucketed a disabled repo
    the same way. Each read as skip, and green. Nothing pinned either until now.
    """
    monkeypatch.setitem(globals(), "REQUIRE_HF", False)  # force skip-mode, not CI mode
    monkeypatch.setattr(
        HfApi, "dataset_info", lambda self, *a, **k: (_ for _ in ()).throw(exc)
    )

    assert _resolve_outcome() == outcome


def test_a_dead_id_is_not_masked_by_a_transport_blip(monkeypatch):
    """`not dead` in the skip guard: one flaky id must not buy silence for a real 404."""
    monkeypatch.setitem(globals(), "REQUIRE_HF", False)
    first, *rest = sorted(_referenced_dataset_ids())
    assert rest, "needs at least two referenced ids to be meaningful"

    def raiser(self, dataset_id, *args, **kwargs):
        if dataset_id == first:
            raise RepositoryNotFoundError("gone", response=_response(404))
        raise requests.exceptions.ConnectionError("no route")

    monkeypatch.setattr(HfApi, "dataset_info", raiser)
    assert _resolve_outcome() == "fail"


def test_require_hf_turns_the_skip_into_a_failure(monkeypatch):
    """CI sets the flag precisely so an unreachable Hub cannot read as green."""
    monkeypatch.setitem(globals(), "REQUIRE_HF", True)
    monkeypatch.setattr(
        HfApi, "dataset_info",
        lambda self, *a, **k: (_ for _ in ()).throw(requests.exceptions.ConnectionError("down")),
    )
    assert _resolve_outcome() == "fail"
