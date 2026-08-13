"""The docs are executed here, because prose cannot be verified by review.

Scope: `from torchtrade... import X` (flat AND parenthesized), plus a syntax check on the
python blocks in the in-package READMEs. Config kwargs are covered below; observation keys still are not.
"""

import ast
import dataclasses
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


# ── Config kwargs ────────────────────────────────────────────────────────────
# The scope note above used to say config kwargs "live in prose and are not covered".
# They were not: `margin_call_threshold`, `rollout_steps` and `trading_mode` sat in the
# docs with ZERO occurrences in the package while all 257 import cases here passed
# (#287). An import check validates that a NAME resolves, not that a CALL is
# constructible -- the boundary was the import line, and every defect lived below it.
# Indentation-tolerant, and single-line calls too. A 4-space-only pattern silently
# skipped both flagship SequentialTradingEnvConfig examples -- they sit inside mkdocs
# `=== "Spot Trading"` tabs, so the whole fence is indented -- plus every Polymarket
# example and a live api_key bug in torchtrade/envs/live/README.md that this check's own
# premise says it should catch.
CONFIG_CALL = re.compile(r"\b(\w*Config)\(([^()]*(?:\([^()]*\)[^()]*)*)\)", re.S)
KWARG = re.compile(r"(?:^|[\s,(])([A-Za-z_]\w*)\s*=(?!=)", re.M)
# Comments are stripped first: a value comment like `# 1 = spot` parses as a kwarg
# named "1" otherwise, and an identifier cannot start with a digit anyway.
COMMENT = re.compile(r"#[^\n]*")
# Nested calls are blanked before kwargs are read: matching "at any depth" pulls the
# INNER call's kwargs out as if they belonged to the config, so
# `reward_function=partial(fn, alpha=0.5)` yields a phantom `alpha`. That either cries
# wolf or -- worse -- passes silently when the inner name collides with a real field.
# No doc does this today. Handles arbitrary nesting depth INSIDE a captured call;
# the depth CONFIG_CALL itself can capture is pinned separately below.
NESTED = re.compile(r"\([^()]*\)")


def _documented_config_kwargs():
    """Every (config class, kwarg) pair written in a tracked markdown code block."""
    for path in _doc_sources():
        for block in PY_BLOCK.findall(path.read_text()):
            for cls_name, body in CONFIG_CALL.findall(COMMENT.sub("", block)):
                flat = body
                while NESTED.search(flat):
                    flat = NESTED.sub("", flat)
                for kwarg in KWARG.findall(flat):
                    yield str(path.relative_to(REPO)), cls_name, kwarg


CONFIG_KWARGS = sorted(set(_documented_config_kwargs()))


def _resolve_config(cls_name):
    """The config class by name, or None when the docs name one we do not ship."""
    for module in ("torchtrade.envs.offline", "torchtrade.envs.live.alpaca",
                   "torchtrade.envs.live.binance", "torchtrade.envs.live.bitget",
                   "torchtrade.envs.live.bybit", "torchtrade.envs.live.okx",
                   "torchtrade.envs.live.polymarket"):
        try:
            found = getattr(importlib.import_module(module), cls_name, None)
        except Exception:
            continue
        if found is not None:
            return found
    return None


# A RATCHET, not an exemption list. Every entry is a documented kwarg that raises
# TypeError today -- the mechanically-derived remainder of #287, which until now was a
# hand-written list in the issue. Shrink it by fixing the doc; never grow it. A NEW bad
# kwarg is not on the list and fails immediately, and an entry that gets FIXED also
# fails until it is removed, so the list cannot drift away from the docs in either
# direction.
KNOWN_BROKEN = {
    ("torchtrade/envs/live/README.md", "AlpacaSLTPTradingEnvConfig", "sl_percent"),
    ("torchtrade/envs/live/README.md", "AlpacaSLTPTradingEnvConfig", "tp_percent"),
    ("torchtrade/envs/live/README.md", "AlpacaTradingEnvConfig", "api_key"),
    ("torchtrade/envs/live/README.md", "AlpacaTradingEnvConfig", "api_secret"),
    ("torchtrade/envs/live/README.md", "AlpacaTradingEnvConfig", "initial_cash"),
    ("torchtrade/envs/live/README.md", "AlpacaTradingEnvConfig", "max_position_size"),
    ("torchtrade/envs/live/README.md", "AlpacaTradingEnvConfig", "timeframe"),
    ("torchtrade/envs/live/README.md", "BinanceFuturesTradingEnvConfig", "api_key"),
    ("torchtrade/envs/live/README.md", "BinanceFuturesTradingEnvConfig", "api_secret"),
    ("torchtrade/envs/live/README.md", "BinanceFuturesTradingEnvConfig", "max_leverage"),
    ("torchtrade/envs/live/README.md", "BinanceFuturesTradingEnvConfig", "testnet"),
    ("torchtrade/envs/live/README.md", "BinanceFuturesTradingEnvConfig", "timeframe"),
    ("torchtrade/envs/live/binance/README.md", "BinanceFuturesSLTPTradingEnvConfig", "api_key"),
    ("torchtrade/envs/live/binance/README.md", "BinanceFuturesSLTPTradingEnvConfig", "api_secret"),
    ("torchtrade/envs/live/binance/README.md", "BinanceFuturesSLTPTradingEnvConfig", "max_leverage"),
    ("torchtrade/envs/live/binance/README.md", "BinanceFuturesSLTPTradingEnvConfig", "sl_percent"),
    ("torchtrade/envs/live/binance/README.md", "BinanceFuturesSLTPTradingEnvConfig", "testnet"),
    ("torchtrade/envs/live/binance/README.md", "BinanceFuturesSLTPTradingEnvConfig", "timeframe"),
    ("torchtrade/envs/live/binance/README.md", "BinanceFuturesSLTPTradingEnvConfig", "tp_percent"),
    ("torchtrade/envs/live/binance/README.md", "BinanceFuturesTradingEnvConfig", "api_key"),
    ("torchtrade/envs/live/binance/README.md", "BinanceFuturesTradingEnvConfig", "api_secret"),
    ("torchtrade/envs/live/binance/README.md", "BinanceFuturesTradingEnvConfig", "max_leverage"),
    ("torchtrade/envs/live/binance/README.md", "BinanceFuturesTradingEnvConfig", "testnet"),
    ("torchtrade/envs/live/binance/README.md", "BinanceFuturesTradingEnvConfig", "timeframe"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesSLTPTradingEnvConfig", "api_key"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesSLTPTradingEnvConfig", "api_secret"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesSLTPTradingEnvConfig", "max_leverage"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesSLTPTradingEnvConfig", "passphrase"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesSLTPTradingEnvConfig", "sl_percent"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesSLTPTradingEnvConfig", "testnet"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesSLTPTradingEnvConfig", "timeframe"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesSLTPTradingEnvConfig", "tp_percent"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesTradingEnvConfig", "api_key"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesTradingEnvConfig", "api_secret"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesTradingEnvConfig", "max_leverage"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesTradingEnvConfig", "passphrase"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesTradingEnvConfig", "testnet"),
    ("torchtrade/envs/live/bitget/README.md", "BitgetFuturesTradingEnvConfig", "timeframe")
}


@pytest.mark.parametrize("source,cls_name,kwarg", CONFIG_KWARGS,
                         ids=[f"{c}.{k}" for _, c, k in CONFIG_KWARGS])
def test_a_documented_config_kwarg_exists(source, cls_name, kwarg):
    """A kwarg the class does not accept is a TypeError for anyone copying the block."""
    config_cls = _resolve_config(cls_name)
    if config_cls is None or not dataclasses.is_dataclass(config_cls):
        pytest.skip(f"{cls_name} is not a resolvable dataclass config")
    fields = {f.name for f in dataclasses.fields(config_cls)}
    if (source, cls_name, kwarg) in KNOWN_BROKEN:
        assert kwarg not in fields, (
            f"{source}: {cls_name}({kwarg}=...) is fixed -- remove it from KNOWN_BROKEN"
        )
        pytest.xfail(f"known #287 remainder: {cls_name}({kwarg}=...)")
    assert kwarg in fields, (
        f"{source} documents {cls_name}({kwarg}=...), which raises TypeError -- "
        f"the class accepts {sorted(fields)}"
    )


def test_no_known_broken_entry_has_gone_stale():
    """A doc line DELETED rather than fixed leaves an entry nothing tests.

    The ratchet is two-directional for "still broken" and "now fixed", but a third case
    slips through: remove the offending line from the docs and its triple vanishes from
    CONFIG_KWARGS, leaving dead weight in KNOWN_BROKEN that no longer guards anything.
    """
    documented = set(CONFIG_KWARGS)
    stale = sorted(entry for entry in KNOWN_BROKEN if entry not in documented)
    assert not stale, (
        f"{len(stale)} KNOWN_BROKEN entries are no longer in the docs -- delete them: "
        f"{stale}"
    )


CONFIG_OPEN = re.compile(r"\b\w*Config\(")


def test_every_documented_config_call_is_actually_captured():
    """A call the pattern cannot parse vanishes silently -- no failure, no xfail, nothing.

    CONFIG_CALL tolerates one level of nesting, so a doubly-nested callable
    (`Config(reward_function=partial(f, w=partial(g, x=1)), initial_cash=1)`) makes the
    WHOLE call unmatchable, taking its legitimate kwargs with it. That is a worse failure
    than the phantom kwarg it replaced, because it produces no signal at all.

    Counting opens against matches is the cheap invariant: it does not care how the
    pattern is written, only that every documented call reached the check.
    """
    missed = []
    for path in _doc_sources():
        for block in PY_BLOCK.findall(path.read_text()):
            block = COMMENT.sub("", block)
            opens = len(CONFIG_OPEN.findall(block))
            matched = len(CONFIG_CALL.findall(block))
            if opens != matched:
                missed.append(f"{path.relative_to(REPO)}: {opens} calls, {matched} parsed")
    assert not missed, (
        "documented config calls the pattern could not parse, so they are unchecked:\n"
        + "\n".join(missed)
    )


def test_the_kwarg_sweep_found_configs_to_check():
    """A regex that silently matched nothing would make the check above vacuous."""
    assert len(CONFIG_KWARGS) > 40, f"only {len(CONFIG_KWARGS)} documented kwargs found"
