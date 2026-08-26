"""Contract tests for behaviour shared by every live env via TorchTradeLiveEnv.

These replace the per-exchange copies of the same assertions. Testing a shared method
once is only sound if every env actually inherits it -- so the unit test here is paired
with a guard that asserts exactly that. If an exchange ever re-adds its own override, the
guard fails and tells you to test that exchange separately.

It also holds the tests for what an unreachable exchange must NOT look like (#270),
which is the same invariant seen from the other side: a flat account must read flat, and
an exchange that did not answer must not read as either.
"""

import ast
import dataclasses
import pathlib
import re
import importlib
import inspect
import textwrap
import math
from types import SimpleNamespace

import pytest
from torchrl.envs.common import EnvBase
from unittest.mock import MagicMock, patch

import logging

import numpy as np
import torch
from tensordict import TensorDict

import torchtrade.envs  # noqa: F401  -- registers every live env as a subclass
from torchtrade.envs.core.live import (
    InvalidActionError,
    LiveObservationHalt,
    ObservationFailurePolicy,
    TorchTradeLiveEnv,
)
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv
from torchtrade.envs.utils.sltp_mixin import SLTPMixin
from torchtrade.envs.utils.liquidation import (
    cross_liquidation_price,
    isolated_liquidation_price,
)
from tests.envs.base_exchange_tests import _sole, wire_outage_state
from tests.envs.test_live_observation_failsafe import _real_futures_env
from torchtrade.envs.core.common_types import PositionStatus
from torchtrade.envs.core.state import (
    POSITION_UNKNOWN,
    PositionState,
    PositionUnknownError,
    position_direction_from_status,
    position_qty_from_status,
)


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


# Discovered, not hand-listed: a hand-listed exchange #6 would silently escape the guard.
# __subclasses__() is a live registry, so the package filter is load-bearing: a test
# harness that subclasses a live base to drive one method in isolation would otherwise
# land in here, import-order dependent, and be asked for specs it never had. That was a
# comment asking test authors to remember; it is now a condition.
LIVE_ENVS = sorted(
    (c for c in _subclasses(TorchTradeLiveEnv) if c.__module__.startswith("torchtrade.")),
    key=lambda c: c.__name__,
)

# The plain envs (env.py). The SLTP ones get their sync from SLTPMixin instead.
NON_SLTP_ENVS = [c for c in LIVE_ENVS if c.__module__.endswith(".env")]

# The 4 futures exchanges (base + SLTP) share ONE _get_observation via TorchTradeFuturesLiveEnv;
# the intermediate base itself is excluded (it IS the shared impl). alpaca (spot) is absent by
# construction -- it does not subclass TorchTradeFuturesLiveEnv.
FUTURES_ENVS = [
    c for c in LIVE_ENVS
    if issubclass(c, TorchTradeFuturesLiveEnv) and c is not TorchTradeFuturesLiveEnv
]

PLAIN_FUTURES_ENVS = [c for c in FUTURES_ENVS if c.__module__.endswith(".env")]
SLTP_FUTURES_ENVS = [c for c in FUTURES_ENVS if c.__module__.endswith(".env_sltp")]
# 4 -> 3 is the hazard: an EMPTY list is already a collection error (pyproject sets
# empty_parameter_set_mark), but a list that merely shrinks is silent.
assert len(PLAIN_FUTURES_ENVS) == len(SLTP_FUTURES_ENVS) == 4


@pytest.mark.parametrize("done_on_bankruptcy,portfolio_value,expected", [
    (True, 50.0, True),    # below 10% of the 1000 initial -> bankrupt
    (True, 100.0, False),  # exactly at the threshold -> NOT bankrupt (the check is a strict <)
    (True, 500.0, False),  # above the threshold -> keep trading
    (False, 0.0, False),   # wiped out, but the check is off -> never terminates
], ids=["below-threshold", "at-threshold", "above-threshold", "disabled"])
def test_check_termination(done_on_bankruptcy, portfolio_value, expected):
    """Terminates iff done_on_bankruptcy and portfolio < bankrupt_threshold * initial.

    Called unbound on a stand-in rather than a real env: instantiating an EnvBase subclass
    via __new__ to skip its __init__ flakes (nn.Module.__init__ never runs, so no _modules).
    The stand-in carries only the three attributes the method reads, so a renamed config
    field raises AttributeError here instead of passing silently.
    """
    env = SimpleNamespace(
        config=SimpleNamespace(
            done_on_bankruptcy=done_on_bankruptcy,
            bankrupt_threshold=0.1,
        ),
        initial_portfolio_value=1000.0,
    )
    assert TorchTradeLiveEnv._check_termination(env, portfolio_value) is expected


@pytest.mark.parametrize("cached_position,cached_level,status,expect_position,expect_level", [
    # The env's own trade already wrote both fields: a matching position must NOT be touched,
    # or the guard could never suppress a genuinely redundant trade.
    (1, 1.0, SimpleNamespace(qty=0.5), 1, 1.0),
    (-1, -1.0, SimpleNamespace(qty=-0.5), -1, -1.0),
    # Position moved behind the env's back -> the level that produced it is unknowable.
    (1, 1.0, None, 0, 0.0),                          # liquidated -> flat, level flat
    (0, 0.0, SimpleNamespace(qty=0.5), 1, math.nan),  # opened externally -> ANY next command runs
    (1, 1.0, SimpleNamespace(qty=-0.5), -1, math.nan),  # flipped long -> short
    (-1, -1.0, SimpleNamespace(qty=0.5), 1, math.nan),  # flipped short -> long
    # A close can leave a float residual instead of an exact zero. Reading that as an open
    # position is what re-froze the guard -- the dust rule is the whole point of the shared
    # position_direction_from_status() rule.
    (1, 1.0, SimpleNamespace(qty=1e-12), 0, 0.0),     # dust after liquidation -> flat
    # qty exactly at the epsilon: the docstring says "at or below" is flat. Pin the boundary.
    (1, 1.0, SimpleNamespace(qty=1e-9), 0, 0.0),
], ids=["long-unchanged", "short-unchanged",
        "liquidated", "opened-externally", "flipped-long-to-short", "flipped-short-to-long",
        "dust-after-liquidation", "at-the-dust-epsilon"])
def test_sync_position_from_exchange(
    cached_position, cached_level, status, expect_position, expect_level
):
    """Exchange truth overwrites the cached position, and an external change NaNs the level.

    The level is the input to _execute_trade_if_needed's duplicate-action guard. If a
    liquidation leaves it stale, the agent re-requesting the level it already holds is
    silently refused -- for the rest of the episode.
    """
    env = SimpleNamespace(position=PositionState())
    env.position.current_position = cached_position
    env.position.current_action_level = cached_level

    TorchTradeLiveEnv._sync_position_from_exchange(env, status)

    assert env.position.current_position == expect_position
    if math.isnan(expect_level):
        assert math.isnan(env.position.current_action_level)
    else:
        assert env.position.current_action_level == expect_level


def test_discovery_covers_every_live_exchange():
    """The override guard below is only as good as this discovery.

    If an exchange ever stops being imported by torchtrade.envs it drops out of LIVE_ENVS
    silently, and the guard would still pass green while covering less. Fail here instead.
    Adding exchange #6 is meant to fail this -- it forces you to confirm the newcomer
    inherits the shared bankruptcy check rather than re-forking it.

    Abstract intermediate bases (e.g. TorchTradeFuturesLiveEnv) live under
    torchtrade/envs/live/shared/ and are not themselves an exchange -- they never define
    _reset or _init_trading_clients, so every other guard in this file already skips or
    passes them via inheritance. Only this set-of-exchanges check needs to filter "shared"
    out explicitly, or a real newcomer exchange could hide behind it.
    """
    exchanges = {cls.__module__.split(".")[-2] for cls in LIVE_ENVS} - {"shared"}
    assert exchanges == {"alpaca", "binance", "bitget", "bybit", "okx"}
    assert len(NON_SLTP_ENVS) == 5
    assert len(FUTURES_ENVS) == 12          # 4 venues x (base, env, env_sltp)


@pytest.mark.parametrize("method", [
    "_check_termination", "_sync_action_level_after_reset", "_build_observation_specs",
    # Everything the shared `_step` calls. Guarding `_step` itself is not enough: a venue
    # that re-forks one of ITS callees still passes the `_step` identity check while the
    # "a shared fix lands once" guarantee is already gone. Verified -- a byte-identical
    "_record_position_after_trade", "_resolve_action_level",
    "_wait_for_next_timestamp", "_finalize_step_flags",
])
@pytest.mark.parametrize("env_cls", LIVE_ENVS, ids=lambda c: c.__name__)
def test_no_live_env_overrides_shared_method(env_cls, method):
    """No live env class overrides a shared money-moving method.

    This is what makes testing each of them once (above) sufficient rather than a coverage
    loss: a re-forked copy fails here.
    """
    assert getattr(env_cls, method) is getattr(TorchTradeLiveEnv, method), (
        f"{env_cls.__name__} overrides {method}. Either drop the override, or give that "
        f"exchange its own tests -- the single shared test above no longer covers it."
    )


@pytest.mark.parametrize(
    "method",
    [
        "_get_observation",
        "_get_portfolio_value",
        "_create_trade_info",
        "_handle_close_action",
        "_execute_market_order",
        # Re-forking either reopens #394 on that venue alone: `_current_mark_price` is
        # what converts a failed fetch into a type `_halting` catches.
        "_current_mark_price",
        "_halting",
        # The shared `_step`'s two state reads. Both apply the halt policy (#295); a venue
        # copy of either is where that policy stops being one policy.
        "_acquire_pre_trade_state",
        "_acquire_post_bar_state",
    ],
)
@pytest.mark.parametrize("env_cls", FUTURES_ENVS, ids=lambda c: c.__name__)
def test_no_futures_env_reforks_the_shared_observation(env_cls, method):
    """The 4 futures exchanges share ONE _get_observation/_get_portfolio_value.

    The whole point of TorchTradeFuturesLiveEnv is that an account_state fix lands ONCE. If an
    exchange re-adds its own copy, that guarantee is silently lost and every per-exchange
    account_state test still passes -- so fail here instead. (alpaca is spot: it keeps its own
    _get_observation and is correctly absent from FUTURES_ENVS.)
    """
    assert getattr(env_cls, method) is getattr(TorchTradeFuturesLiveEnv, method), (
        f"{env_cls.__name__} re-forks {method} instead of sharing TorchTradeFuturesLiveEnv's. "
        f"Drop the override, or the unification no longer covers it."
    )


def _first_call_position(func, names):
    """Source position of the earliest call to any of `names`, or None.

    NOT `ast.walk` order: walk is breadth-first, so a call nested in an earlier `if` is
    visited AFTER a later top-level one, and an index comparison then "passes" while the
    trade actually comes first. Ordering claims have to compare source positions.
    """
    tree = ast.parse(inspect.getsource(func).lstrip())
    positions = [
        (n.lineno, n.col_offset)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in names
    ]
    return min(positions) if positions else None


@pytest.mark.parametrize("env_cls", NON_SLTP_ENVS, ids=lambda c: c.__name__)
def test_non_sltp_step_syncs_before_it_trades(env_cls):
    """Every non-SLTP _step reconciles with the exchange BEFORE the duplicate-action guard.

    This is the only thing guarding the call in bybit/okx: they have no duplicate-action
    guard, so deleting the call there changes nothing observable and no behavioural test
    would notice -- yet it is the call whose absence froze the guard in the three that do.

    Ordering is asserted, not just presence: a sync placed after _execute_trade_if_needed
    would be useless, and the guard would still read the stale cache. AST, not source text,
    so a comment mentioning the method cannot satisfy it.
    """
    step = env_cls._step
    sync = _first_call_position(step, {"_sync_position_from_exchange"})
    # NOT `_dispatch_sltp_trade`: this is parametrized over NON_SLTP_ENVS, which cannot
    # contain an SLTP env, so adding it read as coverage that could never bind.
    trade = _first_call_position(step, {"_execute_trade_if_needed"})

    assert sync is not None, (
        f"{env_cls.__name__}._step never reconciles the cached position with the exchange -- "
        f"a liquidation would leave it stale for the rest of the episode."
    )
    assert sync < trade, (
        f"{env_cls.__name__}._step syncs AFTER it trades: the duplicate-action guard still "
        f"reads the stale position."
    )


# Every CONCRETE live env, discovered by what it RESOLVES rather than what it defines
# locally. `"_step" in c.__dict__` silently dropped all four SLTP futures envs the moment
# #288 moved `_step` onto SLTPMixin -- 10 -> 6 and 7 -> 3 -- taking the #295
# outage-truncation guard and the action-validate-before-trade ordering guard with them.
# The collected test count did not move, because this PR's new cases happened to offset
# the vanished ones. A list keyed on where a method LIVES stops finding its subject the
# moment the method moves, which is the whole point of a refactor.
#
# The length assertions are the other half: without them an emptied list SKIPS rather
# than fails, which is how this would have shipped.
STEPPING_ENVS = [c for c in LIVE_ENVS if not inspect.isabstract(c)]
assert len(STEPPING_ENVS) == 10, (
    f"expected the 10 concrete live envs (5 venues x plain/SLTP), got "
    f"{[c.__name__ for c in STEPPING_ENVS]}"
)


@pytest.mark.parametrize("env_cls", STEPPING_ENVS, ids=lambda c: c.__name__)
def test_every_live_step_validates_its_action_before_it_trades(env_cls):
    """No `_step` may reach an order with an unvalidated action index.

    LIVE_ENVS is discovered via __subclasses__, so this is the only check that extends to
    an exchange #6 nobody wrote a test for -- the ten behavioural tests cover the ten envs
    that exist today. It replaces a substring guard that false-positived on a cosmetic
    re-wrap; AST, so a comment naming the method cannot satisfy it.

    Ordering is the point. Presence alone would pass a `_step` that validated AFTER
    submitting, and the whole contract is that a malformed action costs nothing.
    """
    step = env_cls._step
    resolve = _first_call_position(step, {
        "_resolve_action_level", "_resolve_action_index", "_resolve_action_tuple"})
    # `_dispatch_sltp_trade` is the SLTP trade call as of #288: the shared `_step` hands
    # off through it so bybit/okx can thread the mark. Naming only the executor made this
    # guard find no trade call at all on the four SLTP venues, which is a TypeError rather
    # than a missed ordering -- loud, but only because the list that feeds it was fixed
    # first. Keyed on the wrong name it would simply have stopped constraining anything.
    trade = _first_call_position(
        step, {"_execute_trade_if_needed", "_dispatch_sltp_trade"}
    )

    assert resolve is not None, (
        f"{env_cls.__name__}._step indexes its action space without validating -- on a "
        f"list a negative index wraps to a full LONG, on the SLTP dict it raises KeyError "
        f"after the bracket is priced."
    )
    assert resolve < trade, (
        f"{env_cls.__name__}._step validates AFTER it trades, so a malformed action still "
        f"reaches the exchange."
    )


@pytest.mark.parametrize("env_cls", LIVE_ENVS, ids=lambda c: c.__name__)
def test_position_sync_resolves_to_a_shared_implementation(env_cls):
    """Each env gets the position sync its _step actually expects.

    The base and the mixin share this name but NOT their contract: the base returns None and
    NaNs current_action_level; the mixin returns the `position_closed` bool that every SLTP
    _step reads. Declaring an SLTP env as (Base, SLTPMixin) instead of (SLTPMixin, Base) would
    silently hand it the base version -- position_closed becomes None, falsy, and SL/TP
    brackets are never cleared. Nothing raises. So base-class ORDER is load-bearing; pin it.

    This also restores what the rename cost: without it, an exchange can re-fork its own
    _sync_position_from_exchange (dropping the dust rule, say) with the whole suite green.
    """
    expected = SLTPMixin if issubclass(env_cls, SLTPMixin) else TorchTradeLiveEnv
    assert env_cls._sync_position_from_exchange is expected._sync_position_from_exchange, (
        f"{env_cls.__name__} does not resolve _sync_position_from_exchange to "
        f"{expected.__name__}'s -- check base-class order and any local override."
    )


def test_only_two_resets_derive_the_position():
    """Was five hand-rolled copies; the four futures ones are now one (#288).

    Two of the five silently discarded `cancel_open_orders()` / `close_position()`
    failures that the other two warned about -- a failed cancel leaves live brackets on a
    position the new episode believes is clean. That is what five copies buys you, and it
    is why this counts DERIVATIONS rather than trusting that they agree.

    LIVE_ENVS already recurses through __subclasses__, so intermediate bases are in it --
    an MRO walk here adds only EnvBase, Module and object. Measured, after writing the
    opposite in a docstring.
    """
    deriving = {
        c for c in LIVE_ENVS
        if (r := c.__dict__.get("_reset")) is not None
        and "current_position" in inspect.getsource(r)
    }
    assert {c.__name__ for c in deriving} == {
        "AlpacaBaseTorchTradingEnv", "TorchTradeFuturesLiveEnv"
    }, (
        f"expected the futures venues to share one _reset and alpaca to keep its own, "
        f"found {sorted(c.__name__ for c in deriving)}"
    )


@pytest.mark.parametrize("env_cls", LIVE_ENVS, ids=lambda c: c.__name__)
def test_every_reset_uses_the_shared_direction_rule(env_cls):
    """_reset must derive the position with the SAME dust rule as _step.

    At qty=1e-12 a hand-rolled exact-zero check reports a phantom position in
    account_state that the agent does not hold, while _step reads it as flat.

    Resolved through the MRO, so a venue that re-forks _reset is caught rather than
    skipped -- the SLTP envs' own _reset only delegates to super() and resets brackets.
    """
    reset = next((c.__dict__["_reset"] for c in env_cls.__mro__ if "_reset" in c.__dict__
                  and "current_position" in inspect.getsource(c.__dict__["_reset"])), None)
    if reset is None:
        pytest.skip(f"{env_cls.__name__} inherits _reset")

    tree = ast.parse(inspect.getsource(reset).lstrip())
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "position_direction_from_status" in called, (
        f"{env_cls.__name__}._reset hand-rolls its position direction instead of using the "
        f"shared rule -- a dust residual would read as a phantom position."
    )


@pytest.mark.parametrize("current_position,expected_level", [
    (0, 0.0),          # flat -> a flat command is genuinely redundant, let the guard suppress it
    (1, math.nan),     # a position we did NOT open -> the level behind it is unknowable
    (-1, math.nan),
], ids=["flat", "pre-existing-long", "pre-existing-short"])
def test_sync_action_level_after_reset(current_position, expected_level):
    """A position that predates the episode must not leave the guard able to refuse a close.

    This pins the #243 fix directly. It had exactly ONE guard -- an incidental assertion in
    bitget's test_close_position_action -- so an unrelated edit to that test would have
    silently retired the only cover for a money-moving fix.
    """
    env = SimpleNamespace(position=PositionState())
    env.position.current_position = current_position
    env.position.current_action_level = 0.0          # the stale default the bug relied on

    TorchTradeLiveEnv._sync_action_level_after_reset(env)

    if math.isnan(expected_level):
        assert math.isnan(env.position.current_action_level)
    else:
        assert env.position.current_action_level == expected_level


# --- holding_time on a direct flip (#44) ------------------------------------- #

# Every file that writes position.hold_counter must obey the two guards below. That is the
# live envs, plus the shared live base (core/live.py) and the SLTP mixin (utils/sltp_mixin.py)
# whose _sync_position_from_exchange methods legitimately reset it to 0 -- if those two drifted
# to hand-rolled aging, the live-only rglob would never see it. (#49)
_HOLD_COUNTER_FILES = sorted(pathlib.Path("torchtrade/envs/live").rglob("*.py")) + [
    pathlib.Path("torchtrade/envs/core/live.py"),
    pathlib.Path("torchtrade/envs/utils/sltp_mixin.py"),
]


@pytest.mark.parametrize("path", _HOLD_COUNTER_FILES, ids=str)
def test_only_the_shared_rule_ages_a_position(path):
    """Nothing may age a position except advance_hold_counter().

    This bug shipped FIVE times -- once per exchange, in two spellings -- because no guard
    existed. But the first version of this guard was theatre: a substring search for
    "hold_counter += 1" over a hand-listed set of paths, so it missed both
    `hold_counter = hold_counter + 1` AND any exchange added later.

    This one walks the AST and auto-discovers the files, so neither escape works. Plain
    `hold_counter = 0` resets are still allowed -- they are how a flat position and a new
    episode are expressed; what is banned is anything DERIVING a new age.
    """
    tree = ast.parse(path.read_text())

    offenders = []
    for node in ast.walk(tree):
        target, value = None, None
        if isinstance(node, ast.AugAssign):                  # hold_counter += 1
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value      # hold_counter = <anything>
        if not (isinstance(target, ast.Attribute) and target.attr == "hold_counter"):
            continue
        # a literal reset to 0 is the one legal write
        if isinstance(node, ast.Assign) and isinstance(value, ast.Constant) and value.value == 0:
            continue
        offenders.append(f"line {node.lineno}: {ast.unparse(node)}")

    assert not offenders, (
        f"{path} ages the position itself instead of calling advance_hold_counter():\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("path", _HOLD_COUNTER_FILES, ids=str)
def test_hold_counter_is_only_advanced_inside_get_observation(path):
    """advance_hold_counter() may be called ONLY from _get_observation() (#49).

    The guard above bans hand-rolled aging; this bans MISUSING the sanctioned function. #49
    was a SECOND call site: alpaca/binance aged the counter in _step() off the stale cached
    direction -- a different get_status() snapshot than the one _get_observation() shows the
    policy, so holding_time and position_direction could disagree in the same account_state.
    Pinning the ONE call to _get_observation() keeps them on a single snapshot and lets
    _reset() gate aging with advance_hold=False. A behavioural test is not a reliable backstop
    here -- the aging code is shared per exchange by env.py and env_sltp.py, so a regression in
    just one _step() escapes the other's tests.
    """
    tree = ast.parse(path.read_text())

    def _callee(node):
        f = node.func
        return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)

    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name == "_get_observation":
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and _callee(node) == "advance_hold_counter":
                offenders.append(f"line {node.lineno} (in {fn.name}): {ast.unparse(node)}")

    assert not offenders, (
        f"{path} calls advance_hold_counter() outside _get_observation():\n  "
        + "\n  ".join(offenders)
    )


# ============================================================================
# UNKNOWN EXCHANGE STATUS (#270)
# ============================================================================


def test_unknown_status_is_not_a_direction():
    """The whole bug in one assertion: an unreachable exchange must not read as flat.

    Every caller that has not been taught about an outage falls through to 0 otherwise,
    and 0 means "flat" -- which is how a held position gets re-bought every bar.
    """
    assert position_direction_from_status(None) == 0            # confirmed flat
    # match=: without it the explicit branch is dead weight, since reading .qty off the
    # sentinel raises anyway. The message is the branch's only observable effect.
    with pytest.raises(PositionUnknownError, match="must handle this"):
        position_direction_from_status(POSITION_UNKNOWN)


def test_unknown_status_is_truthy():
    """`if position_status:` is a live-path idiom; falsy would take the flat branch."""
    assert bool(POSITION_UNKNOWN) is True


def test_unknown_status_refuses_to_build_account_state():
    """An outage must not produce an account_state at all, rather than a flat-looking one.

    account_state has no way to say "unknown", and the flat branch would report a held
    position as flat -- invariant #3. Fails closed like get_account_balance() beside it.

    One case, not one per env: all 12 futures envs resolve _get_observation to this same
    implementation, and test_no_futures_env_reforks_the_shared_observation is what proves
    they still do. Parametrizing would run one function twelve times.
    """
    env = SimpleNamespace(
        observer=SimpleNamespace(
            get_observations=lambda **k: {}, get_keys=lambda: []
        ),
        trader=SimpleNamespace(
            get_status=lambda: {"position_status": POSITION_UNKNOWN},
            get_account_balance=lambda: {"total_margin_balance": 1000.0},
            # Present so a regression fails on "DID NOT RAISE" rather than on a gap in
            # the fake, which would say nothing about what the env reported.
            get_mark_price=lambda: 100.0,
        ),
        config=SimpleNamespace(include_base_features=False, leverage=10),
        position=PositionState(),
        account_state_key="account_state",
        market_data_keys=[],
    )
    with pytest.raises(PositionUnknownError):
        TorchTradeFuturesLiveEnv._get_observation(env)


_FAILING_FETCH_EXCHANGES = ["binance", "bitget", "bybit", "okx", "alpaca"]


# Each venue's real set-leverage body at 10x. An executor refuses to construct without
# one (#277): a response the echo cannot be read from confirms nothing, and skipping
# silently is the fail-open the check exists to close.
_LEVERAGE_BODIES = {
    "bybit": {"retCode": 0},
    "okx": {"code": "0", "data": [{"lever": "10"}]},
    "bitget": {"code": "00000", "data": {
        "longLeverage": "10", "shortLeverage": "10",
        "crossMarginLeverage": "10", "marginMode": "isolated"}},
    "binance": {"leverage": 10},
}


def _executor_with_failing_position_fetch(exchange):
    """Build a real order executor whose position fetch raises, as in an outage."""
    def boom(*a, **k):
        raise ConnectionError("simulated outage")


    if exchange == "binance":
        from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass
        client = SimpleNamespace(futures_position_information=boom, futures_exchange_info=boom,
                                 futures_change_leverage=lambda *a, **k: _LEVERAGE_BODIES[exchange], futures_change_margin_type=boom)
        return BinanceFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=10, client=client
        )
    if exchange == "bitget":
        from torchtrade.envs.live.bitget.order_executor import BitgetFuturesOrderClass
        client = SimpleNamespace(fetch_positions=boom, load_markets=boom, markets={},
                                 set_leverage=lambda *a, **k: _LEVERAGE_BODIES[exchange], set_position_mode=boom,
                                 set_margin_mode=boom)
        return BitgetFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=10, client=client
        )
    if exchange == "bybit":
        from torchtrade.envs.live.bybit.order_executor import (
            BybitFuturesOrderClass, MarginMode, PositionMode,
        )
        client = SimpleNamespace(get_positions=boom, get_instruments_info=boom,
                                 set_leverage=lambda *a, **k: _LEVERAGE_BODIES[exchange], switch_position_mode=boom,
                                 switch_margin_mode=boom)
        return BybitFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=10,
            margin_mode=MarginMode.ISOLATED, position_mode=PositionMode.ONE_WAY,
            api_key="k", api_secret="s", client=client,
        )
    if exchange == "okx":
        from torchtrade.envs.live.okx.order_executor import (
            OKXFuturesOrderClass, MarginMode, PositionMode,
        )
        return OKXFuturesOrderClass(
            symbol="BTC-USDT-SWAP", trade_mode="quantity", demo=True, leverage=10,
            margin_mode=MarginMode.ISOLATED, position_mode=PositionMode.NET,
            api_key="k", api_secret="s", passphrase="p",
            client=SimpleNamespace(),
            account_client=SimpleNamespace(get_positions=boom, set_leverage=lambda *a, **k: _LEVERAGE_BODIES[exchange],
                                           set_position_mode=boom),
            public_client=SimpleNamespace(get_instruments=boom),
        )
    if exchange == "alpaca":
        from torchtrade.envs.live.alpaca.order_executor import AlpacaOrderClass
        client = SimpleNamespace(get_open_position=boom, get_account=boom, get_asset=boom)
        return AlpacaOrderClass(symbol="BTCUSD", trade_mode="quantity", client=client)
    raise AssertionError(f"unhandled exchange {exchange}")


@pytest.mark.parametrize("exchange", _FAILING_FETCH_EXCHANGES)
def test_a_failed_position_fetch_does_not_report_flat(exchange):
    """get_status must distinguish "the account is flat" from "the call failed".

    Collapsing both to None is the root cause of #270: every consumer downstream then
    reads an outage as an empty account, and a held position gets re-bought.
    """
    executor = _executor_with_failing_position_fetch(exchange)

    status = executor.get_status()

    assert status.get("position_status") is POSITION_UNKNOWN, (
        f"{exchange} reported {status.get('position_status')!r} for a failed fetch; "
        "None here means flat and would be traded on"
    )


def test_reading_a_field_off_an_unknown_status_says_why():
    """The trade-sizing paths read .qty/.mark_price directly, not via the direction rule.

    They are fail-closed either way -- the sentinel has no such fields -- but a bare
    AttributeError from inside order sizing does not say that the exchange was unreachable.
    """
    # __getattr__ branches only on dunders, so one real field is the whole contract.
    with pytest.raises(PositionUnknownError, match="did not report the position"):
        POSITION_UNKNOWN.qty


# `test_position_sizing_refuses_an_unknown_status` lived here. It guarded an outage that
# began BETWEEN the two get_status() calls inside one step -- a window #295 closed by
# construction: the trade path now takes the qty `_acquire_pre_trade_state` already
# resolved, so there is one PRE-TRADE status read and no second window to fall into.
# `test_an_outage_stops_the_step_before_it_can_trade` below covers what remains.


@pytest.mark.parametrize("module", ["env", "env_sltp"])
@pytest.mark.parametrize("exchange", _FAILING_FETCH_EXCHANGES)
def test_an_outage_stops_the_step_before_it_can_trade(exchange, module):
    """The traced failure from #270, driven through _step rather than reassembled.

    Holding a long, the exchange stops answering. Previously _step read that as an empty
    account, the duplicate-action guard no longer matched, and the env bought the position
    a second time -- every bar of the outage. _step must now fail closed, and above all
    must place no order.

    Asserting through _step is the point. An earlier version of this test called the sync
    and the trade path by hand, in an order _step cannot produce, and so proved nothing
    about production -- _step raises on the status read before it ever reaches the sync.
    """
    import importlib
    env_mod = importlib.import_module(f"torchtrade.envs.live.{exchange}.{module}")
    # By NAME, not by `"_step" in vars(c)`: #288 moved the SLTP `_step` onto the mixin,
    # and a discovery predicate keyed on where the method LIVES silently stops finding
    env_cls = _sole(env_mod, "TorchTradingEnv")

    orders = []
    unexpected = None

    def _sized(*a, **k):
        orders.append("sized")
        return {"executed": True}

    env = SimpleNamespace(
        position=PositionState(),
        trader=SimpleNamespace(
            get_status=lambda: {"position_status": POSITION_UNKNOWN},
            get_mark_price=lambda: 100.0,
            trade=lambda *a, **k: orders.append("trade"),
            close_position=lambda *a, **k: orders.append("close"),
        ),
        action_levels=[-1.0, 0.0, 1.0],
        active_stop_loss=0.0,
        active_take_profit=0.0,
        _create_trade_info=lambda **kw: {"executed": kw.get("executed", False)},
        _execute_fractional_action=_sized,
        _execute_trade_if_needed=_sized,
    )
    env.position.current_position = 1
    env.position.current_action_level = 1.0
    # The real shared sync, so that if the status ever stops raising the step proceeds the
    # way production would -- otherwise a regression surfaces as a missing-attribute error
    # on the fake instead of as the order it placed.
    if hasattr(env_cls, "_get_current_price"):
        env._get_current_price = lambda ps=None: env_cls._get_current_price(env, ps)
        env.observer = SimpleNamespace(get_current_price=lambda: 100.0)
    sync_owner = SLTPMixin if module == "env_sltp" else TorchTradeLiveEnv
    env._sync_position_from_exchange = (
        lambda ps: sync_owner._sync_position_from_exchange(env, ps)
    )
    env._current_mark_price = (
        lambda ps=None: TorchTradeFuturesLiveEnv._current_mark_price(env, ps)
    )
    # Real, for the reason above: a stub without it turns any regression here into a
    # missing-attribute error rather than the order the env placed.
    env._resolve_action_index = (
        lambda td, n: TorchTradeLiveEnv._resolve_action_index(env, td, n)
    )
    env._resolve_action_level = (
        lambda td: TorchTradeLiveEnv._resolve_action_level(env, td)
    )
    # The real halt wrapper: #355 routes the pre-trade read through it, so an outage now
    # surfaces as LiveObservationHalt rather than the bare exception -- which is the point,
    # since `except LiveObservationHalt` is what the docs and the DQN example catch.
    env.config = SimpleNamespace(
        observation_failure_policy=ObservationFailurePolicy.HALT, symbol="TEST"
    )
    env.consecutive_unknown_status = 0
    env._last_confirmed_read = {}
    env._max_unknown_status_steps = 0
    env._halting = lambda read, cache_key=None: TorchTradeFuturesLiveEnv._halting(
        env, read, cache_key
    )
    env._acquire_pre_trade_state = (
        lambda: TorchTradeFuturesLiveEnv._acquire_pre_trade_state(env)
    )

    try:
        env_cls._step(env, {"action": 2})
        failed_closed = False
    except (PositionUnknownError, LiveObservationHalt):
        failed_closed = True
    except Exception as exc:
        # The fake is deliberately too thin to finish a whole step. Whether getting this
        # far mattered is what `orders` below answers -- but name the exception, or a gap
        # in the fake reads as "the env did not fail closed", which it is not.
        failed_closed = False
        unexpected = exc

    # Order matters: the money claim is "no order", and asserting it first means a
    # regression reports that rather than "DID NOT RAISE".
    assert orders == [], f"an outage must not place an order, but {exchange} sent {orders}"
    assert failed_closed, (
        "an outage must fail closed rather than stepping on stale state"
        + (f"; stopped on {unexpected!r} instead" if unexpected else "")
    )


@pytest.mark.parametrize("body,expect_flat", [
    ('{"code":40410000,"message":"position does not exist"}', True),
    ('{"code":42910000,"message":"rate limit exceeded"}', False),
    ('{"code":50010000,"message":"internal server error"}', False),
    ("not the documented json", False),
], ids=["no-position", "rate-limited", "server-error", "unparseable-body"])
def test_alpaca_only_treats_its_not_found_error_as_flat(body, expect_flat):
    """Alpaca raises for a flat account, so the error itself has to be read.

    Every APIError meaning flat would put #270 back inside its own fix: a 429 during a
    volatile minute, or a 5xx, would report a held position as gone.
    """
    from alpaca.common.exceptions import APIError
    from torchtrade.envs.live.alpaca.order_executor import AlpacaOrderClass

    def raise_api_error(*a, **k):
        raise APIError(body)

    executor = AlpacaOrderClass(
        symbol="BTCUSD", trade_mode="quantity",
        client=SimpleNamespace(get_open_position=raise_api_error),
    )

    position_status = executor.get_status()["position_status"]

    if expect_flat:
        assert position_status is None
    else:
        assert position_status is POSITION_UNKNOWN, (
            f"{body[:40]!r} is an answer we cannot read as an empty account"
        )


def test_alpaca_refuses_to_build_account_state_on_an_unknown_status():
    """Alpaca has its own _get_observation, so the futures test above does not reach it.

    Its flat branch would report a held position as flat -- invariant #3 for the spot env.
    """
    from torchtrade.envs.live.alpaca.base import AlpacaBaseTorchTradingEnv

    env = SimpleNamespace(
        observer=SimpleNamespace(get_observations=lambda **k: {}, get_keys=lambda: []),
        trader=SimpleNamespace(
            get_status=lambda: {"position_status": POSITION_UNKNOWN},
            client=SimpleNamespace(get_account=lambda: SimpleNamespace(cash="1000")),
        ),
        config=SimpleNamespace(include_base_features=False),
        position=PositionState(),
        # Present so a regression fails on the assertion below rather than on a gap in
        # the fake, which would say nothing about what the env reported.
        account_state_key="account_state",
        market_data_keys=[],
    )
    with pytest.raises(PositionUnknownError):
        AlpacaBaseTorchTradingEnv._get_observation(env)


def test_alpaca_refuses_to_value_the_portfolio_on_an_unknown_status():
    """Cash alone feeds _check_termination, so an outage would read as a near-total loss.

    The flat branch returns the balance without the position's market value, which for a
    held position is most of the portfolio -- enough to terminate the episode as bankrupt.
    """
    from torchtrade.envs.live.alpaca.base import AlpacaBaseTorchTradingEnv

    env = SimpleNamespace(
        trader=SimpleNamespace(
            get_status=lambda: {"position_status": POSITION_UNKNOWN},
            client=SimpleNamespace(get_account=lambda: SimpleNamespace(cash="1000")),
        ),
        balance=0.0,
    )
    env._read_cash = lambda: AlpacaBaseTorchTradingEnv._read_cash(env)
    with pytest.raises(PositionUnknownError):
        AlpacaBaseTorchTradingEnv._get_portfolio_value(env)


def test_alpaca_outer_failure_does_not_report_flat():
    """get_status wraps everything in a second try, and it used to return {}.

    An absent key reads as flat downstream exactly like an explicit None does, so the
    outer handler needs the sentinel too. Reached by failing the order lookup, which runs
    before the position fetch.
    """
    from torchtrade.envs.live.alpaca.order_executor import AlpacaOrderClass

    def boom(*a, **k):
        raise ConnectionError("simulated outage")

    executor = AlpacaOrderClass(
        symbol="BTCUSD", trade_mode="quantity",
        client=SimpleNamespace(get_order_by_id=boom, get_open_position=boom),
    )
    executor.last_order_id = "some-order-id"   # forces the outer try to run and fail

    assert executor.get_status().get("position_status") is POSITION_UNKNOWN


@pytest.mark.parametrize("sync", [
    TorchTradeLiveEnv._sync_position_from_exchange,
    SLTPMixin._sync_position_from_exchange,
], ids=["base", "sltp"])
def test_neither_sync_fork_treats_an_outage_as_flat(sync):
    """Pins the contract #295 will edit.

    Today _step raises on its own status read before either fork is reached, so making
    these sync an outage to 0 changes nothing observable -- which is exactly why it needs
    saying. #295 sets out to make an outage survivable, and will start here; without this
    it could reintroduce the flat reading with the whole suite still green.
    """
    env = SimpleNamespace(position=PositionState(), active_stop_loss=0.0, active_take_profit=0.0)
    env.position.current_position = 1
    env.position.current_action_level = 1.0

    with pytest.raises(PositionUnknownError):
        sync(env, POSITION_UNKNOWN)


def test_the_hand_listed_exchanges_match_the_discovered_ones():
    """The hand-written exchange list, against this file's "discovered, not hand-listed".

    It cannot be derived -- each exchange's executor takes different constructor kwargs --
    so the list is asserted instead. Both #270 parametrizations that cannot be derived now
    read from it, so exchange #6 fails here rather than quietly skipping them.
    """
    # NON_SLTP_ENVS is one concrete env per exchange; LIVE_ENVS also carries the
    # intermediate futures base, whose module is "shared" rather than an exchange.
    discovered = {c.__module__.split(".")[-2] for c in NON_SLTP_ENVS}
    assert discovered == set(_FAILING_FETCH_EXCHANGES), (
        f"hand-listed {sorted(_FAILING_FETCH_EXCHANGES)} but discovered {sorted(discovered)}"
    )


def test_position_unknown_identity_survives_a_round_trip():
    """`is POSITION_UNKNOWN` is the check at every call site, so identity has to hold.

    __new__ exists for this; without pinning it the docstring is an unverified claim.
    """
    import copy
    import pickle

    assert pickle.loads(pickle.dumps(POSITION_UNKNOWN)) is POSITION_UNKNOWN
    assert copy.copy(POSITION_UNKNOWN) is POSITION_UNKNOWN
    # deepcopy probes the instance for __deepcopy__; without the dunder escape hatch in
    # __getattr__ that probe raises a trading error from inside a copy.
    assert copy.deepcopy(POSITION_UNKNOWN) is POSITION_UNKNOWN
    assert copy.deepcopy({"position_status": POSITION_UNKNOWN})["position_status"] is POSITION_UNKNOWN


@pytest.mark.parametrize("env_cls", FUTURES_ENVS, ids=lambda c: c.__name__)
def test_no_live_env_reforks_the_position_quantity_accessor(env_cls):
    """The size accessor lives on the shared base, and must stay there (#283).

    Three envs carried a byte-identical `position.qty if position is not None else 0.0`,
    which returned a dust residual rather than 0 -- so `abs(current_qty) > 0` fired on a
    flat account, closed nothing, and still advanced current_action_level from a trade
    that never happened.

    Structural, not behavioural: a re-forked copy that has not drifted yet passes every
    behavioural test, which is exactly how three of them survived.

    #295 deleted the shared accessor entirely -- the trade path now takes the qty
    `_acquire_pre_trade_state` resolved under the halt policy, so there is nothing left to
    inherit. That makes re-introducing a private copy MORE tempting, not less, which is
    why this guard outlived the method it was written for.
    """
    assert "_get_current_position_quantity" not in vars(env_cls), (
        f"{env_cls.__name__} defines its own _get_current_position_quantity. The dust "
        f"rule lives in position_qty_from_status, and the qty is already resolved once "
        f"per step in _acquire_pre_trade_state -- take it from there."
    )


def test_a_dust_residual_does_not_look_like_a_position_to_the_trade_path():
    """The concrete failure from #283, at the seam every sizing path reads.

    An exchange can leave a float residual after a full close. Read as a live position it
    makes `abs(current_qty) > 0` true on a flat account, so action 0.0 calls
    close_position() on nothing -- and still advances current_action_level from a trade
    that never happened, which freezes the duplicate-action guard (invariant 2).

    The seam MOVED in #295: the trade path no longer queries the venue itself, it takes
    the qty `_acquire_pre_trade_state` already resolved under the halt policy. So this
    drives that, rather than the per-env accessor it used to -- which #295 left with no
    production callers at all.
    """
    ps = SimpleNamespace(qty=1e-12, mark_price=100.0)
    env = SimpleNamespace(
        config=SimpleNamespace(
            observation_failure_policy=ObservationFailurePolicy.HALT, symbol="T"
        ),
        trader=SimpleNamespace(get_status=lambda: {"position_status": ps},
                               get_mark_price=lambda: 100.0),
        consecutive_unknown_status=0, _status_unknown_this_step=False,
        _last_confirmed_read={}, _max_unknown_status_steps=0,
    )
    env._halting = lambda read, cache_key=None: TorchTradeFuturesLiveEnv._halting(
        env, read, cache_key
    )
    env._current_mark_price = (
        lambda p=None: TorchTradeFuturesLiveEnv._current_mark_price(env, p)
    )

    _, _, _, size = TorchTradeFuturesLiveEnv._acquire_pre_trade_state(env)
    assert size == 0.0, "a 1e-12 residual must read as flat, not as a position to close"


@pytest.mark.parametrize("qty,expected", [
    (None, 0.0),        # no position at all
    (0.0, 0.0),
    (1e-12, 0.0),       # dust left by a full close
    (-1e-12, 0.0),
    (1e-9, 0.0),        # exactly at the epsilon, which is inclusive
    (1.1e-9, 1.1e-9),   # just past it: a real, if tiny, position
    (-2.5, -2.5),      # a real SHORT: nothing else here holds a negative
    ("2.5", 2.5),       # exchanges return strings; the old form passed one through
], ids=["none", "zero", "dust-long", "dust-short", "at-eps", "past-eps", "short", "string"])
def test_position_qty_from_status_is_the_one_size_rule(qty, expected):
    """Direct cover for the helper (#283), which okx reaches without the accessor.

    The string cell is not hypothetical: binance, bybit and okx all read qty off the wire
    as a string. They coerce at construction today, but the deleted hand-rolled form
    returned whatever it was given, so an uncoerced path would have made the downstream
    `abs(current_qty) > 0` raise TypeError rather than size an order.
    """
    status = None if qty is None else SimpleNamespace(qty=qty)
    assert position_qty_from_status(status) == expected


def test_position_qty_from_status_refuses_an_unknown_status():
    """An outage is not flat -- the same rule position_direction_from_status enforces."""
    # match=: without it this passes whether or not the helper has its own guard, because
    # _PositionUnknown.__getattr__("qty") raises the same type anyway.
    with pytest.raises(PositionUnknownError, match="treating it as a size"):
        position_qty_from_status(POSITION_UNKNOWN)


def test_okx_sizes_through_the_dust_rule_in_step():
    """okx sizes in _step, not through the inherited accessor -- drive the real path.

    The accessor cells above resolve on okx but okx never calls it, so reverting okx's own
    read to `position_status.qty if position_status else 0.0` left the whole suite green.
    This is the only cell that fails on that.
    """
    from torchtrade.envs.live.okx.env import OKXFuturesTorchTradingEnv

    seen = {}

    def _capture(desired_action, *, current_qty, current_price):
        seen["current_qty"] = current_qty
        raise RuntimeError("stop here; the stand-in cannot finish a step")

    env = SimpleNamespace(
        position=PositionState(),
        trader=SimpleNamespace(
            get_status=lambda: {
                "position_status": SimpleNamespace(qty=1e-12, mark_price=100.0)
            },
            get_mark_price=lambda: 100.0,
        ),
        action_levels=[-1.0, 0.0, 1.0],
        _sync_position_from_exchange=lambda ps: None,
        _execute_trade_if_needed=_capture,
    )
    env._current_mark_price = (
        lambda ps=None: TorchTradeFuturesLiveEnv._current_mark_price(env, ps)
    )
    env.config = SimpleNamespace(
        observation_failure_policy=ObservationFailurePolicy.HALT, symbol="TEST"
    )
    env.consecutive_unknown_status = 0
    env._last_confirmed_read = {}
    env._max_unknown_status_steps = 0
    env._halting = lambda read, cache_key=None: TorchTradeFuturesLiveEnv._halting(
        env, read, cache_key
    )
    env._acquire_pre_trade_state = (
        lambda: TorchTradeFuturesLiveEnv._acquire_pre_trade_state(env)
    )
    env._resolve_action_index = (
        lambda td, n: TorchTradeLiveEnv._resolve_action_index(env, td, n)
    )
    env._resolve_action_level = (
        lambda td: TorchTradeLiveEnv._resolve_action_level(env, td)
    )
    with pytest.raises(RuntimeError):
        OKXFuturesTorchTradingEnv._step(env, {"action": 1})

    assert seen["current_qty"] == 0.0, (
        "a 1e-12 residual reached okx's sizing path as a live position"
    )


# --- #277: account_state must not fail open on missing venue fields ---------------


def _futures_env_stub(position_status, balance, leverage=5):
    """Minimal stand-in for calling TorchTradeFuturesLiveEnv._get_observation directly.

    Same shape as test_unknown_status_refuses_to_build_account_state above: one call
    site, because all 12 futures envs resolve _get_observation to this one
    implementation and test_no_futures_env_reforks_the_shared_observation proves it.
    """
    return SimpleNamespace(
        observer=SimpleNamespace(get_observations=lambda **k: {}, get_keys=lambda: []),
        trader=SimpleNamespace(
            get_status=lambda: {"position_status": position_status},
            get_account_balance=lambda: balance,
            get_mark_price=lambda: 100.0,
        ),
        # Deliberately NOT the position's leverage: the fallback must use what the
        # venue reports for the position, not what the config asked for.
        config=SimpleNamespace(include_base_features=False, leverage=leverage),
        position=PositionState(),
        account_state_key="account_state",
        market_data_keys=[],
        # _get_observation declares status_unknown; _finalize_step_flags sets it (#295).
        consecutive_unknown_status=0,
    )


def _open_position(qty, *, entry_price=100.0, mark_price=100.0, leverage=20,
                   liquidation_price=0.0, margin_mode="isolated"):
    return SimpleNamespace(
        qty=qty,
        notional_value=qty * mark_price,
        entry_price=entry_price,
        mark_price=mark_price,
        unrealized_pnl_pct=0.0,
        leverage=leverage,
        liquidation_price=liquidation_price,
        margin_mode=margin_mode,
    )


@pytest.mark.parametrize("qty,expected", [
    (1.0, (100.0 - 100.0 * (1 - 1 / 20 + 0.004)) / 100.0),   # long: liq below
    (-1.0, (100.0 * (1 + 1 / 20 - 0.004) - 100.0) / 100.0),  # short: liq above
], ids=["long", "short"])
def test_cross_margin_position_without_liq_price_does_not_read_as_flat(qty, expected):
    """The #277 bug: bybit/OKX send liqPrice="" for cross, which arrived here as 0.0.

    `liquidation_price <= 0` then took the same branch as "no position at all" and
    reported 1.0 -- a 20x position four percent from liquidation reading exactly as safe
    as a flat spot account. The aggregate equals the focal maintenance here, so the
    account-aware and single-position cross formulas reduce to the same expected value.
    """
    env = _futures_env_stub(
        _open_position(qty, margin_mode="cross"),
        {"total_margin_balance": 1000.0, "total_maintenance_margin": 0.4},
    )
    obs = TorchTradeFuturesLiveEnv._get_observation(env)
    distance = obs["account_state"][5].item()

    assert distance == pytest.approx(expected, rel=1e-5)


def test_venue_liq_price_is_used_verbatim_when_present():
    """The fallback is for an absent price, not a second opinion on a published one.

    Isolated-margin venues do publish liqPrice; estimating over the top of it would put
    the live observation at odds with the exchange that will actually do the liquidating.
    90.0 here is deliberately NOT what the isolated formula gives (95.4).
    """
    env = _futures_env_stub(
        _open_position(1.0, liquidation_price=90.0), {"total_margin_balance": 1000.0}
    )
    obs = TorchTradeFuturesLiveEnv._get_observation(env)

    assert obs["account_state"][5].item() == pytest.approx(0.1, rel=1e-6)


def test_flat_account_still_reads_as_maximally_far_from_liquidation():
    """The other half of invariant #3: the fallback must not make a flat account look risky.

    The fallback reads entry_price off position_status, which is None when flat -- so a
    fallback hoisted above the position check would raise here rather than report 1.0.
    """
    env = _futures_env_stub(None, {"total_margin_balance": 1000.0})
    obs = TorchTradeFuturesLiveEnv._get_observation(env)

    assert obs["account_state"][5].item() == 1.0


def test_missing_balance_key_raises_rather_than_reporting_zero_exposure():
    """`.get("total_margin_balance", 0)` made a broken adapter report every position flat.

    All four adapters build the key unconditionally, so its absence is a bug in the
    adapter, and 0.0 exposure is the one answer guaranteed to be wrong.
    """
    env = _futures_env_stub(_open_position(1.0), {"available_balance": 1000.0})
    with pytest.raises(KeyError, match="total_margin_balance"):
        TorchTradeFuturesLiveEnv._get_observation(env)


def test_empty_flat_account_is_zero_exposure_not_an_error():
    """The zero-equity guard must not fire on an account that is simply empty and flat."""
    env = _futures_env_stub(None, {"total_margin_balance": 0.0})
    obs = TorchTradeFuturesLiveEnv._get_observation(env)

    assert obs["account_state"][0].item() == 0.0


@pytest.mark.parametrize("kwargs,match", [
    ({"leverage": 0}, "leverage must be at least 1"),
    ({"leverage": float("nan")}, "leverage must be at least 1"),
    ({"entry_price": 0.0}, "entry_price must be positive"),
    ({"entry_price": float("nan")}, "entry_price must be positive"),
], ids=["zero-leverage", "nan-leverage", "zero-entry", "nan-entry"])
def test_unusable_inputs_raise_instead_of_pricing_a_liquidation(kwargs, match):
    """Without these the venue's own bad data produces 0.0, which reads as distance 1.0.

    That is #277 again by a different route: a zero liquidation price is indistinguishable
    from the flat case downstream, so the caller must never receive one.

    The NaN cells are the reason the guards are spelled `not (x >= 1)` instead of `x < 1`.
    NaN compares False to every operator, so the natural spelling passes it straight
    through to arithmetic that yields NaN, and `max(0.0, nan)` returns 0.0 -- the exact
    "reads as safe" answer, arrived at silently.
    """
    args = {"entry_price": 100.0, "is_long": True, "leverage": 10}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        isolated_liquidation_price(**args)


@pytest.mark.parametrize("position_value,equity,match", [
    (100.0, 0.0, "refusing to report this account as flat"),
    (100.0, -50.0, "refusing to report this account as flat"),
    # NaN is caught earlier and more specifically, by the equity finiteness check: on a
    # FLAT account the held-position guard cannot see it, and the same value reaches
    # is_bankrupt(), where `nan < threshold * initial` is False -- termination disabled.
    (100.0, float("nan"), "non-finite equity"),
], ids=["zero-equity", "negative-equity", "nan-equity"])
def test_position_held_against_unusable_equity_refuses_to_report_flat(position_value, equity, match):
    """A held position must never divide out to 0.0% exposure.

    The NaN cell is not hypothetical pedantry: written `total_balance <= 0`, the raise is
    skipped for NaN and the ternary beside it then takes its `else 0.0` arm, so the
    position reports as flat -- invariant #3 -- with no error anywhere. Only the
    `not (x > 0)` spelling catches all three.
    """
    env = _futures_env_stub(
        _open_position(position_value / 100.0), {"total_margin_balance": equity}
    )
    with pytest.raises(ValueError, match=match):
        TorchTradeFuturesLiveEnv._get_observation(env)


@pytest.mark.parametrize("qty,mark", [(1.0, 94.0), (-1.0, 106.0)], ids=["long", "short"])
def test_estimate_already_crossed_reads_as_at_liquidation(qty, mark):
    """Past the estimated price, the clamp pins distance at 0.0 rather than going negative.

    Worth pinning because the estimate is an isolated-margin one: a cross-margin position
    is backed by the whole account and routinely survives past it, so it will sit at a
    saturated 0.0 rather than degrading smoothly. That is the conservative direction, but
    it is a real limit on what account_state[5] tells the policy under cross margin.
    """
    env = _futures_env_stub(
        _open_position(qty, mark_price=mark, leverage=20), {"total_margin_balance": 1000.0}
    )
    obs = TorchTradeFuturesLiveEnv._get_observation(env)

    assert obs["account_state"][5].item() == 0.0


@pytest.mark.parametrize("leverage", [1, 2, 20], ids=lambda l: f"{l}x")
@pytest.mark.parametrize("is_long", [True, False], ids=["long", "short"])
def test_live_distance_to_liquidation_agrees_with_the_offline_env(leverage, is_long):
    """Same position, same account_state[5], offline and live.

    This is the reason the rule was extracted rather than copied: a policy trained on the
    offline geometry reads account_state[5] live and must be reading the same quantity.

    Compares the two end to end rather than the two liquidation prices, because the gates
    around the arithmetic are where they diverged -- offline returns "no liquidation" at
    1x via has_liquidation, and a live fallback without that gate priced a 1x long at
    mmr*entry and reported 0.996 against offline's 1.0. 1x is the live config default, so
    that divergence was the common case. Nothing here supplies a rate to both sides: the
    offline one comes from its config default and the live one from the helper's.
    """
    from torchtrade.envs.offline.sequential import (
        SequentialTradingEnv, SequentialTradingEnvConfig,
    )

    entry = mark = 100.0
    qty = 1.0 if is_long else -1.0

    offline = SimpleNamespace(
        leverage=leverage,
        maintenance_margin_rate=SequentialTradingEnvConfig().maintenance_margin_rate,
    )
    # The real property, not `leverage > 1` restated here: supplying the gate's own answer
    # made this pass even with offline's has_liquidation mutated to always-False.
    offline.has_liquidation = SequentialTradingEnv.has_liquidation.fget(offline)
    offline_distance = SequentialTradingEnv._calculate_distance_to_liquidation(
        offline, mark, SequentialTradingEnv._calculate_liquidation_price(offline, entry, qty), qty
    )

    env = _futures_env_stub(
        _open_position(qty, entry_price=entry, mark_price=mark, leverage=leverage),
        {"total_margin_balance": 1000.0},
    )
    live_distance = TorchTradeFuturesLiveEnv._get_observation(env)["account_state"][5].item()

    assert live_distance == pytest.approx(offline_distance, rel=1e-6)


def test_portfolio_value_raises_on_missing_balance_key():
    """`_get_portfolio_value` feeds the `current` side of is_bankrupt().

    Defaulting it to 0 makes `current < threshold * initial` true for any funded
    account -- instant false bankruptcy on an adapter bug rather than a loud failure.
    """
    env = SimpleNamespace(
        trader=SimpleNamespace(get_account_balance=lambda: {"available_balance": 1000.0})
    )
    with pytest.raises(KeyError, match="total_margin_balance"):
        TorchTradeFuturesLiveEnv._get_portfolio_value(env)


def test_no_live_env_silently_defaults_a_money_field():
    """Structural: no `.get("<money field>", 0)` anywhere under live/ (#277).

    Five sites defaulted the equity key and it broke is_bankrupt() in both directions --
    0 as the baseline means `current < threshold * 0` never fires and a wiped account
    trades on; 0 as the current value is instant false bankruptcy. `notional` is here for
    the same reason: binance defaulted it to 0, which zeroes exposure_pct and, before the
    raise was re-keyed, also suppressed the non-positive-equity error on a held position.

    Behavioural tests cover the two sites in this file; the four per-exchange baselines
    are set inside _reset scaffolding that needs a live client, so this guard is what
    keeps them fixed.

    Matched by regex over both quote styles: the first version of this guard looked for
    one literal spelling, and bitget's executor quotes its keys the other way -- four
    more sizing paths were hiding behind that. Only a ZERO literal counts: bitget falls
    back to a computed `abs(contracts * mark_price)` and binance's MIN_NOTIONAL filter
    defaults to 100, and neither invents a flat account.
    """
    live_root = pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent / "live"
    pattern = re.compile(r"""\.get\(\s*["'](total_margin_balance|notional)["']\s*,\s*0(\.0+)?\s*\)""")
    scanned, offenders = 0, []
    for path in sorted(live_root.rglob("*.py")):
        scanned += 1
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(live_root.parent)}:{i}")

    # Without this the guard passes by scanning nothing -- it did, from any cwd but the
    # repo root, back when it built the path from a module name.
    assert scanned > 10, f"guard scanned only {scanned} files under {live_root}"
    assert offenders == [], (
        f"money field read with a silent default at {offenders}; index it, or fall back to "
        f"a computed value, so a broken adapter cannot report a wiped or flat account"
    )


# --- the cross-margin half of the estimate (#342 review) -------------------------


@pytest.mark.parametrize("qty,expected_nearer", [
    (1.0, 98.3936),    # long: cross sits ABOVE isolated's 95.40, i.e. nearer the mark
    (-1.0, 101.5936),  # short: cross sits BELOW isolated's 104.60
], ids=["long", "short"])
def test_depleted_collateral_prices_liquidation_nearer_than_isolated(qty, expected_nearer):
    """Isolated alone is not conservative, which is the whole reason cross is consulted.

    The isolated formula sees only this position, so it answers the same 95.40 whether the
    account is flush or nearly empty. Here a 100-notional 20x position is backed by 2.0 of
    equity: cross liquidates at 98.39, well before isolated's 95.40, and reporting the
    isolated distance would OVERSTATE the room left -- the fail-open this change removes,
    reintroduced one level down.
    """
    env = _futures_env_stub(
        _open_position(qty, leverage=20, margin_mode="cross"),
        {"total_margin_balance": 2.0, "total_maintenance_margin": 0.4},
    )
    obs = TorchTradeFuturesLiveEnv._get_observation(env)
    distance = obs["account_state"][5].item()

    assert distance == pytest.approx(abs(100.0 - expected_nearer) / 100.0, rel=1e-4)
    # Strictly tighter than the isolated-only answer of 0.046.
    assert distance < 0.046


@pytest.mark.parametrize("qty", [1.0, -1.0], ids=["long", "short"])
def test_amply_funded_account_falls_back_to_the_isolated_estimate(qty):
    """The other side of `nearest`: cross must not be trusted alone either.

    With equity far exceeding the position, the cross formula prices liquidation at an
    absurd distance (-903 for a long) because it cannot see other positions' own
    maintenance requirements or the venue's risk tiers. Taking the nearer of the two keeps
    the isolated bound in exactly that case.
    """
    env = _futures_env_stub(
        _open_position(qty, leverage=20, margin_mode="cross"),
        {"total_margin_balance": 1_000_000.0, "total_maintenance_margin": 0.4},
    )
    obs = TorchTradeFuturesLiveEnv._get_observation(env)

    assert obs["account_state"][5].item() == pytest.approx(0.046, rel=1e-3)


@pytest.mark.parametrize("kwargs,match", [
    ({"mark_price": 0.0}, "mark_price must be positive"),
    ({"mark_price": float("nan")}, "mark_price must be positive"),
    ({"position_size": 0.0}, "flat position has no liquidation price"),
    ({"equity": 0.0}, "equity must be positive"),
    ({"total_account_maintenance": float("inf")}, "finite non-negative"),
    ({"maintenance_margin_rate": 1.0}, "degenerate denominator"),
], ids=["zero-mark", "nan-mark", "flat", "zero-equity", "infinite-maintenance", "degenerate"])
def test_cross_estimate_refuses_inputs_it_cannot_price(kwargs, match):
    """Same contract as the isolated helper: never hand back a number that reads as safe."""
    args = {
        "position_size": 1.0,
        "mark_price": 100.0,
        "equity": 5.0,
        "total_account_maintenance": 0.4,
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        cross_liquidation_price(**args)


@pytest.mark.parametrize("qty,expected_price", [
    (1.0, 99.0 / 0.996),
    (-1.0, 101.0 / 1.004),
], ids=["long", "short"])
def test_cross_estimate_includes_other_account_maintenance(qty, expected_price):
    """#344: other obligations move both long and short liquidation nearer the mark."""
    env = _futures_env_stub(
        _open_position(qty, margin_mode="cross"),
        {"total_margin_balance": 5.0, "total_maintenance_margin": 4.4},
    )

    distance = TorchTradeFuturesLiveEnv._get_observation(env)["account_state"][5].item()

    assert distance == pytest.approx(abs(expected_price - 100.0) / 100.0, rel=1e-5)


@pytest.mark.parametrize("qty", [1.0, -1.0], ids=["long", "short"])
def test_no_other_obligation_matches_single_position_cross_formula(qty):
    """When aggregate maintenance equals the focal term, #344 reduces to #342."""
    expected_price = (qty * 100.0 - 5.0) / (qty - 0.004 * abs(qty))
    actual = cross_liquidation_price(
        position_size=qty,
        mark_price=100.0,
        equity=5.0,
        total_account_maintenance=0.4,
    )
    assert actual == pytest.approx(expected_price)


@pytest.mark.parametrize("maintenance", [None, float("nan")], ids=["missing", "nan"])
def test_cross_position_without_usable_aggregate_maintenance_fails_closed(maintenance):
    env = _futures_env_stub(
        _open_position(1.0, margin_mode="cross"),
        {"total_margin_balance": 5.0, "total_maintenance_margin": maintenance},
    )

    with pytest.raises(ValueError, match="maintenance"):
        TorchTradeFuturesLiveEnv._get_observation(env)


def test_cross_position_without_aggregate_maintenance_key_fails_closed():
    env = _futures_env_stub(
        _open_position(1.0, margin_mode="cross"),
        {"total_margin_balance": 5.0},
    )

    with pytest.raises(KeyError, match="total_maintenance_margin"):
        TorchTradeFuturesLiveEnv._get_observation(env)


def test_isolated_missing_liquidation_price_does_not_require_account_maintenance():
    env = _futures_env_stub(
        _open_position(1.0, margin_mode="isolated"),
        {"total_margin_balance": 5.0},
    )
    distance = TorchTradeFuturesLiveEnv._get_observation(env)["account_state"][5].item()
    assert distance == pytest.approx(0.046, rel=1e-5)


@pytest.mark.parametrize("field,label", [
    ("margin_mode", "CROSSED"),  # Binance
    ("margin_mode", "cross"),    # OKX
    ("margin_mode", "crossed"),  # Bitget
], ids=["binance", "okx", "bitget"])
def test_concrete_adapter_cross_labels_route_to_account_aware_estimate(field, label):
    position = _open_position(1.0)
    delattr(position, "margin_mode")
    setattr(position, field, label)
    env = _futures_env_stub(
        position,
        {"total_margin_balance": 5.0, "total_maintenance_margin": 4.4},
    )

    distance = TorchTradeFuturesLiveEnv._get_observation(env)["account_state"][5].item()

    assert distance == pytest.approx(abs(99.0 / 0.996 - 100.0) / 100.0, rel=1e-5)


@pytest.mark.parametrize("margin_mode", [None, "portfolio", "NEW_MODE"])
def test_blank_liquidation_price_with_unknown_margin_mode_fails_closed(margin_mode):
    env = _futures_env_stub(
        _open_position(1.0, margin_mode=margin_mode),
        {"total_margin_balance": 5.0, "total_maintenance_margin": 4.4},
    )

    with pytest.raises(ValueError, match="margin mode"):
        TorchTradeFuturesLiveEnv._get_observation(env)


def test_native_liquidation_price_remains_authoritative_for_cross_positions():
    env = _futures_env_stub(
        _open_position(1.0, liquidation_price=98.0, margin_mode="cross"),
        {"total_margin_balance": 5.0, "total_maintenance_margin": None},
    )
    distance = TorchTradeFuturesLiveEnv._get_observation(env)["account_state"][5].item()
    assert distance == pytest.approx(0.02)


@pytest.mark.parametrize("leverage,match", [
    (0, "leverage must be at least 1"),
    (-3, "leverage must be at least 1"),
    (0.5, "leverage must be at least 1"),
    (float("nan"), "non-finite leverage"),
], ids=["zero", "negative", "fractional", "nan"])
def test_nonsense_venue_leverage_refuses_to_report_a_distance(leverage, match):
    """Reachable, not theoretical: OKX blanks `lever` on cross positions.

    Its adapter then substitutes the CONFIG leverage, so an env left at the default 1x
    against an account still on 20x arrives here with leverage 1 and no liquidation
    price. Written `leverage <= 1`, the no-liquidation gate answered "maximally safe" for
    that, and for 0, -3 and 0.5 besides -- #277 one field over. Offline refuses the same
    inputs in __post_init__; live now refuses them here.
    """
    env = _futures_env_stub(_open_position(1.0, leverage=leverage), {"total_margin_balance": 1000.0})
    with pytest.raises(ValueError, match=match):
        TorchTradeFuturesLiveEnv._get_observation(env)


def test_held_position_with_no_reported_notional_still_refuses_zero_equity():
    """The equity raise must key on the position, not on the notional backing it.

    Keyed on `position_value > 0`, a venue that omitted the notional zeroed the second
    conjunct and skipped the raise -- and binance defaulted exactly that field to 0 until
    this PR. position_direction is the authoritative signal, and it goes through the dust
    rule.
    """
    position = _open_position(1.0)
    position.notional_value = 0.0
    env = _futures_env_stub(position, {"total_margin_balance": 0.0})

    with pytest.raises(ValueError, match="refusing to report this account as flat"):
        TorchTradeFuturesLiveEnv._get_observation(env)


@pytest.mark.parametrize("is_long", [True, False], ids=["long", "short"])
def test_one_x_with_a_published_liq_price_deliberately_beats_offline(is_long):
    """A known, accepted divergence -- recorded so it is a decision and not a surprise.

    At 1x the offline env short-circuits on has_liquidation and reports 1.0. Live, when
    the venue actually publishes a price for a 1x position, uses it: 0.996, because a 1x
    long really is liquidated near zero rather than never. Live is the more accurate of
    the two by 0.4%, so this is not forced into parity -- but it does mean
    test_live_distance_to_liquidation_agrees_with_the_offline_env speaks only to the
    no-published-price case.
    """
    qty = 1.0 if is_long else -1.0
    published = 0.4 if is_long else 199.6
    env = _futures_env_stub(
        _open_position(qty, leverage=1, liquidation_price=published),
        {"total_margin_balance": 1000.0},
    )
    obs = TorchTradeFuturesLiveEnv._get_observation(env)

    assert obs["account_state"][5].item() == pytest.approx(0.996, rel=1e-4)


@pytest.mark.parametrize("field", [
    "mark_price", "liquidation_price", "qty",
    # These three reach account_state directly. notional_value is the worst of the set:
    # exposure_pct = nan/equity puts NaN into the tensor handed to the policy, where a
    # NaN liquidation price at least clamped to a number first.
    "notional_value", "unrealized_pnl_pct", "entry_price",
])
def test_non_finite_venue_numbers_refuse_to_build_an_account_state(field):
    """NaN defeats every comparison downstream, so it is caught once at the source.

    A NaN liquidation price passes `<= 0`, skips the fallback, reaches the arithmetic and
    clamps to a distance of 0.0 -- a healthy 20x position reported as AT liquidation on
    one garbage tick. Fail-closed, but silently, and on the wrong side of the truth.
    """
    position = _open_position(1.0, leverage=20)
    setattr(position, field, float("nan"))
    env = _futures_env_stub(position, {"total_margin_balance": 1000.0})

    # qty is caught upstream, by the dust rule, before this loop runs -- it has to be,
    # because the live _steps sync and trade on the direction it returns.
    with pytest.raises(ValueError, match=f"non-finite ({field}|position quantity)"):
        TorchTradeFuturesLiveEnv._get_observation(env)


def test_no_adapter_fabricates_zero_equity():
    """Indexing `total_margin_balance` only helps if the adapter doesn't invent the value.

    Three of the four coerced a missing or blank venue field to 0.0 and published it under
    the key -- `.get('total', 0)`, `.get("totalEquity", 0)`, `.get("totalEq") or "0"` --
    so every downstream guard saw a well-formed 0 rather than an error: the bankruptcy
    baseline became 0 (and `current < threshold * 0` never fires), sizing refused to
    trade, exposure_pct read flat. That is #277's mechanism surviving one layer below the
    code hardened for it.

    Structural rather than behavioural: reaching these lines needs a faithful fake of
    three different venue clients, and a wrong fake proves nothing about the adapter. The
    guard is on the coercion itself, which is the thing that must not come back.

    Note for anyone writing the behavioural version: all three adapters wrap their body in
    `except Exception as e: raise RuntimeError(...) from e`, so the ValueError surfaces to
    the caller as a RuntimeError with the message preserved. It still raises rather than
    returning a fabricated balance, which is the property that matters here.
    """
    live_root = pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent / "live"
    equity_key = r"""["'](total|totalEquity|totalEq|totalMarginBalance)["']"""
    pattern = re.compile(
        rf"""\.get\(\s*{equity_key}\s*,\s*0"""       # .get("totalEquity", 0)
        rf"""|\.get\(\s*{equity_key}\s*\)\s*or\s*["']?0"""  # .get("totalEq") or "0"
    )
    scanned, offenders = 0, []
    for path in sorted(live_root.rglob("order_executor.py")):
        scanned += 1
        for i, line in enumerate(path.read_text().splitlines(), 1):
            # Comment lines are skipped so a comment explaining the banned pattern (there
            # is one, at the bitget fix) does not read as the pattern itself.
            if not line.lstrip().startswith("#") and pattern.search(line):
                offenders.append(f"{path.relative_to(live_root.parent)}:{i}")

    assert scanned >= 4, f"guard scanned only {scanned} executors under {live_root}"
    assert offenders == [], (
        f"equity coerced to zero at {offenders}; raise on a missing or blank venue field "
        f"so it fails where it is diagnosable, not as a plausible 0 three layers down"
    )


@pytest.mark.parametrize("equity,reason", [
    (0.0, "an empty account"),
    (-50.0, "negative equity"),
    (float("nan"), "a non-finite reading"),
    (float("inf"), "a non-finite reading"),
], ids=["zero", "negative", "nan", "inf"])
def test_a_bankruptcy_baseline_that_would_never_fire_is_refused(equity, reason):
    """`is_bankrupt` is `current < threshold * initial`, so a baseline of 0 reduces to
    `current < 0` -- it never fires above zero equity and a wiped account trades on.

    Behavioural now that #345 hoisted this into TorchTradeLiveEnv; it used to be a
    structural grep over four per-exchange copies.
    """
    env = SimpleNamespace(
        trader=SimpleNamespace(
            get_account_balance=lambda: {"total_margin_balance": equity}
        )
    )
    # The real reader, bound to the stand-in -- stubbing it would test nothing.
    env._get_portfolio_value = lambda: TorchTradeFuturesLiveEnv._get_portfolio_value(env)

    with pytest.raises(ValueError):
        TorchTradeFuturesLiveEnv._capture_bankruptcy_baseline(env)


@pytest.mark.parametrize("exchange", ["alpaca", "binance", "bitget", "bybit", "okx"])
def test_every_live_env_delegates_its_baseline(exchange):
    """Alpaca included, not exempted -- excluding it is how the bug survived last time."""
    src = (pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent
           / "live" / exchange / "base.py").read_text()
    # `_finish_futures_init()` calls it: as of #288 the four futures venues share that
    # tail instead of each spelling the call out. Alpaca still calls it directly.
    assert ("_capture_bankruptcy_baseline()" in src
            or "_finish_futures_init()" in src), (
        f"{exchange} neither captures the baseline nor delegates to the shared tail"
    )
    # The load-bearing half, unchanged: delegating is the point, re-forking the READ is
    # the bug. This is what actually caught it last time.
    assert 'balance["total_margin_balance"]' not in src, (
        f"{exchange} re-forked the baseline read instead of delegating"
    )


@pytest.mark.parametrize("exchange", ["binance", "bitget", "bybit", "okx"])
def test_adapters_do_not_truncate_fractional_leverage(exchange):
    """`int(float("1.5"))` is 1, and 1 takes the no-liquidation branch.

    bybit and okx coerced venue leverage with `int(float(...))`, so any leverage in
    [1, 2) arrived as exactly 1 -- and the `== 1` gate then handed that position the
    maximally-safe distance of 1.0, on the very cross-margin path the fallback exists
    for. #277 surviving inside the branch written to stop it.
    """
    src = (pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent
           / "live" / exchange / "order_executor.py").read_text()
    assert "leverage=int(" not in src, (
        f"{exchange} truncates venue leverage to int; 1.5x becomes 1x and reads as unlevered"
    )


@pytest.mark.parametrize("qty", [1.0, -1.0], ids=["long", "short"])
def test_fractional_leverage_still_gets_a_real_distance(qty):
    """The behavioural half of the above: 1.5x must not read as maximally safe."""
    env = _futures_env_stub(
        _open_position(qty, leverage=1.5), {"total_margin_balance": 1000.0}
    )
    distance = TorchTradeFuturesLiveEnv._get_observation(env)["account_state"][5].item()

    assert distance == pytest.approx(1 / 1.5 - 0.004, rel=1e-4)
    assert distance < 1.0


def test_no_adapter_swaps_a_venue_reported_zero_leverage_for_the_config():
    """`float(pos.get("leverage") or self.leverage)` swaps a venue-reported 0 for the config.

    A numeric 0 is falsy, so `or` silently substitutes a leverage the venue never
    confirmed and the position gets a plausible liquidation distance computed from it --
    #277 on the field that was being hardened. bitget reads ccxt's unified `leverage`,
    which is a number rather than a REST string, so `0` arrives falsy there today.

    Structural because the behavioural cells construct PositionStatus directly and so
    cannot see the adapter's parsing at all -- which is why this regression survived a
    mutation run that killed everything else.

    Scope, stated so this is not mistaken for more than it is: a MISSING or BLANK venue
    leverage still falls back to the config value, deliberately, because refusing would
    leave OKX and bybit cross accounts unable to produce an observation at all. That
    fallback is only as good as set_leverage having worked, which #277 now verifies at
    construction. What this forbids is the silent case -- a leverage the venue did report, as 0,
    being swapped for a different number.
    """
    live_root = pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent / "live"
    offenders = [
        f"{path.relative_to(live_root.parent)}:{i}"
        for path in sorted(live_root.rglob("order_executor.py"))
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if not line.lstrip().startswith("#") and "or self.leverage" in line
    ]
    assert offenders == [], (
        f"venue leverage falls back to the config value via `or` at {offenders}; test "
        f"`in (None, \"\")` so a genuine 0 stays 0 and fails loudly downstream"
    )


def test_flat_account_with_non_finite_equity_still_raises():
    """The held-position guard is keyed on direction, so a flat account bypasses it.

    That matters because the same equity reaches is_bankrupt(), where
    `nan < threshold * initial` is False -- bankruptcy silently disabled for the rest of
    the episode, on an account whose equity the venue could not report.
    """
    env = _futures_env_stub(None, {"total_margin_balance": float("nan")})
    with pytest.raises(ValueError, match="non-finite equity"):
        TorchTradeFuturesLiveEnv._get_observation(env)


def _alpaca_env_stub(position_status, cash):
    """Stand-in for AlpacaBaseTorchTradingEnv._get_observation. Spot, so no leverage."""
    from torchtrade.envs.live.alpaca.base import AlpacaBaseTorchTradingEnv  # noqa: F401

    return SimpleNamespace(
        observer=SimpleNamespace(
            get_observations=lambda **k: {}, get_keys=lambda: [],
            get_current_price=lambda: 100.0,
        ),
        trader=SimpleNamespace(
            get_status=lambda: {"position_status": position_status},
            client=SimpleNamespace(get_account=lambda: SimpleNamespace(cash=str(cash))),
        ),
        config=SimpleNamespace(include_base_features=False),
        position=PositionState(),
        account_state_key="account_state",
        market_data_keys=[],
    )


def _alpaca_position(qty=1.0, market_value=1000.0, avg_entry_price=100.0,
                     current_price=102.0, unrealized_plpc=0.02):
    return SimpleNamespace(
        qty=qty, market_value=market_value, avg_entry_price=avg_entry_price,
        current_price=current_price, unrealized_plpc=unrealized_plpc,
    )


@pytest.mark.parametrize("field", [
    "qty", "market_value", "avg_entry_price", "current_price", "unrealized_plpc",
])
def test_alpaca_non_finite_venue_numbers_refuse_to_build_an_account_state(field):
    """Spot is not exempt from any of this (#277).

    Alpaca was the one live env left without the guards after the futures four got them --
    the same shape as the bankruptcy baseline, where the guard excluded alpaca by name and
    so skipped the one env still carrying the bug. A NaN unrealized_plpc here goes
    straight into the observation tensor and on into the policy network.
    """
    from torchtrade.envs.live.alpaca.base import AlpacaBaseTorchTradingEnv

    position = _alpaca_position()
    setattr(position, field, float("nan"))
    env = _alpaca_env_stub(position, cash=10000.0)

    # qty is caught upstream, by the dust rule, before this loop runs -- it has to be,
    # because the live _steps sync and trade on the direction it returns.
    with pytest.raises(ValueError, match=f"non-finite ({field}|position quantity)"):
        AlpacaBaseTorchTradingEnv._get_observation(env)


def test_alpaca_position_held_against_wiped_portfolio_refuses_to_report_flat():
    """Invariant #3 on the spot env: `portfolio_value > 0 else 0.0` read a held position
    as flat whenever margin debt exceeded the position's market value."""
    from torchtrade.envs.live.alpaca.base import AlpacaBaseTorchTradingEnv

    env = _alpaca_env_stub(_alpaca_position(market_value=1000.0), cash=-1000.0)

    with pytest.raises(ValueError, match="refusing to report this account as flat"):
        AlpacaBaseTorchTradingEnv._get_observation(env)


@pytest.mark.parametrize("qty", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "inf", "-inf"])
def test_dust_rule_refuses_a_non_finite_quantity(qty):
    """A NaN qty read as a SHORT, upstream of every finiteness check in the envs.

    `abs(nan) <= eps` is False and `nan > 0` is False, so it fell through to -1. That
    matters more than a bad observation: alpaca's _step syncs the position and executes a
    trade on this direction BEFORE building an observation, so a fabricated short drove a
    trade decision on a long-only spot account.
    """
    with pytest.raises(ValueError, match="non-finite position quantity"):
        position_direction_from_status(SimpleNamespace(qty=qty))
    with pytest.raises(ValueError, match="non-finite position quantity"):
        position_qty_from_status(SimpleNamespace(qty=qty))


def test_portfolio_value_refuses_non_finite_equity():
    """`_get_portfolio_value` is a SECOND fetch and the literal argument to
    _check_termination, so guarding only the observation's read leaves the two able to
    disagree -- a NaN here alone reaches is_bankrupt(), where it is silently False."""
    env = SimpleNamespace(
        trader=SimpleNamespace(
            get_account_balance=lambda: {"total_margin_balance": float("nan")}
        )
    )
    with pytest.raises(ValueError, match="non-finite equity"):
        TorchTradeFuturesLiveEnv._get_portfolio_value(env)


@pytest.mark.parametrize("cash", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_alpaca_non_finite_cash_refuses_to_build_an_account_state(cash):
    """cash IS alpaca's equity -- the whole of portfolio_value when flat, and the
    bankruptcy baseline. Guarded on a FLAT account, where the held-position raise (keyed
    on direction) cannot see it. `+inf` is included because `not (x > 0)` passes it."""
    from torchtrade.envs.live.alpaca.base import AlpacaBaseTorchTradingEnv

    env = _alpaca_env_stub(None, cash=cash)
    with pytest.raises(ValueError, match="non-finite cash balance"):
        AlpacaBaseTorchTradingEnv._get_observation(env)


@pytest.mark.parametrize("price", ["nan", "inf", "-inf", "-0.5", "1.5"],
                         ids=["nan", "inf", "-inf", "negative", "above-one"])
def test_polymarket_refuses_a_price_that_is_not_a_probability(price):
    """`_compute_payoff` guards `fill_price <= 0`, which NaN and +inf compare False to.

    A garbage price then flows into self.cash -- polymarket's equity and its only
    bankruptcy input. Once cash is NaN it stays NaN, is_bankrupt is False for the rest of
    the episode, and every reward is NaN.
    """
    from torchtrade.envs.live.polymarket.market_scanner import _valid_price

    with pytest.raises(ValueError, match="not a probability"):
        _valid_price(price, "yes_price", "some-market")


@pytest.mark.parametrize("price", ["0", "0.0", "1", "1.0", "0.5"])
def test_polymarket_accepts_the_endpoints_of_the_probability_range(price):
    """0 and 1 are legitimate for a near-resolved market, and rejecting them would have
    broken scanning on live data -- the per-market handler did not catch ValueError, so
    one such market would have aborted the entire scan rather than being skipped."""
    from torchtrade.envs.live.polymarket.market_scanner import _valid_price

    assert _valid_price(price, "yes_price", "some-market") == float(price)


@pytest.mark.parametrize("field,value", [
    ("cash", float("nan")),
    ("cash", float("inf")),
    ("market_value", float("nan")),
], ids=["nan-cash", "inf-cash", "nan-market-value"])
def test_alpaca_refuses_to_size_a_trade_from_a_non_finite_read(field, value):
    """The worst path in this PR: these reads size a REAL order.

    A NaN makes delta_qty NaN; `nan > 0` is False so the caller always takes the SELL
    branch; and in trade_mode="notional" a sell is intercepted as close_position(). One
    garbage tick would liquidate the entire position. `total_portfolio <= 0` cannot see
    that, which is why the check is on finiteness.
    """
    from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv

    cash = value if field == "cash" else 10000.0
    market_value = value if field == "market_value" else 1000.0
    env = SimpleNamespace(
        trader=SimpleNamespace(
            client=SimpleNamespace(get_account=lambda: SimpleNamespace(cash=str(cash))),
            get_status=lambda: {"position_status": SimpleNamespace(
                qty=10.0, market_value=market_value)},
        ),
    )
    with pytest.raises(ValueError, match="refusing to size a trade against it"):
        AlpacaTorchTradingEnv._calculate_fractional_position(env, 1.0, 100.0)


@pytest.mark.parametrize("cash,market_value,expected", [
    ("nan", None, "unusable cash balance"),
    ("-250", None, "unusable cash balance"),
    ("1000", float("nan"), "non-finite portfolio value"),
], ids=["nan-cash", "negative-cash", "nan-market-value"])
def test_alpaca_portfolio_value_refuses_a_non_finite_read(cash, market_value, expected):
    """alpaca's _step calls this BEFORE building an observation, and its value feeds the
    reward and _check_termination -- so the observation guard can never catch it.

    Two guards, and the rows prove BOTH are live: cash is now refused at the read (#347,
    because the SLTP env sizes off it directly), while market_value still arrives raw and
    can only be caught here.
    """
    from torchtrade.envs.live.alpaca.base import AlpacaBaseTorchTradingEnv

    position = None if market_value is None else SimpleNamespace(market_value=market_value)
    env = SimpleNamespace(
        trader=SimpleNamespace(
            get_status=lambda: {"position_status": position},
            client=SimpleNamespace(get_account=lambda: SimpleNamespace(cash=cash)),
        ),
    )
    env._read_cash = lambda: AlpacaBaseTorchTradingEnv._read_cash(env)
    with pytest.raises(ValueError, match=expected):
        AlpacaBaseTorchTradingEnv._get_portfolio_value(env)


@pytest.mark.parametrize("field", ["volume24hr", "volume", "liquidity", "spread"])
def test_polymarket_refuses_non_finite_market_numbers(field):
    """Three of these four are the market half of the observation tensor, and
    `_filter_markets` compares them with `<`/`>`, which NaN passes -- so garbage survives
    filtering and reaches the policy network."""
    from torchtrade.envs.live.polymarket.market_scanner import _finite

    with pytest.raises(ValueError, match="not a finite non-negative number"):
        _finite("nan", field, "some-market")


def test_alpaca_sizing_refuses_a_non_finite_position_quantity():
    """Alpaca hand-rolled `float(position_status.qty)` here, bypassing the dust rule.

    The dust half of that is absorbed downstream -- a 1e-12 difference in delta_qty loses
    to the $1 minimum-trade tolerance -- so this asserts the half that survives: a NaN qty
    used to flow into `delta_qty = target_qty - current_qty`, and `nan > 0` is False, so
    the trade fell to the SELL branch. In trade_mode="notional" a sell is intercepted as
    close_position(). Going through position_qty_from_status raises instead.
    """
    from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv

    submitted = []
    env = SimpleNamespace(
        trader=SimpleNamespace(
            get_status=lambda: {"position_status": SimpleNamespace(
                qty=float("nan"), market_value=1000.0, current_price=100.0)},
            trade=lambda **kw: submitted.append(kw) or True,
        ),
        _get_current_price=lambda ps: 100.0,
        _calculate_fractional_position=lambda *a, **k: (1000.0, 10.0),
    )
    with pytest.raises(ValueError, match="non-finite position quantity"):
        AlpacaTorchTradingEnv._execute_fractional_action(env, 1.0)
    assert submitted == [], f"an order was submitted from a NaN quantity: {submitted}"


@pytest.mark.parametrize("balance", [0.0, -5.0, float("nan"), float("inf")],
                         ids=["zero", "negative", "nan", "inf"])
def test_an_unusable_balance_cannot_size_a_trade(balance):
    """The guard BEHAVIOURALLY. Its sibling below only greps the source for two strings.

    Four mutations kept both grep strings intact and passed:
      `isfinite(x) and x <= 0`   NaN and +inf now size a trade -- this IS #277
      `x <= 0` -> `x < 0`        ZERO equity accepted, which is what a venue reports
                                 while liquidating you -- the moment the comment on the
                                 guard says is the worst one to keep trading through
      `raise` -> `logger.warning` nothing is ever rejected
      guard hoisted out of the closure   the #295 regression, named in that same comment

    Raising LiveObservationHalt rather than ValueError is the third and fourth of those:
    `_halting` converts the ValueError only if it is raised INSIDE the closure it wraps.
    A guard hoisted one frame up sends a bare ValueError out of `_step`.
    """
    env, trader = _real_futures_env(budget=0, venue="binance")
    trader.get_account_balance.return_value = {
        "available_balance": balance, "total_margin_balance": balance,
        "total_wallet_balance": balance, "total_maintenance_margin": 0.0,
    }
    with pytest.raises(LiveObservationHalt):
        env._calculate_fractional_position(1.0, 100.0)


def test_the_sizing_uses_the_whole_portfolio_not_the_free_margin():
    """total_margin_balance, not available_balance.

    available_balance shrinks as positions grow, so a held action=1.0 re-sized against it
    buys again every bar. A `.get("available_balance", total)` fallback survives every
    other test in this file: with the two values equal in the fixture, nothing can tell
    which one was read.
    """
    env, trader = _real_futures_env(budget=0, venue="binance")
    trader.get_account_balance.return_value = {
        "available_balance": 1_000.0,          # what is free right now
        "total_margin_balance": 10_000.0,      # what the portfolio is worth
        "total_wallet_balance": 10_000.0, "total_maintenance_margin": 0.0,
    }
    _, notional, _ = env._calculate_fractional_position(1.0, 100.0)

    assert notional > 5_000.0, (
        f"sized a notional of {notional:.2f} against a 10,000 portfolio: that is the free "
        f"margin, and sizing against it makes a held position re-buy every bar"
    )


@pytest.mark.parametrize("exchange", ["binance", "bitget", "bybit", "okx"])
def test_futures_sizing_rejects_a_non_finite_balance(exchange):
    """`not (x > 0)` catches NaN but passes +inf, and these four lines are this PR's own.

    An inf balance sizes an inf target: bitget's amount rounding then yields NaN and hands
    it to create_order, while binance and bybit raise an undiagnosable OverflowError
    mid-step. The alpaca sibling written in the same commit is safe only because it
    isfinite-checks its inputs first.
    """
    # The RESOLVED method, following one hop through `super()`. A file scan of
    # `live/<venue>/env.py` broke the moment #288 folded the sizing onto the shared base:
    # the guard could no longer see its own subject, which is the failure it exists to
    # prevent in the code. binance still overrides, so its own body is scanned too.
    cls = _sole(importlib.import_module(f"torchtrade.envs.live.{exchange}.env"),
                "TorchTradingEnv")
    src = inspect.getsource(cls._calculate_fractional_position)
    if "super()._calculate_fractional_position" in src:
        src += inspect.getsource(
            TorchTradeFuturesLiveEnv._calculate_fractional_position
        )
    assert "if not (total_balance > 0):" not in src, (
        f"{exchange} sizes against a balance guarded by `not (x > 0)`, which +inf passes"
    )
    assert "math.isfinite(total_balance)" in src, (
        f"{exchange} does not check that the sizing balance is finite"
    )


@pytest.mark.parametrize("source", ["fetched", "position"])
@pytest.mark.parametrize("price", [float("nan"), float("inf"), float("-inf"), 0.0, -100.0],
                         ids=["nan", "inf", "-inf", "zero", "negative"])
def test_an_unusable_mark_price_cannot_size_a_trade(source, price):
    """#347: the sizing paths divide by this, and `<= 0` cannot see NaN.

    A NaN price makes delta_qty NaN; `nan > 0` is False, so the trade falls to the SELL
    branch -- which in some trade modes is intercepted as a full close. A negative one
    flips the sign outright. Both SOURCES matter: an open position's mark took the same
    money path as the fetched one but went unvalidated in 7 of 8 envs, reaching
    history.record_step and the reward. Every venue reads it as
    `float(pos.get("markPrice") or entry_price)`, so two blank fields yield 0.0.
    """
    env = SimpleNamespace(trader=SimpleNamespace(get_mark_price=lambda: price))
    position_status = SimpleNamespace(mark_price=price) if source == "position" else None

    with pytest.raises(ValueError, match="unusable mark price"):
        TorchTradeFuturesLiveEnv._current_mark_price(env, position_status)


@pytest.mark.parametrize("venue_error", [
    pytest.param(RuntimeError("venue unreachable"), id="wrapped"),
    pytest.param(ConnectionError("socket closed"), id="raw-sdk"),
    pytest.param(KeyError("markPrice"), id="malformed-payload"),
])
def test_a_mark_price_that_cannot_be_read_raises_what_halting_catches(venue_error):
    """#394: the mark-price fetch raised types `_halting` misses, bypassing the policy.

    `_halting` catches (PositionUnknownError, ValueError) and deliberately not
    RuntimeError, which adapters use for timeouts. Failing to READ the mark is the same
    event as reading an unusable one above, so it must raise the same way. The three
    shapes are three exception families: narrowing the catch to RuntimeError lets two
    escape, to (RuntimeError, OSError) lets one.
    """
    env = SimpleNamespace(
        trader=SimpleNamespace(get_mark_price=MagicMock(side_effect=venue_error)),
        config=SimpleNamespace(symbol="BTCUSDT"),
    )
    with pytest.raises(ValueError, match="could not read the mark price"):
        TorchTradeFuturesLiveEnv._current_mark_price(env, None)


@pytest.mark.parametrize("exchange", ["binance", "bitget", "bybit", "okx"])
@pytest.mark.parametrize("module", ["env", "env_sltp"])
def test_no_futures_env_reads_the_mark_price_unvalidated(exchange, module):
    """The wiring: 13 call sites across 8 files, and the helper alone proves none of them."""
    path = (pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent
            / "live" / exchange / f"{module}.py")
    if not path.exists():
        pytest.skip(f"{exchange} has no {module}")
    src = path.read_text()
    assert "self.trader.get_mark_price()" not in src, (
        f"{exchange}/{module} sizes from an unvalidated mark price"
    )
    # The position's own mark is the same number by another route -- okx was the only env
    # that ever validated it, and it feeds record_step and the reward in all of them.
    assert "position_status.mark_price" not in src, (
        f"{exchange}/{module} reads the position mark raw instead of via "
        f"_current_mark_price(position_status)"
    )


@pytest.mark.parametrize("equity", [0.0, -50.0], ids=["empty", "negative"])
def test_alpaca_refuses_to_size_rather_than_returning_a_flat_target(equity):
    """#348: "I cannot size this" returned the same tuple as "go flat".

    Alpaca has no `target_qty == 0` guard, so a zero target against an open position
    becomes `delta = 0 - current` and SELLS -- from an action that meant maximum long.
    """
    from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv

    env = SimpleNamespace(trader=SimpleNamespace(
        client=SimpleNamespace(get_account=lambda: SimpleNamespace(cash=str(equity))),
        get_status=lambda: {"position_status": None},
    ))
    with pytest.raises(ValueError):
        AlpacaTorchTradingEnv._calculate_fractional_position(env, 1.0, 100.0)


def _bitget_status(**pos_overrides):
    """Real bitget adapter over a stubbed ccxt client."""
    from torchtrade.envs.live.bitget.order_executor import BitgetFuturesOrderClass

    pos = {"symbol": "BTC/USDT:USDT", "contracts": 1.0, "side": "long",
           "entryPrice": 100.0, "markPrice": 101.0}
    pos.update(pos_overrides)
    client = MagicMock()
    client.set_leverage = MagicMock(return_value=_LEVERAGE_BODIES["bitget"])
    client.fetch_positions = MagicMock(return_value=[pos])
    client.load_markets = MagicMock(return_value={})
    client.markets = {}
    ex = BitgetFuturesOrderClass(symbol="BTC/USDT:USDT", trade_mode="quantity",
                                 demo=True, leverage=10, client=client)
    return ex.get_status().get("position_status")


@pytest.mark.parametrize("field", ["liquidationPrice", "entryPrice", "markPrice",
                                   "unrealizedPnl", "notional"])
def test_a_null_venue_field_does_not_turn_a_healthy_position_into_an_outage(field):
    """#341: bare `float(None)` raised TypeError into bitget's broad except, so a cross
    position with `liquidationPrice: None` reported POSITION_UNKNOWN.

    That is the worst shape available: POSITION_UNKNOWN is deliberately fail-closed, so a
    perfectly healthy position froze the env every bar -- on a lie.
    """
    status = _bitget_status(**{field: None})

    assert status is not POSITION_UNKNOWN, f"a null {field} read as an outage"
    assert status.qty == pytest.approx(1.0)


@pytest.mark.parametrize("side", [None, "", "unexpected", "SHORT"])
def test_an_unusable_side_reads_as_unknown_not_as_a_direction(side):
    """`contracts if side == 'long' else -contracts` signed a long as a SHORT for any
    unexpected value, and every consumer reads that sign."""
    assert _bitget_status(side=side) is POSITION_UNKNOWN


@pytest.mark.parametrize("pos_side", [None, "", "SHORT", "unexpected"])
def test_okx_refuses_an_unusable_posside_instead_of_signing_it_long(pos_side):
    """OKX was the worst of the three and my first pass missed it.

    It reports hedge-mode size as a POSITIVE `pos` with direction only in `posSide`, and
    any unrecognised value fell through to the net-mode branch keeping that positive sign
    -- so a short read as a long, with no error, straight into the trade path. bitget and
    bybit at least degraded to POSITION_UNKNOWN.
    """
    from torchtrade.envs.live.okx.order_executor import OKXFuturesOrderClass

    # account_client, not client: okx splits Trade/Account/PublicData, and injecting only
    # `client` left get_positions hitting the real API -- so the test passed on the API
    # error rather than on the guard, with or without the fix.
    account = MagicMock()
    account.get_positions = MagicMock(return_value={"code": "0", "data": [
        {"instId": "BTC-USDT-SWAP", "pos": "1.0", "posSide": pos_side,
         "avgPx": "100", "markPx": "101", "lever": "10", "mgnMode": "cross"}
    ]})
    account.set_position_mode = MagicMock(return_value={"code": "0"})
    account.set_leverage = MagicMock(return_value=_LEVERAGE_BODIES["okx"])
    public = MagicMock()
    public.get_instruments = MagicMock(return_value={"data": []})

    ex = OKXFuturesOrderClass(
        symbol="BTC-USDT-SWAP", trade_mode="quantity", demo=True, leverage=10,
        api_key="k", api_secret="s", passphrase="p",
        client=MagicMock(), account_client=account, public_client=public,
    )
    assert ex.get_status().get("position_status") is POSITION_UNKNOWN


def test_okx_still_signs_a_recognised_short_negative():
    """The guard must reject the unrecognised, not the legitimate."""
    from torchtrade.envs.live.okx.order_executor import OKXFuturesOrderClass

    account = MagicMock()
    account.get_positions = MagicMock(return_value={"code": "0", "data": [
        {"instId": "BTC-USDT-SWAP", "pos": "1.0", "posSide": "short",
         "avgPx": "100", "markPx": "101", "lever": "10", "mgnMode": "cross"}
    ]})
    account.set_position_mode = MagicMock(return_value={"code": "0"})
    account.set_leverage = MagicMock(return_value=_LEVERAGE_BODIES["okx"])
    public = MagicMock()
    public.get_instruments = MagicMock(return_value={"data": []})

    ex = OKXFuturesOrderClass(
        symbol="BTC-USDT-SWAP", trade_mode="quantity", demo=True, leverage=10,
        api_key="k", api_secret="s", passphrase="p",
        client=MagicMock(), account_client=account, public_client=public,
    )
    assert ex.get_status()["position_status"].qty == pytest.approx(-1.0)


@pytest.mark.parametrize("exchange", ["binance", "bitget", "bybit", "okx"])
def test_sltp_sizing_rejects_a_non_finite_price_or_balance(exchange):
    """#347 stopped at env.py: all four SLTP envs still divided by an unguarded value.

    `nan <= 0` is False, so the guard was transparent to exactly the input the issue is
    about -- and binance/bitget size from a candle close, so the validated mark-price
    accessor never entered their path at all.
    """
    src = (pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent
           / "live" / exchange / "env_sltp.py").read_text()
    assert "math.isfinite(balance)" in src, (
        f"{exchange} SLTP sizing still lets a non-finite balance through"
    )
    if exchange in ("binance", "bitget"):
        # These two size from a candle close, so _current_mark_price() never guards them.
        # bybit and okx go through it, which makes a second price check there dead code.
        assert "math.isfinite(current_price)" in src, (
            f"{exchange} sizes from a candle close with no finiteness check"
        )


@pytest.mark.parametrize("exchange,side_key,side", [
    ("bybit", "side", None), ("bybit", "side", ""), ("bybit", "side", "unexpected"),
    ("okx", "posSide", None), ("okx", "posSide", ""), ("okx", "posSide", "unexpected"),
])
def test_an_emergency_close_refuses_an_unusable_side_instead_of_guessing(
    exchange, side_key, side
):
    """#341's other half: the sweep fixed get_status and stopped before close_position.

    FLATTEN now calls this, so it is on the money path the halt work created. bybit
    defaulted to "Buy" and okx to "net" -- either sends a reduce-only close in the wrong
    direction, which the venue rejects. The operator sees flatten_accepted=False having
    believed the position was closed, so the refusal must be explicit, not a rejection.
    """
    client = MagicMock()
    client.set_leverage = MagicMock(return_value=_LEVERAGE_BODIES[exchange])
    if exchange == "bybit":
        from torchtrade.envs.live.bybit.order_executor import BybitFuturesOrderClass as cls
        client.get_positions = MagicMock(return_value={"retCode": 0, "result": {"list": [
            {"symbol": "BTCUSDT", "size": "1.0", side_key: side}
        ]}})
        ex = cls(symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=10,
                 api_key="k", api_secret="s", client=client)
    else:
        from torchtrade.envs.live.okx.order_executor import OKXFuturesOrderClass as cls
        account = MagicMock()
        account.get_positions = MagicMock(return_value={"code": "0", "data": [
            {"instId": "BTC-USDT-SWAP", "pos": "1.0", side_key: side}
        ]})
        account.set_position_mode = MagicMock(return_value={"code": "0"})
        account.set_leverage = MagicMock(return_value=_LEVERAGE_BODIES["okx"])
        public = MagicMock(); public.get_instruments = MagicMock(return_value={"data": []})
        ex = cls(symbol="BTC-USDT-SWAP", trade_mode="quantity", demo=True, leverage=10,
                 api_key="k", api_secret="s", passphrase="p",
                 client=client, account_client=account, public_client=public)

    assert ex.close_position() is False, "an unusable side must not report a clean close"
    client.place_order.assert_not_called()


@pytest.mark.parametrize("mark_price", [0.0, -100.0], ids=["zero", "negative"])
def test_a_non_positive_mark_on_an_open_position_is_refused(mark_price):
    """Finite is not enough, and this PR is what made it load-bearing.

    The flat branch now writes `current_price = 0.0` as a sentinel, which turned a
    near-dead `current_price == 0` clause into the one that decides
    distance_to_liquidation. A venue mark of exactly 0.0 on a HELD 20x position passes
    the finiteness loop and then reads as 1.0 -- as safe as a flat spot account, which is
    the literal text of #277. A negative one computes (-100-95)/-100 = 1.95, and the
    `max(0.0, ...)` clamp only holds the bottom, so the policy is handed a reading better
    than maximally safe.
    """
    env = _futures_env_stub(
        _open_position(1.0, mark_price=mark_price, liquidation_price=95.0),
        {"total_margin_balance": 5.0, "total_maintenance_margin": 4.4},
    )
    with pytest.raises(ValueError, match="non-positive mark price"):
        TorchTradeFuturesLiveEnv._get_observation(env)

@pytest.mark.parametrize("desired_action,expected", [
    (1.0, 1), (0.5, 1), (0.0, 0), (-0.5, -1), (-1.0, -1),
], ids=["full-long", "partial-long", "flat", "partial-short", "full-short"])
def test_direction_comes_from_the_target_not_the_order_side(desired_action, expected):
    """#276: five envs inferred direction from the ORDER SIDE.

    Under fractional sizing a SELL that trims a long from 1.0 to 0.5 is still a long --
    the `partial-long` row is the one that was wrong, and it recorded -1.
    """
    env = SimpleNamespace(position=PositionState())
    TorchTradeLiveEnv._record_position_after_trade(env, desired_action, {"executed": True})

    assert env.position.current_position == expected
    assert env.position.current_action_level == desired_action


@pytest.mark.parametrize("exchange", ["alpaca", "binance", "bitget", "bybit", "okx"])
@pytest.mark.parametrize("module", ["env", "env_sltp"])
def test_no_live_env_infers_direction_from_the_order_side(exchange, module):
    """Five copies drifted into four different spellings of the same wrong idea.

    binance's read `side == "BUY"` (uppercase) with its `closed_position` branch LAST,
    behind an `elif` that already matched -- so binance could never record a close at all.
    """
    src = (pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent
           / "live" / exchange / f"{module}.py").read_text()
    # The mixin too: `_record_sltp_position` and `_sync_position_from_exchange` live there
    # and serve all four SLTP venues, so a venue-file-only scan polices the copies that no
    # longer hold the code. Reintroducing this pattern in the mixin passed 20/20 cases.
    if module == "env_sltp":
        src += (pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent
                / "utils" / "sltp_mixin.py").read_text()
    assert 'trade_info["side"] ==' not in src, (
        f"{exchange}/{module} still infers position direction from the order side"
    )


@pytest.mark.parametrize("position_price,fallback,expected", [
    (float("nan"), 100.0, 100.0),
    (float("inf"), 100.0, 100.0),
    (0.0, 100.0, 100.0),
    (float("nan"), float("nan"), 0.0),
    (50.0, 100.0, 50.0),
], ids=["nan-falls-back", "inf-falls-back", "zero-falls-back", "chain-exhausted", "healthy"])
def test_alpaca_price_fallback_is_not_transparent_to_nan(position_price, fallback, expected):
    """#349: `current_price <= 0` is False for NaN, so a NaN skipped BOTH fallbacks.

    The function whose entire purpose is producing a usable price handed back the exact
    value it exists to avoid, and the last row pins the other half: once the chain is
    exhausted it must return the documented 0.0 rather than the garbage it collected.
    """
    from torchtrade.envs.live.alpaca.base import AlpacaBaseTorchTradingEnv

    env = SimpleNamespace(
        trader=SimpleNamespace(
            get_status=lambda: {
                "position_status": SimpleNamespace(current_price=position_price)
            },
            current_price=fallback,
        ),
        observer=SimpleNamespace(get_current_price=lambda: fallback),
    )
    assert AlpacaBaseTorchTradingEnv._get_current_price(env) == expected


@pytest.mark.parametrize("mid,usable", [
    (float("inf"), False), (float("nan"), False), (1.5, False), (-0.2, False),
    (0.995, True), (0.0, True), (1.0, True),
], ids=["inf", "nan", "above-one", "negative", "resolved", "zero", "one"])
def test_polymarket_midpoint_outside_probability_range_is_unavailable(mid, usable):
    """#349: the caller decides an outcome RESOLVED from this and pays into self.cash.

    `inf >= 0.99` is True, so one garbage midpoint declares a win and books a real payoff.
    A number outside [0, 1] is not a probability, so it reads as unavailable.
    """
    from torchtrade.envs.live.polymarket.env import PolymarketBetEnv

    with patch("torchtrade.envs.live.polymarket.env.requests.get") as get:
        get.return_value = SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"mid": mid}
        )
        result = PolymarketBetEnv._fetch_clob_midpoint("tok")

    assert (result is not None) is usable


@pytest.mark.parametrize("exchange", ["alpaca", "binance", "bitget", "bybit", "okx"])
@pytest.mark.parametrize("module", ["env", "env_sltp"])
def test_no_live_env_reads_a_raw_position_qty(exchange, module):
    """CLAUDE.md invariant 1: every qty read goes through the dust rule, no exceptions.

    A venue can leave a 1e-12 residual after a full close. #283 removed the hand-rolled
    comparisons from the trading paths and these survived in the observation/history
    paths -- latent only because the shipped log_return_reward reads portfolio values
    rather than quantities, so a reward function that used size would inherit the hole.
    A structural guard because the behaviour is currently unobservable: nothing downstream
    reads it yet, which is exactly when a re-fork goes unnoticed.
    """
    path = (pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent
            / "live" / exchange / f"{module}.py")
    if not path.exists():
        pytest.skip(f"{exchange} has no {module}")
    src = path.read_text()
    if module == "env_sltp":
        src += (pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent
                / "utils" / "sltp_mixin.py").read_text()
    assert "position_status.qty" not in src, (
        f"{exchange}/{module} reads a raw qty instead of position_qty_from_status()"
    )


@pytest.mark.parametrize("observed,target,tol,released", [
    (0.50, 0.50, 0.001, False),
    (0.5005, 0.50, 0.001, False),
    (0.95, 0.50, 0.001, True),
    (0.05, 0.50, 0.001, True),
    (0.499, 0.50, 0.01, False),
    (0.50, None, 0.001, False),
], ids=["exact", "below-min-qty", "under-filled", "barely-filled",
        "coarse-lot-step", "no-target"])
def test_a_partial_fill_releases_the_duplicate_action_guard(observed, target, tol, released):
    """#276 follow-up: the direction check alone cannot see a partial fill.

    Fixing #276 made the cached level correct for a COMPLETE fill -- and thereby removed
    the accidental recovery the old bug provided, because a wrong direction used to force
    a mismatch every bar. An under-fill leaves the direction intact, so without this the
    env believes it holds the level it asked for while holding something else, and the
    guard suppresses every corrective retry, permanently and with no log.

    The tolerance is the VENUE's own minimum tradeable size, not an invented percentage:
    a divergence smaller than that cannot be corrected by placing an order, so firing on
    it would release the guard every bar and never converge. `coarse-lot-step` is that
    case -- a complete fill that a flat 1% band would have called a partial one.
    """
    env = SimpleNamespace(position=PositionState())
    # Direction matches by construction, so only the SIZE branch can fire -- otherwise the
    # `closed` row would trip the direction check and prove nothing about this rule.
    env.position.current_position = (observed > 0) - (observed < 0)
    env.position.current_action_level = 0.5
    env.position.target_qty = target
    env.position.target_tol = tol

    TorchTradeLiveEnv._sync_position_from_exchange(
        env, SimpleNamespace(qty=observed)
    )

    assert math.isnan(env.position.current_action_level) is released
    assert env.position.current_position == (observed > 0) - (observed < 0)


@pytest.mark.parametrize("levels", [
    [0.0, float("nan"), 1.0], [0.0, float("inf")], [0.0, float("-inf")],
    [0.0, 5.0], [0.5, 0.5], [],
], ids=["nan", "inf", "-inf", "out-of-range", "duplicate", "empty"])
def test_unusable_action_levels_are_refused(levels):
    """The RULE, once. A NaN level reaches sizing as `target_qty = nan`; `nan > 0` is
    False, so it takes the SELL branch and `trade(side="sell", amount=nan)` goes to the
    venue -- an order the agent never asked for, in the wrong direction."""
    from torchtrade.envs.utils.fractional_sizing import validate_action_levels

    with pytest.raises(ValueError):
        validate_action_levels(levels)


@pytest.mark.parametrize("exchange", ["alpaca", "binance", "bitget", "bybit", "okx"])
def test_every_live_config_validates_its_action_levels(exchange):
    """The WIRING, per exchange. Crossing the two dimensions proved the same thing five
    times over: a missing call kills every value row for that exchange together, so the
    values never distinguish anything the rule test above does not already cover."""
    import importlib

    module = importlib.import_module(f"torchtrade.envs.live.{exchange}.env")
    config_cls = _sole(module, "Config")
    with pytest.raises(ValueError):
        config_cls(symbol="BTC/USD", action_levels=[0.0, float("nan"), 1.0])


@pytest.mark.parametrize("side,expected", [
    ("long", 1), ("short", -1), ("close", 0),
])
def test_the_sltp_side_maps_to_the_direction_it_targets(side, expected):
    """A swapped map passed the WHOLE suite: nothing asserted this mapping at all.

    The four futures SLTP envs each carried their own copy, so an inverted one reported a
    short as long to every account_state read and every reward. One rule on the mixin now,
    and this is the test it was missing.
    """
    from torchtrade.envs.utils.sltp_mixin import SLTPMixin

    env = SimpleNamespace(position=PositionState(), SIDE_DIRECTION=SLTPMixin.SIDE_DIRECTION)
    SLTPMixin._record_sltp_position(env, side)

    assert env.position.current_position == expected


@pytest.mark.parametrize("exchange", ["alpaca", "binance", "bitget", "bybit", "okx"])
def test_every_live_env_reports_the_size_it_asked_for(exchange):
    """The rule was right and the callers disagreed about feeding it (#276 follow-up).

    Round 2 found the reconciliation dead on alpaca (never set a target at all) and off
    for four of binance's six paths -- and no test noticed, because the only coverage
    drove the shared method directly with a hand-built target. A `.get()` that silently
    yields None is exactly the shape that hides this, so assert the writers exist.
    """
    src = (pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent
           / "live" / exchange / "env.py").read_text()

    assert 'target_qty"] = ' in src, f"{exchange} never reports what its action asked for"
    assert 'target_tol"] = ' in src, f"{exchange} reports no tolerance, so any lot-step"


@pytest.mark.parametrize("exchange", ["alpaca", "binance", "bitget", "bybit", "okx"])
def test_the_size_an_env_reports_actually_reaches_the_position(exchange):
    """The tolerance was computed in all five envs and DROPPED at every call site.

    `test_every_live_env_reports_the_size_it_asked_for` proved the assignments exist; it
    could not prove anything read them. Passed as separate arguments, all five forwarded
    target_qty and forgot target_tol, so the tolerance fell back to POSITION_DUST_EPS
    (1e-9) -- and lot quantization moves the filled quantity by far more than that, so
    every COMPLETE fill would have read as a partial one. Exactly the defect the previous
    round fixed, one layer up.
    """

    # AST over the RESOLVED `_step`, not the venue file: #288 moved the plain `_step` onto
    # TorchTradeFuturesLiveEnv, so a file scan reads a module that no longer contains the
    # call. And a substring cannot tell `trade_info` from `trade_info["qty"]` -- which is
    # exactly the half-passing this guard exists to reject.
    import importlib

    mod = importlib.import_module(f"torchtrade.envs.live.{exchange}.env")
    cls = _sole(mod, "TorchTradingEnv")
    call = next(
        (n for n in ast.walk(ast.parse(inspect.getsource(cls._step).lstrip()))
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and n.func.attr == "_record_position_after_trade"),
        None,
    )
    assert call is not None, f"{exchange}'s _step never records the position it traded"
    passed = [a.id for a in call.args if isinstance(a, ast.Name)]
    assert "trade_info" in passed, (
        f"{exchange} hands the recorder {ast.unparse(call)} rather than the whole "
        f"trade_info, so a size value it computes can be silently dropped on the way"
    )


def test_the_recorder_takes_one_argument_that_cannot_be_half_passed():
    """Structural, and deliberately so: the bug was a SIGNATURE that let a caller pass
    one of two paired values. A test on behaviour cannot see that -- only the shape can."""
    import inspect as _inspect

    params = list(_inspect.signature(
        TorchTradeLiveEnv._record_position_after_trade
    ).parameters)

    assert params == ["self", "desired_action", "trade_info"], (
        f"the recorder takes {params}; separate size arguments are what let five call "
        f"sites forward one and drop the other"
    )


@pytest.mark.parametrize("exchange,min_qty", [
    ("bitget", 0.001), ("bybit", 0.001), ("okx", 0.001),
])
def test_a_real_step_lands_both_size_values_on_the_position(exchange, min_qty):
    """Behavioural, because the grep tests could not see the bug that shipped.

    Both size values were computed in every env and BOTH structural tests passed while
    the tolerance was dropped on the way to PositionState -- one asserted the writer
    existed, the other that the call site named the dict. Only reading the value off the
    position after a real step can tell you it arrived, so this drives `_step` and looks.
    """
    import importlib

    module = importlib.import_module(f"torchtrade.envs.live.{exchange}.env")
    env_cls = _sole(module, "TorchTradingEnv")

    trader = MagicMock()
    trader.get_status = MagicMock(return_value={"position_status": None})
    trader.get_mark_price = MagicMock(return_value=50000.0)
    trader.get_lot_size = MagicMock(return_value={"min_qty": min_qty, "qty_step": min_qty})
    trader.get_account_balance = MagicMock(return_value={
        "total_wallet_balance": 10000.0, "available_balance": 10000.0,
        "total_unrealized_profit": 0.0, "total_margin_balance": 10000.0,
    })
    trader.trade = MagicMock(return_value=True)
    trader.cancel_open_orders = MagicMock(return_value=True)
    trader._round_amount = MagicMock(side_effect=lambda amount: amount)

    env = _build_env_with(env_cls, module, trader)
    assert env is not None, (
        f"{exchange} env could not be constructed -- a skip here would make this test "
        f"vacuous, which is exactly how the dropped tolerance survived"
    )

    with patch.object(type(env), "_wait_for_next_timestamp"):
        env.reset()
        env.step(TensorDict({"action": torch.tensor(len(env.action_levels) - 1)}, []))

    assert env.position.target_qty is not None, "the size the action asked for never landed"
    assert env.position.target_tol == pytest.approx(min_qty), (
        "the tolerance never landed: it was computed, written to trade_info, and dropped"
    )


def _build_env_with(env_cls, module, trader):
    """Construct a futures env against the shared mock shape, or None if it does not fit."""
    observer = MagicMock()
    observer.get_keys = MagicMock(return_value=["1m_10"])
    observer.get_observations = MagicMock(side_effect=lambda return_base_ohlc=False: {
        "1m_10": np.zeros((10, 4), dtype=np.float32),
        **({"base_features": np.full((10, 4), 50000.0, dtype=np.float32),
            "base_timestamps": np.arange(10)} if return_base_ohlc else {}),
    })
    observer.get_features = MagicMock(return_value={
        "observation_features": ["a", "b", "c", "d"], "original_features": []})
    observer.intervals, observer.window_sizes = ["1m"], [10]

    config_cls = _sole(module, "Config")
    try:
        config = config_cls(symbol="BTCUSDT", demo=True, time_frames=["1m"],
                            window_sizes=[10], execute_on="1m", leverage=5)
        with patch("time.sleep"), patch.object(env_cls, "_wait_for_next_timestamp"):
            return env_cls(config=config, observer=observer, trader=trader)
    except Exception:
        return None


def test_a_new_episode_does_not_inherit_the_last_one_s_size_target():
    """No live _reset cleared the size target, so episode N+1 started holding episode N's.

    On a flat, fully reconciled account the first sync then compared 0.0 against the old
    position, released the guard reset had just computed, and warned about a discrepancy
    that did not exist. Reset owns "is this position mine?", so it owns clearing this too.
    """
    env = SimpleNamespace(position=PositionState())
    env.position.target_qty = 0.29
    env.position.target_tol = 0.001
    env.position.target_reported = True

    TorchTradeLiveEnv._sync_action_level_after_reset(env)

    assert env.position.target_qty is None, "the new episode inherited the old target"
    assert env.position.target_tol == 0.0
    assert env.position.target_reported is False


def test_a_divergence_is_reported_even_when_reset_already_wrote_nan(caplog):
    """The already-reported flag has to be explicit, not `isnan(current_action_level)`.

    Reset writes NaN to that field to mean "this position predates the episode and its
    level is unknowable". Reusing NaN as the already-reported marker meant a genuine
    divergence in such an episode logged NOWHERE -- an 83% under-fill, silent. Asserting
    the LOG, not the flag: the flag is set either way, so only the log can see the bug.
    """
    env = SimpleNamespace(position=PositionState())
    env.position.current_position = 1
    env.position.current_action_level = float("nan")   # written by reset, not a divergence
    env.position.target_qty = 0.29
    env.position.target_tol = 0.001

    with caplog.at_level(logging.WARNING, logger="torchtrade.envs.core.live"):
        TorchTradeLiveEnv._sync_position_from_exchange(env, SimpleNamespace(qty=0.05))
    assert any("venue holds" in r.message for r in caplog.records), (
        "an 83% under-fill went unreported because reset had already written NaN"
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="torchtrade.envs.core.live"):
        TorchTradeLiveEnv._sync_position_from_exchange(env, SimpleNamespace(qty=0.05))
    assert not caplog.records, "a standing fault must report once, not every bar"


@pytest.mark.parametrize("trade_info,expect_level,expect_target", [
    ({"executed": True, "target_qty": 0.5, "target_tol": 0.001}, 0.5, 0.5),
    ({"executed": False, "at_target": True}, 0.5, None),
    ({"executed": False}, float("nan"), 0.29),
    ({"executed": True, "success": False}, float("nan"), 0.29),
], ids=["traded", "already-there", "refused-for-another-reason", "venue-rejected"])
def test_a_released_guard_re_arms_only_when_the_env_is_at_target(
    trade_info, expect_level, expect_target
):
    """The release had no way back (#276 follow-up, round 4).

    Only an EXECUTED trade could restore a finite level -- but the release compares the
    venue against the snapshot target while the refusal compares it against one recomputed
    from drifting equity, so the two disagree routinely. The level then stayed NaN, the
    duplicate-action guard stayed dead, and the env quietly became a continuous
    rebalancer: re-sizing every bar and trading whenever drift cleared the venue minimum,
    which the policy never asked for and never sees.

    `already-there` is the row that matters: a refusal meaning "we are where we asked to
    be" re-arms. A refusal for any other reason must NOT, or a real divergence is
    forgotten.
    """
    env = SimpleNamespace(position=PositionState())
    env.position.current_action_level = float("nan")
    env.position.target_qty = 0.29
    env.position.target_reported = True

    TorchTradeLiveEnv._record_position_after_trade(env, 0.5, trade_info)

    if math.isnan(expect_level):
        assert math.isnan(env.position.current_action_level)
    else:
        assert env.position.current_action_level == expect_level
    assert env.position.target_qty == expect_target


@pytest.mark.parametrize("exchange", ["binance", "bitget", "bybit", "okx"])
@pytest.mark.parametrize("module", ["env", "env_sltp"])
def test_the_pre_trade_read_halts_like_the_post_bar_one(exchange, module):
    """Every `_step` must acquire its pre-trade state through the halt-policy helper.

    Names the helper, not `_halting(get_status)`: wrapping the fetch alone caught NOTHING,
    because get_status RETURNS the POSITION_UNKNOWN sentinel and the error comes later,
    when the reads that follow touch its attributes.

    Resolved through the MRO rather than read from the venue's FILE (#288): the SLTP
    `_step` is shared now, so a file check would fail on a venue that correctly inherits
    it. Resolution is the stronger check anyway -- a venue that re-forks `_step` without
    the helper still fails, because `inspect.getsource` then returns ITS copy.
    """
    import importlib

    mod = importlib.import_module(f"torchtrade.envs.live.{exchange}.{module}")
    cls = _sole(mod, "TorchTradingEnv")

    # AST call POSITIONS, not a substring. A substring passed on a commented-out call and
    # on the real call moved AFTER the dispatch -- so it proved neither that the read
    # executes nor that it precedes the trade, which is the entire contract. The tool for
    # this already existed two hundred lines up; I reached for `in source` anyway.
    acquire = _first_call_position(cls._step, {"_acquire_pre_trade_state"})
    trade = _first_call_position(
        cls._step, {"_execute_trade_if_needed", "_dispatch_sltp_trade"}
    )
    assert acquire is not None, (
        f"{exchange}/{module}'s resolved _step reads the venue before trading without "
        f"the halt policy"
    )
    assert trade is not None and acquire < trade, (
        f"{exchange}/{module} acquires its pre-trade state AFTER dispatching the trade; "
        f"the halt policy then guards nothing the order depended on"
    )


@pytest.mark.parametrize("policy,expect_flatten", [
    (ObservationFailurePolicy.FLATTEN, True),
    (ObservationFailurePolicy.HALT, False),
])
def test_a_failed_read_runs_the_configured_policy(policy, expect_flatten):
    """The wrapper is the whole point: it is what makes FLATTEN mean anything on a read
    that is not the post-bar one."""
    closed = []
    env = SimpleNamespace(
        config=SimpleNamespace(observation_failure_policy=policy, symbol="TEST"),
        trader=SimpleNamespace(close_position=lambda: closed.append(1) or True),
        consecutive_unknown_status=0,
        _last_confirmed_read={},
        _max_unknown_status_steps=0,
    )

    def boom():
        raise PositionUnknownError("venue unreachable")

    with pytest.raises(LiveObservationHalt) as caught:
        TorchTradeFuturesLiveEnv._halting(env, boom)

    assert bool(closed) is expect_flatten
    assert caught.value.flatten_accepted is (True if expect_flatten else None)


@pytest.mark.parametrize("policy,expect_flatten", [
    (ObservationFailurePolicy.FLATTEN, True),
    (ObservationFailurePolicy.HALT, False),
])
def test_an_unreadable_position_halts_before_the_env_trades(policy, expect_flatten):
    """Drives the REAL pre-trade read with POSITION_UNKNOWN (#355).

    The structural guard next to this one asserts a source substring, so it passes on
    code that wraps `get_status` and halts on nothing -- get_status catches its own venue
    errors and RETURNS the sentinel, so the error comes from the reads that follow. Only
    driving the value through can tell the difference.
    """
    closed = []
    env = SimpleNamespace(
        config=SimpleNamespace(observation_failure_policy=policy, symbol="TEST"),
        trader=SimpleNamespace(
            get_status=lambda: {"position_status": POSITION_UNKNOWN},
            close_position=lambda: closed.append("close") or True,
        ),
    )
    env.consecutive_unknown_status = 0
    env._last_confirmed_read = {}
    env._max_unknown_status_steps = 0
    env._halting = lambda read, cache_key=None: TorchTradeFuturesLiveEnv._halting(
        env, read, cache_key
    )
    env._current_mark_price = (
        lambda ps=None: TorchTradeFuturesLiveEnv._current_mark_price(env, ps)
    )

    with pytest.raises(LiveObservationHalt) as caught:
        TorchTradeFuturesLiveEnv._acquire_pre_trade_state(env)

    assert bool(closed) is expect_flatten
    assert caught.value.flatten_accepted is (True if expect_flatten else None)


def test_a_reset_on_an_unreadable_position_still_raises_the_bare_exception():
    """Pins what reset ACTUALLY does, which is not what the halt promises.

    Routing reset through the halt was reverted deliberately: `_halting` runs
    close_position on any ValueError, and _get_observation raises ValueError for METADATA
    gaps -- a Bybit portfolio-margin account, which always blanks liqPrice, would be
    market-flattened on every reset, overriding close_position_on_reset=False.

    So an unreadable position at reset raises a BARE PositionUnknownError, which the
    documented `except LiveObservationHalt` does not catch. That is a known hole, and
    this test exists so a future refactor cannot silently turn it into a swallow instead.
    """
    from torchtrade.envs.core.state import position_direction_from_status

    with pytest.raises(PositionUnknownError):
        position_direction_from_status(POSITION_UNKNOWN)


def test_an_open_position_never_re_fetches_the_mark_in_the_halt_wrapped_read():
    """The one fact that makes #394's escalation safe, and nothing pinned it.

    `_current_mark_price` converts ANY fetch failure into a ValueError so `_halting`
    catches it -- which under FLATTEN means a market close. That is only acceptable
    because the sole halt-wrapped caller passes `position_status`, so with a position
    open it reads `position_status.mark_price` and never calls the venue at all.

    Drop the `if position_status:` branch and the suite stays green while a read timeout
    starts market-closing real positions. Five OTHER call sites do fetch with a position
    open; they are outside `_halting`, so they crash rather than trade.
    """
    from torchtrade.envs.live.binance import env as binance_env

    trader = MagicMock()
    trader.get_account_balance.return_value = {
        "available_balance": 1000.0, "total_margin_balance": 1000.0,
        "total_wallet_balance": 1000.0}
    trader.cancel_open_orders.return_value = True
    trader.close_position.return_value = True
    trader.get_mark_price.side_effect = ConnectionError("venue unreachable")
    trader.get_status.return_value = {"position_status": SimpleNamespace(
        mark_price=50000.0, qty=0.5, side="long", entry_price=49000.0,
        unrealized_pnl=500.0, leverage=5.0, liquidation_price=40000.0)}

    env = _build_env_with(binance_env.BinanceFuturesTorchTradingEnv, binance_env, trader)
    assert env is not None
    trader.get_mark_price.reset_mock()

    env._acquire_pre_trade_state()
    trader.get_mark_price.assert_not_called()


@pytest.mark.parametrize("flat_at_reset", [True, False],
                         ids=["reset-is-not-halt-wrapped", "step-flattens-a-real-position"])
def test_what_a_short_observation_costs_under_flatten(flat_at_reset):
    """#400's raise is cheap at reset and expensive mid-episode. Pin both.

    `_reset` calls `_get_observation` UNWRAPPED (`futures_live_base.py:568`), so a config
    that can never fill its window surfaces as a bare ValueError while flat -- that is the
    half that makes the escalation acceptable, and nothing drove it before.

    `_acquire_post_bar_state` does wrap it, so the SAME raise mid-episode market-closes an
    open position under FLATTEN. That cost is opt-in (HALT is the default and leaves the
    position alone), but it must not become opt-out by accident: the `+50` fetch buffer is
    a fixed candle count, so on a 1m leg it is ~50 minutes of bad feed, not an outage.
    """
    from torchtrade.envs.live.binance import env as binance_env

    trader = MagicMock()
    trader.get_account_balance.return_value = {
        "available_balance": 1000.0, "total_margin_balance": 1000.0,
        "total_wallet_balance": 1000.0}
    trader.get_mark_price.return_value = 50000.0
    trader.cancel_open_orders.return_value = True
    trader.close_position.return_value = True
    trader.get_status.return_value = {"position_status": None}

    env = _build_env_with(binance_env.BinanceFuturesTorchTradingEnv, binance_env, trader)
    assert env is not None
    env.config.observation_failure_policy = ObservationFailurePolicy.FLATTEN

    # The observer now returns a window SHORT of the declared (10, 4) spec.
    env.observer.get_observations.side_effect = lambda return_base_ohlc=False: (_ for _ in ()
        ).throw(ValueError("only 4 usable candles for BTCUSDT on 1m, need 10"))

    trader.close_position.reset_mock()
    if flat_at_reset:
        with pytest.raises(ValueError) as excinfo:
            env.reset()
        assert not isinstance(excinfo.value, LiveObservationHalt), (
            "reset must NOT route through _halting: a metadata gap would then "
            "market-flatten on every episode start"
        )
        assert trader.close_position.call_count == 0
    else:
        with pytest.raises(LiveObservationHalt):
            env._acquire_post_bar_state()
        assert trader.close_position.call_count == 1


@pytest.mark.parametrize("bad_idx", [
    -1, 99, 1.5, float("nan"), float("inf"), True, torch.tensor([1, 2]),
], ids=["negative", "too-large", "fractional", "nan", "inf", "bool", "multi-element"])
def test_an_invalid_action_index_cannot_pick_a_position(bad_idx):
    """Every kind of malformed index, swept once against the one implementation.

    Clamping was bybit's and okx's behaviour and #288 reversed it. `True` is here because
    bool is an int subclass, so it was resolving to the second level.
    """
    with pytest.raises(InvalidActionError):
        TorchTradeLiveEnv._resolve_action_index(
            SimpleNamespace(), {"action": bad_idx}, 3
        )


def test_a_numpy_index_is_accepted():
    """`np.argmax(probs)` returns np.int64, which is not an `int` subclass.

    main accepted it on three venues; bybit and okx fell through to their index-0
    default, a full SHORT. A strict `isinstance(x, int)` check would have turned the
    canonical hand-rolled live-loop idiom into a killed episode, so this pins acceptance
    rather than leaving it to the next person to rediscover.
    """
    idx = TorchTradeLiveEnv._resolve_action_index(
        SimpleNamespace(), {"action": np.int64(2)}, 3
    )
    assert idx == 2 and type(idx) is int, f"got {idx!r} ({type(idx).__name__})"


def test_a_missing_action_key_cannot_resolve_to_a_tradeable_index():
    """`.get("action", 0)` used to default here, and index 0 is a full SHORT on futures.

    That was the same fail-open this method exists to delete, surviving inside the
    boundary itself -- and nothing in the suite depended on it, so it was pure untested
    surface. The type matters less than that it refuses: it must not return.
    """
    # KeyError, not `raises(Exception)`: the loose form would pass for an AttributeError
    # from a future refactor reading something off the stub, leaving the claim unchecked.
    # It is structurally not a ValueError, so _halting cannot FLATTEN on it.
    with pytest.raises(KeyError):
        TorchTradeLiveEnv._resolve_action_index(SimpleNamespace(), {}, 3)


def test_an_invalid_action_is_not_a_valueerror_that_halting_would_flatten():
    """`_halting` catches ValueError to emergency-close, and `_current_mark_price`
    raises one on the same `_step` path -- a shared type would let both pass for a
    malformed action. See InvalidActionError's own docstring.
    """
    assert not issubclass(InvalidActionError, ValueError)


def test_the_pre_trade_tuple_cannot_be_reordered_unnoticed():
    """Eight sites unpack one 4-tuple, and only okx would notice a swap.

    In the other six the pair reaches `history.record_step(price=, position=)`, which
    feeds the reward and the recorded price series -- so transposing them corrupts the
    training signal rather than crashing, and mutation testing showed 6 of 8 accept it
    with a green suite. Collapsing eight per-file mistakes into one shared line makes the
    mistake cheaper to make, so the shape has to be pinned somewhere.
    """
    ps = SimpleNamespace(qty=0.25, mark_price=50000.0)
    env = SimpleNamespace(
        config=SimpleNamespace(
            observation_failure_policy=ObservationFailurePolicy.HALT, symbol="T"
        ),
        trader=SimpleNamespace(get_status=lambda: {"position_status": ps},
                               get_mark_price=lambda: 50000.0),
    )
    env.consecutive_unknown_status = 0
    env._last_confirmed_read = {}
    env._max_unknown_status_steps = 0
    env._halting = lambda read, cache_key=None: TorchTradeFuturesLiveEnv._halting(
        env, read, cache_key
    )
    env._current_mark_price = (
        lambda p=None: TorchTradeFuturesLiveEnv._current_mark_price(env, p)
    )

    status, position_status, price, size = TorchTradeFuturesLiveEnv._acquire_pre_trade_state(env)

    assert position_status is ps
    assert price == 50000.0, "slot 2 is the PRICE"
    assert size == 0.25, "slot 3 is the SIZE"


def _calls_observer_reset(func) -> bool:
    """True if the body has `self.observer.reset()` as an unconditional statement.

    A substring check on inspect.getsource passes on a comment: replacing okx's live
    line with `# NOTE: self.observer.reset() handled by the caller` left the entire
    suite green while okx's second episode continued mid-stream (#278 review). The five
    older guards in this file already parse; this one had to as well.

    Top-level statements only, not `ast.walk`: walking accepts a call that never runs --
    `if False:`, a nested def, a line after `return` -- and the point is that the reset
    HAPPENS, on every reset, before the reads below it.
    """
    body = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0].body
    for stmt in body:
        # The three shapes a top-level call takes: bare, returned, or assigned. Anything
        # nested (a branch, an inner def, code after a return) is deliberately excluded.
        call = None
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
        elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
            call = stmt.value
        elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            call = stmt.value
        if call is None:
            continue
        f = call.func
        if (isinstance(f, ast.Attribute) and f.attr == "reset"
                and isinstance(f.value, ast.Attribute) and f.value.attr == "observer"
                and isinstance(f.value.value, ast.Name) and f.value.value.id == "self"):
            return True
        # `super()._reset(...)` delegates to the base, which does call it.
        if (isinstance(f, ast.Attribute) and f.attr == "_reset"
                and isinstance(f.value, ast.Call) and isinstance(f.value.func, ast.Name)
                and f.value.func.id == "super"):
            return True
    return False


def test_every_live_reset_rewinds_its_observer():
    """One line, five places, and only ONE venue's tests would notice it missing.

    A mutation sweep replacing `self.observer.reset()` with `pass` at each of the five
    call sites left four of them green (#278 review): the replay-backed env tests happen
    to exist for bybit only. Without this guard a sixth exchange, or a copy-paste that
    drops the line, ships an env whose second episode continues mid-stream with the
    previous episode's balance and an inherited position.
    """
    missing = [
        c.__name__ for c in LIVE_ENVS
        if (r := c.__dict__.get("_reset")) is not None and not _calls_observer_reset(r)
    ]
    assert not missing, (
        f"{len(missing)} live _reset methods never rewind their observer: {missing}"
    )


def test_the_observer_interface_declares_reset():
    """The call above needs no hasattr guard only because every observer HAS the method.

    A hasattr guard would be fail-open -- an observer that renamed it would silently stop
    rewinding, which is the defect #278 fixed. That safety rests on the interface, so the
    interface is asserted rather than assumed: without this, a new exchange's observation
    class fails at runtime on episode 2, not in CI.
    """
    from torchtrade.envs.live.alpaca.observation import AlpacaObservationClass
    from torchtrade.envs.live.binance.observation import BinanceObservationClass
    from torchtrade.envs.live.shared.futures_base_obs import BaseFuturesObservationClass
    from torchtrade.envs.replay.observer import ReplayObserver

    for cls in (AlpacaObservationClass, BinanceObservationClass,
                BaseFuturesObservationClass, ReplayObserver):
        assert callable(getattr(cls, "reset", None)), f"{cls.__name__} has no reset()"

    # A live reset must be a genuine no-op: it takes no arguments beyond self and
    # returns nothing. If one starts rebasing account state, the run-level bankruptcy
    # baseline it protects goes with it.
    for cls in (AlpacaObservationClass, BinanceObservationClass, BaseFuturesObservationClass):
        assert cls.reset(None) is None, f"{cls.__name__}.reset must be a no-op"


def test_a_live_account_keeps_its_run_level_bankruptcy_baseline():
    """The yardstick must not chase the account down.

    Rebasing the baseline every episode looks like offline parity and is not: offline
    resets the BALANCE and the baseline together, so its ratio starts at 1.0 because the
    account was reset too. A live account persists, so rebasing only the yardstick
    removes the cross-episode drawdown circuit breaker entirely -- an account halving
    every episode reached 0.39% of its starting value without ever reporting bankrupt.
    """
    from tests.mocks.alpaca import MockObserver, MockTrader
    from torchtrade.envs.live.alpaca.env import (
        AlpacaTorchTradingEnv,
        AlpacaTradingEnvConfig,
    )

    trader = MockTrader(initial_cash=1000.0)
    env = AlpacaTorchTradingEnv(
        config=AlpacaTradingEnvConfig(
            symbol="BTC/USD", window_sizes=[10], bankrupt_threshold=0.1,
        ),
        observer=MockObserver(window_sizes=[10]),
        trader=trader,
    )
    env._wait_for_next_timestamp = lambda: None

    env.reset()
    assert env.initial_portfolio_value == pytest.approx(1000.0)

    # The account halves each episode. A live observer rewinds nothing, so the baseline
    # must stay at the equity the RUN started from.
    for equity in (500.0, 250.0, 125.0, 62.5):
        trader.cash = equity
        env.reset()
        assert env.initial_portfolio_value == pytest.approx(1000.0), (
            f"baseline rebased to {env.initial_portfolio_value} at equity {equity}; it "
            f"now chases the account down and bankruptcy can never fire (#278)"
        )

    assert env._check_termination(62.5) is True, (
        "62.5 is below 10% of the 1000 this run started with and must terminate"
    )


@pytest.mark.parametrize("cls", LIVE_ENVS, ids=lambda c: c.__name__)
def test_every_live_env_can_actually_run_the_bar_wait(cls):
    """RUN it, do not inspect it. This is the test that was missing.

    `_wait_for_next_timestamp` reads `self.execute_on`, and alpaca's base set only
    `execute_on_value`/`execute_on_unit` -- so both alpaca envs raised AttributeError at
    the first bar, AFTER the order was placed and the position recorded. 834 exchange
    tests and 513 contract tests passed, because every one of them stubs this method
    (`env._wait_for_next_timestamp = lambda: None` or `patch.object(...)`). Nothing ever
    executed the body.

    Constructing a live env needs credentials, so this asserts the attribute the body
    reads is set by __init__ on every exchange -- the cheapest thing that fails when it
    is not.
    """
    # The EXCHANGE base, not the whole MRO: TorchTradeLiveEnv initialises
    # `self.execute_on = None`, so an MRO-wide search passes on a concrete class that
    # never assigns it -- which is exactly the alpaca break, certified clean. First
    # version of this test did that and the mutation survived.
    #
    # #288 moved the assignment into the shared `_finish_futures_init`, so the PREDICATE
    # widens, not the base list. Admitting the futures base as a source was my first
    # attempt and it restored the hole above -- mutation-proven: deleting bybit's
    # `_finish_futures_init()` call still passed. The venue must show one or the other in
    # ITS OWN source.
    exchange_bases = [
        base for base in cls.__mro__
        if base.__module__.startswith("torchtrade.envs.live.")
        and not base.__module__.startswith("torchtrade.envs.live.shared")
    ]
    if not exchange_bases:
        pytest.skip(f"{cls.__name__} is a shared base, not an exchange env")
    assert any(
        "self.execute_on = " in inspect.getsource(base)
        or "_finish_futures_init()" in inspect.getsource(base)
        for base in exchange_bases
    ), (
        f"{cls.__name__} never assigns self.execute_on in its exchange base; the shared "
        f"bar wait reads it and will AttributeError at the first bar, after the order"
    )


def test_the_bar_wait_actually_sleeps_for_the_timeframe():
    """Drive the real body with a stub env, since no exchange test ever does."""
    from unittest.mock import patch

    from torchtrade.envs.core.live import TorchTradeLiveEnv
    from torchtrade.envs.utils import TimeFrame, TimeFrameUnit

    for value, unit, expected in ((1, TimeFrameUnit.Minute, 60),
                                  (5, TimeFrameUnit.Minute, 300),
                                  (4, TimeFrameUnit.Hour, 14400),
                                  (1, TimeFrameUnit.Day, 86400)):
        env = SimpleNamespace(execute_on=TimeFrame(value, unit), timezone=None)
        with patch("torchtrade.envs.core.live.time.sleep") as slept:
            TorchTradeLiveEnv._wait_for_next_timestamp(env)
        assert slept.called, f"{value}{unit.name} did not sleep at all"
        # The truncation to whole minutes means the slept value is within a minute of
        # the period; the point is that it scales with the timeframe, not that it is exact.
        assert 0 < slept.call_args[0][0] <= expected, (
            f"{value}{unit.name} slept {slept.call_args[0][0]}s, not ~{expected}s"
        )


def test_the_bar_wait_derives_from_the_timeframe_not_a_string_alias():
    """One duration rule, not an alias table that grows per spelling (#288).

    `_wait_for_next_timestamp` looked its unit up in a 17-entry map --
    "TimeFrameUnit.Minute", "Minute", "Min", "min", "minute", "h", "H", "D", "d",
    "seconds" -- because five exchanges stringified the same enum four different ways
    and the map grew an entry per spelling rather than the spellings being fixed. A
    sixth exchange spelling it a fifth way would have raised at the first bar, in
    production, on a timer.
    """
    from torchtrade.envs.core import live as live_module

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(live_module.TorchTradeLiveEnv._wait_for_next_timestamp)
    ))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "timeframe_to_seconds" in called, "the wait must derive from the TimeFrame"

    # Literals only -- the explanation above quotes the old spellings, and a check that
    # a comment can trip is a check a comment can also satisfy.
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    regrown = literals & {"TimeFrameUnit.Minute", "Minute", "Min", "min", "minute",
                          "TimeFrameUnit.Hour", "Hour", "h", "H", "seconds",
                          "TimeFrameUnit.Day", "Day", "D", "d", "hour", "day"}
    assert not regrown, f"the unit alias table is regrowing in the wait path: {regrown}"


def test_no_live_env_re_forks_the_shared_accessors():
    """`get_account_state` / `get_market_data_keys` were FIVE byte-identical copies.

    One-line returns, so the copies were harmless in themselves -- but this repo has
    shipped a fix landing on some exchanges and not others three times (lot size #271,
    full_done_spec #272, hedge-mode surface), and a copy that has not drifted YET is
    still a copy (#288).

    Asserts the MRO OWNER, not a module allowlist. The first version filtered on
    `torchtrade.envs.live.*`, which missed `SLTPMixin` -- and that is the one place a
    re-fork actually wins: all four SLTP envs are `class X(SLTPMixin, XBase)`, so the
    mixin precedes the shared base and a method defined there shadows the hoisted one on
    every venue at once, with the guard staying green.
    """
    from torchtrade.envs.core.live import TorchTradeLiveEnv

    for name in ("get_account_state", "get_market_data_keys"):
        wrong_owner = {
            c.__name__: next(b for b in c.__mro__ if name in b.__dict__).__name__
            for c in LIVE_ENVS
            if next((b for b in c.__mro__ if name in b.__dict__), None)
            is not TorchTradeLiveEnv
        }
        assert not wrong_owner, (
            f"{name} does not resolve to TorchTradeLiveEnv on: {wrong_owner}"
        )


def test_every_live_env_reports_the_same_account_state_layout():
    """The accessor is shared; the DATA still lives per exchange, so pin it.

    Every live env, not just the futures ones -- CLAUDE.md specifies this 6-element
    vector as universal, and the first version's `issubclass(..., FuturesLiveEnv)` filter
    excluded alpaca entirely, so replacing its ACCOUNT_STATE with ['a','b','c'] passed.

    The NAMES are pinned, not just agreement: renaming a field consistently across all
    venues satisfied "they all match" while breaking every consumer that indexes
    account_state by meaning. And no `hasattr` filter -- that was fail-open, dropping a
    class that lost the attribute instead of failing it.
    """
    expected = (
        "exposure_pct", "position_direction", "unrealized_pnlpct",
        "holding_time", "leverage", "distance_to_liquidation",
    )
    from torchtrade.envs.core.live import TorchTradeLiveEnv
    from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv

    # Excluded BY IDENTITY, not by `hasattr`: the shared bases legitimately have no
    # ACCOUNT_STATE, but a hasattr filter would also silently drop a concrete venue that
    # LOST it -- and get_account_state() would then raise on every step.
    abstract = {TorchTradeLiveEnv, TorchTradeFuturesLiveEnv}
    layouts = {
        c.__name__: tuple(c.ACCOUNT_STATE) for c in LIVE_ENVS if c not in abstract
    }
    assert len(layouts) >= 10, f"only {len(layouts)} concrete live envs checked"
    wrong = {n: v for n, v in layouts.items() if v != expected}
    assert not wrong, f"venues disagree with the universal account_state vector: {wrong}"


def test_the_trade_info_defaults_are_the_whole_contract():
    """A path that did not trade still returns this dict, so a MISSING key reads
    downstream as an absent fact rather than a default one -- which is what four copies
    risked drifting on."""
    from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv

    info = TorchTradeFuturesLiveEnv._create_trade_info(None)
    assert info == {
        "executed": False, "quantity": 0, "side": None,
        "success": None, "closed_position": False,
    }
    # kwargs override, and do not silently drop a default.
    overridden = TorchTradeFuturesLiveEnv._create_trade_info(None, executed=True, side="BUY")
    assert overridden["executed"] is True and overridden["side"] == "BUY"
    assert overridden["closed_position"] is False and overridden["quantity"] == 0

    # An UNKNOWN key must survive: `at_target` is read by core/live.py to re-arm
    # current_action_level, so a merge that kept only known keys would silently drop a
    # money-moving fact. That is the one semantic this hoist actually changed
    # (`dict(...).update(kwargs)` -> `{**defaults, **kwargs}`) and nothing covered it.
    extra = TorchTradeFuturesLiveEnv._create_trade_info(None, at_target=True, target_qty=0.5)
    assert extra["at_target"] is True and extra["target_qty"] == 0.5
    assert extra["executed"] is False, "an unknown key must not disturb the defaults"


@pytest.mark.parametrize("qty,expected_side", [(0.5, "sell"), (-0.5, "buy")])
def test_closing_reports_the_order_side_that_closed_it(qty, expected_side):
    """binance reported the literal "CLOSE" where its three siblings report the real side.

    `side` is what went to the venue, and `closed_position=True` in the same dict already
    says it was a close -- so the literal added nothing and lost the one fact this field
    carries. That is the silent per-exchange divergence #288 exists to remove, and it is
    why the four copies could not be hoisted until it was resolved: whichever version
    became shared, one venue's observable output changed, and it had to change
    deliberately.
    """
    from types import SimpleNamespace

    from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv

    env = SimpleNamespace(
        trader=SimpleNamespace(close_position=lambda: True),
        config=SimpleNamespace(symbol="BTCUSDT"),
        position=SimpleNamespace(current_position=1),
        _create_trade_info=TorchTradeFuturesLiveEnv._create_trade_info.__get__(object()),
    )
    wire_outage_state(env)   # _handle_close_action clears the balance cache
    info = TorchTradeFuturesLiveEnv._handle_close_action(env, qty)

    assert info["side"] == expected_side, "a close must report the side it sent"
    assert info["closed_position"] is True and info["executed"] is True
    assert info["quantity"] == abs(qty)
    assert env.position.current_position == 0, "a successful close must zero the position"


def test_a_close_with_no_position_does_not_report_a_trade():
    from types import SimpleNamespace

    from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv

    env = SimpleNamespace(_create_trade_info=TorchTradeFuturesLiveEnv._create_trade_info.__get__(object()))
    info = TorchTradeFuturesLiveEnv._handle_close_action(env, 0)
    assert info["executed"] is False and info["side"] is None


@pytest.mark.parametrize("qty,expected", [(0.5, "buy"), (-0.5, "sell")])
def test_the_order_side_sent_is_lowercase_and_matches_the_direction(qty, expected):
    """The VALUE that reaches the trader, not a source-text grep.

    A grep version of this was defeated five ways: single quotes, `"buy".upper()`, a
    module-level helper (getsource returns only the class block), `futures_live_base`
    itself -- the very file this hoist moved side construction INTO, excluded from
    FUTURES_ENVS by construction -- and alpaca, which is in LIVE_ENVS but not
    FUTURES_ENVS, so a test named "every venue" checked 12 of 16 classes.

    It also could not see the value at all: hardcoding `side="buy"` in the hoisted
    method, so a short reported "buy", passed the whole suite.
    """
    from types import SimpleNamespace

    from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv

    sent = {}

    def _trade(**kwargs):
        sent.update(kwargs)
        return True

    env = SimpleNamespace(
        trader=SimpleNamespace(trade=_trade, close_position=lambda: True),
        config=SimpleNamespace(symbol="BTCUSDT"),
        position=SimpleNamespace(current_position=0),
        _create_trade_info=TorchTradeFuturesLiveEnv._create_trade_info.__get__(object()),
    )
    side = "buy" if qty > 0 else "sell"
    info = TorchTradeFuturesLiveEnv._execute_market_order(env, side, abs(qty))

    assert sent["side"] == expected, f"sent {sent['side']!r} to the venue, not {expected!r}"
    assert sent["side"] == sent["side"].lower(), "every venue sends a lowercase side"
    assert info["side"] == expected and info["quantity"] == abs(qty)


@pytest.mark.parametrize("env_cls", LIVE_ENVS, ids=lambda c: c.__name__)
def test_no_live_env_builds_an_uppercase_order_side(env_cls):
    """binance passed "BUY"/"SELL" where its siblings passed "buy"/"sell" (#288).

    Its executor upper-cases whatever it receives, so the venue never saw a difference --
    it surfaced only as a mixed-case `trade_info["side"]` across exchanges, in the field
    a preceding slice had just unified. Normalising it is what made the four
    `_execute_market_order` copies identical enough to hoist.
    """
    from torchtrade.envs.live.shared import futures_live_base

    # Every LIVE env, not just the futures ones -- alpaca is in LIVE_ENVS and was not
    # checked -- plus the shared base module itself, which is where this PR moved side
    # construction and which no env class's getsource covers.
    source = inspect.getsource(env_cls) + inspect.getsource(futures_live_base)
    for upper in ('"BUY"', '"SELL"', "'BUY'", "'SELL'"):
        assert upper not in source, (
            f"{env_cls.__name__} builds an uppercase order side {upper}; every venue "
            f"sends lowercase and the executors normalise"
        )


from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from torchtrade.envs.live.alpaca import AlpacaSLTPTradingEnvConfig
from torchtrade.envs.offline.sequential_sltp import SequentialTradingEnvSLTPConfig
from torchtrade.envs.offline.vectorized_sequential_sltp import (
    VectorizedSequentialTradingEnvSLTPConfig,
)
from torchtrade.envs.live.binance import (
    BinanceFuturesSLTPTradingEnvConfig, BinanceFuturesTradingEnvConfig)
from torchtrade.envs.live.bitget import (
    BitgetFuturesSLTPTradingEnvConfig, BitgetFuturesTradingEnvConfig)
from torchtrade.envs.live.bybit import (
    BybitFuturesSLTPTradingEnvConfig, BybitFuturesTradingEnvConfig)
from torchtrade.envs.live.okx import (
    OKXFuturesSLTPTradingEnvConfig, OKXFuturesTradingEnvConfig)
from torchtrade.envs.live.shared.sltp_config import BaseFuturesSLTPConfig

SLTP_CONFIGS = [
    pytest.param(c, id=c.__name__)
    for c in [
        BinanceFuturesSLTPTradingEnvConfig,
        BitgetFuturesSLTPTradingEnvConfig,
        BybitFuturesSLTPTradingEnvConfig,
        OKXFuturesSLTPTradingEnvConfig,
    ]
]

# Reproduced from main before the extraction. One literal now sets each of these for four
# live venues, so a typo moves all four at once: mutating bankrupt_threshold 0.1 -> 0.5
# (a 5x move in every venue's force-close point) passed 1432 tests before this pin.
SHARED_DEFAULTS = {
    "time_frames": [TimeFrame(1, TimeFrameUnit.Hour)],
    "execute_on": TimeFrame(1, TimeFrameUnit.Hour),
    "window_sizes": [10],
    "leverage": 1,
    "quantity_per_trade": 0.001,
    "trade_mode": "quantity",
    "position_fraction": 1.0,
    "lock_position_until_sltp": False,
    "stoploss_levels": (-0.025, -0.05, -0.1),
    "takeprofit_levels": (0.05, 0.1, 0.2),
    "include_short_positions": True,
    "include_hold_action": True,
    "include_close_action": False,
    "done_on_bankruptcy": True,
    "bankrupt_threshold": 0.1,
    "demo": True,
    "seed": 42,
    "include_base_features": False,
    "close_position_on_init": True,
    "close_position_on_reset": False,
    "max_unknown_status_steps": 0,
}


@pytest.mark.parametrize("config_cls", SLTP_CONFIGS)
def test_every_sltp_config_inherits_the_shared_fields(config_cls):
    """A subclass re-declaring a shared field silently shadows the base default (#288)."""
    assert issubclass(config_cls, BaseFuturesSLTPConfig)
    own = set(vars(config_cls).get("__annotations__", {}))
    assert not (own & set(SHARED_DEFAULTS)), (
        f"{config_cls.__name__} re-declares {sorted(own & set(SHARED_DEFAULTS))}; a "
        f"redeclaration with the same value today silently stops tracking the base "
        f"tomorrow. Only `symbol` is meant to be overridden, and it is not in "
        f"SHARED_DEFAULTS because its value legitimately differs per venue."
    )


@pytest.mark.parametrize("config_cls", SLTP_CONFIGS)
def test_hoisting_a_default_did_not_change_it(config_cls):
    """Values the four venues agreed on before the hoist, and must still agree on."""
    config = config_cls()
    assert {f: getattr(config, f) for f in SHARED_DEFAULTS} == SHARED_DEFAULTS


# Both variants per venue: the SLTP-only list let a base config drift unseen. Imported
# classes, not names resolved at runtime -- a string table re-targets silently.
MARGIN_BEARING_CONFIGS = [
    BinanceFuturesTradingEnvConfig, BinanceFuturesSLTPTradingEnvConfig,
    BitgetFuturesTradingEnvConfig, BitgetFuturesSLTPTradingEnvConfig,
    BybitFuturesTradingEnvConfig, BybitFuturesSLTPTradingEnvConfig,
    OKXFuturesTradingEnvConfig, OKXFuturesSLTPTradingEnvConfig,
]


@pytest.mark.parametrize("config_cls", MARGIN_BEARING_CONFIGS, ids=lambda c: c.__name__)
def test_every_venue_spells_the_margin_field_the_same_way(config_cls):
    """#289's call was made: one spelling, `margin_mode`, on all four venues.

    This used to assert each venue's OWN name, because binance said `margin_type` and the
    other three `margin_mode` -- one concept exchange-agnostic config code could not set.
    It guarded against an accidental unification while a deliberate one was still open.

    Named per venue rather than `margin_type XOR margin_mode`: the XOR form passes the
    very swap it names as the risk, which is how the original divergence survived.
    """
    names = {f.name for f in dataclasses.fields(config_cls)}
    assert "margin_mode" in names
    assert "margin_type" not in names, (
        f"{config_cls.__name__} reintroduced margin_type; the VALUES stay venue-specific "
        f"(okx 'cross' vs bitget/bybit 'crossed'), the field name does not"
    )


SIZING_CONFIGS = SLTP_CONFIGS + [
    pytest.param(c, id=c.__name__) for c in [
        AlpacaSLTPTradingEnvConfig,
        SequentialTradingEnvSLTPConfig,
        VectorizedSequentialTradingEnvSLTPConfig,
    ]
]


@pytest.mark.parametrize("config_cls", SIZING_CONFIGS)
@pytest.mark.parametrize("kwargs,match", [
    (dict(trade_mode="fractional", position_fraction=0.0), "position_fraction"),
    (dict(trade_mode="fractional", position_fraction=1.5), "position_fraction"),
    (dict(trade_mode="notional", quantity_per_trade=0), "quantity_per_trade"),
    (dict(trade_mode="notional", quantity_per_trade=-1), "quantity_per_trade"),
])
def test_a_sizing_config_that_cannot_trade_is_rejected(config_cls, kwargs, match):
    """All seven callers of validate_position_sizing, including both offline SLTP envs.

    Deleting the validator's body outright failed ZERO tests before this. That matters
    most offline: `SequentialTradingEnvSLTP` with `quantity_per_trade=0` runs 20 buy
    actions as silent no-ops, `account_state` flat throughout and reward sum exactly
    0.0 -- a training run that burns compute on a degenerate signal and never says so.

    It is also what catches a venue re-forking `__post_init__` without `super()`, which
    a structural "no subclass defines __post_init__" assertion cannot distinguish from a
    correct override that calls super() and adds venue logic.
    """
    with pytest.raises(ValueError, match=match):
        config_cls(**kwargs)


def test_the_shared_defaults_pin_covers_every_hoisted_field():
    """SHARED_DEFAULTS backs two guards, and is hand-maintained.

    Hoist a field into the base and forget to list it here and it is unpinned on BOTH
    axes at once -- silent default drift and silent subclass shadowing. The two
    exclusions are deliberate: `symbol` legitimately differs per venue, and
    `observation_failure_policy` is covered by test_live_observation_failsafe.py.
    """
    hoisted = {f.name for f in dataclasses.fields(BaseFuturesSLTPConfig)}
    assert set(SHARED_DEFAULTS) | {"symbol", "observation_failure_policy"} == hoisted


# Every class that defines close(), discovered: a sixth exchange cannot opt out.
CLOSE_OWNERS = sorted(
    {c for env in LIVE_ENVS for c in env.__mro__
     if "close" in c.__dict__ and c.__module__.startswith("torchtrade.")},
    key=lambda c: c.__name__,
)

_RESET_LOGGER = TorchTradeFuturesLiveEnv.__module__


class _ResetStub:
    """Minimal stand-in for a futures env, so the shared _reset can be driven directly.

    Not a mock: a MagicMock's methods are truthy, which is exactly the failure mode these
    tests exist to detect -- `if not self.trader.cancel_open_orders()` would never fire.
    """

    def __init__(self, cancel_ok=True, close_ok=True, close_on_reset=False):
        outer = self

        class _Trader:
            def cancel_open_orders(self): return cancel_ok
            def close_position(self): return close_ok
            def get_account_balance(self): return {"available_balance": 1000.0}
            def get_status(self): return {"position_status": None}

        class _Observer:
            def reset(self): pass

        self.history_reset_calls = 0
        self.sync_action_level_calls = 0
        # #295: the shared _reset clears the outage state at the episode boundary.
        wire_outage_state(self)
        self._reset_outage_state = (
            lambda: TorchTradeLiveEnv._reset_outage_state(self)
        )
        self._max_unknown_status_steps = 0
        # The reset read runs under the halt policy as of #295.
        self._halting = lambda read, cache_key=None: (
            TorchTradeFuturesLiveEnv._halting(self, read, cache_key)
        )
        self.trader, self.observer = _Trader(), _Observer()
        self.history = SimpleNamespace(reset=lambda: setattr(
            outer, "history_reset_calls", outer.history_reset_calls + 1))
        self.position = SimpleNamespace(hold_counter=7, current_position=1)
        self.config = SimpleNamespace(close_position_on_reset=close_on_reset)
        self.balance = 0.0

    _reset = TorchTradeFuturesLiveEnv._reset

    def _sync_action_level_after_reset(self):
        # Load-bearing for binance and bitget: their duplicate-action guard compares
        # against current_action_level, so dropping this call makes _reset silently
        # refuse the first trade of the episode. Its only other cover was an incidental
        # assertion in one bitget test.
        self.sync_action_level_calls += 1

    _get_observation = lambda self, advance_hold=True, **_kw: advance_hold


@pytest.mark.parametrize("kwargs,expected", [
    (dict(cancel_ok=False), "cancel_open_orders failed"),
    (dict(close_ok=False, close_on_reset=True), "close_position failed"),
    (dict(cancel_ok=False, close_ok=False, close_on_reset=True), "close_position failed"),
], ids=["stale-brackets", "residual-exposure", "both-fail-still-reports-the-second"])
def test_a_failed_reset_cleanup_is_not_swallowed(caplog, kwargs, expected):
    """binance and bitget discarded both return values; bybit and okx warned (#288).

    A failed cancel leaves live brackets attached to a position the new episode believes
    is clean; a failed close leaves real exposure the account state will not show.
    Neither is recoverable here -- the episode has to start -- but silence made them
    invisible, and the fold had to pick one behaviour for all four venues.
    """
    with caplog.at_level(logging.WARNING, logger=_RESET_LOGGER):
        _ResetStub(**kwargs)._reset(None)
    assert any(expected in r.message for r in caplog.records), (
        f"a failed reset cleanup logged nothing; records={[r.message for r in caplog.records]}"
    )


def test_a_clean_reset_is_silent_and_zeroes_the_hold_counter(caplog):
    """All four venues return True from both calls when flat, so a flat reset must not warn.

    The silence is the point, not a detail: warnings that fire on every episode start are
    noise that hides the one that matters. Nothing pinned it, and "warn unconditionally"
    survived the whole live suite.

    Also pins that the hold counter is zeroed BEFORE the observation is read with
    advance_hold=False -- a reset that counted its own bar would age a fresh position --
    and that the four single lines the fold created still run at all.
    """
    stub = _ResetStub()
    with caplog.at_level(logging.WARNING, logger=_RESET_LOGGER):
        assert stub._reset(None) is False  # advance_hold=False reached _get_observation
    ours = [r for r in caplog.records if r.name == _RESET_LOGGER]
    assert not ours, [r.message for r in ours]
    assert stub.history_reset_calls == 1
    assert stub.sync_action_level_calls == 1
    assert stub.balance == 1000.0
    assert stub.position.hold_counter == 0
    assert stub.position.current_position == 0  # flat; the dust rule itself is pinned
    # by test_every_reset_uses_the_shared_direction_rule, which reads the source.


class _CloseHarness(TorchTradeFuturesLiveEnv):
    """A real subclass, because close() calls zero-arg super().

    Built with object.__new__ so EnvBase.__init__ (specs, device, batch size) is skipped
    -- close() reads nothing but `self.trader`. See _ResetStub on why not a MagicMock.
    """

    def _init_trading_clients(self, *a, **k): raise NotImplementedError
    def _step(self, tensordict): raise NotImplementedError
    def _execute_trade_if_needed(self, action): raise NotImplementedError

    @classmethod
    def build(cls, direction=0, cancel_raises=False, status_raises=False):
        self = object.__new__(cls)
        self.cancelled = False

        class _Trader:
            def get_status(_):
                if status_raises: raise RuntimeError("exchange down")
                return {"position_status": SimpleNamespace(qty=direction)}
            def cancel_open_orders(_):
                if cancel_raises: raise RuntimeError("cancel blew up")
                self.cancelled = True

        self.trader = _Trader()
        return self


@pytest.mark.parametrize("kwargs,expect_log,still_cancels", [
    (dict(direction=1), "Closing environment with open position", True),
    (dict(direction=0), None, True),
    (dict(status_raises=True), None, True),
    (dict(direction=1, cancel_raises=True), "Failed to cancel open orders", False),
], ids=["open-position-warns", "flat-is-silent", "status-failure-still-cleans-up", "cancel-failure-is-logged"])
def test_close_warns_but_never_raises(caplog, kwargs, expect_log, still_cancels):
    """close() was three different versions; binance's warned about nothing (#288).

    It deliberately does not close positions -- automated closure on cleanup could
    liquidate an intended position -- so the warning is the ONLY signal that you are
    walking away from live exposure. Nothing tested it on any venue.

    It must also never raise: close() runs during teardown, where an exception replaces
    whatever error you were actually trying to see.
    """
    stub = _CloseHarness.build(**kwargs)
    with caplog.at_level(logging.WARNING):
        stub.close()
    assert stub.cancelled is still_cancels
    ours = [r for r in caplog.records if r.name == TorchTradeFuturesLiveEnv.__module__]
    if expect_log is None:
        assert not ours, [r.message for r in ours]
    else:
        assert any(expect_log in r.message for r in ours)


@pytest.mark.parametrize("owner", [
    pytest.param(c, id=c.__name__) for c in CLOSE_OWNERS
])
def test_close_accepts_the_keyword_torchrl_passes_it(owner):
    """`TransformedEnv.close` forwards `raise_if_closed` unconditionally.

    Every live env had a bare `def close(self)`, so closing a wrapped env died with
    TypeError -- and `examples/llm/frontier/live.py:150` does exactly that. Discovered
    rather than hardcoded: pinning only the class already fixed is how alpaca kept the
    bug through a PR that fixed the other four.
    """
    ours = inspect.signature(owner.__dict__["close"]).parameters
    theirs = inspect.signature(EnvBase.close).parameters
    assert [(n, p.kind, p.default) for n, p in ours.items()] == [
        (n, p.kind, p.default) for n, p in theirs.items()
    ]


def test_construction_survives_an_observer_that_cannot_reach_the_exchange():
    """binance and bitget built the spec from `get_observations()` -- a live kline fetch
    PER TIMEFRAME, during __init__ -- purely to read `.shape[1]`.

    So an outage made CONSTRUCTION fail rather than the first step, and alpaca did the
    same. Behavioural rather than structural: the AST form this replaced asserted which
    method is called by name, which a rename of the bad path escapes, and it parametrized
    12 envs that all resolve to one function -- 12 copies of one assertion.
    """
    from torchtrade.envs.live.binance.env import (
        BinanceFuturesTorchTradingEnv,
        BinanceFuturesTradingEnvConfig,
    )

    observer = MagicMock()
    observer.get_keys = MagicMock(return_value=["1Minute_10"])
    observer.get_features = MagicMock(return_value={
        "observation_features": ["a", "b", "c", "d"], "original_features": []})
    observer.get_observations = MagicMock(side_effect=ConnectionError("exchange down"))

    trader = MagicMock()
    trader.get_account_balance.return_value = {
        "available_balance": 1000.0, "total_margin_balance": 1000.0,
        "total_wallet_balance": 1000.0,
    }
    trader.get_status.return_value = {"position_status": None}
    trader.cancel_open_orders.return_value = True
    trader.close_position.return_value = True
    trader.get_mark_price.return_value = 100.0

    config = BinanceFuturesTradingEnvConfig(
        symbol="BTCUSDT", time_frames=["1m"], window_sizes=[10], execute_on="1m",
    )
    with patch("time.sleep"), patch.object(
        BinanceFuturesTorchTradingEnv, "_wait_for_next_timestamp"
    ):
        env = BinanceFuturesTorchTradingEnv(
            config=config, observer=observer, trader=trader
        )

    observer.get_observations.assert_not_called()
    assert env.observation_spec["market_data_1Minute_10"].shape == torch.Size([10, 4])
    # Behavioural, not "some assignment exists": only alpaca's copy set account_state and
    # the fold dropped it, and examples/llm/{frontier,local}/live.py read it to label the
    # observation. The AST form this replaced passed with the labels set to ["MUTANT"].
    assert env.account_state == type(env).ACCOUNT_STATE


def test_each_timeframe_declares_its_own_window_size():
    """The loop pairs `get_keys()` with `config.window_sizes` positionally.

    Collapsing every window to `window_sizes[0]` passed the whole suite bar this test:
    every other multi-timeframe env test uses uniform sizes, so the pairing was never
    exercised. This is
    the shape-corruption class the spec guards exist for -- a policy fed a 20-bar window
    declared as 10.
    """
    from torchtrade.envs.live.binance.env import (
        BinanceFuturesTorchTradingEnv,
        BinanceFuturesTradingEnvConfig,
    )

    observer = MagicMock()
    observer.get_keys = MagicMock(return_value=["1Minute_10", "1Hour_20"])
    observer.get_features = MagicMock(return_value={
        "observation_features": ["a", "b", "c"], "original_features": []})

    trader = MagicMock()
    trader.get_account_balance.return_value = {
        "available_balance": 1000.0, "total_margin_balance": 1000.0,
        "total_wallet_balance": 1000.0,
    }
    trader.get_status.return_value = {"position_status": None}
    trader.cancel_open_orders.return_value = True
    trader.close_position.return_value = True
    trader.get_mark_price.return_value = 100.0

    config = BinanceFuturesTradingEnvConfig(
        symbol="BTCUSDT", time_frames=["1m", "1h"], window_sizes=[10, 20], execute_on="1m",
    )
    with patch("time.sleep"), patch.object(
        BinanceFuturesTorchTradingEnv, "_wait_for_next_timestamp"
    ):
        env = BinanceFuturesTorchTradingEnv(
            config=config, observer=observer, trader=trader
        )

    assert env.observation_spec["market_data_1Minute_10"].shape == torch.Size([10, 3])
    assert env.observation_spec["market_data_1Hour_20"].shape == torch.Size([20, 3])


def test_an_observer_disagreeing_with_the_config_raises_rather_than_guessing():
    """`zip(..., strict=True)` -- the fail-open case the old index fallback absorbed.

    Every config's __post_init__ normalizes window_sizes to a list as long as time_frames,
    so a mismatch can only come from an INJECTED observer. The old
    `window_sizes[i] if i < len(window_sizes) else window_sizes[0]` silently declared the
    first window for every extra key, which is a spec that quietly lies about its shape.
    Dropping strict=True failed nothing before this.
    """
    from torchtrade.envs.live.binance.env import (
        BinanceFuturesTorchTradingEnv,
        BinanceFuturesTradingEnvConfig,
    )

    observer = MagicMock()
    observer.get_keys = MagicMock(return_value=["1Minute_10", "1Hour_20"])  # 2 keys...
    observer.get_features = MagicMock(return_value={
        "observation_features": ["a", "b"], "original_features": []})

    trader = MagicMock()
    trader.get_account_balance.return_value = {
        "available_balance": 1000.0, "total_margin_balance": 1000.0,
        "total_wallet_balance": 1000.0,
    }
    trader.get_status.return_value = {"position_status": None}
    trader.cancel_open_orders.return_value = True
    trader.close_position.return_value = True
    trader.get_mark_price.return_value = 100.0

    config = BinanceFuturesTradingEnvConfig(  # ...against 1 window
        symbol="BTCUSDT", time_frames=["1m"], window_sizes=[10], execute_on="1m",
    )
    with patch("time.sleep"), patch.object(
        BinanceFuturesTorchTradingEnv, "_wait_for_next_timestamp"
    ), pytest.raises(ValueError, match="zip"):
        BinanceFuturesTorchTradingEnv(config=config, observer=observer, trader=trader)


@pytest.mark.parametrize("env_cls", STEPPING_ENVS, ids=lambda c: c.__name__)
def test_every_live_step_writes_done_explicitly(env_cls):
    """Every live `_step` writes the done family through the shared writer (#295).

    Not because TorchRL cannot derive `done` -- `EnvBase._complete_done` fills
    `terminated | truncated` when the key is absent. Because these envs emit the whole
    family from `_step` itself, which many tests drive directly, and because a venue
    reimplementing the writes by hand is how `truncated` became a declared constant in
    the first place.
    """
    def _attr_calls(func):
        return [n.func.attr for n in ast.walk(ast.parse(inspect.getsource(func).lstrip()))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]

    # One hop is allowed, and only one: #288 moved the record-then-score tail onto
    # `_record_and_score`, so `_step` now reaches the writer through it. The delegate is
    # pinned below, so "delegates" cannot become "delegates to something that stopped
    # writing the family".
    step_calls = _attr_calls(env_cls._step)
    assert ("_finalize_step_flags" in step_calls
            or "_record_and_score" in step_calls), (
        f"{env_cls.__name__}._step neither calls the shared done-family writer nor the "
        f"tail that does, so its truncation channel is unreachable"
    )
    assert "_finalize_step_flags" in _attr_calls(env_cls._record_and_score), (
        f"{env_cls.__name__}._record_and_score stopped calling the done-family writer; "
        f"every `_step` that delegates to it silently lost its truncation channel"
    )
    # A substring check passed on a commented-out call with the writes reimplemented
    # alongside it. Only binance had an end-to-end backstop for that; nine venues did not.
    # Both bodies, since the writes could be hand-rolled in either.
    hand_written = [
        n for func in (env_cls._step, env_cls._record_and_score)
        for n in ast.walk(ast.parse(inspect.getsource(func).lstrip()))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "set"
        and n.args and isinstance(n.args[0], ast.Constant)
        and n.args[0].value in ("done", "terminated", "truncated")
    ]
    assert not hand_written, (
        f"{env_cls.__name__}._step writes the done family by hand alongside "
        f"_finalize_step_flags; the hand-written value wins and the shared one is a lie"
    )


@pytest.mark.parametrize("terminated,unknown,budget,expected", [
    (False, 0, 3, (False, False, False)),   # healthy
    (True,  0, 3, (True,  True,  False)),   # bankruptcy terminates, never truncates
    (False, 3, 3, (True,  False, True)),    # outage spends the budget -> truncate
    (False, 9, 0, (False, False, False)),   # budget 0 disables the channel entirely
    (True,  3, 3, (True,  True,  True)),    # both: done stays the OR of the two
], ids=["healthy", "bankrupt", "outage", "disabled", "both"])
def test_the_done_family_separates_an_outage_from_a_blown_account(
    terminated, unknown, budget, expected
):
    """A prolonged outage truncates; only a blown account terminates.

    Value estimators read `terminated` as "true return-to-go is 0" and `truncated` as
    "bootstrap from the final observation". Terminating on an unreachable API would teach
    the critic that a broker outage and a liquidated account carry the same value target.
    """
    env = SimpleNamespace(
        consecutive_unknown_status=unknown,
        _status_unknown_this_step=bool(unknown),
        config=SimpleNamespace(max_unknown_status_steps=budget),
    )
    env._max_unknown_status_steps = budget
    td = TensorDict({}, batch_size=())
    TorchTradeLiveEnv._finalize_step_flags(env, td, terminated=terminated)

    got = (bool(td["done"]), bool(td["terminated"]), bool(td["truncated"]))
    assert got == expected, f"done/terminated/truncated = {got}, expected {expected}"


@pytest.mark.parametrize("env_cls", STEPPING_ENVS, ids=lambda c: c.__name__)
def test_every_live_reset_clears_the_outage_state(env_cls):
    """A truncated episode must not poison the next one (#295).

    `_finalize_step_flags` derives `truncated` from the counter, so an episode that ended
    on a spent budget would start the NEXT one already at budget and truncate on its first
    step -- 1-step episodes forever, and a collector that looks busy while collecting
    nothing. AST rather than source text, so a comment naming the method cannot satisfy it.
    """
    calls = [n.func.attr for n in ast.walk(ast.parse(
        inspect.getsource(env_cls._reset).lstrip()))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    # `_reset` accepted as delegation: the SLTP envs override _reset and call
    # super()._reset(), which is where the clear lives. What this rejects is an override
    # that does NEITHER -- reimplementing reset and dropping the clear on the floor.
    assert {"_reset_outage_state", "_reset"} & set(calls), (
        f"{env_cls.__name__}._reset neither clears the outage counter nor delegates to a "
        f"_reset that does; a truncated episode would make every following episode "
        f"truncate on step 1"
    )


def test_a_recovered_venue_does_not_truncate_the_next_episode():
    """The behavioural half of the guard above, driven through a real env."""
    from unittest.mock import MagicMock, patch
    import numpy as np
    from torchtrade.envs.core.state import PositionUnknownError
    from torchtrade.envs.live.binance.env import (
        BinanceFuturesTorchTradingEnv as Env, BinanceFuturesTradingEnvConfig as Config,
    )

    observer = MagicMock()
    observer.get_keys.return_value = ["1m_10"]
    observer.get_observations.return_value = {"1m_10": np.zeros((10, 4), dtype=np.float32)}
    observer.get_features.return_value = {
        "observation_features": list("abcd"), "original_features": []}
    observer.reset.return_value = None
    trader = MagicMock()
    trader.get_account_balance.return_value = {
        "available_balance": 1e4, "total_margin_balance": 1e4,
        "total_wallet_balance": 1e4, "total_maintenance_margin": 0.0}
    trader.get_status.return_value = {"position_status": None}
    trader.get_mark_price.return_value = 100.0
    trader.get_lot_size.return_value = {"min_qty": 0.001, "qty_step": 0.001}

    with patch("time.sleep"), patch.object(Env, "_wait_for_next_timestamp"):
        env = Env(config=Config(
            symbol="BTCUSDT", time_frames=["1m"], window_sizes=[10], execute_on="1m",
            max_unknown_status_steps=2, close_position_on_init=False,
        ), observer=observer, trader=trader)

        td = env.reset()
        td = env.step(td.set("action", torch.tensor(1)))["next"]
        trader.get_status.side_effect = PositionUnknownError("down")
        for _ in range(2):
            td = env.step(td.exclude("done", "terminated", "truncated", "reward")
                          .set("action", torch.tensor(1)))["next"]
        assert bool(td["truncated"]), "setup: the outage should have truncated"

        trader.get_status.side_effect = None          # venue recovers
        fresh = env.reset()
        assert env.consecutive_unknown_status == 0
        nxt = env.step(fresh.set("action", torch.tensor(1)))["next"]

    assert not bool(nxt["truncated"]), (
        "the new episode truncated on its first step: the outage counter survived reset"
    )


@pytest.mark.parametrize("config_cls", [
    BinanceFuturesTradingEnvConfig, BitgetFuturesTradingEnvConfig,
    BybitFuturesTradingEnvConfig, OKXFuturesTradingEnvConfig,
    BinanceFuturesSLTPTradingEnvConfig,
], ids=lambda c: c.__name__)
@pytest.mark.parametrize("bad", [-1, 1.5, True, "3", None], ids=[
    "negative", "fractional", "bool", "str", "none"])
def test_an_unusable_outage_budget_is_rejected_at_the_boundary(config_cls, bad):
    """A negative budget reads as "disabled" through `>= budget > 0`.

    So a typo silently selects the STRICTEST posture rather than erroring -- the shape
    invariant 4 names: a guard that absorbs nonsense is worse than no guard. `True` is
    here because bool is an int subclass, and would otherwise pass as a budget of 1.

    Deleting the validator's body failed ZERO tests before this: it shipped unpinned.

    Four plain configs, because deleting the call from ONE of them fails only its own
    cases -- they are four independent regression points, which is the shape this repo
    keeps shipping. One SLTP config, because deleting it from BaseFuturesSLTPConfig fails
    all four together: they are one call site wearing four names, pinned below.
    """
    with pytest.raises(ValueError, match="max_unknown_status_steps"):
        config_cls(symbol="TEST", max_unknown_status_steps=bad)


@pytest.mark.parametrize("config_cls", [c.values[0] for c in SLTP_CONFIGS],
                         ids=lambda c: c.__name__)
def test_every_sltp_config_shares_the_one_validated_post_init(config_cls):
    """The cheaper half of the sweep above, and a stronger guard than re-running it.

    Re-running five bad inputs against four configs that resolve to the SAME function
    tests one thing four times. What can actually break is a subclass growing its own
    `__post_init__` and dropping the `super()` call -- which this catches and the
    parametrized sweep would not, since the venue would simply stop validating.
    """
    assert config_cls.__post_init__ is BaseFuturesSLTPConfig.__post_init__, (
        f"{config_cls.__name__} defines its own __post_init__; it must call super() or "
        f"it silently stops validating every field the shared one checks"
    )


def test_every_successful_close_invalidates_the_cached_balance():
    """A realised close moves equity, so the cached sizing balance is stale after it.

    Seven close sites exist: the shared `_handle_close_action` (which the direction-switch
    leg routes through) and six across the SLTP envs, in three different shapes. I patched
    one and reported it done -- the exact "landed on some copies, not others" failure this
    PR has now reproduced five times, that time while fixing an instance of it.

    Structural, because the cost is invisible behaviourally: a stale balance sizes an
    order the venue rejects on margin, many bars later, only during an outage.

    Two functions are exempt BY ORDERING, not by oversight. `_halting`'s emergency FLATTEN
    close runs when the venue is already unreadable -- the cache is what grace is standing
    on. `_reset` clears the whole cache before it closes and re-seeds from a fresh read
    after, so its balance already reflects the close.
    """
    import ast
    import pathlib

    # `_finish_futures_init` runs at CONSTRUCTION, before any read has been cached, so
    # its startup flatten has nothing to invalidate -- same ordering argument as `_reset`.
    EXEMPT = {"_halting", "_reset", "_finish_futures_init"}
    root = pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent / "live"
    offenders = []
    for path in sorted(root.glob("*/env_sltp.py")) + [
        root / "shared" / "futures_live_base.py"
    ]:
        for fn in ast.walk(ast.parse(path.read_text())):
            if not isinstance(fn, ast.FunctionDef) or fn.name in EXEMPT:
                continue
            closes = [n for n in ast.walk(fn)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr == "close_position"]
            if not closes:
                continue
            invalidations = [n for n in ast.walk(fn)
                             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr == "pop" and n.args
                             and isinstance(n.args[0], ast.Constant)
                             and n.args[0].value == "balance"]
            if len(invalidations) < len(closes):
                offenders.append(f"{path.parent.name}/{path.name}::{fn.name} "
                                 f"({len(closes)} closes, {len(invalidations)} invalidations)")

    assert not offenders, (
        "a successful close leaves the cached sizing balance stale in: "
        + "; ".join(offenders)
    )


@pytest.mark.parametrize("venue,expect", [
    ("binance", {}),
    ("bitget",  {}),
    ("bybit",   {"client": "SHARED"}),
    ("okx",     {}),
], ids=["binance", "bitget", "bybit", "okx"])
def test_each_venue_builds_its_observer_with_the_arguments_it_needs(venue, expect):
    """The non-DI path, which no test in the repo exercised: 83 constructions inject an
    observer, 2 do not, and those 2 need live credentials so they skip.

    So `_observer_kwargs` -- the whole point of #288 slice 1 -- was unreachable by the
    suite, and it shipped with two bugs. okx inherited a `client=` override copied from
    bybit, handing its observer the trader's `Trade.TradeAPI`, which has no
    `get_candlesticks`: every kline fetch would have raised. bitget lost `product_type`,
    so a COIN-FUTURES deployment would trade one product line and read candles from
    another, silently.

    `client` is asserted as a CATEGORY, not a value: bybit shares one session between the
    two roles and the other three must not, which is the actual contract.
    """
    import importlib

    # The concrete plain env, not the abstract exchange base: the base still declares
    # `_step`/`_execute_trade_if_needed` abstract, so it cannot be instantiated at all.
    module = importlib.import_module(f"torchtrade.envs.live.{venue}.env")
    cls = _sole(module, "TorchTradingEnv")

    # A real instance via __new__, not a SimpleNamespace: `_observer_kwargs` calls
    # zero-arg `super()`, which needs the class in its MRO. __init__ is skipped so the
    # venue is never contacted.
    env = cls.__new__(cls)
    env.config = SimpleNamespace(
        symbol="X", time_frames=["1m"], window_sizes=[10], demo=True,
        leverage=5, margin_mode="isolated", position_mode="one_way",
        product_type="USDT-FUTURES",
    )
    env._feature_preprocessing_fn = None
    env.trader = SimpleNamespace(client="SHARED")
    kwargs = env._observer_kwargs()

    base_keys = {"symbol", "time_frames", "window_sizes",
                 "feature_preprocessing_fn", "demo"}
    assert base_keys <= set(kwargs), f"{venue} dropped {base_keys - set(kwargs)}"

    extras = {k: v for k, v in kwargs.items() if k not in base_keys}
    assert extras == expect, (
        f"{venue} observer extras are {extras}, expected {expect}. A client the venue "
        f"does not share is a broken market-data feed; a missing product_type is a "
        f"silent wrong-market read."
    )


@pytest.mark.parametrize("env_cls", LIVE_ENVS, ids=lambda c: c.__name__)
def test_no_venue_redeclares_the_account_state_contract(env_cls):
    """`ACCOUNT_STATE` lived in five identical copies before #288.

    It is the observation contract itself -- CLAUDE.md's "universal 6 elements", shared
    with the offline envs. A venue re-declaring it can reorder or rename an element and
    every existing test still passes, because each env is only ever compared against its
    OWN list. What breaks is a trained checkpoint, silently, at a different index.

    Structural for the reason this file keeps needing structural tests: a re-forked copy
    that has not drifted yet passes everything.
    """
    offenders = [c.__name__ for c in env_cls.__mro__
                 if "ACCOUNT_STATE" in c.__dict__
                 and c.__name__ != "TorchTradeLiveEnv"]
    assert not offenders, (
        f"{env_cls.__name__} resolves ACCOUNT_STATE from {offenders} rather than the "
        f"shared contract on TorchTradeLiveEnv; a reordered copy is a permuted "
        f"observation that no test compares against anything else"
    )


@pytest.mark.parametrize("venue,expect_order,observer_extras,trader_extras", [
    ("binance", ["observer", "trader"], set(),
     {"trade_mode": "fractional"}),
    ("bitget",  ["observer", "trader"], set(),
     {"trade_mode": "fractional", "position_mode": "one_way",
      "product_type": "COIN-FUTURES", "passphrase": "PASS"}),
    ("bybit",   ["trader", "observer"], {"client"},
     {"position_mode": "one_way"}),
    ("okx",     ["trader", "observer"], set(),
     {"position_mode": "one_way", "passphrase": "PASS"}),
], ids=["binance", "bitget", "bybit", "okx"])
def test_init_trading_clients_wires_each_venue_as_before(
    venue, expect_order, observer_extras, trader_extras
):
    """Characterisation of the code #288 actually refactored, network-free.

    The previous test called `_observer_kwargs()` directly on a `__new__`'d instance, so
    it never ran `_init_trading_clients` -- the method that selects the order, constructs
    both classes and consumes both hooks. A mutation of a hook failed it while a wiring
    regression at any of the eight call sites passed. Same error as the kwargs probe in
    the PR body: verifying something adjacent to the thing that changed.

    Construction ORDER is asserted, not just kwargs. bybit's observer reuses the trader's
    session, so it must be built second; okx must NOT, because its trader client is a
    Trade API and klines live on the MarketData one -- the round-1 regression.
    """
    import importlib

    module = importlib.import_module(f"torchtrade.envs.live.{venue}.env")
    cls = _sole(module, "TorchTradingEnv")

    log = []

    class FakeObserver:
        def __init__(self, **kw):
            log.append(("observer", kw))

    class FakeTrader:
        def __init__(self, **kw):
            log.append(("trader", kw))
            self.client = "TRADER_SESSION"

    env = cls.__new__(cls)
    env.config = SimpleNamespace(
        symbol="X", time_frames=["1m"], window_sizes=[10], demo=True,
        leverage=5, margin_mode="isolated", position_mode="one_way",
        product_type="COIN-FUTURES", trade_mode="fractional",
    )
    env._feature_preprocessing_fn = None
    env._passphrase = env._api_passphrase = "PASS"

    with patch.object(cls, "OBSERVER_CLS", FakeObserver), \
         patch.object(cls, "TRADER_CLS", FakeTrader):
        env._init_trading_clients("KEY", "SECRET", None, None)

    assert [name for name, _ in log] == expect_order, (
        f"{venue} built {[n for n, _ in log]}, expected {expect_order}"
    )

    kw = dict(log)
    base_obs = {"symbol", "time_frames", "window_sizes", "feature_preprocessing_fn", "demo"}
    assert set(kw["observer"]) == base_obs | observer_extras, (
        f"{venue} observer kwargs {sorted(kw['observer'])}, expected "
        f"{sorted(base_obs | observer_extras)}"
    )
    if "client" in observer_extras:
        assert kw["observer"]["client"] == "TRADER_SESSION", (
            f"{venue} shares the trader's session, so it must receive that object"
        )

    # EXACT, not a subset. A subset check passed while binance dropped `trade_mode`,
    # bitget dropped `product_type`, okx dropped `passphrase` and bybit's `position_mode`
    # changed value -- the routing `_trader_kwargs` exists to do was entirely unpinned.
    base_tr = {"symbol": "X", "api_key": "KEY", "api_secret": "SECRET",
               "demo": True, "leverage": 5, "margin_mode": "isolated"}
    assert kw["trader"] == {**base_tr, **trader_extras}, (
        f"{venue} trader kwargs {kw['trader']}, expected {{**base, **{trader_extras}}}"
    )


def test_dependency_injection_still_skips_construction_entirely():
    """A supplied observer/trader must win, on the shared path as on the old per-venue one.

    83 of the suite's 85 env constructions rely on this, so a regression here would show
    up everywhere at once -- but it is the shared method's contract now, and nothing
    asserted it directly.
    """
    from torchtrade.envs.live.binance.env import BinanceFuturesTorchTradingEnv as Env

    def explode(**kw):
        raise AssertionError("constructed a client despite dependency injection")

    env = Env.__new__(Env)
    env.config = SimpleNamespace(
        symbol="X", time_frames=["1m"], window_sizes=[10], demo=True,
        leverage=5, margin_mode="isolated", trade_mode="fractional",
    )
    env._feature_preprocessing_fn = None
    supplied_obs, supplied_tr = object(), object()

    with patch.object(Env, "OBSERVER_CLS", explode), patch.object(Env, "TRADER_CLS", explode):
        env._init_trading_clients("K", "S", supplied_obs, supplied_tr)

    assert env.observer is supplied_obs and env.trader is supplied_tr


# alpaca is absent DELIBERATELY, and that is worth stating rather than leaving to be
# inferred from an omission: it is a SPOT env whose SLTP `_step` is only 35% identical to
# the mixin's (no futures pre-trade acquisition, no mark). It overrides `_step` on purpose
# and inherits `_reset`, which IS identical. Deleting alpaca's `_step` as "redundant"
# would hand a spot env the futures step.
# Per OWNER, because the two own different method sets.
#
# `_dispatch_sltp_trade` is deliberately ABSENT. It is the venue-variation hook -- its own
# docstring calls it "the ONE thing `_step` varies by venue" -- and bybit and okx really do
# override it, to thread the price rather than re-read it (#295). Guarding it would demand
# a fold of the one method that exists not to be folded.
_SHARED_METHOD_OWNERSHIP = [
    # `_record_and_score` is owned by TorchTradeLiveEnv and reached by all TEN stepping
    # envs, alpaca included -- the only tail in this file that alpaca shares (#288).
    *((c, TorchTradeLiveEnv, "_record_and_score") for c in STEPPING_ENVS),
    *((c, TorchTradeFuturesLiveEnv, m)
      for c in PLAIN_FUTURES_ENVS for m in ("_step", "_reset")),
    # binance is absent by design: it EXTENDS the sizing via super() (min-notional refusal
    # and target rounding), which its own behavioural test pins.
    *((c, TorchTradeFuturesLiveEnv, "_calculate_fractional_position")
      for c in PLAIN_FUTURES_ENVS if c.__module__.split(".")[-2] != "binance"),
    *((c, SLTPMixin, m) for c in SLTP_FUTURES_ENVS for m in (
        "_step", "_reset", "_resolve_action_tuple", "_record_sltp_position",
        "_reset_sltp_state",
    )),
]
# By NAME, not by count. Dropping `_reset` from the matrix above passed 936 tests: the
# registry-length pins guard the ENV axis, and nothing guarded the METHOD axis. This also
# says out loud what is meant to be shared, so adding a genuinely shared method is a
# deliberate edit here rather than an arithmetic fix.
assert {(owner.__name__, method) for _, owner, method in _SHARED_METHOD_OWNERSHIP} == {
    ("TorchTradeLiveEnv", "_record_and_score"),
    ("TorchTradeFuturesLiveEnv", "_step"),
    ("TorchTradeFuturesLiveEnv", "_reset"),
    ("TorchTradeFuturesLiveEnv", "_calculate_fractional_position"),
    ("SLTPMixin", "_step"),
    ("SLTPMixin", "_reset"),
    ("SLTPMixin", "_resolve_action_tuple"),
    ("SLTPMixin", "_record_sltp_position"),
    ("SLTPMixin", "_reset_sltp_state"),
}


def test_binance_extends_the_shared_sizing_rather_than_replacing_it():
    """binance's override must still run the shared arithmetic underneath its extras.

    Replacing rather than extending is how the halt-policy balance read, the isfinite
    guard and the 2% buffer would quietly stop applying to the one venue whose sizing has
    the most steps in it.
    """
    cls = _sole(importlib.import_module("torchtrade.envs.live.binance.env"),
                "TorchTradingEnv")
    assert "super()._calculate_fractional_position" in inspect.getsource(
        cls._calculate_fractional_position
    ), "binance's sizing no longer delegates to the shared implementation"

    # By BEHAVIOUR, not by grepping the body for `_get_min_notional`: neutering the `if`
    # that uses it left the name in the source and the string check passed.
    env = cls.__new__(cls)
    env.config = SimpleNamespace(leverage=10)
    env.trader = MagicMock()
    env.trader.get_account_balance.return_value = {"total_margin_balance": 10_000.0}
    env.trader.round_quantity.side_effect = lambda q: round(float(q), 3)
    env._halting = lambda read, cache_key=None: read()

    env._get_min_notional = lambda: 1e12          # nothing can clear this
    assert env._calculate_fractional_position(1.0, 100.0) == (0.0, 0.0, "flat"), (
        "binance opened a position whose notional is below the venue minimum; the "
        "exchange rejects it and the env then believes it holds one"
    )

    env._get_min_notional = lambda: 0.0
    env.trader.round_quantity.side_effect = lambda q: 0.123456
    # Signed, and both directions. `abs(size)` discarded precisely the expression under
    # test -- `position_qty * (1 if position_size > 0 else -1)` -- and only long was run,
    # so the short branch was never entered at all.
    for action, expected in ((1.0, 0.123456), (-1.0, -0.123456)):
        size, _, _ = env._calculate_fractional_position(action, 100.0)
        assert size == pytest.approx(expected), (
            f"binance sized {size} for action {action}, not the executor's rounded "
            f"quantity with the direction re-applied (#271)"
        )


# BOTH module axes. The plain and SLTP classes are siblings, not parent and child -- the
# SLTP MRO runs SLTPMixin -> <Venue>Base -> TorchTradeFuturesLiveEnv and never touches the
# plain leaf. Setting the fee on the leaf resolved it for four of the eight classes that
# inherit the shared sizing and AttributeError'd for the other four, and every test I wrote
# for the fee iterated the plain list only.
# Derived from the registry and pinned by NAME, so a fifth venue fails here rather than
# on its first live trade. The values stay literal: the failure being guarded is every
# venue reading ONE fee, which a derived expected value would reproduce.
_EXPECTED_TAKER_FEES = {
    "binance": 0.0004, "bitget": 0.0006, "bybit": 0.00055, "okx": 0.0005,
}
assert set(_EXPECTED_TAKER_FEES) == {c.__module__.split(".")[-2]
                                     for c in PLAIN_FUTURES_ENVS}


@pytest.mark.parametrize("module", ["env", "env_sltp"])
@pytest.mark.parametrize("venue,fee", sorted(_EXPECTED_TAKER_FEES.items()))
def test_each_venue_keeps_its_own_taker_fee(venue, fee, module):
    """The four sizing bodies were byte-identical TEXT and not identical BEHAVIOUR.

    `TAKER_FEE` resolved to a different value in each module, so folding them without
    lifting the fee onto the class would have silently re-priced three venues. Sizing is
    `1 + leverage * fee`, so at 125x the binance and bitget fees size a position ~2.3%
    apart. Pinned by VALUE, because the failure mode is one shared constant that looks
    right on the venue you happen to test.
    """
    cls = _sole(importlib.import_module(f"torchtrade.envs.live.{venue}.{module}"),
                "TorchTradingEnv")
    assert getattr(cls, "TAKER_FEE", None) == pytest.approx(fee), (
        f"{cls.__name__}'s taker fee is {getattr(cls, 'TAKER_FEE', None)}, not {fee}; "
        f"every venue reading one fee is what a careless fold of the sizing bodies would "
        f"produce, and a fee only the plain sibling can see is what a careless fix does"
    )


@pytest.mark.parametrize("cls", PLAIN_FUTURES_ENVS + SLTP_FUTURES_ENVS,
                         ids=lambda c: c.__name__)
def test_every_class_that_inherits_the_shared_sizing_can_run_it(cls):
    """Resolving `self.TAKER_FEE` is not enough -- the method has to actually execute.

    The SLTP classes inherit `_calculate_fractional_position` from the futures base while
    the fee sat on the plain leaf, so calling it raised AttributeError mid-sizing. Nothing
    reached it yet, which is exactly why no test noticed: a landmine rather than a bug.
    """
    env = cls.__new__(cls)
    env.config = SimpleNamespace(leverage=10)
    env.trader = MagicMock()
    env.trader.get_account_balance.return_value = {"total_margin_balance": 10_000.0}
    env.trader.round_quantity.side_effect = lambda q: round(float(q), 3)
    env._halting = lambda read, cache_key=None: read()
    env._get_min_notional = lambda: 0.0

    size, notional, side = env._calculate_fractional_position(1.0, 100.0)

    assert side == "long" and size > 0 and notional > 0, (
        f"{cls.__name__} inherits the shared sizing but produced {(size, notional, side)}"
    )


def test_a_venue_without_a_taker_fee_fails_at_construction():
    """Boundary, not rule. Without this the failure is an AttributeError raised the first
    time a nonzero action is sized -- mid-`_step`, with a position possibly already open.
    """
    # Everything the tail touches AFTER the check, so the TypeError is what fails rather
    # than the first missing attribute. A venue that forgets the fee has a real config.
    env = SimpleNamespace(
        config=SimpleNamespace(execute_on="1m", close_position_on_init=False),
        trader=MagicMock(),
    )
    with pytest.raises(TypeError, match="does not set TAKER_FEE"):
        TorchTradeFuturesLiveEnv._finish_futures_init(env)


def test_the_sizing_reserves_the_maintenance_margin_buffer():
    """The 2% haircut on the balance. Dropping it failed ZERO tests.

    It is the venue's maintenance-margin headroom: sizing against the full balance leaves
    nothing between the position and a margin call on the first adverse tick. Asserted as
    a ratio between two sizings rather than a magic notional, so it survives a change to
    the fee or the leverage.
    """
    cls = _sole(importlib.import_module("torchtrade.envs.live.bybit.env"),
                "TorchTradingEnv")

    def notional_for(balance):
        env = cls.__new__(cls)
        env.config = SimpleNamespace(leverage=10)
        env.trader = MagicMock()
        env.trader.get_account_balance.return_value = {"total_margin_balance": balance}
        env._halting = lambda read, cache_key=None: read()
        return env._calculate_fractional_position(1.0, 100.0)[1]

    # Against an INDEPENDENT expected value, not a ratio between two of the env's own
    # sizings: sizing is linear in the balance, so `n(B) == n(B/0.98) * 0.98` holds whether
    # or not the buffer exists. That first version passed with the buffer deleted.
    from torchtrade.envs.utils.fractional_sizing import (
        PositionCalculationParams, calculate_fractional_position,
    )
    expected = calculate_fractional_position(PositionCalculationParams(
        balance=10_000.0 * 0.98, action_value=1.0, current_price=100.0,
        leverage=10, transaction_fee=cls.TAKER_FEE,
    ))[1]
    assert notional_for(10_000.0) == pytest.approx(expected), (
        f"sized against the full balance rather than 98% of it: the 2% maintenance-margin "
        f"headroom is what keeps the first adverse tick from being a margin call"
    )


def test_the_shared_sizing_actually_uses_the_venues_fee():
    """And the fee must reach the arithmetic, not merely be declared on the class.

    Pinning the four constants proves they differ; this proves the shared body reads
    `self.TAKER_FEE` rather than a module-level one it closed over.
    """
    sizes = {}
    for venue in ("binance", "bitget", "bybit", "okx"):
        cls = _sole(importlib.import_module(f"torchtrade.envs.live.{venue}.env"),
                    "TorchTradingEnv")
        env = cls.__new__(cls)
        env.config = SimpleNamespace(leverage=125)
        env.trader = MagicMock()
        env.trader.get_account_balance.return_value = {"total_margin_balance": 10_000.0}
        env.trader.round_quantity.side_effect = lambda q: q
        env._halting = lambda read, cache_key=None: read()
        env._get_min_notional = lambda: 0.0
        sizes[venue] = env._calculate_fractional_position(1.0, 100.0)[1]

    assert len(set(round(v, 6) for v in sizes.values())) == 4, (
        f"four venues with four different taker fees produced {sizes}; identical notionals "
        f"mean the shared body is not reading self.TAKER_FEE"
    )


# binance and bitget take the mixin's `_dispatch_sltp_trade`; bybit and okx override it to
# thread the price (#295). That split is intended, so the hook is out of the uniform table
# above -- but "intended for two venues" is not "unguarded for the other two": a later
# private override in binance or bitget would otherwise evade every guard in this file.
_DISPATCH_INHERITORS = [c for c in SLTP_FUTURES_ENVS
                        if c.__module__.split(".")[-2] in ("binance", "bitget")]
_DISPATCH_OVERRIDERS = [c for c in SLTP_FUTURES_ENVS
                        if c.__module__.split(".")[-2] in ("bybit", "okx")]
assert len(_DISPATCH_INHERITORS) == len(_DISPATCH_OVERRIDERS) == 2


@pytest.mark.parametrize("cls", _DISPATCH_INHERITORS, ids=lambda c: c.__name__)
def test_the_dispatch_hook_is_inherited_where_it_is_not_deliberately_overridden(cls):
    """The venue-variation hook is still owned, for the venues that do not vary.

    bybit and okx override `_dispatch_sltp_trade` to pass the threaded price rather than
    let the executor re-read it (#295). binance and bitget price their brackets off a
    candle close and take the mixin's default -- so for them a private copy is a re-fork
    like any other, and nothing said so.
    """
    assert cls._dispatch_sltp_trade is SLTPMixin._dispatch_sltp_trade, (
        f"{cls.__name__} defines its own _dispatch_sltp_trade. Only bybit and okx have a "
        f"reason to (the #295 threaded price); if this venue now needs one too, move it "
        f"into _DISPATCH_OVERRIDERS deliberately rather than letting the hook drift"
    )


@pytest.mark.parametrize("cls", _DISPATCH_OVERRIDERS, ids=lambda c: c.__name__)
def test_the_dispatch_overriders_thread_the_price_they_were_given(cls):
    """And the two that DO override must consume the threaded price, not re-read it.

    Excluding them from the ownership table costs the identity check; this replaces it
    with what identity was standing in for.
    """
    call = next(
        (n for n in ast.walk(ast.parse(
            inspect.getsource(cls._dispatch_sltp_trade).lstrip()))
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and n.func.attr == "_execute_trade_if_needed"),
        None,
    )
    assert call is not None, f"{cls.__name__}'s override never dispatches the trade"
    assert any(kw.arg == "current_price" and isinstance(kw.value, ast.Name)
               for kw in call.keywords), (
        f"{cls.__name__} overrides _dispatch_sltp_trade but does not forward the threaded "
        f"current_price -- the only reason the override exists (#295); re-reading it "
        f"inside the trade path bypasses the halt policy"
    )


@pytest.mark.parametrize("cls,owner,method", _SHARED_METHOD_OWNERSHIP,
                         ids=[f"{c.__name__}-{m}" for c, _, m in _SHARED_METHOD_OWNERSHIP])
def test_no_futures_env_reforks_a_shared_step_or_reset(cls, owner, method):
    """An incomplete fold is invisible behaviourally, right up until a fix lands on the
    shared copy and one venue does not get it.

    Identity against the owner, not `"_step" not in vars(cls)`: those are equivalent only
    while SLTPMixin sits directly after the class in the MRO. Insert one intermediate base
    and absence-from-the-subclass starts passing on a copy that really does shadow the
    owner; identity does not care where the copy is planted.
    """
    assert getattr(cls, method) is getattr(owner, method), (
        f"{cls.__name__}.{method} resolves to {getattr(cls, method).__qualname__} rather "
        f"than {owner.__name__}.{method}; a private copy is where a shared fix silently "
        f"fails to land"
    )


def test_no_sltp_env_writes_the_dead_position_closed_field():
    """`trade_info["position_closed"]` had five writers and zero readers.

    `_sync_position_from_exchange` already acts on its own return value; the field was
    pure metadata. I removed it from the shared copy and left alpaca's -- this issue's own
    defect, committed while reviewing the change that created it, which is why the guard
    is here rather than a note in a commit message.
    """
    import pathlib

    root = pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent
    # AST, not substring: my first version matched the COMMENTS explaining the removal
    # and failed on a clean tree. That is the same defect the pre-trade guard had -- a
    # source-text check cannot tell code from prose about the code.
    offenders = []
    for path in list(root.glob("live/*/env_sltp.py")) + [root / "utils" / "sltp_mixin.py"]:
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Subscript)
                            and isinstance(t.slice, ast.Constant)
                            and t.slice.value == "position_closed"
                            for t in node.targets)):
                offenders.append(path.relative_to(root).as_posix())
    assert not offenders, (
        f"{offenders} write a field nothing reads; if it gains a reader, delete this test "
        f"deliberately rather than letting the write drift back in"
    )


@pytest.mark.parametrize("side_idx,expected", [(1, 1.0), (-1, -1.0), (0, 0.0)],
                         ids=["long", "short", "hold"])
def test_the_history_row_records_the_side_actually_traded(side_idx, expected):
    """`action_value` is what the reward function reads out of `history.actions`.

    Flipping the long/short sign here fails ZERO of 4196 tests -- found while mutating the
    freshly-folded `_step`, and pre-existing rather than introduced by the fold. It is the
    worst shape of bug this repo records: plausible numbers at the wrong sign, so the
    reward inverts silently and nothing crashes.

    Driven through the real shared `_step` rather than the expression, because the
    expression is not what can drift -- the mapping from a bracket SIDE to a numeric
    action is.
    """
    import torch
    env, trader = _real_futures_env(budget=0, venue="binance", sltp=True)
    td = env.reset()

    # The action index whose bracket side is long / short / hold.
    want = {1: "long", -1: "short", 0: None}[side_idx]
    action = next(i for i, tup in sorted(env.action_map.items()) if tup[0] == want)

    env.step(td.set("action", torch.tensor(action)))
    assert env.history.actions[-1] == pytest.approx(expected), (
        f"a {want or 'hold'} bracket recorded {env.history.actions[-1]} in the history "
        f"row the reward function reads, expected {expected}"
    )


@pytest.mark.parametrize("trade_info,expect_recorded", [
    ({"executed": True, "success": True}, True),
    ({"executed": True, "success": False}, False),   # the venue REFUSED the bracket
    ({"executed": False}, False),                    # nothing was sent
], ids=["accepted", "refused", "not-sent"])
def test_a_refused_sltp_bracket_does_not_write_a_phantom_position(trade_info,
                                                                  expect_recorded):
    """`executed and success is not False` gates the SLTP position write.

    Replacing that condition with `if True:` survives the whole suite: a bracket the venue
    REFUSED then writes `position.current_position`, so the cache reads long while the
    account is flat. That is invariant 2 inverted -- the cache overriding exchange truth --
    and the next bar's duplicate-action guard trusts it.

    The plain owner's identical guard in `_record_position_after_trade` is covered; this
    one was not. Same rule, two owners, one pinned -- which is #288's thesis about what
    duplication costs, in the gate rather than in the code.
    """
    env, trader = _real_futures_env(budget=0, venue="binance", sltp=True)
    env._dispatch_sltp_trade = lambda action_tuple, current_price: trade_info

    td = env.reset()
    assert env.position.current_position == 0, "setup: expected to start flat"
    env.active_stop_loss, env.active_take_profit = 111.0, 222.0
    env.step(td.set("action", torch.tensor(1)))     # a bracket action, not the hold

    recorded = env.position.current_position != 0
    assert recorded is expect_recorded, (
        f"trade_info={trade_info} left current_position="
        f"{env.position.current_position}; a refused or unsent bracket must leave the "
        f"cache flat, or the next bar's duplicate-action guard trusts a position that "
        f"does not exist"
    )


def test_a_reset_clears_the_bracket_on_the_shared_sltp_owner():
    """`_reset_sltp_state` is pinned for alpaca only, and alpaca does not share this copy.

    alpaca is spot: it is deliberately absent from SLTP_FUTURES_ENVS, so the identity
    guard does not reach it. Neuter the mixin copy the four FUTURES venues actually use
    and the suite stays green -- stale SL/TP levels then survive into the next episode,
    where they arm brackets the agent never chose.
    """
    env, _ = _real_futures_env(budget=0, venue="binance", sltp=True)
    env.reset()
    env.active_stop_loss, env.active_take_profit = 111.0, 222.0

    env.reset()

    assert (env.active_stop_loss, env.active_take_profit) == (0.0, 0.0), (
        f"reset left SL/TP at ({env.active_stop_loss}, {env.active_take_profit}); a live "
        f"bracket carried into the next episode arms on a position it never opened"
    )


def test_the_trade_takes_the_qty_and_price_the_pre_trade_read_acquired():
    """#295: the shared `_step` FORWARDS the qty and price its pre-trade read acquired.

    Scope, stated because the stub sets it: `_execute_trade_if_needed` is replaced here,
    so this pins the shared CALLER -- what it threads, and that it does not read again.
    It does not prove a venue's executor then uses those values rather than re-reading;
    that is the venues' own contract, covered behaviourally by the SLTP grace-bar tests.

    `_execute_trade_if_needed` is NOT shared, so this contract belongs where the shared
    `_step` is owned rather than in one venue's file.

    The venue reports something DIFFERENT after the first read. That is the whole design:
    with the fixture's constant mocks a re-read returns the identical value, so the first
    version of this test passed under the very mutation it names -- swapping the threaded
    price for `self._current_mark_price()` changed nothing observable. Constants cannot
    distinguish "read once and thread it" from "read again"; only a changing venue can.

    The read COUNT is pinned too, and it is the stricter half: two status reads per step,
    the pre-trade one and the post-bar one. A third is a re-read, which is the window an
    outage slips through even when the values happen to agree.
    """
    # mark 137.0 against entry 100.0 and a fetched mark of 100.0: with an open position
    # the threaded price comes from `position_status.mark_price`, so any of the three
    # plausible re-reads -- `_current_mark_price()` with no status, `get_mark_price()`,
    # `entry_price` -- lands on 100.0 and is distinguishable. They were not when every
    # number in the fixture was 100.0.
    held = PositionStatus(
        qty=0.05, notional_value=6000.0, entry_price=100.0, unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0, mark_price=137.0, leverage=10,
        margin_mode="isolated", liquidation_price=91.0,
    )
    moved = PositionStatus(
        qty=0.99, notional_value=99000.0, entry_price=100.0, unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0, mark_price=999.0, leverage=10,
        margin_mode="isolated", liquidation_price=91.0,
    )
    env, trader = _real_futures_env(budget=0, venue="binance", position_status=held)

    reads = {"status": 0, "mark": 0}

    def status():
        reads["status"] += 1
        return {"position_status": held if reads["status"] == 1 else moved}

    def mark():
        reads["mark"] += 1
        return 100.0 if reads["mark"] == 1 else 999.0

    trader.get_status.side_effect = status
    trader.get_mark_price.side_effect = mark

    seen = {}

    def spy(action, **kw):
        seen.update(kw)
        return {"executed": False}          # a real return: five sites in binance alone

    env._execute_trade_if_needed = spy

    td = env.reset()
    reads["status"] = reads["mark"] = 0     # count the STEP's reads, not reset's
    env.step(td.set("action", torch.tensor(0)))

    assert seen, "setup: the trade was never dispatched, so nothing was threaded to it"
    assert seen.get("current_qty") == pytest.approx(0.05), (
        f"the trade was handed current_qty={seen.get('current_qty')!r}; 0.99 means it "
        f"re-read the venue instead of taking what the pre-trade read acquired (#295), "
        f"and 0.0 means it was dropped"
    )
    assert seen.get("current_price") == pytest.approx(137.0), (
        f"the trade was handed current_price={seen.get('current_price')!r}; 100.0 means "
        f"it re-read the mark or fell back to entry_price rather than taking the threaded "
        f"one (#295)"
    )
    assert reads["mark"] == 0, (
        f"{reads['mark']} mark fetches in a step that holds a position; the mark comes "
        f"off the status snapshot already read, and a fetch is a second round-trip that "
        f"can halt the episode"
    )
    assert reads["status"] == 2, (
        f"{reads['status']} status reads in one step, not 2 (pre-trade and post-bar); an "
        f"extra read is the outage window #295 closed by threading"
    )


@pytest.mark.parametrize("sltp", [False, True], ids=["plain", "sltp"])
def test_the_history_row_is_the_one_the_reward_function_scores(sltp):
    """The history row the reward function scores. Each assertion below kills a mutation
    of the shared `_step` that failed zero tests.

    Parametrised on plain vs SLTP, NOT on venue. Since #288 folded the tail itself onto
    `TorchTradeLiveEnv._record_and_score` there is ONE copy for all ten live envs, so the
    reward contracts die here once; what still differs per owner is what each `_step`
    THREADS into it -- the price, the qty, and how the action level is derived. That is
    what these two cases pin. Which venue resolves to which owner is the re-fork guard's
    job, and alpaca is covered by that guard rather than by a case here.

    ONE step, not two: the pre/post price asymmetry below is armed once, so by step two
    both reads return the post-bar value and the price and qty assertions stop
    discriminating.
    """

    env, trader = _real_futures_env(budget=0, venue="binance", sltp=sltp)
    env.reward_function = lambda history: float(len(history.actions))

    opened = PositionStatus(
        qty=0.05, notional_value=6000.0, entry_price=100.0, unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0, mark_price=120.0, leverage=10,
        margin_mode="isolated", liquidation_price=91.0,
    )
    bar = {"done": False}
    trader.get_status.side_effect = lambda: {
        "position_status": opened if bar["done"] else None
    }
    trader.get_mark_price.side_effect = lambda: 120.0 if bar["done"] else 100.0

    real_wait = env._wait_for_next_timestamp

    def wait():
        bar["done"] = True
        return real_wait()

    env._wait_for_next_timestamp = wait

    td = env.reset()
    if sltp:
        env.active_stop_loss, env.active_take_profit = 111.0, 222.0
    action = 0 if sltp else len(env.action_levels) - 1
    out = env.step(td.set("action", torch.tensor(action)))["next"]

    # The literal 1, not `len(history.actions)` re-read here: a count taken after the step
    # moves with the bug, so a doubled `record_step` would score 2 against 2 rows and pass.
    assert out["reward"].item() == pytest.approx(1.0), (
        f"the step returned {out['reward'].item()}, not reward_function's score of the 1 "
        f"row that existed when it was called; 0.0 is either the placeholder or a reward "
        f"computed before `record_step` wrote its row"
    )
    assert env.history.rewards[-1] == pytest.approx(1.0), (
        "history kept `record_step`'s 0.0 placeholder; the reward function reads this "
        "row on the NEXT step"
    )
    assert env.history.base_prices[-1] == pytest.approx(120.0), (
        "the row carries the pre-trade price 100.0, so price[t] would pair with "
        "portfolio_value[t] from a different bar (#278)"
    )
    assert env.history.positions[-1] == pytest.approx(0.05), (
        "the row records the pre-trade qty, so a position held across the bar reads flat"
    )


# binance ships [-1, 0, 1]; the other three ship [-1, -0.5, 0, 0.5, 1]. The venue axis is
# NOT theatre here, unlike elsewhere in this file: venues differ in DATA,
# not only in code, and on binance a mutation flattening the level to its sign is a no-op.
#
@pytest.mark.parametrize("venue,want", [
    ("binance", 1), ("binance", -1), ("bitget", 0.5), ("bitget", -0.5),
], ids=["full-long", "full-short", "half-long", "half-short"])
def test_the_plain_history_row_records_the_action_actually_traded(venue, want):
    """`action` in the history row is what the reward function scores -- the LEVEL, not
    its sign.
    """
    env, _ = _real_futures_env(budget=0, venue=venue)
    # Index by value, not position: the action_levels lists differ in length.
    action = env.action_levels.index(want)
    td = env.reset()
    env.step(td.set("action", torch.tensor(action)))

    recorded = env.history.actions[-1]
    assert recorded == pytest.approx(float(want)), (
        f"the row recorded {recorded} rather than the action level {float(want)} actually "
        f"traded; a sign-only check passes on a halved or a flattened level"
    )


@pytest.mark.parametrize("sltp", [False, True], ids=["plain", "sltp"])
def test_a_flat_bar_falls_back_to_the_pre_trade_price(sltp):
    """A flat bar has no post-bar mark; the row must carry the pre-trade price.

    `_acquire_post_bar_state` returns None for the mark when the account is flat -- there
    is no position to read one from. Dropping the fallback feeds None into
    `history.record_step(price=...)`, and every downstream reader of that row gets it.

    137.0, not the fixture's 100.0: `_real_futures_env` also fills `base_features` with
    100.0, so asserting 100.0 could not tell the pre-trade MARK from the observer's close
    -- and the close is the plausible wrong implementation, since the SLTP path prices its
    brackets off exactly that.

    The `setup:` assertion is not ceremony. The scenario is armed by a one-shot flag
    flipped inside `_wait_for_next_timestamp`; delete that call from the shared `_step`
    and the flag never flips, the post-bar mark never fails, the fallback never runs --
    and without this assertion the test still passed.
    """
    env, trader = _real_futures_env(budget=0, venue="binance", sltp=sltp)
    td = env.reset()

    # The mark fails only for the POST-bar read: the pre-trade read must succeed, or the
    # step halts before it ever reaches the fallback.
    state = {"traded": False, "post_bar_failed": False}

    def mark():
        if state["traded"]:
            state["post_bar_failed"] = True
            raise RuntimeError("post-bar mark unavailable")
        return 137.0

    real_wait = env._wait_for_next_timestamp

    def wait():
        state["traded"] = True
        return real_wait()

    trader.get_mark_price.side_effect = mark
    env._wait_for_next_timestamp = wait
    env.step(td.set("action", torch.tensor(0)))

    assert state["post_bar_failed"], (
        "setup: the post-bar mark fetch never failed, so the fallback under test never "
        "ran -- this test would pass on any implementation"
    )
    # The value, not the type: `isinstance(..., (int, float))` passes on an `else 0.0`.
    assert env.history.base_prices[-1] == pytest.approx(137.0), (
        f"a flat bar recorded {env.history.base_prices[-1]!r} rather than the pre-trade "
        f"mark of 137.0 (100.0 would mean it took the observer's close instead)"
    )
