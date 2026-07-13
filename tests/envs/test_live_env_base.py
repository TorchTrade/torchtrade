"""Contract tests for behaviour shared by every live env via TorchTradeLiveEnv.

These replace the per-exchange copies of the same assertions. Testing a shared method
once is only sound if every env actually inherits it -- so the unit test here is paired
with a guard that asserts exactly that. If an exchange ever re-adds its own override, the
guard fails and tells you to test that exchange separately.
"""

import ast
import inspect
import math
from types import SimpleNamespace

import pytest

import torchtrade.envs  # noqa: F401  -- registers every live env as a subclass
from torchtrade.envs.core.live import TorchTradeLiveEnv
from torchtrade.envs.utils.sltp_mixin import SLTPMixin
from torchtrade.envs.core.state import PositionState


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


# Discovered, not hand-listed: a hand-listed exchange #6 would silently escape the guard.
# __subclasses__() is a live registry, so do NOT define a TorchTradeLiveEnv subclass in any
# test module -- it would land in here, import-order dependent.
LIVE_ENVS = sorted(_subclasses(TorchTradeLiveEnv), key=lambda c: c.__name__)

# The plain envs (env.py). The SLTP ones get their sync from SLTPMixin instead.
NON_SLTP_ENVS = [c for c in LIVE_ENVS if c.__module__.endswith(".env")]


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
    (0, 0.0, None, 0, 0.0),
    # Position moved behind the env's back -> the level that produced it is unknowable.
    (1, 1.0, None, 0, 0.0),                          # liquidated -> flat, level flat
    (0, 0.0, SimpleNamespace(qty=0.5), 1, math.nan),  # opened externally -> ANY next command runs
    (1, 1.0, SimpleNamespace(qty=-0.5), -1, math.nan),  # flipped long -> short
    (-1, -1.0, SimpleNamespace(qty=0.5), 1, math.nan),  # flipped short -> long
    # A close can leave a float residual instead of an exact zero. Reading that as an open
    # position is what re-froze the guard -- the dust rule is the whole point of the shared
    # position_direction() helper.
    (1, 1.0, SimpleNamespace(qty=1e-12), 0, 0.0),     # dust after liquidation -> flat
    # Exchanges also report a flat position as a zero-qty object rather than None; every
    # live _reset already has an explicit branch for it.
    (1, 1.0, SimpleNamespace(qty=0.0), 0, 0.0),
], ids=["long-unchanged", "short-unchanged", "flat-unchanged",
        "liquidated", "opened-externally", "flipped-long-to-short", "flipped-short-to-long",
        "dust-after-liquidation", "zero-qty-status-not-none"])
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
    """
    exchanges = {cls.__module__.split(".")[-2] for cls in LIVE_ENVS}
    assert exchanges == {"alpaca", "binance", "bitget", "bybit", "okx"}
    # NON_SLTP_ENVS drives the call-site guard below, and an empty parametrize SKIPS rather
    # than fails -- a module rename would silently retire that guard.
    assert len(NON_SLTP_ENVS) == 5


@pytest.mark.parametrize("method", [
    "_check_termination", "_sync_action_level_after_reset",
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
    tree = ast.parse(inspect.getsource(env_cls.__dict__["_step"]).lstrip())
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]

    assert "_sync_position_from_exchange" in calls, (
        f"{env_cls.__name__}._step never reconciles the cached position with the exchange -- "
        f"a liquidation would leave it stale for the rest of the episode."
    )
    assert calls.index("_sync_position_from_exchange") < calls.index("_execute_trade_if_needed"), (
        f"{env_cls.__name__}._step syncs AFTER it trades: the duplicate-action guard still "
        f"reads the stale position."
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
