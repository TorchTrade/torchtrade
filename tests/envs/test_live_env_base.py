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
import pathlib
import re
import inspect
import math
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock, patch

import torchtrade.envs  # noqa: F401  -- registers every live env as a subclass
from torchtrade.envs.core.live import TorchTradeLiveEnv
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv
from torchtrade.envs.utils.sltp_mixin import SLTPMixin
from torchtrade.envs.utils.liquidation import (
    cross_liquidation_price,
    isolated_liquidation_price,
)
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
# __subclasses__() is a live registry, so do NOT define a TorchTradeLiveEnv subclass in any
# test module -- it would land in here, import-order dependent.
LIVE_ENVS = sorted(_subclasses(TorchTradeLiveEnv), key=lambda c: c.__name__)

# The plain envs (env.py). The SLTP ones get their sync from SLTPMixin instead.
NON_SLTP_ENVS = [c for c in LIVE_ENVS if c.__module__.endswith(".env")]

# The 4 futures exchanges (base + SLTP) share ONE _get_observation via TorchTradeFuturesLiveEnv;
# the intermediate base itself is excluded (it IS the shared impl). alpaca (spot) is absent by
# construction -- it does not subclass TorchTradeFuturesLiveEnv.
FUTURES_ENVS = [
    c for c in LIVE_ENVS
    if issubclass(c, TorchTradeFuturesLiveEnv) and c is not TorchTradeFuturesLiveEnv
]


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


@pytest.mark.parametrize("method", ["_get_observation", "_get_portfolio_value"])
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


# Every live env that defines its own _build_observation_specs -- auto-discovered from LIVE_ENVS
# (base + SLTP variants collapse to their defining base classes), so a future exchange, or a new
# non-futures env, is covered without editing this list. This spans both the 4 futures exchanges
# AND alpaca (spot) -- alpaca declares base_features via the same shared helper too.
_BASE_FEATURES_SPEC_CLASSES = sorted(
    {c for c in LIVE_ENVS if "_build_observation_specs" in vars(c)},
    key=lambda c: c.__module__.split(".")[-2],
)


@pytest.mark.parametrize(
    "env_cls", _BASE_FEATURES_SPEC_CLASSES, ids=lambda c: c.__module__.split(".")[-2]
)
def test_every_live_env_declares_base_features_via_the_shared_helper(env_cls):
    """Every live env's _build_observation_specs must call the shared _declare_base_features_spec.

    #61 was a class-level defect: base_features is EMITTED by the shared _get_observation but was
    DECLARED in observation_spec only by okx (3 of 4 futures exchanges forgot), so spec and
    observation disagreed and a collector pre-allocating from the spec silently dropped it. The
    helper now lives on TorchTradeLiveEnv (the common ancestor of both the futures base and
    alpaca), so this guard spans every live env, not just futures. The per-exchange behavioural
    tests each guard only their own exchange; this catches a FUTURE exchange that forgets the call.
    AST, not source text (like the guards above), so a comment mentioning the method can't satisfy it.
    """
    tree = ast.parse(inspect.getsource(env_cls.__dict__["_build_observation_specs"]).lstrip())
    called = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_declare_base_features_spec" in called, (
        f"{env_cls.__name__}._build_observation_specs never calls _declare_base_features_spec -- "
        f"base_features would be emitted but not declared in observation_spec (#61)."
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


def test_exactly_five_resets_derive_the_position():
    """One position-deriving _reset per exchange, and the guard below sees all five.

    That guard skips a _reset that only delegates. Move a derivation into a helper and the
    class starts skipping too -- the guard would then cover less while staying green. A review
    demonstrated exactly that: the old exact-zero rule went back into okx behind a helper and
    the suite stayed green, with only the skip count moving.
    """
    deriving = [
        c for c in LIVE_ENVS
        if (r := c.__dict__.get("_reset")) is not None
        and "current_position" in inspect.getsource(r)
    ]
    assert len(deriving) == 5, (
        f"expected one position-deriving _reset per exchange, found {len(deriving)}: "
        f"{[c.__name__ for c in deriving]}"
    )


@pytest.mark.parametrize("env_cls", LIVE_ENVS, ids=lambda c: c.__name__)
def test_every_reset_uses_the_shared_direction_rule(env_cls):
    """_reset must derive the position with the SAME dust rule as _step.

    The five resets each hand-rolled an exact-zero check until now: at qty=1e-12 reset
    reported a phantom position in account_state that the agent does not hold, while _step
    read it as flat. One rule, or they disagree.

    Applies to whichever class actually DERIVES the position; the SLTP envs' _reset only
    delegates to super() and then resets brackets, so it is not one of them.
    """
    reset = env_cls.__dict__.get("_reset")
    if reset is None:
        pytest.skip(f"{env_cls.__name__} inherits _reset")

    source = inspect.getsource(reset).lstrip()
    if "current_position" not in source:
        pytest.skip(f"{env_cls.__name__}._reset delegates the position derivation")

    tree = ast.parse(source)
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


# Envs that RESOLVE the shared accessor, which replaced three byte-identical copies
# (#283). Note okx resolves it but sizes in _step instead, so these cells prove wiring
# rather than okx's own path. A rename would empty this and pytest would SKIP rather
# than fail -- the hazard this file guards against elsewhere with its own len()
# assertions.
_SIZING_ENVS = [c for c in NON_SLTP_ENVS if hasattr(c, "_get_current_position_quantity")]
assert len(_SIZING_ENVS) == 4, f"expected 4 envs that size from a live query, got {_SIZING_ENVS}"

_FAILING_FETCH_EXCHANGES = ["binance", "bitget", "bybit", "okx", "alpaca"]


def _executor_with_failing_position_fetch(exchange):
    """Build a real order executor whose position fetch raises, as in an outage."""
    def boom(*a, **k):
        raise ConnectionError("simulated outage")

    if exchange == "binance":
        from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass
        client = SimpleNamespace(futures_position_information=boom, futures_exchange_info=boom)
        return BinanceFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=10, client=client
        )
    if exchange == "bitget":
        from torchtrade.envs.live.bitget.order_executor import BitgetFuturesOrderClass
        client = SimpleNamespace(fetch_positions=boom, load_markets=boom, markets={})
        return BitgetFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=10, client=client
        )
    if exchange == "bybit":
        from torchtrade.envs.live.bybit.order_executor import (
            BybitFuturesOrderClass, MarginMode, PositionMode,
        )
        client = SimpleNamespace(get_positions=boom, get_instruments_info=boom)
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
            account_client=SimpleNamespace(get_positions=boom, set_leverage=boom),
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


@pytest.mark.parametrize(
    "env_cls",
    _SIZING_ENVS,
    ids=lambda c: c.__name__,
)
def test_position_sizing_refuses_an_unknown_status(env_cls):
    """_get_current_position_quantity must not size an order off a phantom flat account.

    The hand-rolled `position.qty if position is not None else 0.0` read an outage as 0
    quantity, so the delta was computed against a position the exchange never said was
    gone. _step normally raises earlier, on its own status read -- this is the path when
    the outage begins between the two get_status() calls inside a single step.

    All four resolve the accessor since it was shared (#283), though okx sizes in _step
    and never calls it -- these cells prove the MRO wiring, and okx's real path is covered
    by test_okx_sizes_through_the_dust_rule_in_step. Alpaca spells the same second query
    inline and is covered through _step by the composite test.
    """
    env = SimpleNamespace(
        trader=SimpleNamespace(get_status=lambda: {"position_status": POSITION_UNKNOWN})
    )
    with pytest.raises(PositionUnknownError):
        env_cls._get_current_position_quantity(env)


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
    env_cls = next(
        c for c in vars(env_mod).values()
        if isinstance(c, type) and c.__module__ == env_mod.__name__ and "_step" in vars(c)
    )

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

    try:
        env_cls._step(env, {"action": 2})
        failed_closed = False
    except PositionUnknownError:
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
    behavioural test, which is exactly how three of them survived. Over FUTURES_ENVS, not
    just the non-SLTP ones: the SLTP variants inherit the accessor without calling it
    today, so a fork there would be dead code -- and dead-but-wrong is the state the three
    originals were in.
    """
    assert "_get_current_position_quantity" not in vars(env_cls), (
        f"{env_cls.__name__} redefines _get_current_position_quantity. The dust rule "
        "lives in position_qty_from_status -- inherit it rather than re-deriving qty."
    )


@pytest.mark.parametrize("env_cls", _SIZING_ENVS, ids=lambda c: c.__name__)
def test_a_dust_residual_does_not_look_like_a_position_to_the_trade_path(env_cls):
    """The concrete failure from #283, at the seam every sizing path reads.

    An exchange can leave a float residual after a full close. Read as a live position it
    makes `abs(current_qty) > 0` true on a flat account, so action 0.0 calls
    close_position() on nothing -- and still advances current_action_level from a trade
    that never happened, which freezes the duplicate-action guard (invariant 2).

    The account_state paths already honoured the dust rule; only the TRADE paths
    hand-rolled it, which is why nothing caught this.
    """
    env = SimpleNamespace(
        trader=SimpleNamespace(
            get_status=lambda: {"position_status": SimpleNamespace(qty=1e-12)}
        )
    )
    assert env_cls._get_current_position_quantity(env) == 0.0, (
        "a 1e-12 residual must read as flat, not as a position to close"
    )


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
    ("margin_type", "CROSSED"),  # Binance
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
    assert "_capture_bankruptcy_baseline()" in src
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
    fallback is only as good as set_leverage having worked, which #277's unfixed half is
    about. What this forbids is the silent case -- a leverage the venue did report, as 0,
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


@pytest.mark.parametrize("exchange", ["binance", "bitget", "bybit", "okx"])
def test_futures_sizing_rejects_a_non_finite_balance(exchange):
    """`not (x > 0)` catches NaN but passes +inf, and these four lines are this PR's own.

    An inf balance sizes an inf target: bitget's amount rounding then yields NaN and hands
    it to create_order, while binance and bybit raise an undiagnosable OverflowError
    mid-step. The alpaca sibling written in the same commit is safe only because it
    isfinite-checks its inputs first.
    """
    src = (pathlib.Path(inspect.getfile(TorchTradeLiveEnv)).parent.parent
           / "live" / exchange / "env.py").read_text()
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
    account.set_leverage = MagicMock(return_value={"code": "0"})
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
    account.set_leverage = MagicMock(return_value={"code": "0"})
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
        account.set_leverage = MagicMock(return_value={"code": "0"})
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
    TorchTradeLiveEnv._record_position_after_trade(env, desired_action)

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
    assert "position_status.qty" not in path.read_text(), (
        f"{exchange}/{module} reads a raw qty instead of position_qty_from_status()"
    )


@pytest.mark.parametrize("observed,target,tol,released", [
    (0.50, 0.50, 0.001, False),
    (0.5005, 0.50, 0.001, False),
    (0.95, 0.50, 0.001, True),
    (0.05, 0.50, 0.001, True),
    (0.499, 0.50, 0.01, False),
    (0.0, 0.0, 0.001, False),
    (0.50, None, 0.001, False),
], ids=["exact", "below-min-qty", "under-filled", "barely-filled",
        "coarse-lot-step", "closed", "no-target"])
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
    config_cls = next(
        v for k, v in vars(module).items()
        if k.endswith("Config") and hasattr(v, "__dataclass_fields__")
    )
    with pytest.raises(ValueError):
        config_cls(symbol="BTC/USD", action_levels=[0.0, float("nan"), 1.0])


@pytest.mark.parametrize("side,expected", [
    ("long", 1), ("short", -1), ("close", 0), (None, 0), ("unexpected", 0),
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
