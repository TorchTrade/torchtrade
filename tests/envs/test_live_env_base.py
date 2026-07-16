"""Contract tests for behaviour shared by every live env via TorchTradeLiveEnv.

These replace the per-exchange copies of the same assertions. Testing a shared method
once is only sound if every env actually inherits it -- so the unit test here is paired
with a guard that asserts exactly that. If an exchange ever re-adds its own override, the
guard fails and tells you to test that exchange separately.
"""

import ast
import pathlib
import inspect
import math
from types import SimpleNamespace

import pytest

import torchtrade.envs  # noqa: F401  -- registers every live env as a subclass
from torchtrade.envs.core.live import TorchTradeLiveEnv
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv
from torchtrade.envs.utils.sltp_mixin import SLTPMixin
from torchtrade.envs.core.state import PositionState, advance_hold_counter


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
