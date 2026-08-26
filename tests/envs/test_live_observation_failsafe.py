"""#343: a live env that cannot read its own account state must halt, not improvise."""

import pytest
from types import SimpleNamespace

from torchtrade.envs.core.live import (
    LiveObservationHalt,
    ObservationFailurePolicy,
)
from torchtrade.envs.core.state import PositionUnknownError
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv


def _env(error, policy=ObservationFailurePolicy.HALT, close_result=True):
    closed = []

    def close_position():
        closed.append(True)
        return close_result

    env = SimpleNamespace(
        _get_portfolio_value=lambda: (_ for _ in ()).throw(error),
        _get_observation=lambda **k: None,
        trader=SimpleNamespace(close_position=close_position),
        config=SimpleNamespace(symbol="BTCUSDT", observation_failure_policy=policy),
    )
    # The post-bar read delegates to the shared policy now rather than restating it, so
    # these tests exercise `_halting` itself -- which is the point of extracting it.
    env.consecutive_unknown_status = 0
    env._last_confirmed_read = {}
    env._max_unknown_status_steps = 0
    env._halting = lambda read, cache_key=None: TorchTradeFuturesLiveEnv._halting(
        env, read, cache_key
    )
    return env, closed


@pytest.mark.parametrize("error", [
    PositionUnknownError("exchange status is unknown"),
    ValueError("venue reported a non-finite mark_price"),
], ids=["unknown-position", "impossible-account-state"])
def test_a_terminal_failure_halts_and_raises(error):
    """Terminal means the venue told us something impossible about our own money."""
    env, _ = _env(error)

    with pytest.raises(LiveObservationHalt) as exc:
        TorchTradeFuturesLiveEnv._acquire_post_bar_state(env)

    assert exc.value.original_exception is error
    assert exc.value.policy is ObservationFailurePolicy.HALT


@pytest.mark.parametrize("error", [
    RuntimeError("Failed to get account balance: read timeout"),
    KeyError("features_1m"),
], ids=["adapter-wrapped-transient", "config-error"])
def test_a_non_terminal_failure_propagates_without_halting(error):
    """The money-critical distinction.

    Every adapter wraps any exception in `RuntimeError("Failed to get account balance:
    ...")`, so treating RuntimeError as terminal would classify a read timeout as fatal
    and -- under FLATTEN -- market-close a live position on a blip. KeyError is a
    config/programmer error and should crash rather than end the episode.
    """
    env, closed = _env(error, policy=ObservationFailurePolicy.FLATTEN)

    with pytest.raises(type(error)):
        TorchTradeFuturesLiveEnv._acquire_post_bar_state(env)

    assert closed == [], "a transient failure closed a live position"


@pytest.mark.parametrize("policy,close_result,expect_closed,expect_accepted", [
    (ObservationFailurePolicy.HALT, True, False, None),
    (ObservationFailurePolicy.FLATTEN, True, True, True),
    # The venue can REJECT the close without raising. Reporting that as accepted is the
    # "operator believes the position is flat when it is not" bug the exception warns of.
    (ObservationFailurePolicy.FLATTEN, False, True, False),
], ids=["halt", "flatten-accepted", "flatten-rejected"])
def test_flatten_policy_controls_the_emergency_close(
    policy, close_result, expect_closed, expect_accepted
):
    env, closed = _env(PositionUnknownError("outage"), policy=policy,
                       close_result=close_result)

    with pytest.raises(LiveObservationHalt) as exc:
        TorchTradeFuturesLiveEnv._acquire_post_bar_state(env)

    assert bool(closed) is expect_closed
    assert exc.value.flatten_accepted is expect_accepted


def test_a_failed_emergency_close_is_recorded_not_swallowed():
    """The operator has to know the position may still be open."""
    env, _ = _env(PositionUnknownError("outage"), policy=ObservationFailurePolicy.FLATTEN)
    env.trader.close_position = lambda: (_ for _ in ()).throw(RuntimeError("venue down"))

    with pytest.raises(LiveObservationHalt) as exc:
        TorchTradeFuturesLiveEnv._acquire_post_bar_state(env)

    assert isinstance(exc.value.flatten_error, RuntimeError)
    assert exc.value.flatten_accepted is not True


@pytest.mark.parametrize("exchange", ["binance", "bitget", "bybit", "okx"])
@pytest.mark.parametrize("module", ["env", "env_sltp"])
def test_every_futures_step_reads_state_through_the_halting_helper(exchange, module):
    """The wiring, not the helper.

    A `_step` reading the venue directly gets the raw exception -- no halt, no emergency
    flatten -- and reverting one env to do so fails nothing else in the suite. Structural
    because the alternative is eight live-env harnesses to assert one call site each.
    """
    import importlib
    import inspect

    ns = importlib.import_module(f"torchtrade.envs.live.{exchange}.{module}").__dict__
    env_cls = next(v for k, v in ns.items()
                   if inspect.isclass(v) and k.endswith("TorchTradingEnv")
                   and v.__module__.endswith(module))
    source = inspect.getsource(env_cls._step)

    assert "_acquire_post_bar_state()" in source, f"{exchange}/{module} bypasses the halt"
    assert "self._get_observation()" not in source, f"{exchange}/{module} reads directly"


@pytest.mark.parametrize("exchange", ["binance", "bitget", "bybit", "okx"])
@pytest.mark.parametrize("module", ["env", "env_sltp"])
def test_every_futures_config_coerces_its_failure_policy(exchange, module):
    """A bad policy string must be rejected at the boundary, not in production.

    The `__module__` filter matches the env_cls discovery above and is load-bearing: the
    SLTP configs now import their shared base into the module namespace, and without it
    `next()` returns that base for all four venues -- so this test silently stopped
    exercising the subclass coercion it exists to pin (#288 review).
    """
    import importlib
    import inspect

    ns = importlib.import_module(f"torchtrade.envs.live.{exchange}.{module}").__dict__
    cfg_cls = next(v for k, v in ns.items()
                   if inspect.isclass(v) and k.endswith("Config")
                   and hasattr(v, "observation_failure_policy")
                   and v.__module__.endswith(module))

    assert cfg_cls().observation_failure_policy is ObservationFailurePolicy.HALT
    assert cfg_cls(observation_failure_policy="flatten").observation_failure_policy is (
        ObservationFailurePolicy.FLATTEN
    )
    with pytest.raises(ValueError):
        cfg_cls(observation_failure_policy="nonsense")


# --- #295: the grace period, and the truncation that ends it --------------------------

VENUES = ["binance", "bitget", "bybit", "okx"]


def _real_futures_env(budget, venue="binance", sltp=False, position_status=None,
                      **config_kw):
    """A real futures env on mocks, for any of the four venues.

    Parametrised because round 3 fixed the sizing read on all four and tested it on
    ONE -- reverting it on bitget/bybit/okx failed zero tests, which is the same
    shape as the defect it was fixing.
    """
    import importlib
    from unittest.mock import MagicMock, patch
    import numpy as np

    module = importlib.import_module(
        f"torchtrade.envs.live.{venue}.env_sltp" if sltp
        else f"torchtrade.envs.live.{venue}.env"
    )
    from tests.envs.test_live_env_base import _sole

    Env = _sole(module, "TorchTradingEnv")
    Config = _sole(module, "TradingEnvConfig")

    observer = MagicMock()
    observer.get_keys.return_value = ["1m_10"]
    # base_features carries OHLC; SLTP prices its brackets off the close column.
    observer.get_observations.side_effect = lambda **kw: {
        "1m_10": np.zeros((10, 4), dtype=np.float32),
        **({"base_features": np.full((10, 4), 100.0, dtype=np.float32)}
           if kw.get("return_base_ohlc") else {}),
    }
    observer.get_features.return_value = {
        "observation_features": list("abcd"), "original_features": []}
    observer.reset.return_value = None

    trader = MagicMock()
    trader.get_account_balance.return_value = {
        "available_balance": 1e4, "total_margin_balance": 1e4,
        "total_wallet_balance": 1e4, "total_maintenance_margin": 0.0}
    trader.get_status.return_value = {"position_status": position_status}
    trader.get_mark_price.return_value = 100.0
    trader.get_lot_size.return_value = {"min_qty": 0.001, "qty_step": 0.001}
    trader.round_quantity.side_effect = lambda q: round(float(q), 3)
    trader._round_amount.side_effect = lambda q: round(float(q), 3)   # bitget/okx spelling
    trader.trade.return_value = True
    trader.close_position.return_value = True
    trader.client.futures_exchange_info.return_value = {"symbols": [{
        "symbol": "BTCUSDT",
        "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "10"}],
    }]}

    symbol = "BTC-USDT-SWAP" if venue == "okx" else "BTCUSDT"
    trader.get_status.return_value = {"position_status": position_status}
    trader.client.futures_exchange_info.return_value = {"symbols": [{
        "symbol": symbol,
        "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "10"}],
    }]}
    config = Config(symbol=symbol, time_frames=["1m"], window_sizes=[10],
                    execute_on="1m", max_unknown_status_steps=budget,
                    close_position_on_init=False, **config_kw)
    with patch("time.sleep"), patch.object(Env, "_wait_for_next_timestamp"):
        env = Env(config=config, observer=observer, trader=trader)
    env._wait_for_next_timestamp = lambda: None
    return env, trader


def _grace_env(budget, policy=ObservationFailurePolicy.HALT):
    """A stub wired to the real `_halting`, with a configurable outage budget."""
    closed = []
    env = SimpleNamespace(
        trader=SimpleNamespace(close_position=lambda: closed.append(True) or True),
        config=SimpleNamespace(symbol="BTCUSDT", observation_failure_policy=policy),
        consecutive_unknown_status=0,
        _last_confirmed_read={},
        _max_unknown_status_steps=budget,
    )
    env._halting = lambda read, cache_key=None: TorchTradeFuturesLiveEnv._halting(
        env, read, cache_key
    )
    return env, closed


def _boom():
    raise PositionUnknownError("venue unreachable")


def test_a_confirmed_read_is_what_the_grace_period_stands_on():
    """The budget alone is not enough -- there must be a real prior read to fall back on.

    A failure on the FIRST read of an episode has no last-known truth, so honouring the
    budget there would mean inventing an account state. It raises regardless of budget.
    """
    env, _ = _grace_env(budget=3)

    with pytest.raises(LiveObservationHalt):
        env._halting(_boom, cache_key="pre_trade")

    assert env._status_unknown_this_step, "the bar is still marked unconfirmed"


def test_the_grace_period_rides_out_an_outage_on_the_last_confirmed_read():
    """`_halting` serves the cached read and never raises once grace applies.

    It deliberately does NOT consult the budget: whether the outage has outlasted it is a
    question about the BAR, answered once in `_finalize_step_flags`. Raising here on the
    spent bar would reintroduce the process crash #295 exists to remove.
    """
    env, _ = _grace_env(budget=3)
    assert env._halting(lambda: "GOOD", cache_key="pre_trade") == "GOOD"

    for _ in range(5):        # well past the budget: still cached, still no raise
        assert env._halting(_boom, cache_key="pre_trade") == "GOOD"
        assert env._status_unknown_this_step


def test_a_successful_read_refreshes_what_the_grace_period_will_serve():
    env, _ = _grace_env(budget=3)
    env._halting(lambda: "GOOD", cache_key="pre_trade")
    assert env._halting(_boom, cache_key="pre_trade") == "GOOD"

    env._halting(lambda: "FRESH", cache_key="pre_trade")
    assert env._halting(_boom, cache_key="pre_trade") == "FRESH"


def test_the_default_budget_keeps_the_pre_295_posture():
    """0 is the default on every venue: refuse to act on state you cannot confirm."""
    env, _ = _grace_env(budget=0)
    env._halting(lambda: "GOOD", cache_key="pre_trade")

    with pytest.raises(LiveObservationHalt):
        env._halting(_boom, cache_key="pre_trade")


def test_flatten_does_not_honour_the_grace_period():
    """FLATTEN means "get me out while I cannot see the account". Riding out the outage
    would defeat the only thing it is for, so grace is a HALT-only concession."""
    env, closed = _grace_env(budget=3, policy=ObservationFailurePolicy.FLATTEN)
    env._halting(lambda: "GOOD", cache_key="pre_trade")

    with pytest.raises(LiveObservationHalt):
        env._halting(_boom, cache_key="pre_trade")
    assert closed == [True], "the emergency close must still run"


def test_the_two_read_sites_do_not_share_a_cache_slot():
    """`_acquire_pre_trade_state` and `_acquire_post_bar_state` return different shapes.
    One slot would hand the post-bar caller a pre-trade tuple during an outage."""
    env, _ = _grace_env(budget=3)
    env._halting(lambda: "PRE", cache_key="pre_trade")
    env._halting(lambda: "POST", cache_key="post_bar")

    assert env._halting(_boom, cache_key="pre_trade") == "PRE"
    assert env._halting(_boom, cache_key="post_bar") == "POST"


def test_a_real_env_flags_the_outage_and_truncates_on_the_budgeted_bar():
    """Drives a REAL env through an outage. The unit tests above missed two defects here.

    The first: `status_unknown` was stamped at observation-BUILD time, so the grace period
    served a cached observation carrying the healthy 0.0 -- the policy was told "all
    confirmed" on exactly the bars it was not.

    The second: both read sites advanced the counter, so a budget of 3 truncated after two
    bars. `max_unknown_status_steps` has to mean bars, or it means nothing a trader can
    reason about.
    """
    import torch

    env, trader = _real_futures_env(budget=3)
    td = env.reset()
    td = env.step(td.set("action", torch.tensor(1)))["next"]
    assert td["status_unknown"].item() == 0.0

    trader.get_status.side_effect = PositionUnknownError("venue unreachable")
    seen = []
    for _ in range(3):
        td = td.exclude("done", "terminated", "truncated", "reward")
        nxt = env.step(td.set("action", torch.tensor(1)))["next"]
        seen.append((nxt["status_unknown"].item(), bool(nxt["done"]),
                     bool(nxt["terminated"]), bool(nxt["truncated"])))
        td = nxt

    assert seen == [
        (1.0, False, False, False),
        (1.0, False, False, False),
        (1.0, True, False, True),
    ], f"outage bars were {seen}"



def test_reset_halts_typed_rather_than_relocating_the_crash():
    """The reset read must go through the halt policy too (#295).

    Truncation fires precisely BECAUSE the outage is ongoing, so the very next `reset()`
    is the likeliest moment for the venue to still be down. With reset's reads outside
    `_halting`, the crash this work exists to remove did not go away -- it moved from
    `_step` to the following `reset`, as a bare PositionUnknownError that
    `except LiveObservationHalt` does not catch and FLATTEN does not act on.

    Raising here is still correct: an episode must not START on unconfirmed state.
    """
    env, trader = _real_futures_env(
        budget=3, observation_failure_policy=ObservationFailurePolicy.FLATTEN
    )
    trader.get_status.side_effect = PositionUnknownError("still down")
    trader.close_position.reset_mock()

    with pytest.raises(LiveObservationHalt):
        env.reset()

    assert trader.close_position.called, (
        "FLATTEN did not act on the reset read: it was outside the halt policy"
    )



def test_the_cache_holds_a_copy_not_the_object_the_collector_mutates():
    """`TensorDict.set()` stores references, so caching the read by reference aliases it.

    The post-bar tuple carries the observation tensordict that `_step` returns -- the same
    object the collector then stamps with reward and the done family. Cached by reference,
    a grace bar re-serves LAST bar's flags, which is how a stale `done=False` ends up in a
    tensordict that `EnvBase._complete_done` then declines to fill.

    Currently invisible behaviourally, because `_finalize_step_flags` overwrites the whole
    family anyway. That is precisely why it needs pinning directly: the aliasing is a live
    hazard that today's code happens to paper over, and removing the clone fails nothing
    without this test.
    """
    import torch
    from tensordict import TensorDict

    env, _ = _grace_env(budget=3)
    observation = TensorDict({"done": torch.zeros(1, dtype=torch.bool)}, batch_size=())
    env._halting(lambda: ("status", observation), cache_key="post_bar")

    # Two grace bars in a row. Each must get its OWN object: `_step` stamps reward and
    # the done family onto whatever it is handed, so one shared object means bar 1 and
    # bar 2 in a collector's rollout both end up carrying bar 2's values.
    _, first = env._halting(_boom, cache_key="post_bar")
    _, second = env._halting(_boom, cache_key="post_bar")
    assert first is not second, "two grace bars were served the same tensordict"

    first.set("done", torch.ones(1, dtype=torch.bool))
    assert not bool(second["done"]), "stamping one grace bar mutated another"
    assert not bool(observation["done"]), "stamping a grace bar mutated the cache"


def test_an_open_position_is_frozen_but_flagged_through_a_grace_outage():
    """The scenario the whole feature exists for, and the one nothing else covered.

    Every other test here drives the outage FLAT. With a position open, `account_state`
    -- exposure, unrealised PnL, distance to liquidation -- is served from the cached
    read and freezes at the last confirmed values while the real account keeps moving.
    That is the accepted cost of the opt-in posture, but it must be exactly that: frozen
    and FLAGGED, never silently re-derived from a stale mark as though it were fresh.
    """
    import torch
    from torchtrade.envs.core.common_types import PositionStatus

    open_long = PositionStatus(
        qty=0.05, notional_value=5000.0, entry_price=100.0, unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0, mark_price=100.0, leverage=10,
        margin_mode="isolated", liquidation_price=91.0,
    )
    env, trader = _real_futures_env(budget=3, position_status=open_long)

    td = env.reset()
    td = env.step(td.set("action", torch.tensor(2)))["next"]
    confirmed = td["account_state"].clone()
    assert confirmed[1].item() != 0.0, "setup: expected an open position"

    trader.get_status.side_effect = PositionUnknownError("venue unreachable")
    trader.get_mark_price.side_effect = PositionUnknownError("venue unreachable")

    frozen = []
    for _ in range(2):
        td = env.step(td.exclude("done", "terminated", "truncated", "reward")
                      .set("action", torch.tensor(2)))["next"]
        frozen.append((td["account_state"].clone(), td["status_unknown"].item()))

    for state, flag in frozen:
        assert flag == 1.0, "an unconfirmed bar must say so"
        assert torch.equal(state, confirmed), (
            "account_state moved during an outage: it must be the last CONFIRMED read, "
            "not a value re-derived from a stale mark"
        )


def test_a_new_episode_cannot_be_served_the_previous_one_s_account():
    """`_reset_outage_state` clears the cache; nothing forced the case it protects.

    Episode N truncates with a populated cache. Episode N+1 resets successfully, then its
    FIRST pre-trade read fails. With the cache uncleared, grace would serve episode N's
    account -- a position that may since have been closed, at a price hours old.
    """
    import torch
    from torchtrade.envs.core.common_types import PositionStatus

    open_long = PositionStatus(
        qty=0.05, notional_value=5000.0, entry_price=100.0, unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0, mark_price=100.0, leverage=10,
        margin_mode="isolated", liquidation_price=91.0,
    )
    env, trader = _real_futures_env(budget=3, position_status=open_long)
    td = env.reset()
    env.step(td.set("action", torch.tensor(2)))
    assert env._last_confirmed_read, "setup: the cache should be populated"

    env.reset()                                   # episode N+1, venue healthy
    # `balance` is legitimately re-seeded by reset's own confirmed read. What must NOT
    # survive is the previous episode's POSITION and mark -- a position that may since
    # have been closed, at a price from before the outage.
    assert "pre_trade" not in env._last_confirmed_read
    assert "post_bar" not in env._last_confirmed_read

    trader.get_status.side_effect = PositionUnknownError("down again")
    with pytest.raises(LiveObservationHalt):       # nothing confirmed yet this episode
        env.step(td.set("action", torch.tensor(2)))


def test_alpaca_reports_its_status_as_known_because_it_has_no_failure_policy():
    """Alpaca is the third hand-written copy of the same one-line tensor.

    It has no `observation_failure_policy` and no `_halting`, so its counter never
    advances and the flag is a constant. Declared anyway, so the observation contract does
    not fork by venue -- and pinned here, because this repo's recorded failure mode is a
    third copy quietly drifting from the other two.
    """
    import torch
    from tests.mocks.alpaca import MockObserver, MockTrader
    from torchtrade.envs.live.alpaca.env import (
        AlpacaTorchTradingEnv, AlpacaTradingEnvConfig,
    )

    env = AlpacaTorchTradingEnv(
        config=AlpacaTradingEnvConfig(symbol="BTC/USD", window_sizes=[10]),
        observer=MockObserver(window_sizes=[10]), trader=MockTrader(initial_cash=10000.0),
    )
    env._wait_for_next_timestamp = lambda: None

    assert not hasattr(env.config, "max_unknown_status_steps"), (
        "alpaca grew an outage budget; this test and the docs both need revisiting"
    )
    td = env.reset()
    assert td["status_unknown"].item() == 0.0
    for _ in range(3):
        td = env.step(td.set("action", torch.tensor(1)))["next"]
        assert td["status_unknown"].item() == 0.0
        assert not bool(td["truncated"])
        td = td.exclude("done", "terminated", "truncated", "reward")


@pytest.mark.parametrize("venue", VENUES)
def test_a_grace_bar_can_still_SIZE_a_trade_with_the_venue_down(venue):
    """Every other grace test holds the action constant, so sizing never ran.

    `_execute_trade_if_needed` early-returns when the action equals the current level, so
    a test that never CHANGES the action never reaches `_calculate_fractional_position` --
    and that is where the balance read lives. Threading qty and price fixed the decision
    to trade; the read that SIZES the trade was still outside `_halting`, so the first
    grace bar on which a policy actually wanted to open or resize died with a bare error
    instead of trading on cached state and truncating on budget.
    """
    import torch

    env, trader = _real_futures_env(budget=3, venue=venue)
    # Index-by-VALUE, not by position: binance ships [-1, 0, 1] and the others
    # [-1, -0.5, 0, 0.5, 1], so a hardcoded index 2 is "flat" on three of the four and
    # takes the close path instead of the sizing path. That is exactly how this fix went
    # untested on three venues in the first place.
    flat = env.action_levels.index(0.0)
    full_long = len(env.action_levels) - 1

    td = env.reset()
    td = env.step(td.set("action", torch.tensor(flat)))["next"]       # flat, confirmed

    trader.get_status.side_effect = PositionUnknownError("venue unreachable")
    trader.get_mark_price.side_effect = PositionUnknownError("venue unreachable")
    trader.get_account_balance.side_effect = PositionUnknownError("venue unreachable")

    # CHANGE the action, so the early-return cannot hide the sizing path.
    nxt = env.step(td.exclude("done", "terminated", "truncated", "reward")
                   .set("action", torch.tensor(full_long)))["next"]

    assert nxt["status_unknown"].item() == 1.0, "the bar must be flagged, not crash"
    assert not bool(nxt["terminated"]), "an outage is never a terminated episode"


@pytest.mark.parametrize("venue", VENUES)
def test_an_sltp_grace_bar_can_still_size_a_bracket_with_the_venue_down(venue):
    """No test drove an SLTP env through an outage at all, on any venue.

    So when the sizing read came under `_halting` on the plain envs, the SLTP envs' own
    `get_account_balance()` -- a separate line, in a separate file, doing the same thing
    -- was simply not part of the change. #288's thesis stated as a bug: the same fix
    landing on some copies and not others, with nothing able to tell.
    """
    import torch

    env, trader = _real_futures_env(budget=3, venue=venue, sltp=True,
                                    trade_mode="fractional", position_fraction=0.5)
    td = env.reset()
    td = env.step(td.set("action", torch.tensor(0)))["next"]          # HOLD, confirmed

    trader.get_status.side_effect = PositionUnknownError("venue unreachable")
    trader.get_mark_price.side_effect = PositionUnknownError("venue unreachable")
    trader.get_account_balance.side_effect = PositionUnknownError("venue unreachable")

    # A real bracket action, so the sizing path runs rather than the hold early-return.
    nxt = env.step(td.exclude("done", "terminated", "truncated", "reward")
                   .set("action", torch.tensor(1)))["next"]

    assert nxt["status_unknown"].item() == 1.0, "the bar must be flagged, not crash"
    assert not bool(nxt["terminated"]), "an outage is never a terminated episode"


@pytest.mark.parametrize("sltp", [False, True], ids=["plain", "sltp"])
@pytest.mark.parametrize("venue", VENUES)
def test_reset_halts_on_the_sentinel_venues_actually_return(venue, sltp):
    """`get_status` does not RAISE on an outage -- it returns POSITION_UNKNOWN.

    All four adapters assign the sentinel and return normally; the error surfaces at the
    first attribute touch. So wrapping the CALL in `_halting` catches nothing, which is
    the trap `_acquire_pre_trade_state`'s own docstring warns about twelve lines above the
    code that fell into it -- the conversion has to be inside the closure too.

    The existing reset test drove `side_effect = PositionUnknownError`, a raise. That
    shape does not occur in production: POSITION_UNKNOWN appeared ZERO times in this file,
    so the test asserted a halt on a failure the venues never generate.
    """
    from torchtrade.envs.core.state import POSITION_UNKNOWN

    env, trader = _real_futures_env(
        budget=0, venue=venue, sltp=sltp,
        observation_failure_policy=ObservationFailurePolicy.FLATTEN,
    )
    trader.get_status.return_value = {"position_status": POSITION_UNKNOWN}
    trader.close_position.reset_mock()

    with pytest.raises(LiveObservationHalt):
        env.reset()
    assert trader.close_position.called, (
        "FLATTEN did not act: the sentinel escaped as a bare PositionUnknownError"
    )


@pytest.mark.parametrize("sltp", [False, True], ids=["plain", "sltp"])
@pytest.mark.parametrize("venue", VENUES)
def test_grace_covers_the_exception_the_adapters_actually_raise(venue, sltp):
    """All four adapters wrap a failed balance read in `RuntimeError`, not ValueError.

    `binance/order_executor.py`, `bitget`, `bybit`, `okx` all do
    `raise RuntimeError(f"Failed to get account balance: {e}")`. `_halting` deliberately
    did not catch RuntimeError, so the grace period never engaged for the failure mode
    production actually produces -- every test here injected PositionUnknownError, which
    only the STATUS path raises.

    RuntimeError is grace-eligible ONLY. `test_a_non_terminal_failure_propagates_without
    _halting` pins the other half: outside grace it still escapes untouched, because a
    read timeout arrives as RuntimeError too and #394 refused to flatten on those.
    """
    import torch

    env, trader = _real_futures_env(budget=3, venue=venue, sltp=sltp,
                                    **({"trade_mode": "fractional",
                                        "position_fraction": 0.5} if sltp else {}))
    td = env.reset()
    td = env.step(td.set("action", torch.tensor(0 if sltp else 1)))["next"]

    trader.get_account_balance.side_effect = RuntimeError(
        "Failed to get account balance: connection reset"
    )
    trade_action = 1 if sltp else len(env.action_levels) - 1
    nxt = env.step(td.exclude("done", "terminated", "truncated", "reward")
                   .set("action", torch.tensor(trade_action)))["next"]

    assert nxt["status_unknown"].item() == 1.0, (
        "a real adapter balance failure was not recognised as an outage"
    )
    assert not bool(nxt["terminated"])


@pytest.mark.parametrize("succeeded,still_cached", [(True, False), (False, True)],
                         ids=["close-succeeds", "close-fails"])
def test_a_realised_close_invalidates_the_cached_balance(succeeded, still_cached):
    """A close moves equity, so the cached balance is wrong by the trade's P&L.

    The sizing path early-returns BEFORE the balance read, so nothing refreshes it: reset
    seeds $10k, the policy opens, holds, closes at a loss, and a grace bar many bars later
    sizes against the pre-close $10k. At 10x that is an order the venue rejects on margin.

    Only on SUCCESS -- a failed close leaves the position, and with it the equity the
    cache already describes.
    """
    env, trader = _real_futures_env(budget=3)
    env.reset()
    assert "balance" in env._last_confirmed_read, "setup: reset seeds the cache"

    env.position.current_position = 1
    trader.close_position.return_value = succeeded
    env._handle_close_action(0.05)

    assert ("balance" in env._last_confirmed_read) is still_cached


@pytest.mark.parametrize("venue", ["binance", "bitget"])
def test_a_candle_close_failure_is_ridden_out_rather_than_raised(venue):
    """`_halting(read_close)` without a cache_key still RAISES -- just with a nicer type.

    Grace needs a slot to serve from. Wrapping the read made the exception well-typed and
    changed nothing about the contract the docs claim: serve the last CONFIRMED close and
    flag the bar. Its own slot, because binance/bitget price brackets off a candle close
    rather than the mark, deliberately.
    """
    import torch

    env, trader = _real_futures_env(budget=3, venue=venue, sltp=True,
                                    trade_mode="fractional", position_fraction=0.5)
    td = env.reset()
    # A TRADING action, not a hold: the hold path early-returns before `read_close`, so a
    # hold bar never seeds the slot. That is a real residual boundary -- hold for N bars,
    # then trade during an outage, and there is still no confirmed close to serve -- and
    # it is reported on the PR rather than papered over here.
    td = env.step(td.set("action", torch.tensor(1)))["next"]

    trader.get_status.side_effect = PositionUnknownError("down")
    env.observer.get_observations.side_effect = ValueError(
        "most recent candle did not survive preprocessing"
    )

    nxt = env.step(td.exclude("done", "terminated", "truncated", "reward")
                   .set("action", torch.tensor(1)))["next"]
    assert nxt["status_unknown"].item() == 1.0, (
        "the bar raised instead of being ridden out: no cached candle close to serve"
    )


@pytest.mark.parametrize("venue", VENUES)
def test_reset_survives_its_guarded_reads_then_a_failing_reread(venue):
    """The reads were confirmed, then thrown away and taken again raw.

    A failure on that SECOND pair escaped the policy entirely -- FLATTEN could not act on
    it -- while the confirmed snapshot sat unused. `_get_observation` is deliberately NOT
    halt-wrapped (it reads the observer too, so a config gap must not flatten); the fix is
    to stop re-reading, not to wrap.
    """
    calls = {"n": 0}
    env, trader = _real_futures_env(
        budget=0, venue=venue, observation_failure_policy=ObservationFailurePolicy.FLATTEN
    )

    def failing_second_read():
        calls["n"] += 1
        if calls["n"] > 1:
            raise PositionUnknownError("venue went down between the two reads")
        return {"position_status": None}

    trader.get_status.side_effect = failing_second_read
    env.reset()      # must not raise: there IS no second read any more

    assert calls["n"] == 1, (
        f"reset read the venue {calls['n']} times; the second read is what escaped policy"
    )


def test_an_aborted_runtimeerror_does_not_flag_the_retried_bar():
    """`_status_unknown_this_step` is set before the no-grace re-raise, and consumed by
    `_finalize_step_flags` -- which that path never reaches.

    So a caller that catches the timeout and retries would have the next HEALTHY bar
    report status_unknown=1 and count toward truncation. The flag is cleared on abort.
    """
    import torch

    env, trader = _real_futures_env(budget=0)     # no grace: RuntimeError re-raises
    td = env.reset()

    trader.get_account_balance.side_effect = RuntimeError("Failed to get account balance")
    with pytest.raises(RuntimeError):
        env.step(td.set("action", torch.tensor(len(env.action_levels) - 1)))
    assert not env._status_unknown_this_step, "the aborted bar left the flag set"

    trader.get_account_balance.side_effect = None                  # venue recovers
    nxt = env.step(td.set("action", torch.tensor(1)))["next"]
    assert nxt["status_unknown"].item() == 0.0, "a healthy retry was flagged as unknown"


def test_an_exchange_detected_closure_invalidates_the_cached_balance():
    """A bracket firing has no `close_position()` call for the close-site guard to find.

    The env never asked for this close, so none of the seven close sites run -- but the
    realised P&L has moved equity all the same, and a later grace bar would size against
    the pre-close figure.
    """
    from torchtrade.envs.core.common_types import PositionStatus

    open_long = PositionStatus(
        qty=0.05, notional_value=5000.0, entry_price=100.0, unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0, mark_price=100.0, leverage=10,
        margin_mode="isolated", liquidation_price=91.0,
    )
    env, trader = _real_futures_env(budget=3, sltp=True, position_status=open_long)
    env.reset()
    env.position.current_position = 1
    env._last_confirmed_read["balance"] = {"total_margin_balance": 1e4}

    env._sync_position_from_exchange(None)        # the bracket fired; venue reports flat

    assert "balance" not in env._last_confirmed_read, (
        "an SL/TP closure left pre-close equity cached for the next grace bar to size on"
    )
