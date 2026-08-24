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
    env._flatten_if_policy_says_so = (
        lambda: TorchTradeFuturesLiveEnv._flatten_if_policy_says_so(env)
    )
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
    env._flatten_if_policy_says_so = (
        lambda: TorchTradeFuturesLiveEnv._flatten_if_policy_says_so(env)
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

    assert env.consecutive_unknown_status == 1, "the failure still counts"


def test_the_grace_period_rides_out_an_outage_on_the_last_confirmed_read():
    env, _ = _grace_env(budget=3)
    assert env._halting(lambda: "GOOD", cache_key="pre_trade") == "GOOD"

    # Two failures inside the budget: the cached value stands in, nothing raises.
    for expected_count in (1, 2):
        assert env._halting(_boom, cache_key="pre_trade") == "GOOD"
        assert env.consecutive_unknown_status == expected_count

    # The third spends the budget. It must NOT raise -- an exception here is the process
    # crash #295 exists to remove; the step finishes so _finalize_step_flags can truncate.
    assert env._halting(_boom, cache_key="pre_trade") == "GOOD"
    assert env.consecutive_unknown_status == 3


def test_a_successful_read_ends_the_outage_rather_than_denting_it():
    """It counts CONSECUTIVE unknowns. A lifetime counter would truncate a healthy
    session that had accumulated unrelated blips hours apart."""
    env, _ = _grace_env(budget=3)
    env._halting(lambda: "GOOD", cache_key="pre_trade")
    env._halting(_boom, cache_key="pre_trade")
    env._halting(_boom, cache_key="pre_trade")
    assert env.consecutive_unknown_status == 2

    env._halting(lambda: "FRESH", cache_key="pre_trade")
    assert env.consecutive_unknown_status == 0
    assert env._halting(_boom, cache_key="pre_trade") == "FRESH", "cache refreshed too"


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

    Neither is visible from `_halting` in isolation, which is why this drives `env.step`.
    """
    from unittest.mock import MagicMock, patch
    import numpy as np
    import torch
    from torchtrade.envs.live.binance.env import (
        BinanceFuturesTorchTradingEnv as Env,
        BinanceFuturesTradingEnvConfig as Config,
    )

    observer = MagicMock()
    observer.get_keys.return_value = ["1m_10"]
    observer.get_observations.return_value = {"1m_10": np.zeros((10, 4), dtype=np.float32)}
    observer.get_features.return_value = {
        "observation_features": list("abcd"), "original_features": []
    }
    observer.reset.return_value = None

    trader = MagicMock()
    trader.get_account_balance.return_value = {
        "available_balance": 1e4, "total_margin_balance": 1e4,
        "total_wallet_balance": 1e4, "total_maintenance_margin": 0.0,
    }
    trader.get_status.return_value = {"position_status": None}
    trader.get_mark_price.return_value = 100.0
    trader.get_lot_size.return_value = {"min_qty": 0.001, "qty_step": 0.001}

    config = Config(symbol="BTCUSDT", time_frames=["1m"], window_sizes=[10],
                    execute_on="1m", max_unknown_status_steps=3,
                    close_position_on_init=False)
    with patch("time.sleep"), patch.object(Env, "_wait_for_next_timestamp"):
        env = Env(config=config, observer=observer, trader=trader)
        td = env.reset()

        # One healthy bar, so the grace period has a confirmed read to stand on.
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
    `_halting`, the process crash this work exists to remove did not go away -- it moved
    from `_step` to the following `reset`, one bar later, as a bare PositionUnknownError
    that `except LiveObservationHalt` does not catch and FLATTEN does not act on.

    Raising here is still correct: an episode must not START on unconfirmed state. What
    the grace period buys is riding out an outage mid-episode, not beginning one blind.
    """
    from unittest.mock import MagicMock, patch
    import numpy as np
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
    trader.get_mark_price.return_value = 100.0
    trader.get_lot_size.return_value = {"min_qty": 0.001, "qty_step": 0.001}
    trader.get_status.return_value = {"position_status": None}

    config = Config(symbol="BTCUSDT", time_frames=["1m"], window_sizes=[10],
                    execute_on="1m", max_unknown_status_steps=3,
                    observation_failure_policy=ObservationFailurePolicy.FLATTEN,
                    close_position_on_init=False)
    with patch("time.sleep"), patch.object(Env, "_wait_for_next_timestamp"):
        env = Env(config=config, observer=observer, trader=trader)
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

    # The collector stamps the object it was handed.
    observation.set("done", torch.ones(1, dtype=torch.bool))

    _, cached_obs = env._halting(_boom, cache_key="post_bar")
    assert cached_obs is not observation, "the cache aliases the returned tensordict"
    assert not bool(cached_obs["done"]), (
        "the cached observation inherited the flag stamped after it was cached"
    )
