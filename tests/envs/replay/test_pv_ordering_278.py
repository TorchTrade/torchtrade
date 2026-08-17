"""The recorded portfolio value and price must belong to the bar the action moved to (#278)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from torchtrade.envs.live.bybit.env_sltp import (
    BybitFuturesSLTPTorchTradingEnv,
    BybitFuturesSLTPTradingEnvConfig,
)
from torchtrade.envs.replay.observer import ReplayObserver
from torchtrade.envs.replay.order_executor import ReplayOrderExecutor


def _df(n=80):
    ts = pd.date_range("2024-01-01", periods=n, freq="1min")
    prices = np.linspace(50000, 55000, n)  # steep, so consecutive bars differ visibly
    return pd.DataFrame({"timestamp": ts, "open": prices, "high": prices + 10,
                         "low": prices - 10, "close": prices, "volume": np.ones(n) * 100})


def test_the_recorded_portfolio_value_is_the_bar_the_action_moved_to():
    """Behavioural, not a source-text check.

    The first version of this test asserted that `_get_observation()` appeared before
    `_get_portfolio_value()` in the module source. That is one docstring away from
    vacuous -- `str.find` takes the first occurrence anywhere, prose included, and this
    fix's own comments name both methods. Drive the value through instead.

    The bug: a tuple evaluates left to right, and under a ReplayObserver the clock
    advances only inside `get_observations()`, so reading PV first recorded the PREVIOUS
    bar's equity against this bar's action -- 8/8 against the decision bar, 0/8 against
    the next.
    """
    import torch

    df = _df()
    config = BybitFuturesSLTPTradingEnvConfig(
        symbol="BTCUSDT", time_frames=["1m"], window_sizes=[10], execute_on="1m",
        stoploss_levels=(-0.02,), takeprofit_levels=(0.03,), leverage=5,
        trade_mode="quantity", quantity_per_trade=0.01,
    )
    executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5)
    observer = ReplayObserver(df=df, time_frames=config.time_frames,
                              window_sizes=config.window_sizes,
                              execute_on=config.execute_on, executor=executor)

    equity = lambda: executor.get_account_balance()["total_margin_balance"]
    at_decision, at_next, recorded = [], [], []

    # A live env sleeps until the next real bar boundary; replay supplies the bars.
    with patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
        env = BybitFuturesSLTPTorchTradingEnv(config=config, observer=observer, trader=executor)
        td = env.reset()
        for step in range(8):
            at_decision.append(equity())            # the bar the policy is looking at
            acted = td.clone()
            acted["action"] = torch.tensor(1 if step == 0 else 0)  # open, then hold
            td = env.step(acted)["next"]
            at_next.append(equity())                # the bar the action moved us to
            recorded.append(env.history.portfolio_values[-1])

    stale = sum(r == pytest.approx(d) for r, d in zip(recorded, at_decision))
    correct = sum(r == pytest.approx(n) for r, n in zip(recorded, at_next))
    assert correct == len(recorded) and stale == 0, (
        f"{stale}/{len(recorded)} recorded PVs are the DECISION bar's equity (stale) "
        f"and {correct}/{len(recorded)} are the next bar's -- the reward at step t would "
        f"belong to the action at t-1 (#278)"
    )


def test_the_recorded_price_and_portfolio_value_describe_the_same_bar():
    """Both facts in a history row must come from one bar.

    The first version of this test asserted only that the recorded price was within 1.0
    of SOME close in the frame -- which every bar satisfies, so reverting `price=` to the
    pre-trade mark left it passing. It tested nothing. This pins the price and the equity
    to the SAME post-step snapshot, and fails under either mutation: price reverted to
    pre-trade, or the pair read before the observation.
    """
    import torch

    df = _df()
    config = BybitFuturesSLTPTradingEnvConfig(
        symbol="BTCUSDT", time_frames=["1m"], window_sizes=[10], execute_on="1m",
        stoploss_levels=(-0.02,), takeprofit_levels=(0.03,), leverage=5,
        trade_mode="quantity", quantity_per_trade=0.01,
    )
    executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5)
    observer = ReplayObserver(df=df, time_frames=config.time_frames,
                              window_sizes=config.window_sizes,
                              execute_on=config.execute_on, executor=executor)

    snapshot = lambda: (executor.get_account_balance()["total_margin_balance"],
                        executor.get_mark_price())
    at_decision, at_next, recorded = [], [], []

    with patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
        env = BybitFuturesSLTPTorchTradingEnv(config=config, observer=observer, trader=executor)
        td = env.reset()
        for step in range(8):
            at_decision.append(snapshot())
            acted = td.clone()
            # Fixed, not action_spec.rand(): an unseeded action sequence makes a failure
            # unreproducible, and holding a position open is what exposes the lag.
            acted["action"] = torch.tensor(1 if step == 0 else 0)
            td = env.step(acted)["next"]
            at_next.append(snapshot())
            recorded.append((env.history.portfolio_values[-1], env.history.base_prices[-1]))

    stale = [i for i, (r, d) in enumerate(zip(recorded, at_decision))
             if r[1] == pytest.approx(d[1])]
    correct = [i for i, (r, n) in enumerate(zip(recorded, at_next)) if r == pytest.approx(n)]
    assert len(correct) == len(recorded) and not stale, (
        f"{len(correct)}/{len(recorded)} rows have BOTH facts at the post-bar snapshot; "
        f"rows {stale} carry the decision bar's price against the next bar's equity (#278)"
    )
