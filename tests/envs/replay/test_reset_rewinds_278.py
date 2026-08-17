"""A second episode must start from the beginning, not mid-stream (#278)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.live.bybit.env_sltp import (
    BybitFuturesSLTPTorchTradingEnv,
    BybitFuturesSLTPTradingEnvConfig,
)
from torchtrade.envs.replay.observer import ReplayObserver
from torchtrade.envs.replay.order_executor import ReplayOrderExecutor


def _df(n=120):
    ts = pd.date_range("2024-01-01", periods=n, freq="1min")
    prices = np.linspace(50000, 55000, n)
    return pd.DataFrame({"timestamp": ts, "open": prices, "high": prices + 10,
                         "low": prices - 10, "close": prices, "volume": np.ones(n) * 100})


def _run_episode(env, steps=6):
    td = env.reset()
    for step in range(steps):
        acted = td.clone()
        acted["action"] = torch.tensor(1 if step == 0 else 0)
        td = env.step(acted)["next"]


def test_a_second_episode_rewinds_the_data_the_balance_and_the_baseline():
    """`ReplayObserver.reset()` existed but nothing except a test ever called it.

    Under a collector that made every episode after the first invalid three ways at
    once: it continued mid-stream through the data, it started from the previous
    episode's balance, and it measured bankruptcy against the FIRST episode's starting
    equity -- so a run already down 40% read as 40% down on its opening step.
    """
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

    with patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
        env = BybitFuturesSLTPTorchTradingEnv(config=config, observer=observer, trader=executor)

        _run_episode(env)
        after_first = (observer.sampler._sequential_idx,
                       executor.get_account_balance()["total_margin_balance"],
                       executor.position_qty)
        assert after_first[1] != pytest.approx(10000.0), (
            "the first episode must actually move the balance, or this test proves "
            "nothing about the rewind"
        )

        env.reset()
        idx, balance, qty = (observer.sampler._sequential_idx,
                             executor.get_account_balance()["total_margin_balance"],
                             executor.position_qty)

    assert idx < after_first[0], f"episode 2 starts at index {idx}, mid-stream after {after_first[0]}"
    assert balance == pytest.approx(10000.0), f"episode 2 inherited balance {balance}"
    assert qty == pytest.approx(0.0), f"episode 2 inherited a {qty} position with cancelled brackets"
    assert env.initial_portfolio_value == pytest.approx(balance), (
        f"bankruptcy baseline {env.initial_portfolio_value} is not this episode's "
        f"starting equity {balance}"
    )


def test_the_bankruptcy_baseline_follows_a_live_account_between_episodes():
    """Separate from the rewind test, because replay cannot show this.

    A replayed episode rewinds the balance to its starting value, so a baseline captured
    once in __init__ and one re-captured per episode are numerically identical -- the
    assertion passes either way and proves nothing. A LIVE account persists, and that is
    where capturing once was wrong: episode 2 of a run already down 40% was measured
    against episode 1's opening equity and read as 40% down on its first step.
    """
    from unittest.mock import MagicMock

    from tests.mocks.alpaca import MockObserver, MockTrader
    from torchtrade.envs.live.alpaca.env import (
        AlpacaTorchTradingEnv,
        AlpacaTradingEnvConfig,
    )

    trader = MockTrader(initial_cash=10000.0)
    env = AlpacaTorchTradingEnv(
        config=AlpacaTradingEnvConfig(symbol="BTC/USD", window_sizes=[10]),
        observer=MockObserver(window_sizes=[10]),
        trader=trader,
    )
    env._wait_for_next_timestamp = lambda: None

    env.reset()
    assert env.initial_portfolio_value == pytest.approx(10000.0)

    # The account has lost 40% by the time the next episode starts.
    trader.cash = 6000.0
    env.reset()
    assert env.initial_portfolio_value == pytest.approx(6000.0), (
        "the baseline is still the previous episode's equity, so this episode begins "
        "40% down against a yardstick it never had (#278)"
    )
