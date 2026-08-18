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
        assert after_first[2] != pytest.approx(0.0), (
            "the first episode must END holding a position, or the qty assertion below "
            "compares zero to zero -- a round trip moves the balance too"
        )

        env.reset()
        idx, balance, qty = (observer.sampler._sequential_idx,
                             executor.get_account_balance()["total_margin_balance"],
                             executor.position_qty)

    assert idx < after_first[0], f"episode 2 starts at index {idx}, mid-stream after {after_first[0]}"
    assert balance == pytest.approx(10000.0), f"episode 2 inherited balance {balance}"
    assert qty == pytest.approx(0.0), f"episode 2 inherited a {qty} position with cancelled brackets"
    # No baseline assertion here: replay rewinds the balance to the SAME number, so
    # capturing once and capturing per episode are indistinguishable in this scenario.
    # An assertion that cannot fail is worse than none -- the sibling test carries it.
