"""The recorded portfolio value and price must belong to the bar the action moved to (#278)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from torchtrade.envs.live.binance.env_sltp import (
    BinanceFuturesSLTPTorchTradingEnv,
    BinanceFuturesSLTPTradingEnvConfig,
)
from torchtrade.envs.live.bitget.env_sltp import (
    BitgetFuturesSLTPTorchTradingEnv,
    BitgetFuturesSLTPTradingEnvConfig,
)
from torchtrade.envs.live.bybit.env_sltp import (
    BybitFuturesSLTPTorchTradingEnv,
    BybitFuturesSLTPTradingEnvConfig,
)
from torchtrade.envs.live.okx.env_sltp import (
    OKXFuturesSLTPTorchTradingEnv,
    OKXFuturesSLTPTradingEnvConfig,
)

# The change is byte-identical across eight files, so it is checked on all four venues.
# Covering only one is how the flat-row regression survived a round.
VENUES = [
    pytest.param(BybitFuturesSLTPTorchTradingEnv, BybitFuturesSLTPTradingEnvConfig, id="bybit"),
    pytest.param(BinanceFuturesSLTPTorchTradingEnv, BinanceFuturesSLTPTradingEnvConfig, id="binance"),
    pytest.param(BitgetFuturesSLTPTorchTradingEnv, BitgetFuturesSLTPTradingEnvConfig, id="bitget"),
    pytest.param(OKXFuturesSLTPTorchTradingEnv, OKXFuturesSLTPTradingEnvConfig, id="okx"),
]


def _build(Env, Cfg, df):
    config = Cfg(
        symbol="BTCUSDT", time_frames=["1m"], window_sizes=[10], execute_on="1m",
        stoploss_levels=(-0.02,), takeprofit_levels=(0.001,), leverage=5,
        trade_mode="quantity", quantity_per_trade=0.01,
    )
    executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5)
    observer = ReplayObserver(df=df, time_frames=config.time_frames,
                              window_sizes=config.window_sizes,
                              execute_on=config.execute_on, executor=executor)
    return config, executor, observer
from torchtrade.envs.replay.observer import ReplayObserver
from torchtrade.envs.replay.order_executor import ReplayOrderExecutor


def _df(n=80):
    ts = pd.date_range("2024-01-01", periods=n, freq="1min")
    prices = np.linspace(50000, 55000, n)  # steep, so consecutive bars differ visibly
    return pd.DataFrame({"timestamp": ts, "open": prices, "high": prices + 10,
                         "low": prices - 10, "close": prices, "volume": np.ones(n) * 100})


@pytest.mark.parametrize("Env,Cfg", VENUES)
def test_the_recorded_price_and_portfolio_value_describe_the_same_bar(Env, Cfg):
    """Every history row must carry one bar's facts -- including the rows that end FLAT.

    Behavioural, not a source-text check: the first version asserted
    `source.find("_get_observation()") < source.find("_get_portfolio_value()")`, which
    `str.find` satisfies from any comment. The version after that asserted only that the
    recorded price was within 1.0 of SOME close, which every bar satisfies.

    The flat rows are the point. `_last_observed_mark` is set only when a position is
    open, so an earlier fix let every flat row fall back to the PRE-trade mark -- and the
    EXIT row is flat and carries the realized PnL, giving `price[t] == close[t-1]`
    against `portfolio_value[t] == equity[t]` on 12 of 14 rows. A test that only opens
    and holds never sees it, which is exactly how it survived a review round.
    """
    import torch

    df = _df()
    config, executor, observer = _build(Env, Cfg, df)
    snapshot = lambda: (executor.get_account_balance()["total_margin_balance"],
                        executor.get_mark_price())
    at_decision, at_next, recorded = [], [], []

    # A live env sleeps until the next real bar boundary; replay supplies the bars.
    with patch.object(Env, "_wait_for_next_timestamp"):
        env = Env(config=config, observer=observer, trader=executor)
        td = env.reset()
        # Open, hold, then CLOSE -- so the sequence contains flat rows, and one of them
        # is the exit row whose PnL is realized.
        for action in (1, 0, 0, 0, 0, 0, 0, 0):
            at_decision.append(snapshot())
            acted = td.clone()
            # Fixed rather than action_spec.rand(): an unseeded sequence makes a failure
            # unreproducible and may never reach the flat rows this test exists for.
            acted["action"] = torch.tensor(action)
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
