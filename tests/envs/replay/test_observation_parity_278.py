"""Replay must emit the observation offline and live emit, not a lookalike (#278)."""

import numpy as np
import pandas as pd
import pytest

from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.replay.observer import ReplayObserver
from torchtrade.envs.utils import TimeFrame, TimeFrameUnit

TF = TimeFrame(1, TimeFrameUnit.Minute)


def _df(n=160):
    ts = pd.date_range("2024-01-01", periods=n, freq="1min")
    prices = np.linspace(100, 110, n)
    return pd.DataFrame({"timestamp": ts, "open": prices, "high": prices + 0.2,
                         "low": prices - 0.2, "close": prices, "volume": np.ones(n) * 1000})


def _observer(**kw):
    return ReplayObserver(df=_df(), time_frames=[TF], window_sizes=[10], execute_on=TF, **kw)


def test_a_replay_backed_env_emits_the_offline_env_keys():
    """Compared at the ENV, not the observer -- the layer the first version got wrong.

    The env bases build their spec key as `"market_data_" + observer.get_keys()`, and the
    real observers return the UNPREFIXED `{timeframe}_{window}`. So the sampler's bare
    timeframe names gave `market_data_1Minute` where offline gives
    `market_data_1Minute_10`, and adding the prefix in the observer to compensate gave
    `market_data_market_data_1Minute_10`. Asserting the observer against itself passed
    for both of those; only the env's own spec keys settle it.
    """
    from unittest.mock import patch

    from torchtrade.envs.live.bybit.env_sltp import (
        BybitFuturesSLTPTorchTradingEnv,
        BybitFuturesSLTPTradingEnvConfig,
    )
    from torchtrade.envs.replay.order_executor import ReplayOrderExecutor

    df = _df()
    offline = SequentialTradingEnv(
        df, SequentialTradingEnvConfig(time_frames=[TF], window_sizes=[10], execute_on=TF),
    )
    offline_keys = {k for k in offline.reset().keys() if k.startswith("market_data_")}

    config = BybitFuturesSLTPTradingEnvConfig(
        symbol="BTCUSDT", time_frames=[TF], window_sizes=[10], execute_on=TF,
        stoploss_levels=(-0.02,), takeprofit_levels=(0.03,), leverage=5,
        trade_mode="quantity", quantity_per_trade=0.01,
    )
    executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5)
    observer = ReplayObserver(df=df, time_frames=[TF], window_sizes=[10],
                              execute_on=TF, executor=executor)
    with patch.object(BybitFuturesSLTPTorchTradingEnv, "_wait_for_next_timestamp"):
        env = BybitFuturesSLTPTorchTradingEnv(config=config, observer=observer, trader=executor)
        replay_keys = {k for k in env.reset().keys() if k.startswith("market_data_")}

    assert replay_keys == offline_keys, (
        f"replay env emits {sorted(replay_keys)}, offline emits {sorted(offline_keys)}"
    )


def test_features_are_reported_when_no_preprocessing_fn_is_given():
    """Filtering to names starting with `features_` reported ZERO columns.

    The sampler still emits the five raw OHLCV columns, so the declared spec came out
    `(window, 0)` against an emitted `(window, 5)` and check_env_specs failed on a size
    mismatch. Only the no-preprocessing-fn config was affected, which is why it survived.
    """
    observer = _observer()
    reported = observer.get_features()["observation_features"]
    emitted = observer.get_observations()[observer.get_keys()[0]].shape[1]
    assert len(reported) == emitted, (
        f"spec would declare {len(reported)} features against {emitted} emitted"
    )


@pytest.mark.parametrize("steps", [1, 2, 10, 14])
def test_base_features_is_as_full_as_the_market_data_from_the_first_step(steps):
    """`include_base_features=True` declares a (window, 4) spec and puts it in the
    OBSERVATION, so a policy consuming that key read 9 of 10 rows as zero on every step.
    Only the newest row was written.

    Filling from the sampler's TRUNCATED execution frame fixed nine steps out of ten and
    left step 1 at 9/10 zeros -- while `market_data_*`, same timeframe and window, was
    full of real bars from step 1. Those bars exist; they were outside the frame. This
    compares the two keys rather than a hardcoded count, so no zero-padding regime can
    be codified as expected again.
    """
    observer = _observer()
    for _ in range(steps):
        obs = observer.get_observations(return_base_ohlc=True)
    base = obs["base_features"]
    market = obs[observer.get_keys()[0]]

    base_zero = int((base == 0).all(axis=1).sum())
    market_zero = int((market == 0).all(axis=1).sum())
    assert base_zero == market_zero, (
        f"base_features has {base_zero} empty rows against market_data's {market_zero}"
    )
    closes = base[:, 3]
    assert np.all(np.diff(closes) > 0), f"rows are not consecutive bars in order: {closes}"
