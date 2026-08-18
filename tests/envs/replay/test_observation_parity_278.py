"""Replay must emit the observation offline and live emit, not a lookalike (#278)."""

import numpy as np
import pandas as pd
import pytest

from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.replay.observer import ReplayObserver
from torchtrade.envs.replay.order_executor import ReplayOrderExecutor
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


# The timeframe layout is the axis both base_features bugs lived on, and the original
# test parametrized only `steps` on a single-timeframe fixture -- so it could not see
# either. execute_on COARSER than time_frames[0] is deliberately absent: market_data
# there extends past the window this builds, and asserting a relationship I have not
# pinned down would be worse than saying so.
LAYOUTS = [
    pytest.param([TF], [10], TF, id="single-1Min"),
    pytest.param([TimeFrame(5, TimeFrameUnit.Minute), TF], [6, 10], TF, id="5Min-first"),
    pytest.param([TimeFrame(15, TimeFrameUnit.Minute), TF], [4, 10], TF, id="15Min-first"),
]


@pytest.mark.parametrize("time_frames,window_sizes,execute_on", LAYOUTS)
@pytest.mark.parametrize("steps", [1, 30])
def test_base_features_is_the_first_timeframes_bars(time_frames, window_sizes, execute_on, steps):
    """base_features is `time_frames[0]`'s raw OHLC -- not the execution timeframe's.

    Live gates on `timeframe == self.time_frames[0]` and every venue sizes the spec off
    `window_sizes[0]`. Filling from `execute_on` gave a window of the right SHAPE holding
    entirely different bars: with `[15Min, 1Min]` it spanned 4 minutes where live spans
    an hour. That is worse than the original defect, which left the mismatched rows
    visibly zero -- this one looks plausible and `check_env_specs` passes.

    Asserted against `market_data_*` for the same timeframe rather than a hardcoded
    count, so neither a zero-padding regime nor a wrong-timeframe window can be written
    down as expected.
    """
    executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5)
    observer = ReplayObserver(df=_df(1200), time_frames=time_frames,
                              window_sizes=window_sizes, execute_on=execute_on,
                              executor=executor)
    for _ in range(steps):
        obs = observer.get_observations(return_base_ohlc=True)
    base, market = obs["base_features"], obs[observer.get_keys()[0]]

    assert base.shape == (window_sizes[0], 4)
    assert int((base == 0).all(axis=1).sum()) == int((market == 0).all(axis=1).sum())
    # All but the last row are completed tf0 bars and must match exactly.
    assert np.allclose(base[:-1, 3], market[:-1, 3], atol=1e-3), (
        f"base_features closes {base[:-1, 3]} are not {time_frames[0]}'s bars "
        f"{market[:-1, 3]} -- wrong timeframe or shifted by the end-time relabel"
    )
    # The last row is the FORMING bar, as live's is: its close is the latest trade.
    # `base_features[-1, 3]` prices entries and BOTH brackets in three SLTP envs
    # (`*/env_sltp.py: current_price = float(obs["base_features"][-1, 3])`), so leaving
    # it at the last CLOSED tf0 close sizes and brackets off a stale number -- measured
    # at -1.94%/+3.07% against a configured -2%/+3%. Asserted against the executor,
    # which is what the venue would actually fill at.
    assert base[-1, 3] == pytest.approx(executor.current_price, abs=1e-3), (
        f"base_features[-1, 3] is {base[-1, 3]}, not the execution price "
        f"{executor.current_price} -- entries and brackets would price off a stale bar"
    )
