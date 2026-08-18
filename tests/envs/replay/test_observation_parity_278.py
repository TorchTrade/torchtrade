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


def test_the_market_data_key_matches_the_offline_env():
    """`market_data_1Minute` vs `market_data_1Minute_10`.

    The sampler's raw keys are bare timeframe names; offline and live both append the
    window size. A policy trained offline could not be fed the replay env without
    rewriting its in_keys -- which defeats the point of replaying through the live
    pipeline.
    """
    offline = SequentialTradingEnv(
        _df(), SequentialTradingEnvConfig(
            time_frames=[TF], window_sizes=[10], execute_on=TF,
        ),
    )
    offline_keys = {k for k in offline.reset().keys() if k.startswith("market_data_")}
    assert set(_observer().get_keys()) == offline_keys, (
        f"replay emits {_observer().get_keys()}, offline emits {sorted(offline_keys)}"
    )


def test_features_are_reported_when_no_preprocessing_fn_is_given():
    """Filtering to names starting with `features_` reported ZERO columns.

    The sampler still emits the five raw OHLCV columns, so the declared spec came out
    `(window, 0)` against an emitted `(window, 5)` and check_env_specs failed on a size
    mismatch. Only the no-preprocessing-fn config was affected, which is why it survived.
    """
    reported = _observer().get_features()["observation_features"]
    emitted = _observer().get_observations()["market_data_1Minute_10"].shape[1]
    assert len(reported) == emitted, (
        f"spec would declare {len(reported)} features against {emitted} emitted"
    )


@pytest.mark.parametrize("steps,expected_zero_rows", [
    (1, 9),   # one bar of history exists, so nine rows genuinely have no data
    (10, 0),  # once ten bars have passed the window is full
    (14, 0),
])
def test_base_features_fills_the_window_not_only_the_last_row(steps, expected_zero_rows):
    """`include_base_features=True` declares a (window, 4) spec and puts it in the
    OBSERVATION, so a policy consuming that key read 90% zeros -- 9 of 10 rows on every
    step, forever. Only the newest row was written.

    Rows before the start of the data stay zero deliberately: there is no bar to report.
    """
    observer = _observer()
    for _ in range(steps):
        base = observer.get_observations(return_base_ohlc=True)["base_features"]

    assert int((base == 0).all(axis=1).sum()) == expected_zero_rows
    if expected_zero_rows == 0:
        closes = base[:, 3]
        assert np.all(np.diff(closes) > 0), (
            f"rows are not consecutive bars in order: {closes}"
        )
