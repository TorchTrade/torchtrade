"""Unit tests for AlpacaObservationClass.

Common observation-class behavior (init, keys, shapes, float32, no-NaN, features,
window sizes, custom preprocessing) is inherited from BaseObservationClassTests.
Only Alpaca-specific tests (client injection, data integrity, feature semantics)
and stricter assertions live here.
"""

import pytest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

from torchtrade.envs.live.alpaca.observation import AlpacaObservationClass
from tests.envs.base_exchange_tests import BaseObservationClassTests
from .mocks import MockCryptoHistoricalDataClient


class TestAlpacaObservationClass(BaseObservationClassTests):
    """Alpaca observation class — common tests inherited from the base."""

    def create_observer(self, symbol, timeframes, window_sizes, **kwargs):
        # num_bars generous so the base's window=100 test is fully populated.
        client = kwargs.pop("client", None) or MockCryptoHistoricalDataClient(num_bars=600)
        return AlpacaObservationClass(
            symbol=symbol, timeframes=timeframes, window_sizes=window_sizes,
            client=client, **kwargs,
        )

    def get_expected_symbol_format(self, symbol):
        return symbol  # Alpaca stores the symbol unchanged

    # --- Alpaca-specific / stricter tests ---

    def test_init_with_mock_client(self):
        """Injected client is stored as-is."""
        mock_client = MockCryptoHistoricalDataClient()
        obs = AlpacaObservationClass(
            symbol="BTC/USD", timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10, client=mock_client)
        assert obs.symbol == "BTC/USD"
        assert obs.client is mock_client

    def test_a_daily_timeframe_is_refused_at_construction(self):
        """The 60-day fetch window cannot fill a daily observation, and says so early.

        This check used to live inside `_fetch_single_timeframe`, so the config built
        fine and failed on the first fetch -- during `reset()`, on a live account. The
        shared base calls `_validate_timeframe` from `__init__`, which is where the
        futures venues have always validated theirs (#288). Untested either way before.
        """
        with pytest.raises(ValueError, match="not allowed for daily data"):
            self.create_observer(
                symbol="BTC/USD", timeframes=TimeFrame(1, TimeFrameUnit.Day),
                window_sizes=5)

    def test_get_keys_multiple_timeframes(self):
        obs = self.create_observer(
            symbol="BTC/USD",
            timeframes=[TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(1, TimeFrameUnit.Hour)],
            window_sizes=[10, 20])
        keys = obs.get_keys()
        assert "1Minute_10" in keys and "1Hour_20" in keys

    def test_get_observations_exact_shapes(self):
        """Alpaca observations have exact (window, 4) shapes (stricter than base >=4)."""
        obs = self.create_observer(
            symbol="BTC/USD",
            timeframes=[TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute)],
            window_sizes=[10, 20])
        observations = obs.get_observations()
        keys = obs.get_keys()
        assert observations[keys[0]].shape == (10, 4)
        assert observations[keys[1]].shape == (20, 4)

    def test_get_observations_with_base_ohlc(self):
        """base_features + base_timestamps present (Alpaca also exposes base_timestamps)."""
        obs = self.create_observer(
            symbol="BTC/USD", timeframes=TimeFrame(1, TimeFrameUnit.Minute), window_sizes=10)
        observations = obs.get_observations(return_base_ohlc=True)
        assert "base_features" in observations
        assert "base_timestamps" in observations
        # (window, 4), not the full 600-bar lookback: base_features must be windowed like
        # market_data, or it disagrees with the env's declared (window, 4) spec (#69).
        assert observations["base_features"].shape == (10, 4)
        assert observations["base_timestamps"].shape == (10,)

    def test_get_features_default_names(self):
        """Default preprocessing yields the four named OHLC features."""
        obs = self.create_observer(
            symbol="BTC/USD", timeframes=TimeFrame(1, TimeFrameUnit.Minute), window_sizes=10)
        features = obs.get_features()
        for feat in ["feature_close", "feature_open", "feature_high", "feature_low"]:
            assert feat in features["observation_features"]

    def test_preprocessing_preserves_window_size(self):
        """A custom preprocessing fn still yields the requested window size."""
        def simple_preprocessing(df):
            df = df.reset_index()
            df.dropna(inplace=True)
            df["feature_return"] = df["close"].pct_change().fillna(0)
            df.dropna(inplace=True)
            return df

        obs = self.create_observer(
            symbol="BTC/USD", timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=15, feature_preprocessing_fn=simple_preprocessing)
        key = obs.get_keys()[0]
        assert obs.get_observations()[key].shape[0] == 15

    def test_different_timeframe_units(self):
        """Hourly timeframe produces the expected key."""
        obs = self.create_observer(
            symbol="BTC/USD", timeframes=TimeFrame(1, TimeFrameUnit.Hour), window_sizes=10)
        assert "1Hour_10" in obs.get_keys()

    def test_multiple_calls_consistency(self):
        """Repeated calls return the same-shaped output (structural consistency)."""
        obs = self.create_observer(
            symbol="BTC/USD", timeframes=TimeFrame(1, TimeFrameUnit.Minute), window_sizes=10)
        key = obs.get_keys()[0]
        obs1 = obs.get_observations()[key]
        obs2 = obs.get_observations()[key]
        assert obs1.shape == obs2.shape == (10, 4)


def test_a_malformed_candle_never_reaches_alpaca_base_features(monkeypatch):
    """Alpaca already dropped NaN before slicing base_features; nothing tested it.

    Deleting that dropna failed zero tests, because every mock client returns clean
    candles -- so the protection that the futures base was missing (#395) was itself
    unguarded on the one venue that had it. Injected at the frame level, since which
    malformed payload produces a NaN is venue-specific.
    """
    observer = AlpacaObservationClass(
        symbol="BTC/USD",
        timeframes=TimeFrame(1, TimeFrameUnit.Minute),
        window_sizes=10,
        client=MagicMock(),
    )

    def frame_with_a_hole(timeframe, limit=None):
        n = 60
        df = pd.DataFrame({
            "symbol": ["BTC/USD"] * n,  # alpaca's default preprocessing drops this column
            "open": np.full(n, 100.0), "high": np.full(n, 101.0),
            "low": np.full(n, 99.0), "close": np.full(n, 100.5),
            "volume": np.full(n, 10.0),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        })
        df.loc[57, "close"] = np.nan  # inside the last 10, (alpaca's SDK returns floats, so a
        # gap arrives already NaN rather than via a coerced parse)
        return df

    monkeypatch.setattr(observer, "_fetch_single_timeframe", frame_with_a_hole)
    observations = observer.get_observations(return_base_ohlc=True)
    assert not np.isnan(observations["base_features"]).any()


def test_alpaca_base_features_survive_a_fn_that_leaves_the_timestamp_in_the_index():
    """Alpaca's SDK returns a (symbol, timestamp) MultiIndex.

    The default preprocessing calls reset_index, but `feature_preprocessing_fn` is a
    public constructor argument and a custom fn need not. Main handled this because it
    did its own `df.reset_index()`; slicing the processed frame instead read `timestamp`
    as a COLUMN and regressed that path to a KeyError (#395 review).

    Rows still come from the processed frame, so alignment is unaffected -- only where
    the timestamps are read from changes.
    """
    def keeps_the_multiindex(df):
        df = df.copy()
        df["feature_close"] = df["close"].pct_change().fillna(0)
        return df.dropna()

    observer = AlpacaObservationClass(
        symbol="BTC/USD",
        timeframes=TimeFrame(1, TimeFrameUnit.Minute),
        window_sizes=10,
        feature_preprocessing_fn=keeps_the_multiindex,
        client=MagicMock(),
    )
    n = 60
    stamps = pd.date_range("2024-01-01", periods=n, freq="1min")
    frame = pd.DataFrame(
        {"open": np.full(n, 100.0), "high": np.full(n, 101.0),
         "low": np.full(n, 99.0), "close": np.linspace(100.0, 130.0, n),
         "volume": np.full(n, 10.0)},
        index=pd.MultiIndex.from_arrays(
            [["BTC/USD"] * n, stamps], names=["symbol", "timestamp"]
        ),
    )
    observer._fetch_single_timeframe = lambda timeframe, limit=None: frame

    observations = observer.get_observations(return_base_ohlc=True)
    assert observations["base_features"].shape == (10, 4)
    assert list(pd.to_datetime(observations["base_timestamps"])) == list(stamps[-10:])


def test_alpaca_refuses_a_fn_that_loses_the_timestamp_entirely():
    """Neither a column nor an index level: raise, do not invent positions as times.

    Falling back to `df.index.values` would hand back 0..n for a frame whose OHLC came
    from real bars -- a spec-shaped lie. Validate at the boundary (invariant 4).
    """
    import numpy as np
    import pandas as pd
    import pytest as _pytest
    from torchtrade.envs.live.alpaca.observation import AlpacaObservationClass
    from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

    observer = AlpacaObservationClass(
        symbol="BTC/USD",
        timeframes=TimeFrame(1, TimeFrameUnit.Minute),
        window_sizes=10,
        feature_preprocessing_fn=lambda df: df.reset_index(drop=True).assign(
            feature_close=0.0
        ),
        client=MagicMock(),
    )
    n = 60
    observer._fetch_single_timeframe = lambda timeframe, limit=None: pd.DataFrame(
        {"open": np.full(n, 100.0), "high": np.full(n, 101.0),
         "low": np.full(n, 99.0), "close": np.full(n, 100.5),
         "volume": np.full(n, 10.0)},
        index=pd.MultiIndex.from_arrays(
            [["BTC/USD"] * n, pd.date_range("2024-01-01", periods=n, freq="1min")],
            names=["symbol", "timestamp"],
        ),
    )
    with _pytest.raises(KeyError, match="timestamp"):
        observer.get_observations(return_base_ohlc=True)


def test_alpaca_drops_a_zero_priced_bar_rather_than_emitting_inf():
    """`close.pct_change()` off a zero close is inf, and dropna does not remove it.

    The shared harness assertion cannot catch this on its own: every mock client returns
    clean candles, so the invariant holds there whether or not the filter exists. The
    zero-priced bar has to be injected (#398).
    """
    observer = AlpacaObservationClass(
        symbol="BTC/USD",
        timeframes=TimeFrame(1, TimeFrameUnit.Minute),
        window_sizes=5,
        client=MagicMock(),
    )
    n = 60
    close = np.linspace(100.0, 130.0, n)
    close[54] = 0.0  # a halted/malformed candle, inside the window
    frame = pd.DataFrame(
        {"open": np.full(n, 100.0), "high": np.full(n, 101.0),
         "low": np.full(n, 99.0), "close": close, "volume": np.full(n, 10.0)},
        index=pd.MultiIndex.from_arrays(
            [["BTC/USD"] * n, pd.date_range("2024-01-01", periods=n, freq="1min")],
            names=["symbol", "timestamp"],
        ),
    )
    observer._fetch_single_timeframe = lambda timeframe, limit=None: frame

    observations = observer.get_observations(return_base_ohlc=True)
    for key, array in observations.items():
        if np.asarray(array).dtype.kind in "fi":
            assert np.isfinite(array).all(), f"{key} carries a non-finite value"


def test_alpaca_refuses_a_stale_last_bar():
    """Alpaca has its own copy of the guard, and removing it whole failed zero tests.

    The zero-bar test above injects at index 54 with window 5 -- outside the retained
    window -- so it never exercises the case where the LAST row is the one dropped. That
    is the case that matters: `base_features[-1, 3]` is the live price at
    `alpaca/env_sltp.py:275`, and a stale one passes its isfinite/`<= 0` guard (#398).
    """
    import numpy as np
    import pandas as pd
    from torchtrade.envs.live.alpaca.observation import AlpacaObservationClass
    from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

    observer = AlpacaObservationClass(
        symbol="BTC/USD",
        timeframes=TimeFrame(1, TimeFrameUnit.Minute),
        window_sizes=5,
        client=MagicMock(),
    )
    n = 60
    close = np.linspace(100.0, 130.0, n)
    close[-1] = 0.0  # the most recent candle is unusable
    observer._fetch_single_timeframe = lambda timeframe, limit=None: pd.DataFrame(
        {"open": np.full(n, 100.0), "high": np.full(n, 101.0),
         "low": np.full(n, 99.0), "close": close, "volume": np.full(n, 10.0)},
        index=pd.MultiIndex.from_arrays(
            [["BTC/USD"] * n, pd.date_range("2024-01-01", periods=n, freq="1min")],
            names=["symbol", "timestamp"],
        ),
    )
    with pytest.raises(ValueError, match="most recent candle"):
        observer.get_observations(return_base_ohlc=True)
