"""Unit tests for AlpacaObservationClass.

Common observation-class behavior (init, keys, shapes, float32, no-NaN, features,
window sizes, custom preprocessing) is inherited from BaseObservationClassTests.
Only Alpaca-specific tests (client injection, data integrity, feature semantics)
and stricter assertions live here.
"""

import numpy as np
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
        assert observations["base_features"].shape[1] == 4

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

    def test_feature_close_is_pct_change(self):
        """feature_close is a percentage change (small values around 0)."""
        obs = self.create_observer(
            symbol="BTC/USD", timeframes=TimeFrame(1, TimeFrameUnit.Minute), window_sizes=10)
        key = obs.get_keys()[0]
        feature_close = obs.get_observations()[key][:, 0]
        assert np.abs(feature_close).max() < 0.1

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

    def test_ohlc_relationship(self):
        """OHLC high >= low holds in the base features."""
        obs = self.create_observer(
            symbol="BTC/USD", timeframes=TimeFrame(1, TimeFrameUnit.Minute), window_sizes=10)
        base = obs.get_observations(return_base_ohlc=True)["base_features"]
        highs, lows = base[:, 1], base[:, 2]
        assert np.all(highs >= lows)

    def test_timestamps_are_ordered(self):
        """base_timestamps are chronological."""
        obs = self.create_observer(
            symbol="BTC/USD", timeframes=TimeFrame(1, TimeFrameUnit.Minute), window_sizes=10)
        timestamps = obs.get_observations(return_base_ohlc=True)["base_timestamps"]
        for i in range(len(timestamps) - 1):
            assert timestamps[i] < timestamps[i + 1]
