"""
Unit tests for AlpacaObservationClass.

Tests observation fetching, preprocessing, and feature extraction using mock clients.
Inherits common tests from BaseObservationClassTests.
"""

import pytest
import numpy as np
import pandas as pd
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

from torchtrade.envs.live.alpaca.observation import AlpacaObservationClass
from tests.mocks.alpaca import MockCryptoHistoricalDataClient
from tests.envs.base_exchange_tests import BaseObservationClassTests


class TestAlpacaObservationClass(BaseObservationClassTests):
    """Tests for AlpacaObservationClass - inherits common tests from base."""

    def create_observer(self, symbol, timeframes, window_sizes, **kwargs):
        """Create an AlpacaObservationClass instance."""
        mock_client = kwargs.get('client', MockCryptoHistoricalDataClient(
            num_bars=kwargs.get('num_bars', 100)
        ))

        return AlpacaObservationClass(
            symbol=symbol,
            timeframes=timeframes,
            window_sizes=window_sizes,
            client=mock_client,
            feature_preprocessing_fn=kwargs.get('feature_preprocessing_fn'),
        )

    def get_expected_symbol_format(self, symbol):
        """Alpaca doesn't modify symbols (slashes handled at order level)."""
        return symbol

    # Alpaca-specific tests

    def test_init_with_mock_client(self):
        """Test initialization with injected mock client."""
        mock_client = MockCryptoHistoricalDataClient()
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        assert obs_class.symbol == "BTC/USD"
        assert len(obs_class.timeframes) == 1
        assert len(obs_class.window_sizes) == 1
        assert obs_class.client is mock_client

    def test_get_keys_format(self):
        """Test that get_keys returns Alpaca-specific format."""
        mock_client = MockCryptoHistoricalDataClient()
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        keys = obs_class.get_keys()
        assert len(keys) == 1
        assert keys[0] == "15Minute_10"

    def test_get_observations_with_base_ohlc_includes_timestamps(self):
        """Test that base OHLC includes timestamps."""
        mock_client = MockCryptoHistoricalDataClient(num_bars=100)
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        observations = obs_class.get_observations(return_base_ohlc=True)

        assert "base_features" in observations
        assert "base_timestamps" in observations
        assert observations["base_features"].shape[1] == 4  # OHLC

    def test_feature_close_is_pct_change(self):
        """Test that default preprocessing creates pct_change features."""
        mock_client = MockCryptoHistoricalDataClient()
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        observations = obs_class.get_observations()
        key = obs_class.get_keys()[0]

        # feature_close should be percentage changes (small values around 0)
        feature_close = observations[key][:, 0]  # First column
        assert np.abs(feature_close).max() < 0.1  # Should be small percentages

    def test_preprocessing_preserves_window_size(self):
        """Test that preprocessing respects window size."""
        mock_client = MockCryptoHistoricalDataClient(num_bars=100)

        def simple_preprocessing(df):
            df = df.reset_index()
            df.dropna(inplace=True)
            df["feature_return"] = df["close"].pct_change().fillna(0)
            df.dropna(inplace=True)
            return df

        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=15,
            feature_preprocessing_fn=simple_preprocessing,
            client=mock_client,
        )

        observations = obs_class.get_observations()
        key = obs_class.get_keys()[0]

        assert observations[key].shape[0] == 15

    def test_default_preprocessing_features(self):
        """Test that default preprocessing creates expected features."""
        mock_client = MockCryptoHistoricalDataClient()
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        features = obs_class.get_features()
        obs_features = features["observation_features"]

        expected_features = ["feature_close", "feature_open", "feature_high", "feature_low"]
        for feat in expected_features:
            assert feat in obs_features

    def test_ohlc_relationship(self):
        """Test that OHLC relationships are maintained."""
        mock_client = MockCryptoHistoricalDataClient()
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        observations = obs_class.get_observations(return_base_ohlc=True)
        base = observations["base_features"]

        # base_features columns: open, high, low, close
        opens = base[:, 0]
        highs = base[:, 1]
        lows = base[:, 2]
        closes = base[:, 3]

        # High should be >= low
        assert np.all(highs >= lows)

    def test_timestamps_are_ordered(self):
        """Test that timestamps are in chronological order."""
        mock_client = MockCryptoHistoricalDataClient()
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        observations = obs_class.get_observations(return_base_ohlc=True)
        timestamps = observations["base_timestamps"]

        # Convert to comparable format if needed
        for i in range(len(timestamps) - 1):
            assert timestamps[i] < timestamps[i + 1]

    def test_different_timeframe_units(self):
        """Test with different timeframe units."""
        mock_client = MockCryptoHistoricalDataClient(num_bars=100)

        # Test hourly
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Hour),
            window_sizes=10,
            client=mock_client,
        )

        keys = obs_class.get_keys()
        assert "1Hour_10" in keys

    def test_multiple_calls_consistency(self):
        """Test that multiple calls return consistent structure."""
        mock_client = MockCryptoHistoricalDataClient()
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        obs1 = obs_class.get_observations()
        obs2 = obs_class.get_observations()

        key = obs_class.get_keys()[0]
        assert obs1[key].shape == obs2[key].shape
