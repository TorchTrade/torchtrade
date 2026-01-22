"""
Unit tests for AlpacaObservationClass.

Tests observation fetching, preprocessing, and feature extraction using mock clients.
"""

import pytest
import numpy as np
import pandas as pd
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

from torchtrade.envs.live.alpaca.observation import AlpacaObservationClass
from .mocks import MockCryptoHistoricalDataClient


class TestAlpacaObservationClassInitialization:
    """Tests for AlpacaObservationClass initialization."""

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

    def test_init_single_timeframe(self):
        """Test initialization with single timeframe."""
        mock_client = MockCryptoHistoricalDataClient()
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=20,
            client=mock_client,
        )

        assert len(obs_class.timeframes) == 1
        assert obs_class.timeframes[0].value == 15
        assert len(obs_class.window_sizes) == 1
        assert obs_class.window_sizes[0] == 20

    def test_init_multiple_timeframes(self):
        """Test initialization with multiple timeframes."""
        mock_client = MockCryptoHistoricalDataClient()
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
                TimeFrame(1, TimeFrameUnit.Hour),
            ],
            window_sizes=[10, 20, 30],
            client=mock_client,
        )

        assert len(obs_class.timeframes) == 3
        assert len(obs_class.window_sizes) == 3

    def test_init_mismatched_lengths_raises_error(self):
        """Test that mismatched timeframes and window_sizes raises ValueError."""
        mock_client = MockCryptoHistoricalDataClient()

        with pytest.raises(ValueError, match="must have the same length"):
            AlpacaObservationClass(
                symbol="BTC/USD",
                timeframes=[
                    TimeFrame(1, TimeFrameUnit.Minute),
                    TimeFrame(5, TimeFrameUnit.Minute),
                ],
                window_sizes=[10, 20, 30],  # 3 sizes for 2 timeframes
                client=mock_client,
            )


class TestAlpacaObservationClassGetKeys:
    """Tests for get_keys method."""

    def test_get_keys_single_timeframe(self):
        """Test get_keys with single timeframe."""
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

    def test_get_keys_multiple_timeframes(self):
        """Test get_keys with multiple timeframes."""
        mock_client = MockCryptoHistoricalDataClient()
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(1, TimeFrameUnit.Hour),
            ],
            window_sizes=[10, 20],
            client=mock_client,
        )

        keys = obs_class.get_keys()
        assert len(keys) == 2
        assert "1Minute_10" in keys
        assert "1Hour_20" in keys


class TestAlpacaObservationClassGetObservations:
    """Tests for get_observations method."""

    def test_get_observations_single_timeframe(self):
        """Test getting observations for single timeframe."""
        mock_client = MockCryptoHistoricalDataClient(num_bars=100)
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        observations = obs_class.get_observations()

        assert isinstance(observations, dict)
        assert len(observations) == 1

        key = obs_class.get_keys()[0]
        assert key in observations
        assert isinstance(observations[key], np.ndarray)
        assert observations[key].shape[0] == 10  # window_size
        assert observations[key].shape[1] == 4  # default 4 features

    def test_get_observations_multiple_timeframes(self):
        """Test getting observations for multiple timeframes."""
        mock_client = MockCryptoHistoricalDataClient(num_bars=100)
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 20],
            client=mock_client,
        )

        observations = obs_class.get_observations()

        assert len(observations) == 2
        keys = obs_class.get_keys()
        assert observations[keys[0]].shape == (10, 4)
        assert observations[keys[1]].shape == (20, 4)

    def test_get_observations_with_base_ohlc(self):
        """Test getting observations with base OHLC data."""
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

    def test_observations_are_float32(self):
        """Test that observations are float32."""
        mock_client = MockCryptoHistoricalDataClient()
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        observations = obs_class.get_observations()
        key = obs_class.get_keys()[0]

        assert observations[key].dtype == np.float32

    def test_observations_no_nan_values(self):
        """Test that observations don't contain NaN values."""
        mock_client = MockCryptoHistoricalDataClient()
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        observations = obs_class.get_observations()
        key = obs_class.get_keys()[0]

        assert not np.isnan(observations[key]).any()


class TestAlpacaObservationClassGetFeatures:
    """Tests for get_features method."""

    def test_get_features_default_preprocessing(self):
        """Test get_features with default preprocessing."""
        mock_client = MockCryptoHistoricalDataClient()
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        features = obs_class.get_features()

        assert "observation_features" in features
        assert "original_features" in features
        assert len(features["observation_features"]) == 4  # close, open, high, low
        assert "feature_close" in features["observation_features"]
        assert "feature_open" in features["observation_features"]
        assert "feature_high" in features["observation_features"]
        assert "feature_low" in features["observation_features"]


class TestAlpacaObservationClassCustomPreprocessing:
    """Tests for custom preprocessing functions."""

    def test_custom_preprocessing(self):
        """Test with custom preprocessing function."""
        mock_client = MockCryptoHistoricalDataClient()

        def custom_preprocessing(df):
            df = df.reset_index()
            df.dropna(inplace=True)
            df["feature_volatility"] = df["high"] - df["low"]
            df["feature_volume_ma"] = df["volume"].rolling(window=3).mean()
            df.dropna(inplace=True)
            return df

        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            feature_preprocessing_fn=custom_preprocessing,
            client=mock_client,
        )

        observations = obs_class.get_observations()
        key = obs_class.get_keys()[0]

        # Custom preprocessing has 2 features
        assert observations[key].shape[1] == 2

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


class TestAlpacaObservationClassDefaultPreprocessing:
    """Tests for default preprocessing behavior."""

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

    def test_feature_close_is_pct_change(self):
        """Test that feature_close is percentage change."""
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


class TestAlpacaObservationClassEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_window_size_one(self):
        """Test with window size of 1."""
        mock_client = MockCryptoHistoricalDataClient(num_bars=100)
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=1,
            client=mock_client,
        )

        observations = obs_class.get_observations()
        key = obs_class.get_keys()[0]

        assert observations[key].shape[0] == 1

    def test_large_window_size(self):
        """Test with large window size."""
        mock_client = MockCryptoHistoricalDataClient(num_bars=500)
        obs_class = AlpacaObservationClass(
            symbol="BTC/USD",
            timeframes=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=100,
            client=mock_client,
        )

        observations = obs_class.get_observations()
        key = obs_class.get_keys()[0]

        assert observations[key].shape[0] == 100

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


class TestAlpacaObservationClassDataIntegrity:
    """Tests for data integrity and correctness."""

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
