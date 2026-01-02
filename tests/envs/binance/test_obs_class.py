"""Tests for BinanceObservationClass."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


class TestBinanceObservationClass:
    """Tests for BinanceObservationClass."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Binance client."""
        client = MagicMock()

        # Mock klines data (12 columns)
        def mock_get_klines(symbol, interval, limit=500):
            klines = []
            base_time = 1700000000000  # Base timestamp in ms
            for i in range(limit):
                klines.append([
                    base_time + i * 60000,  # open_time
                    "50000.0",  # open
                    "50100.0",  # high
                    "49900.0",  # low
                    "50050.0",  # close
                    "100.0",    # volume
                    base_time + i * 60000 + 59999,  # close_time
                    "5000000.0",  # quote_volume
                    100,        # trades
                    "50.0",     # taker_buy_base
                    "2500000.0",  # taker_buy_quote
                    "0",        # ignore
                ])
            return klines

        client.get_klines = MagicMock(side_effect=mock_get_klines)
        return client

    @pytest.fixture
    def observer_single(self, mock_client):
        """Create observer with single interval."""
        from torchtrade.envs.binance.obs_class import BinanceObservationClass

        return BinanceObservationClass(
            symbol="BTCUSDT",
            intervals="15m",
            window_sizes=10,
            client=mock_client,
        )

    @pytest.fixture
    def observer_multi(self, mock_client):
        """Create observer with multiple intervals."""
        from torchtrade.envs.binance.obs_class import BinanceObservationClass

        return BinanceObservationClass(
            symbol="BTCUSDT",
            intervals=["1m", "5m", "1h"],
            window_sizes=[10, 20, 15],
            client=mock_client,
        )

    def test_single_interval_initialization(self, observer_single):
        """Test initialization with single interval."""
        assert observer_single.symbol == "BTCUSDT"
        assert observer_single.intervals == ["15m"]
        assert observer_single.window_sizes == [10]

    def test_multi_interval_initialization(self, observer_multi):
        """Test initialization with multiple intervals."""
        assert observer_multi.symbol == "BTCUSDT"
        assert observer_multi.intervals == ["1m", "5m", "1h"]
        assert observer_multi.window_sizes == [10, 20, 15]

    def test_symbol_normalization(self, mock_client):
        """Test that symbol with slash is normalized."""
        from torchtrade.envs.binance.obs_class import BinanceObservationClass

        observer = BinanceObservationClass(
            symbol="BTC/USDT",
            intervals="1m",
            window_sizes=10,
            client=mock_client,
        )
        assert observer.symbol == "BTCUSDT"

    def test_invalid_interval_raises_error(self, mock_client):
        """Test that invalid interval raises ValueError."""
        from torchtrade.envs.binance.obs_class import BinanceObservationClass

        with pytest.raises(ValueError, match="Invalid interval"):
            BinanceObservationClass(
                symbol="BTCUSDT",
                intervals="2m",  # Invalid interval
                window_sizes=10,
                client=mock_client,
            )

    def test_mismatched_lengths_raises_error(self, mock_client):
        """Test that mismatched intervals and window_sizes raises error."""
        from torchtrade.envs.binance.obs_class import BinanceObservationClass

        with pytest.raises(ValueError, match="same length"):
            BinanceObservationClass(
                symbol="BTCUSDT",
                intervals=["1m", "5m"],
                window_sizes=[10],  # Mismatched length
                client=mock_client,
            )

    def test_get_keys_single(self, observer_single):
        """Test get_keys with single interval."""
        keys = observer_single.get_keys()
        assert keys == ["15m_10"]

    def test_get_keys_multi(self, observer_multi):
        """Test get_keys with multiple intervals."""
        keys = observer_multi.get_keys()
        assert keys == ["1m_10", "5m_20", "1h_15"]

    def test_get_observations_single(self, observer_single):
        """Test get_observations with single interval."""
        observations = observer_single.get_observations()

        assert "15m_10" in observations
        assert observations["15m_10"].shape == (10, 4)  # window_size x num_features
        assert observations["15m_10"].dtype == np.float32

    def test_get_observations_multi(self, observer_multi):
        """Test get_observations with multiple intervals."""
        observations = observer_multi.get_observations()

        assert "1m_10" in observations
        assert "5m_20" in observations
        assert "1h_15" in observations

        assert observations["1m_10"].shape == (10, 4)
        assert observations["5m_20"].shape == (20, 4)
        assert observations["1h_15"].shape == (15, 4)

    def test_get_observations_with_base_ohlc(self, observer_single):
        """Test get_observations with base OHLC data."""
        observations = observer_single.get_observations(return_base_ohlc=True)

        assert "15m_10" in observations
        assert "base_features" in observations
        assert "base_timestamps" in observations

        assert observations["base_features"].shape == (10, 4)

    def test_custom_preprocessing(self, mock_client):
        """Test custom preprocessing function."""
        from torchtrade.envs.binance.obs_class import BinanceObservationClass

        def custom_preprocessing(df):
            df = df.copy()
            df["feature_custom1"] = df["close"].pct_change().fillna(0)
            df["feature_custom2"] = df["volume"].pct_change().fillna(0)
            df.dropna(inplace=True)
            return df

        observer = BinanceObservationClass(
            symbol="BTCUSDT",
            intervals="1m",
            window_sizes=10,
            client=mock_client,
            feature_preprocessing_fn=custom_preprocessing,
        )

        observations = observer.get_observations()
        assert observations["1m_10"].shape == (10, 2)  # 2 custom features

    def test_get_features(self, observer_single):
        """Test get_features returns feature information."""
        features = observer_single.get_features()

        assert "observation_features" in features
        assert "original_features" in features
        assert len(features["observation_features"]) > 0

    def test_default_preprocessing_output(self, observer_single):
        """Test that default preprocessing produces expected features."""
        features = observer_single.get_features()

        expected_features = ["feature_close", "feature_open", "feature_high", "feature_low"]
        for feat in expected_features:
            assert feat in features["observation_features"]


class TestBinanceObservationClassIntegration:
    """Integration tests that would require actual API (skipped by default)."""

    @pytest.mark.skip(reason="Requires live Binance API connection")
    def test_live_data_fetch(self):
        """Test fetching live data from Binance."""
        from torchtrade.envs.binance.obs_class import BinanceObservationClass

        observer = BinanceObservationClass(
            symbol="BTCUSDT",
            intervals=["1m", "5m"],
            window_sizes=[10, 10],
        )

        observations = observer.get_observations()
        assert "1m_10" in observations
        assert "5m_10" in observations
