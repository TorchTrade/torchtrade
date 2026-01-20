"""Tests for BitgetObservationClass with CCXT."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from torchtrade.envs.timeframe import TimeFrame, TimeFrameUnit


class TestBitgetObservationClass:
    """Tests for BitgetObservationClass using CCXT."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock CCXT Bitget client."""
        client = MagicMock()

        # Mock OHLCV data (CCXT fetch_ohlcv returns list of [timestamp, open, high, low, close, volume])
        def mock_fetch_ohlcv(symbol, timeframe, limit=200):
            candles = []
            base_time = 1700000000000  # Base timestamp in ms
            for i in range(limit):
                candles.append([
                    base_time + i * 60000,  # timestamp (int)
                    50000.0,  # open
                    50100.0,  # high
                    49900.0,  # low
                    50050.0,  # close
                    100.0,    # volume
                ])
            return candles

        client.fetch_ohlcv = MagicMock(side_effect=mock_fetch_ohlcv)
        return client

    @pytest.fixture
    def observer_single(self, mock_client):
        """Create observer with single timeframe."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        with patch('torchtrade.envs.bitget.obs_class.ccxt.bitget', return_value=mock_client):
            observer = BitgetObservationClass(
                symbol="BTC/USDT:USDT",
                time_frames=TimeFrame(15, TimeFrameUnit.Minute),
                window_sizes=10,
            )
            observer.client = mock_client
            return observer

    @pytest.fixture
    def observer_multi(self, mock_client):
        """Create observer with multiple timeframes."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        with patch('torchtrade.envs.bitget.obs_class.ccxt.bitget', return_value=mock_client):
            observer = BitgetObservationClass(
                symbol="BTC/USDT:USDT",
                time_frames=[
                    TimeFrame(1, TimeFrameUnit.Minute),
                    TimeFrame(5, TimeFrameUnit.Minute),
                    TimeFrame(1, TimeFrameUnit.Hour),
                ],
                window_sizes=[10, 20, 15],
            )
            observer.client = mock_client
            return observer

    def test_single_interval_initialization(self, observer_single):
        """Test initialization with single timeframe."""
        # Symbol gets normalized
        assert "USDT" in observer_single.symbol
        assert len(observer_single.time_frames) == 1
        assert observer_single.time_frames[0].value == 15
        assert observer_single.time_frames[0].unit == TimeFrameUnit.Minute
        assert observer_single.window_sizes == [10]

    def test_multi_interval_initialization(self, observer_multi):
        """Test initialization with multiple timeframes."""
        assert "USDT" in observer_multi.symbol
        assert len(observer_multi.time_frames) == 3
        assert observer_multi.time_frames[0].value == 1
        assert observer_multi.time_frames[1].value == 5
        assert observer_multi.time_frames[2].value == 1
        assert observer_multi.window_sizes == [10, 20, 15]

    def test_symbol_normalization(self, mock_client):
        """Test that various symbol formats are handled."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        with patch('torchtrade.envs.bitget.obs_class.ccxt.bitget', return_value=mock_client):
            observer = BitgetObservationClass(
                symbol="BTCUSDT:USDT",
                time_frames=TimeFrame(1, TimeFrameUnit.Minute),
                window_sizes=10,
            )
            observer.client = mock_client
            # Symbol gets normalized to CCXT format
            assert "USDT" in observer.symbol

    def test_invalid_interval_raises_error(self, mock_client):
        """Test that invalid timeframe raises ValueError."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            with patch('torchtrade.envs.bitget.obs_class.ccxt.bitget', return_value=mock_client):
                BitgetObservationClass(
                    symbol="BTC/USDT:USDT",
                    time_frames=TimeFrame(2, TimeFrameUnit.Minute),  # 2 minutes not supported
                    window_sizes=10,
                )

    def test_mismatched_lengths_raises_error(self, mock_client):
        """Test that mismatched time_frames and window_sizes raises error."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        with pytest.raises(ValueError, match="same length"):
            with patch('torchtrade.envs.bitget.obs_class.ccxt.bitget', return_value=mock_client):
                BitgetObservationClass(
                    symbol="BTC/USDT:USDT",
                    time_frames=[
                        TimeFrame(1, TimeFrameUnit.Minute),
                        TimeFrame(5, TimeFrameUnit.Minute),
                    ],
                    window_sizes=[10],  # Mismatched length
                )

    def test_product_type_demo(self, mock_client):
        """Test that demo=True sets product type correctly."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        with patch('torchtrade.envs.bitget.obs_class.ccxt.bitget', return_value=mock_client):
            observer = BitgetObservationClass(
                symbol="BTC/USDT:USDT",
                time_frames=TimeFrame(1, TimeFrameUnit.Minute),
                window_sizes=10,
                product_type="USDT-FUTURES",
                demo=True,
            )
            observer.client = mock_client
            assert observer.product_type == "USDT-FUTURES"

    def test_get_keys_single(self, observer_single):
        """Test get_keys with single timeframe."""
        keys = observer_single.get_keys()
        assert keys == ["15Minute_10"]

    def test_get_keys_multi(self, observer_multi):
        """Test get_keys with multiple timeframes."""
        keys = observer_multi.get_keys()
        assert keys == ["1Minute_10", "5Minute_20", "1Hour_15"]

    def test_get_observations_single(self, observer_single):
        """Test get_observations with single timeframe."""
        observations = observer_single.get_observations()

        assert "15Minute_10" in observations
        assert observations["15Minute_10"].shape == (10, 4)  # window_size x num_features

    def test_get_observations_multi(self, observer_multi):
        """Test get_observations with multiple timeframes."""
        observations = observer_multi.get_observations()

        assert "1Minute_10" in observations
        assert "5Minute_20" in observations
        assert "1Hour_15" in observations

        assert observations["1Minute_10"].shape == (10, 4)
        assert observations["5Minute_20"].shape == (20, 4)
        assert observations["1Hour_15"].shape == (15, 4)

    def test_get_observations_with_base_ohlc(self, observer_single):
        """Test get_observations with return_base_ohlc=True."""
        observations = observer_single.get_observations(return_base_ohlc=True)

        assert "15Minute_10" in observations
        assert "base_features" in observations
        # Base features should have all history
        assert observations["base_features"].shape[1] == 4  # OHLC

    def test_custom_preprocessing(self, mock_client):
        """Test custom preprocessing function."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        def custom_preprocess(df):
            """Add a custom feature (matching naming convention: feature_X)."""
            df = df.copy()
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)

            # Create default features
            df["feature_close"] = df["close"].pct_change().fillna(0)
            df["feature_open"] = df["open"] / df["close"]
            df["feature_high"] = df["high"] / df["close"]
            df["feature_low"] = df["low"] / df["close"]

            # Add custom feature
            df["feature_custom"] = df["close"] * 2

            df.dropna(inplace=True)
            return df

        with patch('torchtrade.envs.bitget.obs_class.ccxt.bitget', return_value=mock_client):
            observer = BitgetObservationClass(
                symbol="BTC/USDT:USDT",
                time_frames=TimeFrame(1, TimeFrameUnit.Minute),
                window_sizes=10,
                feature_preprocessing_fn=custom_preprocess,
            )
            observer.client = mock_client

            observations = observer.get_observations()
            # Should have 5 features (OHLC + custom)
            assert observations["1Minute_10"].shape[1] == 5

    def test_get_features(self, observer_single):
        """Test get_features returns feature names."""
        features = observer_single.get_features()

        # get_features() now returns a dict with 'observation_features' and 'original_features'
        assert isinstance(features, dict)
        assert "observation_features" in features
        assert "original_features" in features

        # Check that observation features contain the expected OHLC features
        obs_features = features["observation_features"]
        assert len(obs_features) == 4
        assert all("feature" in f for f in obs_features)
        assert any("open" in f for f in obs_features)
        assert any("high" in f for f in obs_features)
        assert any("low" in f for f in obs_features)
        assert any("close" in f for f in obs_features)

    def test_default_preprocessing_output(self, observer_single):
        """Test that default preprocessing outputs correct features."""
        observations = observer_single.get_observations()
        data = observations["15Minute_10"]

        # Check that data is valid
        assert not np.isnan(data).any(), "Data contains NaN values"
        assert data.shape == (10, 4)

    def test_api_call_parameters(self, observer_single, mock_client):
        """Test that CCXT fetch_ohlcv is called with correct parameters."""
        observer_single.get_observations()

        # Check that fetch_ohlcv was called
        mock_client.fetch_ohlcv.assert_called()

        # Get the call arguments
        call_args = mock_client.fetch_ohlcv.call_args
        assert call_args is not None

    def test_empty_candles_raises_error(self, mock_client):
        """Test that empty candles raises RuntimeError."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        # Mock fetch_ohlcv to return empty list
        mock_client.fetch_ohlcv = MagicMock(return_value=[])

        with patch('torchtrade.envs.bitget.obs_class.ccxt.bitget', return_value=mock_client):
            observer = BitgetObservationClass(
                symbol="BTC/USDT:USDT",
                time_frames=TimeFrame(1, TimeFrameUnit.Minute),
                window_sizes=10,
            )
            observer.client = mock_client

            with pytest.raises(RuntimeError, match="No candle data"):
                observer.get_observations()


class TestBitgetObservationClassIntegration:
    """Integration tests that would require actual API (skipped by default)."""

    @pytest.mark.skip(reason="Requires live Bitget API connection")
    def test_live_data_fetch(self):
        """Test fetching real data from Bitget."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        observer = BitgetObservationClass(
            symbol="BTC/USDT:USDT",
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 20],
            demo=True,
        )

        observations = observer.get_observations()

        assert "1Minute_10" in observations
        assert "5Minute_20" in observations
        assert observations["1Minute_10"].shape == (10, 4)
        assert not np.isnan(observations["1Minute_10"]).any()
