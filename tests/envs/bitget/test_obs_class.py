"""Tests for BitgetObservationClass."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from torchtrade.envs.timeframe import TimeFrame, TimeFrameUnit


class TestBitgetObservationClass:
    """Tests for BitgetObservationClass."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Bitget client."""
        client = MagicMock()

        # Mock candles data (7 columns for Bitget)
        def mock_get_candles(symbol, granularity, productType, limit=200):
            candles = []
            base_time = 1700000000000  # Base timestamp in ms
            for i in range(limit):
                candles.append([
                    str(base_time + i * 60000),  # timestamp (string)
                    "50000.0",  # open
                    "50100.0",  # high
                    "49900.0",  # low
                    "50050.0",  # close
                    "100.0",    # volume
                    "5000000.0",  # quote_volume
                ])
            return candles

        client.mix_get_candles = MagicMock(side_effect=mock_get_candles)
        return client

    @pytest.fixture
    def observer_single(self, mock_client):
        """Create observer with single timeframe."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        return BitgetObservationClass(
            symbol="BTCUSDT",
            time_frames=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

    @pytest.fixture
    def observer_multi(self, mock_client):
        """Create observer with multiple timeframes."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        return BitgetObservationClass(
            symbol="BTCUSDT",
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
                TimeFrame(1, TimeFrameUnit.Hour),
            ],
            window_sizes=[10, 20, 15],
            client=mock_client,
        )

    def test_single_interval_initialization(self, observer_single):
        """Test initialization with single timeframe."""
        assert observer_single.symbol == "BTCUSDT"
        assert len(observer_single.time_frames) == 1
        assert observer_single.time_frames[0].value == 15
        assert observer_single.time_frames[0].unit == TimeFrameUnit.Minute
        assert observer_single.window_sizes == [10]

    def test_multi_interval_initialization(self, observer_multi):
        """Test initialization with multiple timeframes."""
        assert observer_multi.symbol == "BTCUSDT"
        assert len(observer_multi.time_frames) == 3
        assert observer_multi.time_frames[0].value == 1
        assert observer_multi.time_frames[1].value == 5
        assert observer_multi.time_frames[2].value == 1
        assert observer_multi.window_sizes == [10, 20, 15]

    def test_symbol_normalization(self, mock_client):
        """Test that symbol with slash is normalized."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        observer = BitgetObservationClass(
            symbol="BTC/USDT",
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )
        assert observer.symbol == "BTCUSDT"

    def test_invalid_interval_raises_error(self, mock_client):
        """Test that invalid timeframe raises ValueError."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            BitgetObservationClass(
                symbol="BTCUSDT",
                time_frames=TimeFrame(2, TimeFrameUnit.Minute),  # 2 minutes not supported by Bitget
                window_sizes=10,
                client=mock_client,
            )

    def test_mismatched_lengths_raises_error(self, mock_client):
        """Test that mismatched time_frames and window_sizes raises error."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        with pytest.raises(ValueError, match="same length"):
            BitgetObservationClass(
                symbol="BTCUSDT",
                time_frames=[
                    TimeFrame(1, TimeFrameUnit.Minute),
                    TimeFrame(5, TimeFrameUnit.Minute),
                ],
                window_sizes=[10],  # Mismatched length
                client=mock_client,
            )

    def test_product_type_demo(self, mock_client):
        """Test that demo=True forces SUMCBL product type."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        observer = BitgetObservationClass(
            symbol="BTCUSDT",
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            product_type="UMCBL",  # Try production
            demo=True,  # But demo is True
            client=mock_client,
        )
        assert observer.product_type == "SUMCBL"  # Should be forced to testnet

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
        assert observations["15Minute_10"].dtype == np.float32

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
        """Test get_observations with base OHLC data."""
        observations = observer_single.get_observations(return_base_ohlc=True)

        assert "15Minute_10" in observations
        assert "base_features" in observations
        assert "base_timestamps" in observations

        assert observations["base_features"].shape == (10, 4)

    def test_custom_preprocessing(self, mock_client):
        """Test custom preprocessing function."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        def custom_preprocessing(df):
            df = df.copy()
            df["feature_custom1"] = df["close"].pct_change().fillna(0)
            df["feature_custom2"] = df["volume"].pct_change().fillna(0)
            df.dropna(inplace=True)
            return df

        observer = BitgetObservationClass(
            symbol="BTCUSDT",
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
            feature_preprocessing_fn=custom_preprocessing,
        )

        observations = observer.get_observations()
        assert observations["1Minute_10"].shape == (10, 2)  # 2 custom features

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

    def test_api_call_parameters(self, observer_single, mock_client):
        """Test that API is called with correct parameters."""
        observer_single.get_observations()

        # Check that mix_get_candles was called
        mock_client.mix_get_candles.assert_called()
        call_kwargs = mock_client.mix_get_candles.call_args[1]

        assert call_kwargs["symbol"] == "BTCUSDT"
        assert call_kwargs["granularity"] == "15m"
        assert call_kwargs["productType"] == "SUMCBL"

    def test_empty_candles_raises_error(self, mock_client):
        """Test that empty candles data raises RuntimeError."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        # Mock empty candles
        mock_client.mix_get_candles = MagicMock(return_value=[])

        observer = BitgetObservationClass(
            symbol="BTCUSDT",
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        with pytest.raises(RuntimeError, match="Failed to fetch candles"):
            observer.get_observations()


class TestBitgetObservationClassIntegration:
    """Integration tests that would require actual API (skipped by default)."""

    @pytest.mark.skip(reason="Requires live Bitget API connection")
    def test_live_data_fetch(self):
        """Test fetching live data from Bitget."""
        from torchtrade.envs.bitget.obs_class import BitgetObservationClass

        observer = BitgetObservationClass(
            symbol="BTCUSDT",
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 10],
        )

        observations = observer.get_observations()
        assert "1Minute_10" in observations
        assert "5Minute_10" in observations
