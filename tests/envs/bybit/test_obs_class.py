"""Tests for BybitObservationClass with pybit."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


class TestBybitObservationClass:
    """Tests for BybitObservationClass using pybit."""

    @pytest.fixture
    def observer_single(self, mock_pybit_client):
        """Create observer with single timeframe."""
        from torchtrade.envs.live.bybit.observation import BybitObservationClass

        observer = BybitObservationClass(
            symbol="BTCUSDT",
            time_frames=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_pybit_client,
        )
        return observer

    @pytest.fixture
    def observer_multi(self, mock_pybit_client):
        """Create observer with multiple timeframes."""
        from torchtrade.envs.live.bybit.observation import BybitObservationClass

        observer = BybitObservationClass(
            symbol="BTCUSDT",
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
                TimeFrame(1, TimeFrameUnit.Hour),
            ],
            window_sizes=[10, 20, 15],
            client=mock_pybit_client,
        )
        return observer

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
        assert observer_multi.window_sizes == [10, 20, 15]

    @pytest.mark.parametrize("symbol,expected", [
        ("BTCUSDT", "BTCUSDT"),
        ("BTC/USDT", "BTCUSDT"),
        ("BTC/USDT:USDT", "BTCUSDT"),
    ])
    def test_symbol_normalization(self, mock_pybit_client, symbol, expected):
        """Test that various symbol formats are normalized to Bybit format."""
        from torchtrade.envs.live.bybit.observation import BybitObservationClass

        observer = BybitObservationClass(
            symbol=symbol,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_pybit_client,
        )
        assert observer.symbol == expected

    def test_invalid_interval_raises_error(self, mock_pybit_client):
        """Test that invalid timeframe raises ValueError."""
        from torchtrade.envs.live.bybit.observation import BybitObservationClass

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            BybitObservationClass(
                symbol="BTCUSDT",
                time_frames=TimeFrame(2, TimeFrameUnit.Minute),  # 2 minutes not supported
                window_sizes=10,
                client=mock_pybit_client,
            )

    def test_mismatched_lengths_raises_error(self, mock_pybit_client):
        """Test that mismatched time_frames and window_sizes raises error."""
        from torchtrade.envs.live.bybit.observation import BybitObservationClass

        with pytest.raises(ValueError, match="same length"):
            BybitObservationClass(
                symbol="BTCUSDT",
                time_frames=[
                    TimeFrame(1, TimeFrameUnit.Minute),
                    TimeFrame(5, TimeFrameUnit.Minute),
                ],
                window_sizes=[10],  # Mismatched length
                client=mock_pybit_client,
            )

    def test_get_keys_single(self, observer_single):
        """Test get_keys with single timeframe."""
        keys = observer_single.get_keys()
        assert keys == ["15Minute_10"]

    def test_get_keys_multi(self, observer_multi):
        """Test get_keys with multiple timeframes."""
        keys = observer_multi.get_keys()
        assert keys == ["1Minute_10", "5Minute_20", "1Hour_15"]

    def test_get_observations_single(self, observer_single):
        """Test get_observations with single timeframe returns correct shape."""
        observations = observer_single.get_observations()

        assert "15Minute_10" in observations
        assert observations["15Minute_10"].shape == (10, 4)  # window_size x num_features

    def test_get_observations_multi(self, observer_multi):
        """Test get_observations with multiple timeframes returns correct shapes."""
        observations = observer_multi.get_observations()

        assert observations["1Minute_10"].shape == (10, 4)
        assert observations["5Minute_20"].shape == (20, 4)
        assert observations["1Hour_15"].shape == (15, 4)

    def test_get_observations_with_base_ohlc(self, observer_single):
        """Test get_observations with return_base_ohlc=True."""
        observations = observer_single.get_observations(return_base_ohlc=True)

        assert "15Minute_10" in observations
        assert "base_features" in observations
        assert observations["base_features"].shape[1] == 4  # OHLC

    def test_custom_preprocessing(self, mock_pybit_client):
        """Test custom preprocessing function."""
        from torchtrade.envs.live.bybit.observation import BybitObservationClass

        def custom_preprocess(df):
            df = df.copy()
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            df["feature_close"] = df["close"].pct_change().fillna(0)
            df["feature_open"] = df["open"] / df["close"]
            df["feature_high"] = df["high"] / df["close"]
            df["feature_low"] = df["low"] / df["close"]
            df["feature_custom"] = df["close"] * 2
            df.dropna(inplace=True)
            return df

        observer = BybitObservationClass(
            symbol="BTCUSDT",
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            feature_preprocessing_fn=custom_preprocess,
            client=mock_pybit_client,
        )

        observations = observer.get_observations()
        assert observations["1Minute_10"].shape[1] == 5  # 4 default + 1 custom

    def test_api_call_parameters(self, observer_single, mock_pybit_client):
        """Test that pybit get_kline is called with correct parameters."""
        observer_single.get_observations()

        mock_pybit_client.get_kline.assert_called()
        call_kwargs = mock_pybit_client.get_kline.call_args[1]
        assert call_kwargs["category"] == "linear"
        assert call_kwargs["symbol"] == "BTCUSDT"

    def test_empty_candles_raises_error(self, mock_pybit_client):
        """Test that empty candles raises RuntimeError."""
        from torchtrade.envs.live.bybit.observation import BybitObservationClass

        mock_pybit_client.get_kline = MagicMock(return_value={
            "retCode": 0,
            "result": {"list": []},
        })

        observer = BybitObservationClass(
            symbol="BTCUSDT",
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_pybit_client,
        )

        with pytest.raises(RuntimeError, match="No candle data"):
            observer.get_observations()

    def test_default_preprocessing_output(self, observer_single):
        """Test that default preprocessing outputs valid data."""
        observations = observer_single.get_observations()
        data = observations["15Minute_10"]

        assert not np.isnan(data).any(), "Data contains NaN values"
        assert data.shape == (10, 4)
