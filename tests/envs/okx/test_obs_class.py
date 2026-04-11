"""Tests for OKXObservationClass."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


class TestOKXObservationClass:
    """Tests for OKXObservationClass."""

    @pytest.fixture
    def observer_single(self, mock_okx_market_client):
        """Create observer with single timeframe."""
        from torchtrade.envs.live.okx.observation import OKXObservationClass

        observer = OKXObservationClass(
            symbol="BTC-USDT-SWAP",
            time_frames=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_okx_market_client,
        )
        return observer

    @pytest.fixture
    def observer_multi(self, mock_okx_market_client):
        """Create observer with multiple timeframes."""
        from torchtrade.envs.live.okx.observation import OKXObservationClass

        observer = OKXObservationClass(
            symbol="BTC-USDT-SWAP",
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
                TimeFrame(1, TimeFrameUnit.Hour),
            ],
            window_sizes=[10, 20, 15],
            client=mock_okx_market_client,
        )
        return observer

    def test_single_interval_initialization(self, observer_single):
        """Test initialization with single timeframe."""
        assert observer_single.symbol == "BTC-USDT-SWAP"
        assert len(observer_single.time_frames) == 1
        assert observer_single.time_frames[0].value == 15
        assert observer_single.time_frames[0].unit == TimeFrameUnit.Minute
        assert observer_single.window_sizes == [10]

    def test_multi_interval_initialization(self, observer_multi):
        """Test initialization with multiple timeframes."""
        assert observer_multi.symbol == "BTC-USDT-SWAP"
        assert len(observer_multi.time_frames) == 3
        assert observer_multi.window_sizes == [10, 20, 15]

    @pytest.mark.parametrize("symbol,expected", [
        ("BTC-USDT-SWAP", "BTC-USDT-SWAP"),
        ("BTC-USDT", "BTC-USDT-SWAP"),
        ("BTCUSDT", "BTC-USDT-SWAP"),
        ("BTC/USDT", "BTC-USDT-SWAP"),
        ("BTC/USDT:USDT", "BTC-USDT-SWAP"),
        (" btcusdt ", "BTC-USDT-SWAP"),
    ])
    def test_symbol_normalization(self, mock_okx_market_client, symbol, expected):
        """Test that various symbol formats are normalized to OKX format."""
        from torchtrade.envs.live.okx.observation import OKXObservationClass

        observer = OKXObservationClass(
            symbol=symbol,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_okx_market_client,
        )
        assert observer.symbol == expected

    def test_invalid_interval_raises_error(self, mock_okx_market_client):
        """Test that invalid timeframe raises ValueError."""
        from torchtrade.envs.live.okx.observation import OKXObservationClass

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            OKXObservationClass(
                symbol="BTC-USDT-SWAP",
                time_frames=TimeFrame(2, TimeFrameUnit.Minute),  # 2 minutes not supported
                window_sizes=10,
                client=mock_okx_market_client,
            )

    def test_mismatched_lengths_raises_error(self, mock_okx_market_client):
        """Test that mismatched time_frames and window_sizes raises error."""
        from torchtrade.envs.live.okx.observation import OKXObservationClass

        with pytest.raises(ValueError, match="same length"):
            OKXObservationClass(
                symbol="BTC-USDT-SWAP",
                time_frames=[
                    TimeFrame(1, TimeFrameUnit.Minute),
                    TimeFrame(5, TimeFrameUnit.Minute),
                ],
                window_sizes=[10],  # Mismatched length
                client=mock_okx_market_client,
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

    def test_custom_preprocessing(self, mock_okx_market_client):
        """Test custom preprocessing function."""
        from torchtrade.envs.live.okx.observation import OKXObservationClass

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

        observer = OKXObservationClass(
            symbol="BTC-USDT-SWAP",
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            feature_preprocessing_fn=custom_preprocess,
            client=mock_okx_market_client,
        )

        observations = observer.get_observations()
        assert observations["1Minute_10"].shape[1] == 5  # 4 default + 1 custom

    def test_api_call_parameters(self, observer_single, mock_okx_market_client):
        """Test that OKX get_candlesticks is called with correct parameters."""
        observer_single.get_observations()

        mock_okx_market_client.get_candlesticks.assert_called()
        call_kwargs = mock_okx_market_client.get_candlesticks.call_args[1]
        assert call_kwargs["instId"] == "BTC-USDT-SWAP"

    def test_empty_candles_raises_error(self, mock_okx_market_client):
        """Test that empty candles raises RuntimeError."""
        from torchtrade.envs.live.okx.observation import OKXObservationClass

        mock_okx_market_client.get_candlesticks = MagicMock(return_value={
            "code": "0",
            "msg": "",
            "data": [],
        })

        observer = OKXObservationClass(
            symbol="BTC-USDT-SWAP",
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_okx_market_client,
        )

        with pytest.raises(RuntimeError, match="No candle data"):
            observer.get_observations()

    def test_default_preprocessing_output(self, observer_single):
        """Test that default preprocessing outputs valid data."""
        observations = observer_single.get_observations()
        data = observations["15Minute_10"]

        assert not np.isnan(data).any(), "Data contains NaN values"
        assert data.shape == (10, 4)

    def test_fetch_klines_validates_code(self, mock_okx_market_client):
        """_fetch_klines must raise RuntimeError on non-zero code."""
        from torchtrade.envs.live.okx.observation import OKXObservationClass

        mock_okx_market_client.get_candlesticks = MagicMock(return_value={
            "code": "51001",
            "msg": "Invalid parameter",
            "data": [],
        })

        observer = OKXObservationClass(
            symbol="BTC-USDT-SWAP",
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_okx_market_client,
        )

        with pytest.raises(RuntimeError, match="code=51001"):
            observer._fetch_klines("BTC-USDT-SWAP", "1m", 200)

    def test_parse_klines_sorts_chronologically(self, mock_okx_market_client):
        """Reverse-ordered klines from OKX must be sorted oldest-first."""
        from torchtrade.envs.live.okx.observation import OKXObservationClass

        observer = OKXObservationClass(
            symbol="BTC-USDT-SWAP",
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=5,
            client=mock_okx_market_client,
        )

        # Feed reverse-chronological data (newest first, like OKX returns)
        raw_klines = [
            [str(1700000000000 + i * 60000), "50000", "50100", "49900", "50050", "100", "5000000", "5000000", "1"]
            for i in [4, 3, 2, 1, 0]
        ]
        df = observer._parse_klines(raw_klines)
        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps), "Klines must be in chronological order"


class TestOKXUtils:
    """Tests for OKX utility functions."""

    @pytest.mark.parametrize("tf,expected", [
        (TimeFrame(1, TimeFrameUnit.Minute), "1m"),
        (TimeFrame(15, TimeFrameUnit.Minute), "15m"),
        (TimeFrame(1, TimeFrameUnit.Hour), "1H"),
        (TimeFrame(4, TimeFrameUnit.Hour), "4H"),
        (TimeFrame(1, TimeFrameUnit.Day), "1D"),
    ])
    def test_timeframe_to_okx(self, tf, expected):
        """Test TimeFrame to OKX interval conversion."""
        from torchtrade.envs.live.okx.utils import timeframe_to_okx
        assert timeframe_to_okx(tf) == expected

    @pytest.mark.parametrize("interval,expected_value,expected_unit", [
        ("1m", 1, TimeFrameUnit.Minute),
        ("15m", 15, TimeFrameUnit.Minute),
        ("1H", 1, TimeFrameUnit.Hour),
        ("4H", 4, TimeFrameUnit.Hour),
        ("1D", 1, TimeFrameUnit.Day),
    ])
    def test_okx_to_timeframe(self, interval, expected_value, expected_unit):
        """Test OKX interval string to TimeFrame conversion."""
        from torchtrade.envs.live.okx.utils import okx_to_timeframe
        tf = okx_to_timeframe(interval)
        assert tf.value == expected_value
        assert tf.unit == expected_unit

    @pytest.mark.parametrize("bad_tf", [
        TimeFrame(2, TimeFrameUnit.Minute),
        TimeFrame(7, TimeFrameUnit.Hour),
    ])
    def test_timeframe_to_okx_invalid(self, bad_tf):
        """Test that unsupported timeframes raise ValueError."""
        from torchtrade.envs.live.okx.utils import timeframe_to_okx
        with pytest.raises(ValueError):
            timeframe_to_okx(bad_tf)

    def test_okx_to_timeframe_invalid(self):
        """Test that invalid interval string raises ValueError."""
        from torchtrade.envs.live.okx.utils import okx_to_timeframe
        with pytest.raises(ValueError):
            okx_to_timeframe("99m")

    @pytest.mark.parametrize("symbol,expected", [
        ("BTC-USDT-SWAP", "BTC-USDT-SWAP"),
        ("BTC-USDT", "BTC-USDT-SWAP"),
        ("BTCUSDT", "BTC-USDT-SWAP"),
        ("BTC/USDT", "BTC-USDT-SWAP"),
        ("BTC/USDT:USDT", "BTC-USDT-SWAP"),
        (" btcusdt ", "BTC-USDT-SWAP"),
        ("ETH-USDT", "ETH-USDT-SWAP"),
        ("ETHUSDT", "ETH-USDT-SWAP"),
    ])
    def test_normalize_symbol(self, symbol, expected):
        """Test symbol normalization to OKX swap format."""
        from torchtrade.envs.live.okx.utils import normalize_symbol
        assert normalize_symbol(symbol) == expected

    @pytest.mark.parametrize("bad_symbol", ["", "  "])
    def test_normalize_symbol_empty_raises(self, bad_symbol):
        """Empty or whitespace-only symbols must raise ValueError."""
        from torchtrade.envs.live.okx.utils import normalize_symbol
        with pytest.raises(ValueError):
            normalize_symbol(bad_symbol)

    @pytest.mark.parametrize("tf", [
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(1, TimeFrameUnit.Hour),
        TimeFrame(1, TimeFrameUnit.Day),
    ])
    def test_roundtrip_conversion(self, tf):
        """Test that TimeFrame -> OKX -> TimeFrame is identity."""
        from torchtrade.envs.live.okx.utils import timeframe_to_okx, okx_to_timeframe
        result = okx_to_timeframe(timeframe_to_okx(tf))
        assert result.value == tf.value
        assert result.unit == tf.unit
