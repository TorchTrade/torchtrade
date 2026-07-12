"""Tests for BybitObservationClass with pybit.

Common observation-class behavior is inherited from BaseObservationClassTests.
Only Bybit-specific tests (symbol normalization, pybit envelope validation,
chronological parsing, utils) and stricter exact-shape assertions live here.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.envs.base_exchange_tests import BaseObservationClassTests


def _make_pybit_client():
    """Mock pybit HTTP client returning `limit` reverse-chronological klines."""
    client = MagicMock()

    def mock_get_kline(category, symbol, interval, limit=200):
        base_time = 1700000000000
        candles = [
            [str(base_time + i * 60000), "50000.0", "50100.0", "49900.0",
             "50050.0", "100.0", "5005000.0"]
            for i in range(limit - 1, -1, -1)  # reverse order
        ]
        return {"retCode": 0, "result": {"list": candles}}

    client.get_kline = MagicMock(side_effect=mock_get_kline)
    return client


class TestBybitObservationClass(BaseObservationClassTests):
    """Bybit observation class — common tests inherited from the base."""

    def create_observer(self, symbol, timeframes, window_sizes, **kwargs):
        from torchtrade.envs.live.bybit.observation import BybitObservationClass
        client = kwargs.pop("client", None) or _make_pybit_client()
        return BybitObservationClass(
            symbol=symbol, time_frames=timeframes, window_sizes=window_sizes,
            client=client, **kwargs,
        )

    def get_expected_symbol_format(self, symbol):
        from torchtrade.envs.live.bybit.utils import normalize_symbol
        return normalize_symbol(symbol)

    # --- Bybit-specific / stricter tests ---

    def test_symbol_and_unit_on_init(self):
        """Bybit normalizes symbol and preserves the timeframe unit (base checks neither)."""
        observer = self.create_observer(
            symbol="BTC/USDT", timeframes=TimeFrame(15, TimeFrameUnit.Minute), window_sizes=10)
        assert observer.symbol == "BTCUSDT"
        assert observer.time_frames[0].unit == TimeFrameUnit.Minute

    @pytest.mark.parametrize("symbol,expected", [
        ("BTCUSDT", "BTCUSDT"),
        ("BTC/USDT", "BTCUSDT"),
        ("BTC/USDT:USDT", "BTCUSDT"),
        (" btcusdt ", "BTCUSDT"),
    ])
    def test_symbol_normalization(self, symbol, expected):
        """Various symbol formats normalize to Bybit format."""
        observer = self.create_observer(
            symbol=symbol, timeframes=TimeFrame(1, TimeFrameUnit.Minute), window_sizes=10)
        assert observer.symbol == expected

    def test_invalid_interval_raises_error(self):
        """Unsupported timeframe raises ValueError."""
        from torchtrade.envs.live.bybit.observation import BybitObservationClass
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            BybitObservationClass(
                symbol="BTCUSDT", time_frames=TimeFrame(2, TimeFrameUnit.Minute),
                window_sizes=10, client=_make_pybit_client())

    def test_get_observations_exact_shapes(self):
        """Bybit observations have exact (window, 4) shapes (stricter than base >=4)."""
        observer = self.create_observer(
            symbol="BTCUSDT",
            timeframes=[TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute),
                        TimeFrame(1, TimeFrameUnit.Hour)],
            window_sizes=[10, 20, 15])
        obs = observer.get_observations()
        assert obs["1Minute_10"].shape == (10, 4)
        assert obs["5Minute_20"].shape == (20, 4)
        assert obs["1Hour_15"].shape == (15, 4)

    def test_custom_preprocessing_five_features(self):
        """Custom preprocessing adding OHLC-derived + custom feature -> 5 cols."""
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
            symbol="BTCUSDT", time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10, feature_preprocessing_fn=custom_preprocess,
            client=_make_pybit_client())
        assert observer.get_observations()["1Minute_10"].shape[1] == 5

    def test_api_call_parameters(self):
        """pybit get_kline is called with category=linear and the symbol."""
        client = _make_pybit_client()
        observer = self.create_observer(
            symbol="BTCUSDT", timeframes=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=10, client=client)
        observer.get_observations()
        client.get_kline.assert_called()
        call_kwargs = client.get_kline.call_args[1]
        assert call_kwargs["category"] == "linear"
        assert call_kwargs["symbol"] == "BTCUSDT"

    def test_empty_candles_raises_error(self):
        """Empty candle data raises RuntimeError."""
        from torchtrade.envs.live.bybit.observation import BybitObservationClass
        client = _make_pybit_client()
        client.get_kline = MagicMock(return_value={"retCode": 0, "result": {"list": []}})
        observer = BybitObservationClass(
            symbol="BTCUSDT", time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10, client=client)
        with pytest.raises(RuntimeError, match="No candle data"):
            observer.get_observations()

    def test_fetch_klines_validates_retcode(self):
        """_fetch_klines must raise RuntimeError on non-zero retCode."""
        from torchtrade.envs.live.bybit.observation import BybitObservationClass
        client = _make_pybit_client()
        client.get_kline = MagicMock(return_value={
            "retCode": 10001, "retMsg": "Invalid parameter", "result": {"list": []}})
        observer = BybitObservationClass(
            symbol="BTCUSDT", time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10, client=client)
        with pytest.raises(RuntimeError, match="retCode=10001"):
            observer._fetch_klines("BTCUSDT", "1", 200)

    def test_parse_klines_sorts_chronologically(self):
        """Reverse-ordered klines from Bybit must be sorted oldest-first."""
        from torchtrade.envs.live.bybit.observation import BybitObservationClass
        observer = BybitObservationClass(
            symbol="BTCUSDT", time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=5, client=_make_pybit_client())
        raw_klines = [
            [str(1700000000000 + i * 60000), "50000", "50100", "49900", "50050", "100", "5000000"]
            for i in [4, 3, 2, 1, 0]
        ]
        df = observer._parse_klines(raw_klines)
        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps), "Klines must be in chronological order"


class TestBybitUtils:
    """Tests for Bybit utility functions."""

    @pytest.mark.parametrize("tf,expected", [
        (TimeFrame(1, TimeFrameUnit.Minute), "1"),
        (TimeFrame(15, TimeFrameUnit.Minute), "15"),
        (TimeFrame(1, TimeFrameUnit.Hour), "60"),
        (TimeFrame(4, TimeFrameUnit.Hour), "240"),
        (TimeFrame(1, TimeFrameUnit.Day), "D"),
    ])
    def test_timeframe_to_bybit(self, tf, expected):
        """Test TimeFrame to Bybit interval conversion."""
        from torchtrade.envs.live.bybit.utils import timeframe_to_bybit
        assert timeframe_to_bybit(tf) == expected

    @pytest.mark.parametrize("interval,expected_value,expected_unit", [
        ("1", 1, TimeFrameUnit.Minute),
        ("15", 15, TimeFrameUnit.Minute),
        ("60", 1, TimeFrameUnit.Hour),
        ("240", 4, TimeFrameUnit.Hour),
        ("D", 1, TimeFrameUnit.Day),
    ])
    def test_bybit_to_timeframe(self, interval, expected_value, expected_unit):
        """Test Bybit interval string to TimeFrame conversion."""
        from torchtrade.envs.live.bybit.utils import bybit_to_timeframe
        tf = bybit_to_timeframe(interval)
        assert tf.value == expected_value
        assert tf.unit == expected_unit

    @pytest.mark.parametrize("bad_tf", [
        TimeFrame(2, TimeFrameUnit.Minute),
        TimeFrame(7, TimeFrameUnit.Hour),
    ])
    def test_timeframe_to_bybit_invalid(self, bad_tf):
        """Test that unsupported timeframes raise ValueError."""
        from torchtrade.envs.live.bybit.utils import timeframe_to_bybit
        with pytest.raises(ValueError):
            timeframe_to_bybit(bad_tf)

    def test_bybit_to_timeframe_invalid(self):
        """Test that invalid interval string raises ValueError."""
        from torchtrade.envs.live.bybit.utils import bybit_to_timeframe
        with pytest.raises(ValueError):
            bybit_to_timeframe("99")

    @pytest.mark.parametrize("bad_symbol", ["", "  ", "  :  "])
    def test_normalize_symbol_empty_raises(self, bad_symbol):
        """Empty or whitespace-only symbols must raise ValueError."""
        from torchtrade.envs.live.bybit.utils import normalize_symbol
        with pytest.raises(ValueError, match="empty"):
            normalize_symbol(bad_symbol)

    @pytest.mark.parametrize("tf", [
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(1, TimeFrameUnit.Hour),
        TimeFrame(1, TimeFrameUnit.Day),
    ])
    def test_roundtrip_conversion(self, tf):
        """Test that TimeFrame -> Bybit -> TimeFrame is identity."""
        from torchtrade.envs.live.bybit.utils import timeframe_to_bybit, bybit_to_timeframe
        result = bybit_to_timeframe(timeframe_to_bybit(tf))
        assert result.value == tf.value
        assert result.unit == tf.unit
