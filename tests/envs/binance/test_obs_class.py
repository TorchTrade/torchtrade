"""Tests for BinanceObservationClass.

Common observation-class behavior is inherited from BaseObservationClassTests.
Only Binance-specific tests (symbol normalization, exchange-specific kline columns,
default feature names) and stricter assertions live here.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.envs.base_exchange_tests import BaseObservationClassTests


def _make_binance_client():
    """Mock Binance client returning `limit` 12-column klines (chronological)."""
    client = MagicMock()

    def mock_get_klines(symbol, interval, limit=500):
        base_time = 1700000000000
        return [
            [base_time + i * 60000, "50000.0", "50100.0", "49900.0", "50050.0", "100.0",
             base_time + i * 60000 + 59999, "5000000.0", "100", "50.0", "2500000.0", "0"]
            for i in range(limit)
        ]

    client.get_klines = MagicMock(side_effect=mock_get_klines)
    return client


class TestBinanceObservationClass(BaseObservationClassTests):
    """Binance observation class — common tests inherited from the base."""

    def create_observer(self, symbol, timeframes, window_sizes, **kwargs):
        from torchtrade.envs.live.binance.observation import BinanceObservationClass
        client = kwargs.pop("client", None) or _make_binance_client()
        return BinanceObservationClass(
            symbol=symbol, time_frames=timeframes, window_sizes=window_sizes,
            client=client, **kwargs,
        )

    def get_expected_symbol_format(self, symbol):
        return symbol.replace("/", "")

    @pytest.fixture
    def mock_client(self):
        return _make_binance_client()

    @pytest.fixture
    def observer_single(self, mock_client):
        from torchtrade.envs.live.binance.observation import BinanceObservationClass
        return BinanceObservationClass(
            symbol="BTCUSDT", time_frames=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=10, client=mock_client)

    @pytest.fixture
    def observer_multi(self, mock_client):
        from torchtrade.envs.live.binance.observation import BinanceObservationClass
        return BinanceObservationClass(
            symbol="BTCUSDT",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute),
                         TimeFrame(1, TimeFrameUnit.Hour)],
            window_sizes=[10, 20, 15], client=mock_client)

    # --- Binance-specific / stricter tests ---

    def test_single_interval_initialization(self, observer_single):
        """Symbol + timeframe unit preserved (base checks neither)."""
        assert observer_single.symbol == "BTCUSDT"
        assert len(observer_single.time_frames) == 1
        assert observer_single.time_frames[0].value == 15
        assert observer_single.time_frames[0].unit == TimeFrameUnit.Minute
        assert observer_single.window_sizes == [10]

    def test_multi_interval_initialization(self, observer_multi):
        assert observer_multi.symbol == "BTCUSDT"
        assert len(observer_multi.time_frames) == 3
        assert observer_multi.window_sizes == [10, 20, 15]

    def test_symbol_normalization(self, mock_client):
        """Slash is stripped from the symbol."""
        from torchtrade.envs.live.binance.observation import BinanceObservationClass
        observer = BinanceObservationClass(
            symbol="BTC/USDT", time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10, client=mock_client)
        assert observer.symbol == "BTCUSDT"

    def test_get_keys_multi(self, observer_multi):
        assert observer_multi.get_keys() == ["1Minute_10", "5Minute_20", "1Hour_15"]

    def test_get_observations_single_dtype(self, observer_single):
        """Single-timeframe observation is exactly (10, 4) float32 (stricter than base)."""
        obs = observer_single.get_observations()
        assert obs["15Minute_10"].shape == (10, 4)
        assert obs["15Minute_10"].dtype == np.float32

    def test_get_observations_multi_exact_shapes(self, observer_multi):
        obs = observer_multi.get_observations()
        assert obs["1Minute_10"].shape == (10, 4)
        assert obs["5Minute_20"].shape == (20, 4)
        assert obs["1Hour_15"].shape == (15, 4)

    def test_get_observations_with_base_ohlc(self, observer_single):
        """base_features + base_timestamps present with exact shape (adds base_timestamps)."""
        obs = observer_single.get_observations(return_base_ohlc=True)
        assert "15Minute_10" in obs
        assert "base_features" in obs
        assert "base_timestamps" in obs
        assert obs["base_features"].shape == (10, 4)

    def test_custom_preprocessing_with_kline_extra_fields(self, mock_client):
        """Custom preprocessing can derive features from Binance-specific kline fields."""
        from torchtrade.envs.live.binance.observation import BinanceObservationClass

        def kline_preprocessing(df):
            df = df.copy()
            df["feature_taker_ratio"] = df["taker_buy_base"] / (df["volume"] + 1e-9)
            df["feature_quote_vol"] = df["quote_volume"].pct_change().fillna(0)
            df["feature_avg_trade_size"] = df["volume"] / df["trades"]
            df["feature_close"] = df["close"].pct_change().fillna(0)
            df.dropna(inplace=True)
            return df

        observer = BinanceObservationClass(
            symbol="BTCUSDT", time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10, client=mock_client, feature_preprocessing_fn=kline_preprocessing)

        features = observer.get_features()
        for f in ["feature_taker_ratio", "feature_quote_vol", "feature_avg_trade_size"]:
            assert f in features["observation_features"]
        assert observer.get_observations()["1Minute_10"].shape == (10, 4)

    def test_default_preprocessing_output(self, observer_single):
        """Default preprocessing produces the expected named features."""
        features = observer_single.get_features()
        for feat in ["feature_close", "feature_open", "feature_high", "feature_low"]:
            assert feat in features["observation_features"]


class TestBinanceObservationClassIntegration:
    """Integration tests that would require actual API (skipped by default)."""

    @pytest.mark.skip(reason="Requires live Binance API connection")
    def test_live_data_fetch(self):
        """Test fetching live data from Binance."""
        from torchtrade.envs.live.binance.observation import BinanceObservationClass
        observer = BinanceObservationClass(
            symbol="BTCUSDT",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute)],
            window_sizes=[10, 10])
        observations = observer.get_observations()
        assert "1Minute_10" in observations
        assert "5Minute_10" in observations
