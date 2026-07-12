"""Tests for BitgetObservationClass with CCXT.

Common observation-class behavior is inherited from BaseObservationClassTests.
Only Bitget-specific tests (symbol normalization, product type, CCXT call/empty
handling) and stricter assertions live here.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from torchtrade.envs.live.bitget.observation import BitgetObservationClass
from tests.envs.base_exchange_tests import BaseObservationClassTests


def _make_bitget_client():
    """Mock CCXT bitget client returning `limit` 6-column OHLCV candles."""
    client = MagicMock()

    def mock_fetch_ohlcv(symbol, timeframe, limit=200):
        base_time = 1700000000000
        return [
            [base_time + i * 60000, 50000.0, 50100.0, 49900.0, 50050.0, 100.0]
            for i in range(limit)
        ]

    client.fetch_ohlcv = MagicMock(side_effect=mock_fetch_ohlcv)
    client.load_markets = MagicMock(return_value={})
    return client


class TestBitgetObservationClass(BaseObservationClassTests):
    """Bitget observation class — common tests inherited from the base."""

    def create_observer(self, symbol, timeframes, window_sizes, **kwargs):
        # Injecting client= skips BitgetObservationClass's ccxt.bitget() creation.
        client = kwargs.pop("client", None) or _make_bitget_client()
        return BitgetObservationClass(
            symbol=symbol, time_frames=timeframes, window_sizes=window_sizes,
            client=client, **kwargs,
        )

    def get_expected_symbol_format(self, symbol):
        from torchtrade.envs.live.bitget.utils import normalize_symbol
        return normalize_symbol(symbol)

    @pytest.fixture
    def observer_single(self, mock_ccxt_client):
        return BitgetObservationClass(
            symbol="BTC/USDT:USDT", time_frames=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=10, client=mock_ccxt_client)

    @pytest.fixture
    def observer_multi(self, mock_ccxt_client):
        return BitgetObservationClass(
            symbol="BTC/USDT:USDT",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute),
                         TimeFrame(1, TimeFrameUnit.Hour)],
            window_sizes=[10, 20, 15], client=mock_ccxt_client)

    # --- Bitget-specific / stricter tests ---

    def test_single_interval_initialization(self, observer_single):
        assert "USDT" in observer_single.symbol
        assert len(observer_single.time_frames) == 1
        assert observer_single.time_frames[0].value == 15
        assert observer_single.time_frames[0].unit == TimeFrameUnit.Minute
        assert observer_single.window_sizes == [10]

    def test_multi_interval_initialization(self, observer_multi):
        assert "USDT" in observer_multi.symbol
        assert len(observer_multi.time_frames) == 3
        assert observer_multi.window_sizes == [10, 20, 15]

    def test_symbol_normalization(self, mock_ccxt_client):
        """Various symbol formats normalize to CCXT format."""
        observer = BitgetObservationClass(
            symbol="BTCUSDT:USDT", time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10, client=mock_ccxt_client)
        assert "USDT" in observer.symbol

    def test_invalid_interval_raises_error(self, mock_ccxt_client):
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            BitgetObservationClass(
                symbol="BTC/USDT:USDT", time_frames=TimeFrame(2, TimeFrameUnit.Minute),
                window_sizes=10, client=mock_ccxt_client)

    def test_product_type_demo(self, mock_ccxt_client):
        """demo=True sets the product type."""
        observer = BitgetObservationClass(
            symbol="BTC/USDT:USDT", time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10, product_type="USDT-FUTURES", demo=True, client=mock_ccxt_client)
        assert observer.product_type == "USDT-FUTURES"

    def test_get_keys_multi(self, observer_multi):
        assert observer_multi.get_keys() == ["1Minute_10", "5Minute_20", "1Hour_15"]

    def test_get_observations_exact_shapes(self, observer_single, observer_multi):
        """Bitget observations have exact (window, 4) shapes (stricter than base)."""
        assert observer_single.get_observations()["15Minute_10"].shape == (10, 4)
        obs = observer_multi.get_observations()
        assert obs["1Minute_10"].shape == (10, 4)
        assert obs["5Minute_20"].shape == (20, 4)
        assert obs["1Hour_15"].shape == (15, 4)

    def test_custom_preprocessing_five_features(self, mock_ccxt_client):
        """Custom preprocessing adding OHLC-derived + custom feature -> 5 cols."""
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

        observer = BitgetObservationClass(
            symbol="BTC/USDT:USDT", time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10, feature_preprocessing_fn=custom_preprocess, client=mock_ccxt_client)
        assert observer.get_observations()["1Minute_10"].shape[1] == 5

    def test_get_features_names(self, observer_single):
        """get_features returns the 4 named OHLC features."""
        features = observer_single.get_features()
        obs_features = features["observation_features"]
        assert len(obs_features) == 4
        assert all("feature" in f for f in obs_features)
        for name in ("open", "high", "low", "close"):
            assert any(name in f for f in obs_features)

    def test_default_preprocessing_output(self, observer_single):
        """Default preprocessing produces valid no-NaN (10, 4) data."""
        data = observer_single.get_observations()["15Minute_10"]
        assert not np.isnan(data).any()
        assert data.shape == (10, 4)

    def test_api_call_parameters(self, observer_single, mock_ccxt_client):
        """CCXT fetch_ohlcv is called."""
        observer_single.get_observations()
        mock_ccxt_client.fetch_ohlcv.assert_called()

    def test_empty_candles_raises_error(self, mock_ccxt_client):
        """Empty candle data raises RuntimeError."""
        mock_ccxt_client.fetch_ohlcv = MagicMock(return_value=[])
        observer = BitgetObservationClass(
            symbol="BTC/USDT:USDT", time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10, client=mock_ccxt_client)
        with pytest.raises(RuntimeError, match="No candle data"):
            observer.get_observations()


class TestBitgetObservationClassIntegration:
    """Integration tests that would require actual API (skipped by default)."""

    @pytest.mark.skip(reason="Requires live Bitget API connection")
    def test_live_data_fetch(self):
        """Test fetching real data from Bitget."""
        observer = BitgetObservationClass(
            symbol="BTC/USDT:USDT",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute)],
            window_sizes=[10, 20], demo=True)
        observations = observer.get_observations()
        assert "1Minute_10" in observations
        assert "5Minute_20" in observations
        assert observations["1Minute_10"].shape == (10, 4)
        assert not np.isnan(observations["1Minute_10"]).any()
