"""Tests for BinanceObservationClass.

Inherits common tests from BaseObservationClassTests.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from torchtrade.envs.live.binance.observation import BinanceObservationClass
from tests.envs.base_exchange_tests import BaseObservationClassTests


class TestBinanceObservationClass(BaseObservationClassTests):
    """Tests for BinanceObservationClass - inherits common tests from base."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Binance client."""
        client = MagicMock()

        def mock_get_klines(symbol, interval, limit=500):
            klines = []
            base_time = 1700000000000
            for i in range(limit):
                klines.append([
                    base_time + i * 60000, "50000.0", "50100.0", "49900.0",
                    "50050.0", "100.0", base_time + i * 60000 + 59999,
                    "5000000.0", 100, "50.0", "2500000.0", "0",
                ])
            return klines

        client.get_klines = MagicMock(side_effect=mock_get_klines)
        return client

    def create_observer(self, symbol, timeframes, window_sizes, **kwargs):
        """Create a BinanceObservationClass instance."""
        client = kwargs.get('client')
        if client is None:
            client = MagicMock()
            def mock_get_klines(symbol, interval, limit=500):
                klines = []
                base_time = 1700000000000
                for i in range(limit):
                    klines.append([
                        base_time + i * 60000, "50000.0", "50100.0", "49900.0",
                        "50050.0", "100.0", base_time + i * 60000 + 59999,
                        "5000000.0", 100, "50.0", "2500000.0", "0",
                    ])
                return klines
            client.get_klines = MagicMock(side_effect=mock_get_klines)

        return BinanceObservationClass(
            symbol=symbol,
            time_frames=timeframes,
            window_sizes=window_sizes,
            client=client,
            feature_preprocessing_fn=kwargs.get('feature_preprocessing_fn'),
        )

    def get_expected_symbol_format(self, symbol):
        """Binance removes slashes from symbols."""
        return symbol.replace('/', '')

    # Binance-specific tests

    def test_symbol_normalization(self, mock_client):
        """Test that symbol with slash is normalized."""
        observer = BinanceObservationClass(
            symbol="BTC/USDT",
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )
        assert observer.symbol == "BTCUSDT"

    def test_get_keys_format(self, mock_client):
        """Test that get_keys returns Binance-specific format."""
        observer = BinanceObservationClass(
            symbol="BTCUSDT",
            time_frames=TimeFrame(15, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        keys = observer.get_keys()
        assert keys == ["15Minute_10"]

    def test_default_preprocessing_output(self, mock_client):
        """Test that default preprocessing produces expected features."""
        observer = BinanceObservationClass(
            symbol="BTCUSDT",
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            client=mock_client,
        )

        features = observer.get_features()
        expected_features = ["feature_close", "feature_open", "feature_high", "feature_low"]
        for feat in expected_features:
            assert feat in features["observation_features"]
