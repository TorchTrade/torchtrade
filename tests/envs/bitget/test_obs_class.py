"""Tests for BitgetObservationClass with CCXT.

Inherits common tests from BaseObservationClassTests.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from torchtrade.envs.live.bitget.observation import BitgetObservationClass
from tests.envs.base_exchange_tests import BaseObservationClassTests
from tests.mocks.bitget import mock_ccxt_client


class TestBitgetObservationClass(BaseObservationClassTests):
    """Tests for BitgetObservationClass - inherits common tests from base."""

    def create_observer(self, symbol, timeframes, window_sizes, **kwargs):
        """Create a BitgetObservationClass instance."""
        client = kwargs.get('client', mock_ccxt_client())

        with patch('torchtrade.envs.bitget.obs_class.ccxt.bitget', return_value=client):
            observer = BitgetObservationClass(
                symbol=symbol if ':' in symbol else f"{symbol}:USDT",
                time_frames=timeframes,
                window_sizes=window_sizes,
                feature_preprocessing_fn=kwargs.get('feature_preprocessing_fn'),
            )
            observer.client = client
            return observer

    def get_expected_symbol_format(self, symbol):
        """Bitget uses CCXT perpetual swap format."""
        if ':' not in symbol:
            return f"{symbol.replace('/', '')}:USDT"
        return symbol

    # Bitget-specific tests

    def test_symbol_normalization(self):
        """Test that various symbol formats are handled."""
        client = mock_ccxt_client()

        with patch('torchtrade.envs.bitget.obs_class.ccxt.bitget', return_value=client):
            observer = BitgetObservationClass(
                symbol="BTCUSDT:USDT",
                time_frames=TimeFrame(1, TimeFrameUnit.Minute),
                window_sizes=10,
            )
            observer.client = client
            assert "USDT" in observer.symbol

    def test_invalid_interval_raises_error(self):
        """Test that unsupported timeframe raises ValueError."""
        client = mock_ccxt_client()

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            with patch('torchtrade.envs.bitget.obs_class.ccxt.bitget', return_value=client):
                BitgetObservationClass(
                    symbol="BTC/USDT:USDT",
                    time_frames=TimeFrame(2, TimeFrameUnit.Minute),
                    window_sizes=10,
                )
