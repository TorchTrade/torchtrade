"""Tests for BitgetFuturesOrderClass with CCXT.

Inherits common tests from BaseOrderExecutorTests.
"""

import pytest
from unittest.mock import MagicMock, patch
from torchtrade.envs.live.bitget.order_executor import (
    BitgetFuturesOrderClass,
    TradeMode,
    PositionMode,
    MarginMode,
)
from tests.envs.base_exchange_tests import BaseOrderExecutorTests
from tests.mocks.bitget import mock_ccxt_client


class TestBitgetFuturesOrderClass(BaseOrderExecutorTests):
    """Tests for BitgetFuturesOrderClass - inherits common tests from base."""

    def create_order_executor(self, symbol, trade_mode, **kwargs):
        """Create a BitgetFuturesOrderClass instance."""
        client = kwargs.get('client', mock_ccxt_client())

        with patch('torchtrade.envs.bitget.order_executor.ccxt.bitget', return_value=client):
            executor = BitgetFuturesOrderClass(
                symbol=symbol if ':' in symbol else f"{symbol}:USDT",
                trade_mode=trade_mode,
                leverage=kwargs.get('leverage', 10),
                position_mode=kwargs.get('position_mode', PositionMode.ONE_WAY),
                margin_mode=kwargs.get('margin_mode', MarginMode.ISOLATED),
                product_type=kwargs.get('product_type', 'USDT-FUTURES'),
            )
            executor.client = client
            return executor

    def get_trade_mode_enum(self):
        """Get the TradeMode enum for Bitget."""
        return TradeMode

    # Bitget-specific tests

    def test_position_mode_configuration(self):
        """Test position mode configuration."""
        client = mock_ccxt_client()

        with patch('torchtrade.envs.bitget.order_executor.ccxt.bitget', return_value=client):
            executor = BitgetFuturesOrderClass(
                symbol="BTCUSDT:USDT",
                trade_mode=TradeMode.QUANTITY,
                position_mode=PositionMode.HEDGE,
            )
            executor.client = client

            assert executor.position_mode == PositionMode.HEDGE

    def test_margin_mode_configuration(self):
        """Test margin mode configuration."""
        client = mock_ccxt_client()

        with patch('torchtrade.envs.bitget.order_executor.ccxt.bitget', return_value=client):
            executor = BitgetFuturesOrderClass(
                symbol="BTCUSDT:USDT",
                trade_mode=TradeMode.QUANTITY,
                margin_mode=MarginMode.CROSSED,
            )
            executor.client = client

            assert executor.margin_mode == MarginMode.CROSSED

    def test_product_type_configuration(self):
        """Test product type configuration."""
        client = mock_ccxt_client()

        with patch('torchtrade.envs.bitget.order_executor.ccxt.bitget', return_value=client):
            executor = BitgetFuturesOrderClass(
                symbol="BTCUSDT:USDT",
                trade_mode=TradeMode.QUANTITY,
                product_type='USDC-FUTURES',
            )
            executor.client = client

            assert executor.product_type == 'USDC-FUTURES'
