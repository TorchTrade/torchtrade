"""Tests for BinanceFuturesOrderClass.

Inherits common tests from BaseOrderExecutorTests.
"""

import pytest
from unittest.mock import MagicMock
from torchtrade.envs.binance.futures_order_executor import (
    BinanceFuturesOrderClass,
    TradeMode,
    PositionStatus,
    OrderStatus,
)
from tests.envs.base_exchange_tests import BaseOrderExecutorTests


class TestBinanceFuturesOrderClass(BaseOrderExecutorTests):
    """Tests for BinanceFuturesOrderClass - inherits common tests from base."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Binance futures client."""
        client = MagicMock()
        client.futures_change_leverage = MagicMock(return_value={"leverage": 10})
        client.futures_change_margin_type = MagicMock(return_value={})
        client.futures_create_order = MagicMock(return_value={
            "orderId": 12345, "symbol": "BTCUSDT", "status": "FILLED",
            "side": "BUY", "type": "MARKET", "executedQty": "0.001",
            "avgPrice": "50000.0",
        })
        client.futures_get_order = MagicMock(return_value={
            "orderId": 12345, "symbol": "BTCUSDT", "status": "FILLED",
            "side": "BUY", "type": "MARKET", "executedQty": "0.001",
            "avgPrice": "50000.0",
        })
        client.futures_position_information = MagicMock(return_value=[{
            "symbol": "BTCUSDT", "positionAmt": "0.001", "entryPrice": "50000.0",
            "markPrice": "50100.0", "unRealizedProfit": "0.1", "notional": "50.1",
            "leverage": "10", "marginType": "isolated", "liquidationPrice": "45000.0",
        }])
        client.futures_account = MagicMock(return_value={
            "totalWalletBalance": "1000.0", "availableBalance": "900.0",
            "totalUnrealizedProfit": "0.1", "totalMarginBalance": "1000.1",
        })
        client.futures_mark_price = MagicMock(return_value={"markPrice": "50100.0"})
        client.futures_get_open_orders = MagicMock(return_value=[])
        client.futures_cancel_all_open_orders = MagicMock(return_value={})
        return client

    def create_order_executor(self, symbol, trade_mode, **kwargs):
        """Create a BinanceFuturesOrderClass instance."""
        client = kwargs.get('client', self.mock_client())

        return BinanceFuturesOrderClass(
            symbol=symbol.replace('/', ''),
            trade_mode=trade_mode,
            demo=kwargs.get('demo', True),
            leverage=kwargs.get('leverage', 10),
            client=client,
        )

    def get_trade_mode_enum(self):
        """Get the TradeMode enum for Binance."""
        return TradeMode

    # Binance-specific tests

    def test_initialization_with_leverage(self, mock_client):
        """Test order executor initialization with leverage."""
        executor = BinanceFuturesOrderClass(
            symbol="BTCUSDT",
            trade_mode=TradeMode.QUANTITY,
            demo=True,
            leverage=10,
            client=mock_client,
        )

        assert executor.symbol == "BTCUSDT"
        assert executor.leverage == 10
        assert executor.demo is True
        mock_client.futures_change_leverage.assert_called_once()

    def test_symbol_normalization(self, mock_client):
        """Test that symbol with slash is normalized."""
        executor = BinanceFuturesOrderClass(
            symbol="BTC/USDT",
            trade_mode=TradeMode.QUANTITY,
            client=mock_client,
        )
        assert executor.symbol == "BTCUSDT"

    def test_order_with_bracket(self, mock_client):
        """Test order with both take profit and stop loss."""
        executor = BinanceFuturesOrderClass(
            symbol="BTCUSDT",
            trade_mode=TradeMode.QUANTITY,
            client=mock_client,
        )

        success = executor.trade(
            side="BUY",
            quantity=0.001,
            order_type="market",
            take_profit=52000.0,
            stop_loss=48000.0,
        )

        assert success is True
        assert mock_client.futures_create_order.call_count >= 3

    def test_reduce_only_order(self, mock_client):
        """Test reduce only order."""
        executor = BinanceFuturesOrderClass(
            symbol="BTCUSDT",
            trade_mode=TradeMode.QUANTITY,
            client=mock_client,
        )

        success = executor.trade(
            side="SELL",
            quantity=0.001,
            order_type="market",
            reduce_only=True,
        )

        assert success is True
        call_kwargs = mock_client.futures_create_order.call_args[1]
        assert call_kwargs["reduceOnly"] == "true"

    def test_set_leverage(self, mock_client):
        """Test changing leverage."""
        executor = BinanceFuturesOrderClass(
            symbol="BTCUSDT",
            trade_mode=TradeMode.QUANTITY,
            leverage=10,
            client=mock_client,
        )

        success = executor.set_leverage(20)
        assert success is True
        assert executor.leverage == 20

    def test_get_mark_price(self, mock_client):
        """Test getting mark price."""
        executor = BinanceFuturesOrderClass(
            symbol="BTCUSDT",
            trade_mode=TradeMode.QUANTITY,
            client=mock_client,
        )

        price = executor.get_mark_price()
        assert price == 50100.0

    def test_get_account_balance(self, mock_client):
        """Test getting account balance."""
        executor = BinanceFuturesOrderClass(
            symbol="BTCUSDT",
            trade_mode=TradeMode.QUANTITY,
            client=mock_client,
        )

        balance = executor.get_account_balance()
        assert balance["total_wallet_balance"] == 1000.0
        assert balance["available_balance"] == 900.0
