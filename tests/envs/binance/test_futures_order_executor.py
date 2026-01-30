"""Tests for BinanceFuturesOrderClass."""

import pytest
from unittest.mock import MagicMock


class TestBinanceFuturesOrderClass:
    """Tests for BinanceFuturesOrderClass."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Binance futures client."""
        client = MagicMock()

        # Mock futures methods
        client.futures_change_leverage = MagicMock(return_value={"leverage": 10})
        client.futures_change_margin_type = MagicMock(return_value={})

        client.futures_create_order = MagicMock(return_value={
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "side": "BUY",
            "type": "MARKET",
            "executedQty": "0.001",
            "avgPrice": "50000.0",
        })

        client.futures_get_order = MagicMock(return_value={
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "side": "BUY",
            "type": "MARKET",
            "executedQty": "0.001",
            "avgPrice": "50000.0",
        })

        client.futures_position_information = MagicMock(return_value=[{
            "symbol": "BTCUSDT",
            "positionAmt": "0.001",
            "entryPrice": "50000.0",
            "markPrice": "50100.0",
            "unRealizedProfit": "0.1",
            "notional": "50.1",
            "leverage": "10",
            "marginType": "isolated",
            "liquidationPrice": "45000.0",
        }])

        client.futures_account = MagicMock(return_value={
            "totalWalletBalance": "1000.0",
            "availableBalance": "900.0",
            "totalUnrealizedProfit": "0.1",
            "totalMarginBalance": "1000.1",
        })

        client.futures_mark_price = MagicMock(return_value={
            "markPrice": "50100.0",
        })

        client.futures_get_open_orders = MagicMock(return_value=[])
        client.futures_cancel_all_open_orders = MagicMock(return_value={})

        return client

    @pytest.fixture
    def order_executor(self, mock_client):
        """Create order executor with mock client."""
        from torchtrade.envs.live.binance.order_executor import (
            BinanceFuturesOrderClass,
        )

        return BinanceFuturesOrderClass(
            symbol="BTCUSDT",
            trade_mode="quantity",
            demo=True,
            leverage=10,
            client=mock_client,
        )

    def test_initialization(self, order_executor, mock_client):
        """Test order executor initialization."""
        assert order_executor.symbol == "BTCUSDT"
        assert order_executor.leverage == 10
        assert order_executor.demo is True

        # Verify setup was called
        mock_client.futures_change_leverage.assert_called_once()

    def test_symbol_normalization(self, mock_client):
        """Test that symbol with slash is normalized."""
        from torchtrade.envs.live.binance.order_executor import (
            BinanceFuturesOrderClass,
        )

        executor = BinanceFuturesOrderClass(
            symbol="BTC/USDT",
            trade_mode="quantity",
            client=mock_client,
        )
        assert executor.symbol == "BTCUSDT"

    def test_market_buy_order(self, order_executor, mock_client):
        """Test placing a market buy order."""
        success = order_executor.trade(
            side="BUY",
            quantity=0.001,
            order_type="market",
        )

        assert success is True
        mock_client.futures_create_order.assert_called()

        call_kwargs = mock_client.futures_create_order.call_args[1]
        assert call_kwargs["symbol"] == "BTCUSDT"
        assert call_kwargs["side"] == "BUY"
        assert call_kwargs["type"] == "MARKET"

    def test_market_sell_order(self, order_executor, mock_client):
        """Test placing a market sell order (short)."""
        success = order_executor.trade(
            side="SELL",
            quantity=0.001,
            order_type="market",
        )

        assert success is True

        call_kwargs = mock_client.futures_create_order.call_args[1]
        assert call_kwargs["side"] == "SELL"

    def test_limit_order(self, order_executor, mock_client):
        """Test placing a limit order."""
        success = order_executor.trade(
            side="BUY",
            quantity=0.001,
            order_type="limit",
            limit_price=49000.0,
        )

        assert success is True

        call_kwargs = mock_client.futures_create_order.call_args[1]
        assert call_kwargs["type"] == "LIMIT"
        assert call_kwargs["price"] == 49000.0

    def test_limit_order_without_price_fails(self, order_executor):
        """Test that limit order without price raises error."""
        success = order_executor.trade(
            side="BUY",
            quantity=0.001,
            order_type="limit",
            # No limit_price provided
        )

        assert success is False

    def test_order_with_take_profit(self, order_executor, mock_client):
        """Test order with take profit."""
        success = order_executor.trade(
            side="BUY",
            quantity=0.001,
            order_type="market",
            take_profit=52000.0,
        )

        assert success is True

        # Should have called futures_create_order twice (main + TP)
        assert mock_client.futures_create_order.call_count >= 2

    def test_order_with_stop_loss(self, order_executor, mock_client):
        """Test order with stop loss."""
        success = order_executor.trade(
            side="BUY",
            quantity=0.001,
            order_type="market",
            stop_loss=48000.0,
        )

        assert success is True

        # Should have called futures_create_order twice (main + SL)
        assert mock_client.futures_create_order.call_count >= 2

    def test_order_with_bracket(self, order_executor, mock_client):
        """Test order with both take profit and stop loss."""
        success = order_executor.trade(
            side="BUY",
            quantity=0.001,
            order_type="market",
            take_profit=52000.0,
            stop_loss=48000.0,
        )

        assert success is True

        # Should have called futures_create_order three times (main + TP + SL)
        assert mock_client.futures_create_order.call_count >= 3

    def test_get_status(self, order_executor, mock_client):
        """Test getting account/position status."""
        # First execute an order to set last_order_id
        order_executor.trade(side="BUY", quantity=0.001)

        status = order_executor.get_status()

        assert "position_status" in status
        assert status["position_status"] is not None
        assert status["position_status"].qty == 0.001

    def test_get_status_no_position(self, order_executor, mock_client):
        """Test getting status with no position."""
        mock_client.futures_position_information = MagicMock(return_value=[{
            "symbol": "BTCUSDT",
            "positionAmt": "0",
            "entryPrice": "0",
            "markPrice": "50000.0",
            "unRealizedProfit": "0",
            "notional": "0",
            "leverage": "10",
            "marginType": "isolated",
            "liquidationPrice": "0",
        }])

        status = order_executor.get_status()
        assert status["position_status"] is None

    def test_get_account_balance(self, order_executor, mock_client):
        """Test getting account balance."""
        balance = order_executor.get_account_balance()

        assert balance["total_wallet_balance"] == 1000.0
        assert balance["available_balance"] == 900.0

    def test_get_mark_price(self, order_executor, mock_client):
        """Test getting mark price."""
        price = order_executor.get_mark_price()
        assert price == 50100.0

    def test_close_position(self, order_executor, mock_client):
        """Test closing a position."""
        success = order_executor.close_position()

        assert success is True

        # Should have called futures_create_order with reduceOnly
        call_kwargs = mock_client.futures_create_order.call_args[1]
        assert call_kwargs["reduceOnly"] == "true"

    def test_close_position_no_position(self, order_executor, mock_client):
        """Test closing when no position exists."""
        mock_client.futures_position_information = MagicMock(return_value=[{
            "symbol": "BTCUSDT",
            "positionAmt": "0",
            "entryPrice": "0",
            "markPrice": "50000.0",
            "unRealizedProfit": "0",
            "notional": "0",
            "leverage": "10",
            "marginType": "isolated",
            "liquidationPrice": "0",
        }])

        success = order_executor.close_position()
        assert success is True

    def test_cancel_open_orders(self, order_executor, mock_client):
        """Test cancelling open orders."""
        success = order_executor.cancel_open_orders()

        assert success is True
        mock_client.futures_cancel_all_open_orders.assert_called_once()

    def test_set_leverage(self, order_executor, mock_client):
        """Test changing leverage."""
        success = order_executor.set_leverage(20)

        assert success is True
        assert order_executor.leverage == 20

    def test_reduce_only_order(self, order_executor, mock_client):
        """Test reduce only order."""
        success = order_executor.trade(
            side="SELL",
            quantity=0.001,
            order_type="market",
            reduce_only=True,
        )

        assert success is True

        call_kwargs = mock_client.futures_create_order.call_args[1]
        assert call_kwargs["reduceOnly"] == "true"


class TestPositionStatusDataclass:
    """Tests for PositionStatus dataclass."""

    def test_position_status_creation(self):
        """Test creating PositionStatus."""
        from torchtrade.envs.live.binance.order_executor import PositionStatus

        pos = PositionStatus(
            qty=0.001,
            notional_value=50.0,
            entry_price=50000.0,
            unrealized_pnl=0.1,
            unrealized_pnl_pct=0.002,
            mark_price=50100.0,
            leverage=10,
            margin_type="isolated",
            liquidation_price=45000.0,
        )

        assert pos.qty == 0.001
        assert pos.leverage == 10


class TestOrderStatusDataclass:
    """Tests for OrderStatus dataclass."""

    def test_order_status_creation(self):
        """Test creating OrderStatus."""
        from torchtrade.envs.live.binance.order_executor import OrderStatus

        order = OrderStatus(
            is_open=False,
            order_id="12345",
            filled_qty=0.001,
            filled_avg_price=50000.0,
            status="FILLED",
            side="BUY",
            order_type="MARKET",
        )

        assert order.is_open is False
        assert order.filled_qty == 0.001
