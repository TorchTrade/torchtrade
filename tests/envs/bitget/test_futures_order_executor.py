"""Tests for BitgetFuturesOrderClass."""

import pytest
from unittest.mock import MagicMock, patch


class TestBitgetFuturesOrderClass:
    """Tests for BitgetFuturesOrderClass."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Bitget futures client."""
        client = MagicMock()

        # Mock account setup methods
        client.mix_adjust_leverage = MagicMock(return_value={"data": {"leverage": "10"}})
        client.mix_change_margin_mode = MagicMock(return_value={"data": {}})

        # Mock order placement
        client.mix_place_order = MagicMock(return_value={
            "data": {
                "orderId": "12345",
                "symbol": "BTCUSDT",
                "status": "filled",
            }
        })

        # Mock order details
        client.mix_get_order_details = MagicMock(return_value={
            "data": {
                "orderId": "12345",
                "symbol": "BTCUSDT",
                "status": "filled",
                "side": "buy",
                "orderType": "market",
                "filledQty": "0.001",
                "priceAvg": "50000.0",
            }
        })

        # Mock position information
        client.mix_get_single_position = MagicMock(return_value={
            "data": [{
                "symbol": "BTCUSDT",
                "total": "0.001",
                "holdSide": "long",
                "averageOpenPrice": "50000.0",
                "markPrice": "50100.0",
                "unrealizedPL": "0.1",
                "leverage": "10",
                "marginMode": "isolated",
                "liquidationPrice": "45000.0",
            }]
        })

        # Mock account balance
        client.mix_get_accounts = MagicMock(return_value={
            "data": [{
                "equity": "1000.0",
                "available": "900.0",
                "unrealizedPL": "0.1",
            }]
        })

        # Mock mark price
        client.mix_get_market_price = MagicMock(return_value={
            "data": {
                "markPrice": "50100.0",
            }
        })

        # Mock open orders and cancel
        client.mix_get_open_orders = MagicMock(return_value={"data": []})
        client.mix_cancel_all_orders = MagicMock(return_value={"data": {}})

        return client

    @pytest.fixture
    def order_executor(self, mock_client):
        """Create order executor with mock client."""
        from torchtrade.envs.bitget.futures_order_executor import (
            BitgetFuturesOrderClass,
            TradeMode,
        )

        return BitgetFuturesOrderClass(
            symbol="BTCUSDT",
            trade_mode=TradeMode.QUANTITY,
            demo=True,
            leverage=10,
            client=mock_client,
        )

    def test_initialization(self, order_executor, mock_client):
        """Test order executor initialization."""
        assert order_executor.symbol == "BTCUSDT"
        assert order_executor.leverage == 10
        assert order_executor.demo is True
        assert order_executor.product_type == "SUMCBL"  # Demo forces testnet

        # Verify setup was called
        mock_client.mix_adjust_leverage.assert_called_once()

    def test_symbol_normalization(self, mock_client):
        """Test that symbol with slash is normalized."""
        from torchtrade.envs.bitget.futures_order_executor import (
            BitgetFuturesOrderClass,
            TradeMode,
        )

        executor = BitgetFuturesOrderClass(
            symbol="BTC/USDT",
            trade_mode=TradeMode.QUANTITY,
            client=mock_client,
        )
        assert executor.symbol == "BTCUSDT"

    def test_product_type_demo(self, mock_client):
        """Test that demo=True forces SUMCBL product type."""
        from torchtrade.envs.bitget.futures_order_executor import (
            BitgetFuturesOrderClass,
            TradeMode,
        )

        executor = BitgetFuturesOrderClass(
            symbol="BTCUSDT",
            product_type="UMCBL",  # Try production
            demo=True,  # But demo is True
            client=mock_client,
        )
        assert executor.product_type == "SUMCBL"  # Should be forced to testnet

    def test_market_buy_order(self, order_executor, mock_client):
        """Test placing a market buy order."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="market",
        )

        assert success is True
        mock_client.mix_place_order.assert_called()

        call_kwargs = mock_client.mix_place_order.call_args[1]
        assert call_kwargs["symbol"] == "BTCUSDT"
        assert call_kwargs["side"] == "buy"
        assert call_kwargs["orderType"] == "market"
        assert call_kwargs["productType"] == "SUMCBL"

    def test_market_sell_order(self, order_executor, mock_client):
        """Test placing a market sell order (short)."""
        success = order_executor.trade(
            side="sell",
            quantity=0.001,
            order_type="market",
        )

        assert success is True

        call_kwargs = mock_client.mix_place_order.call_args[1]
        assert call_kwargs["side"] == "sell"

    def test_limit_order(self, order_executor, mock_client):
        """Test placing a limit order."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="limit",
            limit_price=49000.0,
        )

        assert success is True

        call_kwargs = mock_client.mix_place_order.call_args[1]
        assert call_kwargs["orderType"] == "limit"
        assert call_kwargs["price"] == "49000.0"

    def test_limit_order_without_price_fails(self, order_executor):
        """Test that limit order without price returns False."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="limit",
            # Missing limit_price
        )
        assert success is False

    def test_stop_market_order(self, order_executor, mock_client):
        """Test placing a stop market order."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="stop_market",
            stop_price=48000.0,
        )

        assert success is True

        call_kwargs = mock_client.mix_place_order.call_args[1]
        assert call_kwargs["orderType"] == "stop_market"
        assert call_kwargs["triggerPrice"] == "48000.0"

    def test_stop_order_without_price_fails(self, order_executor):
        """Test that stop order without price returns False."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="stop_market",
            # Missing stop_price
        )
        assert success is False

    def test_bracket_order_with_tp_sl(self, order_executor, mock_client):
        """Test placing bracket order with take profit and stop loss."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="market",
            take_profit=51000.0,
            stop_loss=49000.0,
        )

        assert success is True

        # Should place 3 orders: main + TP + SL
        assert mock_client.mix_place_order.call_count == 3

    def test_get_status(self, order_executor, mock_client):
        """Test getting order and position status."""
        # Place an order first
        order_executor.trade(side="buy", quantity=0.001)

        status = order_executor.get_status()

        assert "order_status" in status
        assert "position_status" in status
        assert status["order_status"].order_id == "12345"
        assert status["position_status"].qty == 0.001
        assert status["position_status"].entry_price == 50000.0

    def test_get_status_no_position(self, order_executor, mock_client):
        """Test get_status when no position exists."""
        # Mock empty position
        mock_client.mix_get_single_position = MagicMock(return_value={
            "data": [{
                "total": "0",
                "holdSide": "long",
            }]
        })

        status = order_executor.get_status()
        assert status["position_status"] is None

    def test_get_status_short_position(self, order_executor, mock_client):
        """Test get_status with short position (negative qty)."""
        # Mock short position
        mock_client.mix_get_single_position = MagicMock(return_value={
            "data": [{
                "symbol": "BTCUSDT",
                "total": "0.001",
                "holdSide": "short",  # Short position
                "averageOpenPrice": "50000.0",
                "markPrice": "49900.0",
                "unrealizedPL": "0.1",
                "leverage": "10",
                "marginMode": "isolated",
                "liquidationPrice": "55000.0",
            }]
        })

        status = order_executor.get_status()
        assert status["position_status"].qty < 0  # Should be negative for short

    def test_get_account_balance(self, order_executor, mock_client):
        """Test getting account balance."""
        balance = order_executor.get_account_balance()

        assert "total_wallet_balance" in balance
        assert "available_balance" in balance
        assert "total_unrealized_profit" in balance
        assert "total_margin_balance" in balance

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
        mock_client.mix_place_order.assert_called()

        # Should place a close order
        call_kwargs = mock_client.mix_place_order.call_args[1]
        assert call_kwargs["reduceOnly"] == "YES"

    def test_close_position_no_position(self, order_executor, mock_client):
        """Test closing when no position exists."""
        # Mock no position
        mock_client.mix_get_single_position = MagicMock(return_value={
            "data": [{
                "total": "0",
                "holdSide": "long",
            }]
        })

        success = order_executor.close_position()
        assert success is True  # Should succeed without error

    def test_cancel_open_orders(self, order_executor, mock_client):
        """Test cancelling all open orders."""
        success = order_executor.cancel_open_orders()

        assert success is True
        mock_client.mix_cancel_all_orders.assert_called_once()

    def test_set_leverage(self, order_executor, mock_client):
        """Test changing leverage."""
        success = order_executor.set_leverage(20)

        assert success is True
        assert order_executor.leverage == 20
        mock_client.mix_adjust_leverage.assert_called()

    def test_set_margin_mode(self, order_executor, mock_client):
        """Test changing margin mode."""
        from torchtrade.envs.bitget.futures_order_executor import MarginMode

        success = order_executor.set_margin_mode(MarginMode.CROSSED)

        assert success is True
        assert order_executor.margin_mode == MarginMode.CROSSED
        mock_client.mix_change_margin_mode.assert_called()

    def test_trade_failure_handling(self, order_executor, mock_client):
        """Test that trade failures are handled gracefully."""
        # Mock API failure
        mock_client.mix_place_order = MagicMock(side_effect=Exception("API Error"))

        success = order_executor.trade(side="buy", quantity=0.001)

        assert success is False  # Should return False, not raise

    def test_get_open_orders(self, order_executor, mock_client):
        """Test getting open orders."""
        orders = order_executor.get_open_orders()
        assert isinstance(orders, list)

    def test_position_mode_enum(self):
        """Test PositionMode enum values."""
        from torchtrade.envs.bitget.futures_order_executor import PositionMode

        assert PositionMode.ONE_WAY.value == "one_way_mode"
        assert PositionMode.HEDGE.value == "hedge_mode"

    def test_margin_mode_enum(self):
        """Test MarginMode enum values."""
        from torchtrade.envs.bitget.futures_order_executor import MarginMode

        assert MarginMode.ISOLATED.value == "isolated"
        assert MarginMode.CROSSED.value == "crossed"

    def test_trade_mode_enum(self):
        """Test TradeMode enum values."""
        from torchtrade.envs.bitget.futures_order_executor import TradeMode

        assert TradeMode.QUANTITY.value == "quantity"
        assert TradeMode.NOTIONAL.value == "notional"


class TestBitgetFuturesOrderClassIntegration:
    """Integration tests that would require actual API (skipped by default)."""

    @pytest.mark.skip(reason="Requires live Bitget API connection and credentials")
    def test_live_order_placement(self):
        """Test placing a real order on Bitget testnet."""
        import os
        from torchtrade.envs.bitget.futures_order_executor import (
            BitgetFuturesOrderClass,
            TradeMode,
        )

        executor = BitgetFuturesOrderClass(
            symbol="BTCUSDT",
            trade_mode=TradeMode.QUANTITY,
            api_key=os.getenv("BITGET_API_KEY"),
            api_secret=os.getenv("BITGET_SECRET"),
            passphrase=os.getenv("BITGET_PASSPHRASE"),
            demo=True,
            leverage=5,
        )

        # Test getting balance
        balance = executor.get_account_balance()
        assert balance["total_wallet_balance"] > 0

        # Test getting mark price
        price = executor.get_mark_price()
        assert price > 0
