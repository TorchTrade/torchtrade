"""Tests for BitgetFuturesOrderClass with CCXT."""

import pytest
from unittest.mock import MagicMock, patch


class TestBitgetFuturesOrderClass:
    """Tests for BitgetFuturesOrderClass using CCXT."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock CCXT Bitget client."""
        client = MagicMock()

        # Mock CCXT methods
        client.set_leverage = MagicMock(return_value={"leverage": 10})
        client.set_margin_mode = MagicMock(return_value={"marginMode": "isolated"})
        client.set_position_mode = MagicMock(return_value={"posMode": "one_way_mode"})

        # Mock order placement (CCXT unified API)
        client.create_order = MagicMock(return_value={
            "id": "12345",
            "symbol": "BTC/USDT:USDT",
            "status": "closed",
            "side": "buy",
            "type": "market",
            "filled": 0.001,
            "average": 50000.0,
        })

        # Mock bracket order method
        client.create_order_with_take_profit_and_stop_loss = MagicMock(return_value={
            "id": "12345",
            "symbol": "BTC/USDT:USDT",
            "status": "closed",
        })

        # Mock stop market order
        client.create_stop_market_order = MagicMock(return_value={
            "id": "12346",
            "symbol": "BTC/USDT:USDT",
            "type": "stop_market",
        })

        # Mock position information (CCXT fetch_positions)
        client.fetch_positions = MagicMock(return_value=[{
            "symbol": "BTC/USDT:USDT",
            "contracts": 0.001,
            "side": "long",
            "entryPrice": 50000.0,
            "markPrice": 50100.0,
            "unrealizedPnl": 0.1,
            "leverage": 10,
            "marginMode": "isolated",
            "liquidationPrice": 45000.0,
            "notional": 50.1,
        }])

        # Mock account balance (CCXT fetch_balance)
        client.fetch_balance = MagicMock(return_value={
            "info": {
                "totalEquity": "1000.0",
                "available": "900.0",
                "totalUnrealizedProfit": "0.1",
                "totalMarginBalance": "1000.1",
            },
            "total": {"USDT": 1000.0},
            "free": {"USDT": 900.0},
        })

        # Mock mark price (CCXT fetch_ticker)
        client.fetch_ticker = MagicMock(return_value={
            "symbol": "BTC/USDT:USDT",
            "info": {"markPrice": "50100.0"},
        })

        # Mock open orders
        client.fetch_open_orders = MagicMock(return_value=[])
        client.cancel_order = MagicMock(return_value={"id": "12345", "status": "canceled"})

        return client

    @pytest.fixture
    def order_executor(self, mock_client):
        """Create order executor with mock CCXT client."""
        from torchtrade.envs.bitget.futures_order_executor import (
            BitgetFuturesOrderClass,
            TradeMode,
            MarginMode,
            PositionMode,
        )

        # Patch CCXT client creation
        with patch('torchtrade.envs.bitget.futures_order_executor.ccxt.bitget', return_value=mock_client):
            executor = BitgetFuturesOrderClass(
                symbol="BTC/USDT:USDT",
                trade_mode=TradeMode.QUANTITY,
                demo=True,
                leverage=10,
                margin_mode=MarginMode.ISOLATED,
                position_mode=PositionMode.ONE_WAY,
                api_key="test_key",
                api_secret="test_secret",
                passphrase="test_pass",
            )
            # Override client with mock for testing
            executor.client = mock_client
            return executor

    def test_initialization(self, order_executor, mock_client):
        """Test order executor initialization."""
        assert order_executor.symbol == "BTC/USDT:USDT"
        assert order_executor.leverage == 10
        assert order_executor.demo is True
        assert order_executor.product_type == "USDT-FUTURES"

    def test_symbol_normalization(self, mock_client):
        """Test that symbol formats are accepted."""
        from torchtrade.envs.bitget.futures_order_executor import (
            BitgetFuturesOrderClass,
            TradeMode,
        )

        with patch('torchtrade.envs.bitget.futures_order_executor.ccxt.bitget', return_value=mock_client):
            # CCXT format should work
            executor = BitgetFuturesOrderClass(
                symbol="BTC/USDT:USDT",
                trade_mode=TradeMode.QUANTITY,
                api_key="test_key",
                api_secret="test_secret",
                passphrase="test_pass",
            )
            executor.client = mock_client
            assert executor.symbol == "BTC/USDT:USDT"

    def test_product_type_demo(self, mock_client):
        """Test that demo=True uses USDT-FUTURES product type."""
        from torchtrade.envs.bitget.futures_order_executor import (
            BitgetFuturesOrderClass,
            TradeMode,
        )

        with patch('torchtrade.envs.bitget.futures_order_executor.ccxt.bitget', return_value=mock_client):
            executor = BitgetFuturesOrderClass(
                symbol="BTC/USDT:USDT",
                product_type="USDT-FUTURES",
                demo=True,
                api_key="test_key",
                api_secret="test_secret",
                passphrase="test_pass",
            )
            executor.client = mock_client
            assert executor.product_type == "USDT-FUTURES"

    def test_market_buy_order(self, order_executor, mock_client):
        """Test placing a market buy order."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="market",
        )

        assert success is True
        mock_client.create_order.assert_called()

        # Check call arguments
        call_args = mock_client.create_order.call_args
        assert call_args[1]["symbol"] == "BTC/USDT:USDT"
        assert call_args[1]["side"] == "buy"
        assert call_args[1]["type"] == "market"
        assert call_args[1]["amount"] == 0.001
        # Should include marginMode in params
        assert "marginMode" in call_args[1]["params"]

    def test_market_sell_order(self, order_executor, mock_client):
        """Test placing a market sell order (short)."""
        success = order_executor.trade(
            side="sell",
            quantity=0.001,
            order_type="market",
        )

        assert success is True
        call_args = mock_client.create_order.call_args
        assert call_args[1]["side"] == "sell"

    def test_limit_order(self, order_executor, mock_client):
        """Test placing a limit order."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="limit",
            limit_price=49000.0,
        )

        assert success is True
        call_args = mock_client.create_order.call_args
        assert call_args[1]["type"] == "limit"
        assert call_args[1]["price"] == 49000.0

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
        # Should use create_order with stopPrice param
        assert mock_client.create_order.called

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
        # Should use CCXT's bracket order method
        mock_client.create_order_with_take_profit_and_stop_loss.assert_called_once()

    def test_get_status(self, order_executor, mock_client):
        """Test getting order and position status."""
        # Place an order first
        order_executor.trade(side="buy", quantity=0.001)

        status = order_executor.get_status()

        assert "position_status" in status
        # Position status should have attributes
        if status["position_status"]:
            assert hasattr(status["position_status"], "qty")
            assert hasattr(status["position_status"], "entry_price")

    def test_get_status_no_position(self, order_executor, mock_client):
        """Test get_status when no position exists."""
        # Mock empty positions
        mock_client.fetch_positions = MagicMock(return_value=[{
            "symbol": "BTC/USDT:USDT",
            "contracts": 0,
            "side": "long",
        }])

        status = order_executor.get_status()
        assert status["position_status"] is None

    def test_get_status_short_position(self, order_executor, mock_client):
        """Test get_status with short position (negative qty)."""
        # Mock short position
        mock_client.fetch_positions = MagicMock(return_value=[{
            "symbol": "BTC/USDT:USDT",
            "contracts": 0.001,
            "side": "short",
            "entryPrice": 50000.0,
            "markPrice": 49900.0,
            "unrealizedPnl": 0.1,
            "leverage": 10,
            "marginMode": "isolated",
            "liquidationPrice": 55000.0,
            "notional": 49.9,
        }])

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
        mock_client.create_order.assert_called()

        # Should place a close order with reduceOnly
        call_args = mock_client.create_order.call_args
        assert call_args[1]["params"]["reduceOnly"] is True

    def test_close_position_no_position(self, order_executor, mock_client):
        """Test closing when no position exists."""
        # Mock no position
        mock_client.fetch_positions = MagicMock(return_value=[{
            "symbol": "BTC/USDT:USDT",
            "contracts": 0,
            "side": "long",
        }])

        success = order_executor.close_position()
        assert success is True  # Should succeed without error

    def test_cancel_open_orders(self, order_executor, mock_client):
        """Test cancelling all open orders."""
        # Mock some open orders
        mock_client.fetch_open_orders = MagicMock(return_value=[
            {"id": "123", "symbol": "BTC/USDT:USDT"},
            {"id": "456", "symbol": "BTC/USDT:USDT"},
        ])

        success = order_executor.cancel_open_orders()

        assert success is True
        # Should call cancel for each order
        assert mock_client.cancel_order.call_count == 2

    def test_set_leverage(self, order_executor, mock_client):
        """Test changing leverage."""
        success = order_executor.set_leverage(20)

        assert success is True
        assert order_executor.leverage == 20
        mock_client.set_leverage.assert_called()

    def test_set_margin_mode(self, order_executor, mock_client):
        """Test changing margin mode."""
        from torchtrade.envs.bitget.futures_order_executor import MarginMode

        success = order_executor.set_margin_mode(MarginMode.CROSSED)

        assert success is True
        assert order_executor.margin_mode == MarginMode.CROSSED
        # Note: set_margin_mode may not work reliably on Bitget, but we test the call
        mock_client.set_margin_mode.assert_called()

    def test_trade_failure_handling(self, order_executor, mock_client):
        """Test that trade failures are handled gracefully."""
        # Mock API failure
        mock_client.create_order = MagicMock(side_effect=Exception("API Error"))

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
            symbol="BTC/USDT:USDT",
            trade_mode=TradeMode.QUANTITY,
            api_key=os.getenv("BITGETACCESSAPIKEY"),
            api_secret=os.getenv("BITGETSECRETKEY"),
            passphrase=os.getenv("BITGETPASSPHRASE"),
            demo=True,
            leverage=5,
        )

        # Test getting balance
        balance = executor.get_account_balance()
        assert balance["total_wallet_balance"] > 0

        # Test getting mark price
        price = executor.get_mark_price()
        assert price > 0
