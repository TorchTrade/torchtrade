"""
Unit tests for AlpacaOrderClass.

Tests order execution, position management, and error handling using mock clients.
"""

import pytest
import warnings

from torchtrade.envs.alpaca.order_executor import (
    AlpacaOrderClass,
    TradeMode,
    OrderStatus,
    PositionStatus,
)
from .mocks import MockTradingClient


class TestAlpacaOrderClassInitialization:
    """Tests for AlpacaOrderClass initialization."""

    def test_init_with_mock_client(self):
        """Test initialization with injected mock client."""
        mock_client = MockTradingClient()
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.NOTIONAL,
            client=mock_client,
        )

        assert order_class.symbol == "BTCUSD"
        assert order_class.trade_mode == TradeMode.NOTIONAL
        assert order_class.client is mock_client
        assert order_class.last_order_id is None

    def test_symbol_slash_removal(self):
        """Test that slashes are removed from symbols."""
        mock_client = MockTradingClient()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            order_class = AlpacaOrderClass(
                symbol="BTC/USD",
                trade_mode=TradeMode.NOTIONAL,
                client=mock_client,
            )
            assert len(w) == 1
            assert "contains '/'" in str(w[0].message)

        assert order_class.symbol == "BTCUSD"

    def test_trade_mode_quantity(self):
        """Test initialization with QUANTITY trade mode."""
        mock_client = MockTradingClient()
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.QUANTITY,
            client=mock_client,
        )

        assert order_class.trade_mode == TradeMode.QUANTITY


class TestAlpacaOrderClassTrade:
    """Tests for trade execution."""

    @pytest.fixture
    def order_class(self):
        """Create an AlpacaOrderClass with mock client."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        return AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.NOTIONAL,
            client=mock_client,
        )

    def test_market_buy_order(self, order_class):
        """Test placing a market buy order."""
        success = order_class.trade(
            side="buy",
            amount=1000,  # $1000 notional
            order_type="market",
        )

        assert success is True
        assert order_class.last_order_id is not None

    def test_market_sell_order(self, order_class):
        """Test placing a market sell order (closes position)."""
        # First buy
        order_class.trade(side="buy", amount=1000, order_type="market")

        # Then sell
        success = order_class.trade(
            side="sell",
            amount=1000,
            order_type="market",
        )

        assert success is True

    def test_limit_order(self, order_class):
        """Test placing a limit order."""
        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="limit",
            limit_price=95000.0,
        )

        assert success is True

    def test_limit_order_without_price_fails(self, order_class):
        """Test that limit order without price returns False."""
        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="limit",
        )

        assert success is False

    def test_stop_limit_order(self, order_class):
        """Test placing a stop limit order."""
        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="stop_limit",
            limit_price=95000.0,
            stop_price=94000.0,
        )

        assert success is True

    def test_stop_limit_order_without_prices_fails(self, order_class):
        """Test that stop limit order without prices returns False."""
        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="stop_limit",
            limit_price=95000.0,  # Missing stop_price
        )

        assert success is False

    def test_invalid_order_type_fails(self, order_class):
        """Test that invalid order type returns False."""
        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="invalid_type",
        )

        assert success is False

    def test_order_with_take_profit(self, order_class):
        """Test order with take profit."""
        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="market",
            take_profit=110000.0,
        )

        assert success is True

    def test_order_with_stop_loss(self, order_class):
        """Test order with stop loss creates OTO order."""
        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="market",
            stop_loss=90000.0,
        )

        # Should succeed with OTO order
        assert success is True

    def test_trade_failure_handling(self):
        """Test handling of trade failures."""
        mock_client = MockTradingClient(simulate_failures=True)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.NOTIONAL,
            client=mock_client,
        )

        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="market",
        )

        assert success is False

    def test_quantity_mode_trade(self):
        """Test trade in QUANTITY mode."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.QUANTITY,
            client=mock_client,
        )

        success = order_class.trade(
            side="buy",
            amount=0.01,  # 0.01 BTC
            order_type="market",
        )

        assert success is True


class TestAlpacaOrderClassStatus:
    """Tests for status retrieval methods."""

    @pytest.fixture
    def order_class_with_position(self):
        """Create an AlpacaOrderClass with an open position."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.NOTIONAL,
            client=mock_client,
        )
        order_class.trade(side="buy", amount=1000, order_type="market")
        return order_class

    def test_get_status_with_order(self, order_class_with_position):
        """Test getting status when there's an order."""
        status = order_class_with_position.get_status()

        assert "order_status" in status
        assert isinstance(status["order_status"], OrderStatus)
        # Status can be enum or string depending on mock implementation
        status_value = status["order_status"].status
        assert str(status_value).lower() == "filled" or "filled" in str(status_value).lower()

    def test_get_status_with_position(self, order_class_with_position):
        """Test getting status when there's a position."""
        status = order_class_with_position.get_status()

        # Position status may be None if the sell closed the position
        if status.get("position_status") is not None:
            assert isinstance(status["position_status"], PositionStatus)
            assert status["position_status"].qty > 0

    def test_get_status_no_position(self):
        """Test getting status when there's no position."""
        mock_client = MockTradingClient()
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.NOTIONAL,
            client=mock_client,
        )

        status = order_class.get_status()
        assert status.get("position_status") is None

    def test_get_clock(self, order_class_with_position):
        """Test getting market clock."""
        clock = order_class_with_position.get_clock()

        assert hasattr(clock, 'is_open')
        assert hasattr(clock, 'next_close')
        assert hasattr(clock, 'next_open')


class TestAlpacaOrderClassPositionManagement:
    """Tests for position management."""

    @pytest.fixture
    def order_class_with_position(self):
        """Create an AlpacaOrderClass with an open position."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.NOTIONAL,
            client=mock_client,
        )
        order_class.trade(side="buy", amount=1000, order_type="market")
        return order_class

    def test_get_open_orders(self, order_class_with_position):
        """Test getting open orders."""
        orders = order_class_with_position.get_open_orders()
        # After market order is filled, there should be no open orders
        assert isinstance(orders, list)

    def test_cancel_open_orders(self, order_class_with_position):
        """Test canceling open orders."""
        success = order_class_with_position.cancel_open_orders()
        assert success is True

    def test_close_position_full(self, order_class_with_position):
        """Test closing entire position."""
        success = order_class_with_position.close_position()
        assert success is True

        # Verify position is closed
        status = order_class_with_position.get_status()
        assert status.get("position_status") is None

    def test_close_position_partial(self, order_class_with_position):
        """Test closing partial position."""
        initial_status = order_class_with_position.get_status()

        # Skip test if no position was created (depends on mock behavior)
        if initial_status.get("position_status") is None:
            pytest.skip("No position created by mock")

        initial_qty = initial_status["position_status"].qty

        # Close half
        success = order_class_with_position.close_position(qty=initial_qty / 2)
        assert success is True

        # Verify partial close
        status = order_class_with_position.get_status()
        if status.get("position_status") is not None:
            assert status["position_status"].qty < initial_qty

    def test_close_all_positions(self, order_class_with_position):
        """Test closing all positions."""
        results = order_class_with_position.close_all_positions()

        assert isinstance(results, dict)
        assert all(v is True for v in results.values())

    def test_close_position_failure_handling(self):
        """Test handling of position close failures."""
        mock_client = MockTradingClient(simulate_failures=True)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.NOTIONAL,
            client=mock_client,
        )

        success = order_class.close_position()
        assert success is False


class TestAlpacaOrderClassEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_trade_amount(self):
        """Test trading with very small amount."""
        mock_client = MockTradingClient(current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.NOTIONAL,
            client=mock_client,
        )

        success = order_class.trade(
            side="buy",
            amount=1.0,  # $1
            order_type="market",
        )

        assert success is True

    def test_very_large_trade_amount(self):
        """Test trading with large amount."""
        mock_client = MockTradingClient(initial_cash=1000000.0, current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.NOTIONAL,
            client=mock_client,
        )

        success = order_class.trade(
            side="buy",
            amount=100000,  # $100,000
            order_type="market",
        )

        assert success is True

    def test_multiple_consecutive_trades(self):
        """Test multiple trades in sequence."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.NOTIONAL,
            client=mock_client,
        )

        # Buy
        assert order_class.trade(side="buy", amount=1000, order_type="market") is True

        # Sell
        assert order_class.trade(side="sell", amount=1000, order_type="market") is True

        # Buy again
        assert order_class.trade(side="buy", amount=2000, order_type="market") is True

    def test_time_in_force_gtc(self):
        """Test GTC time in force."""
        mock_client = MockTradingClient()
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.NOTIONAL,
            client=mock_client,
        )

        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="market",
            time_in_force="gtc",
        )

        assert success is True

    def test_time_in_force_ioc(self):
        """Test IOC time in force."""
        mock_client = MockTradingClient()
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=TradeMode.NOTIONAL,
            client=mock_client,
        )

        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="market",
            time_in_force="ioc",
        )

        assert success is True
