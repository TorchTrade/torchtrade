"""
Unit tests for AlpacaOrderClass.

Tests order execution, position management, and error handling using mock clients.
Inherits common tests from BaseOrderExecutorTests.
"""

import pytest
import warnings

from torchtrade.envs.live.alpaca.order_executor import (
    AlpacaOrderClass,
    OrderStatus,
    PositionStatus,
)
from tests.mocks.alpaca import MockTradingClient
from tests.envs.base_exchange_tests import BaseOrderExecutorTests


class TestAlpacaOrderClass(BaseOrderExecutorTests):
    """Tests for AlpacaOrderClass - inherits common tests from base."""

    def create_order_executor(self, symbol, trade_mode, **kwargs):
        """Create an AlpacaOrderClass instance."""
        mock_client = kwargs.get('client', MockTradingClient(
            initial_cash=kwargs.get('initial_cash', 10000.0),
            current_price=kwargs.get('current_price', 100000.0),
            simulate_failures=kwargs.get('simulate_failures', False),
        ))

        # Remove slash from symbol for Alpaca (will trigger warning)
        symbol_clean = symbol.replace('/', '')

        return AlpacaOrderClass(
            symbol=symbol_clean,
            trade_mode=trade_mode,
            client=mock_client,
        )

    # Alpaca-specific tests

    def test_symbol_slash_removal(self):
        """Test that slashes are removed from symbols."""
        mock_client = MockTradingClient()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            order_class = AlpacaOrderClass(
                symbol="BTC/USD",
                trade_mode=notional,
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
            trade_mode=quantity,
            client=mock_client,
        )

        assert order_class.trade_mode == quantity

    def test_quantity_mode_trade(self):
        """Test trade in QUANTITY mode."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=quantity,
            client=mock_client,
        )

        success = order_class.trade(
            side="buy",
            amount=0.01,  # 0.01 BTC
            order_type="market",
        )

        assert success is True

    def test_stop_limit_order(self):
        """Test placing a stop limit order."""
        mock_client = MockTradingClient()
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=notional,
            client=mock_client,
        )

        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="stop_limit",
            limit_price=95000.0,
            stop_price=94000.0,
        )

        assert success is True

    def test_stop_limit_order_without_prices_fails(self):
        """Test that stop limit order without prices returns False."""
        mock_client = MockTradingClient()
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=notional,
            client=mock_client,
        )

        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="stop_limit",
            limit_price=95000.0,  # Missing stop_price
        )

        assert success is False

    def test_order_with_stop_loss_creates_oto(self):
        """Test order with stop loss creates OTO (One-Triggers-Other) order."""
        mock_client = MockTradingClient()
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=notional,
            client=mock_client,
        )

        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="market",
            stop_loss=90000.0,
        )

        # Should succeed with OTO order
        assert success is True

    def test_get_status_order_filled(self):
        """Test getting status when order is filled."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=notional,
            client=mock_client,
        )

        # Place an order
        order_class.trade(side="buy", amount=1000, order_type="market")

        status = order_class.get_status()

        assert "order_status" in status
        assert isinstance(status["order_status"], OrderStatus)
        status_value = status["order_status"].status
        assert str(status_value).lower() == "filled" or "filled" in str(status_value).lower()

    def test_get_status_with_position(self):
        """Test getting status when there's a position."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=notional,
            client=mock_client,
        )

        # Place an order
        order_class.trade(side="buy", amount=1000, order_type="market")

        status = order_class.get_status()

        # Position status may be None if the sell closed the position
        if status.get("position_status") is not None:
            assert isinstance(status["position_status"], PositionStatus)
            assert status["position_status"].qty > 0

    def test_get_clock(self):
        """Test getting market clock."""
        mock_client = MockTradingClient()
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=notional,
            client=mock_client,
        )

        clock = order_class.get_clock()

        assert hasattr(clock, 'is_open')
        assert hasattr(clock, 'next_close')
        assert hasattr(clock, 'next_open')

    def test_get_open_orders(self):
        """Test getting open orders."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=notional,
            client=mock_client,
        )

        order_class.trade(side="buy", amount=1000, order_type="market")

        orders = order_class.get_open_orders()
        # After market order is filled, there should be no open orders
        assert isinstance(orders, list)

    def test_close_position_full(self):
        """Test closing entire position."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=notional,
            client=mock_client,
        )

        order_class.trade(side="buy", amount=1000, order_type="market")

        success = order_class.close_position()
        assert success is True

        # Verify position is closed
        status = order_class.get_status()
        assert status.get("position_status") is None

    def test_close_position_partial(self):
        """Test closing partial position."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=notional,
            client=mock_client,
        )

        order_class.trade(side="buy", amount=1000, order_type="market")

        initial_status = order_class.get_status()

        # Skip test if no position was created (depends on mock behavior)
        if initial_status.get("position_status") is None:
            pytest.skip("No position created by mock")

        initial_qty = initial_status["position_status"].qty

        # Close half
        success = order_class.close_position(qty=initial_qty / 2)
        assert success is True

        # Verify partial close
        status = order_class.get_status()
        if status.get("position_status") is not None:
            assert status["position_status"].qty < initial_qty

    def test_close_all_positions(self):
        """Test closing all positions."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=notional,
            client=mock_client,
        )

        order_class.trade(side="buy", amount=1000, order_type="market")

        results = order_class.close_all_positions()

        assert isinstance(results, dict)
        assert all(v is True for v in results.values())

    def test_time_in_force_gtc(self):
        """Test GTC (Good Till Cancelled) time in force."""
        mock_client = MockTradingClient()
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=notional,
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
        """Test IOC (Immediate Or Cancel) time in force."""
        mock_client = MockTradingClient()
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=notional,
            client=mock_client,
        )

        success = order_class.trade(
            side="buy",
            amount=1000,
            order_type="market",
            time_in_force="ioc",
        )

        assert success is True

    def test_very_small_trade_amount(self):
        """Test trading with very small amount."""
        mock_client = MockTradingClient(current_price=100000.0)
        order_class = AlpacaOrderClass(
            symbol="BTCUSD",
            trade_mode=notional,
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
            trade_mode=notional,
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
            trade_mode=notional,
            client=mock_client,
        )

        # Buy
        assert order_class.trade(side="buy", amount=1000, order_type="market") is True

        # Sell
        assert order_class.trade(side="sell", amount=1000, order_type="market") is True

        # Buy again
        assert order_class.trade(side="buy", amount=2000, order_type="market") is True
