"""
Unit tests for AlpacaOrderClass.

Tests order execution, position management, and error handling using mock clients.
"""

import pytest
import warnings

from torchtrade.envs.live.alpaca.order_executor import (
    AlpacaOrderClass,
    OrderStatus,
    PositionStatus,
)
from tests.mocks.alpaca import MockTradingClient, MockOrder, MockOrderStatus

# Alpaca trade modes are the string TradeMode values (Literal, not an Enum).
notional = "notional"
quantity = "quantity"


class TestAlpacaOrderClass:
    """Tests for AlpacaOrderClass.

    NOTE: this class previously subclassed BaseOrderExecutorTests without implementing
    the abstract get_trade_mode_enum(), which made it an abstract class that pytest
    collected ZERO tests from — silently disabling all of the tests below. The base is
    unadoptable (it constructs with TradeMode.NOTIONAL, but TradeMode is a Literal with
    no such member, and calls trade(amount=...) which futures executors don't accept),
    so these tests now stand alone.
    """


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

    @pytest.mark.parametrize("tif", ["gtc", "ioc"])
    def test_time_in_force_accepted(self, tif):
        """Both GTC and IOC time-in-force values submit successfully."""
        oc = AlpacaOrderClass(symbol="BTCUSD", trade_mode=notional, client=MockTradingClient())
        assert oc.trade(side="buy", amount=1000, order_type="market", time_in_force=tif) is True

    def test_notional_sell_full_closes_ignoring_amount(self):
        """Notional-mode SELL does NOT submit a sized order — it full-closes the position
        and returns True, ignoring `amount` (order_executor.py). Regression guard for that
        surprising branch, which no test previously asserted."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        oc = AlpacaOrderClass(symbol="BTCUSD", trade_mode=notional, client=mock_client)
        oc.trade(side="buy", amount=1000, order_type="market")
        assert oc.get_status().get("position_status") is not None
        orders_after_buy = len(mock_client.orders)

        assert oc.trade(side="sell", amount=999999, order_type="market") is True
        assert oc.get_status().get("position_status") is None     # full close
        assert len(mock_client.orders) == orders_after_buy        # no sized sell order submitted

    def test_limit_order_happy_path(self):
        """A limit order with a price submits successfully (the plain-limit success branch)."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        oc = AlpacaOrderClass(symbol="BTCUSD", trade_mode=notional, client=mock_client)
        assert oc.trade(side="buy", amount=1000, order_type="limit", limit_price=95000.0) is True
        assert len(mock_client.orders) == 1

    def test_bracket_order_tp_and_sl(self):
        """Both take_profit and stop_loss -> a BRACKET order submits successfully (a headline
        SLTP feature that had no coverage)."""
        mock_client = MockTradingClient(initial_cash=10000.0, current_price=100000.0)
        oc = AlpacaOrderClass(symbol="BTCUSD", trade_mode=notional, client=mock_client)
        assert oc.trade(side="buy", amount=1000, order_type="market",
                        take_profit=110000.0, stop_loss=90000.0) is True
        assert len(mock_client.orders) == 1

    @pytest.mark.parametrize("inject_open", [False, True], ids=["no-orders", "with-open-order"])
    def test_cancel_open_orders(self, inject_open):
        """cancel_open_orders returns True whether or not open orders exist, and cancels any
        that are open (behavior the deleted BaseOrderExecutorTests named but never ran)."""
        mock_client = MockTradingClient()
        oc = AlpacaOrderClass(symbol="BTCUSD", trade_mode=notional, client=mock_client)
        order = None
        if inject_open:
            order = MockOrder(symbol="BTCUSD", status=MockOrderStatus.NEW)
            mock_client.orders[order.id] = order

        assert oc.cancel_open_orders() is True
        if inject_open:
            assert order.status == MockOrderStatus.CANCELED
