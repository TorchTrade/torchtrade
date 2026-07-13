"""Tests for AlpacaOrderClass.

Order execution, position management, and error handling against a mock client.
"""

import pytest
import warnings

from alpaca.trading.enums import OrderClass

from torchtrade.envs.live.alpaca.order_executor import (
    AlpacaOrderClass,
    OrderStatus,
    PositionStatus,
)
from tests.mocks.alpaca import MockTradingClient, MockOrder, MockOrderStatus, MockAsset


# Standalone by design: the shared exchange test base assumes an enum TradeMode and a
# futures-style trade(quantity=...) signature, neither of which Alpaca has.
class TestAlpacaOrderClass:
    """Tests for AlpacaOrderClass."""

    @pytest.fixture
    def client(self):
        return MockTradingClient()  # defaults: cash=10_000, price=100_000

    @pytest.fixture
    def oc(self, client):
        return AlpacaOrderClass(symbol="BTCUSD", trade_mode="notional", client=client)

    # --- construction ---

    def test_symbol_slash_removal(self, client):
        """A '/' in the symbol is stripped, with a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            order_class = AlpacaOrderClass(symbol="BTC/USD", trade_mode="notional", client=client)
            assert len(w) == 1
            assert "contains '/'" in str(w[0].message)

        assert order_class.symbol == "BTCUSD"

    def test_quantity_mode_trade(self, client):
        """In quantity mode the amount is sent as units (qty), not notional."""
        order_class = AlpacaOrderClass(symbol="BTCUSD", trade_mode="quantity", client=client)
        assert order_class.trade(side="buy", amount=0.5, order_type="market") is True
        req = client.requests[-1]
        assert req.qty == 0.5
        assert req.notional is None

    def test_quantity_mode_does_not_zero_small_quantities(self, client):
        """A small crypto quantity must survive to the request (0.01 BTC, not 0.0).

        Regression: quantity mode did round(amount, 1), so anything under 0.05 became 0.0 --
        an order for 0.01 BTC submitted qty=0.0. BTC's real increment is 0.0001.
        """
        order_class = AlpacaOrderClass(symbol="BTCUSD", trade_mode="quantity", client=client)
        assert order_class.trade(side="buy", amount=0.01, order_type="market") is True
        assert client.requests[-1].qty == 0.01

    @pytest.mark.parametrize("amount,expected_qty", [
        (0.01, 0.01),          # survives -- round(_, 1) used to zero this
        (0.000123456, 0.0001),  # floored ONTO the 0.0001 grid, never up
        (1.23456789, 1.2345),   # ditto, larger
    ], ids=["small-crypto-qty", "floors-to-increment", "floors-larger"])
    def test_quantity_floors_onto_the_asset_increment(self, client, amount, expected_qty):
        """Quantities land on Alpaca's grid for the asset, and always by flooring.

        Flooring, not rounding: rounding up can exceed the size (and the buying power) the
        caller asked for.
        """
        oc = AlpacaOrderClass(symbol="BTCUSD", trade_mode="quantity", client=client)
        assert oc.trade(side="buy", amount=amount, order_type="market") is True
        assert client.requests[-1].qty == pytest.approx(expected_qty)

    def test_quantity_below_min_order_size_is_refused(self, client):
        """Sub-minimum quantities are refused, not submitted as a zero-qty order.

        The old code floored them to 0.0 and submitted anyway.
        """
        oc = AlpacaOrderClass(symbol="BTCUSD", trade_mode="quantity", client=client)
        assert oc.trade(side="buy", amount=0.00001, order_type="market") is False
        assert client.requests == []

    def test_quantity_mode_sell_all_sentinel_closes_position(self, client):
        """env.py signals "sell everything" with amount=-1.

        Notional mode intercepts sells and full-closes; quantity mode had no such intercept,
        so the sentinel would have been submitted as a literal qty=-1.
        """
        oc = AlpacaOrderClass(symbol="BTCUSD", trade_mode="quantity", client=client)
        oc.trade(side="buy", amount=0.01, order_type="market")
        assert oc.get_status()["position_status"] is not None

        assert oc.trade(side="sell", amount=-1, order_type="market") is True
        assert oc.get_status()["position_status"] is None
        assert all(getattr(r, "qty", 0) >= 0 for r in client.requests)  # never a negative qty

    def test_non_fractionable_equity_floors_to_whole_shares(self, client):
        """A non-fractionable stock takes whole shares only."""
        client.asset = MockAsset(
            fractionable=False, min_order_size=None, min_trade_increment=None,
        )
        oc = AlpacaOrderClass(symbol="AAPL", trade_mode="quantity", client=client)
        assert oc.trade(side="buy", amount=3.7, order_type="market") is True
        assert client.requests[-1].qty == 3.0

    def test_asset_lookup_failure_fails_closed(self, client):
        """If the asset's rules cannot be read, refuse the order -- do not guess a precision.

        Falling back to 9dp is wrong for an asset whose real increment is 0.01 or 1: the order
        just gets rejected by the exchange, and a cached fallback would poison every order after
        it.
        """
        def boom(_):
            raise RuntimeError("assets endpoint down")
        client.get_asset = boom

        oc = AlpacaOrderClass(symbol="BTCUSD", trade_mode="quantity", client=client)
        assert oc.trade(side="buy", amount=0.01, order_type="market") is False
        assert client.requests == []

    def test_quantity_lands_on_the_min_plus_n_steps_lattice(self, client):
        """Valid crypto quantities are min + N*step, NOT N*step.

        With a minimum that is not a whole multiple of the step, flooring from zero lands above
        the minimum yet off the lattice -- and Alpaca rejects it.
        """
        client.asset = MockAsset(min_order_size=0.000166, min_trade_increment=0.0001)
        oc = AlpacaOrderClass(symbol="BTCUSD", trade_mode="quantity", client=client)

        assert oc.trade(side="buy", amount=0.00027, order_type="market") is True
        # floor-from-zero would give 0.0002 (above the min, but not on the lattice)
        assert client.requests[-1].qty == pytest.approx(0.000266)   # 0.000166 + 1 * 0.0001

    def test_min_order_size_is_not_cached_across_price_moves(self, client):
        """min_order_size is dynamic (it tracks ~10/price). Caching it goes stale.

        A quantity that was valid at one price must be refused once the minimum rises past it.
        """
        client.asset = MockAsset(min_order_size=0.0001, min_trade_increment=0.0001)
        oc = AlpacaOrderClass(symbol="BTCUSD", trade_mode="quantity", client=client)
        assert oc.trade(side="buy", amount=0.0002, order_type="market") is True

        client.asset = MockAsset(min_order_size=0.001, min_trade_increment=0.0001)  # price fell
        assert oc.trade(side="buy", amount=0.0002, order_type="market") is False

    # --- order types ---

    def test_stop_limit_order(self, oc, client):
        """stop_limit with both prices submits a StopLimitOrderRequest."""
        assert oc.trade(side="buy", amount=1000, order_type="stop_limit",
                        limit_price=95000.0, stop_price=94000.0) is True
        req = client.requests[-1]
        assert req.limit_price == 95000.0
        assert req.stop_price == 94000.0

    def test_limit_order(self, oc, client):
        """limit with a price submits a LimitOrderRequest."""
        assert oc.trade(side="buy", amount=1000, order_type="limit", limit_price=95000.0) is True
        assert client.requests[-1].limit_price == 95000.0

    @pytest.mark.parametrize("order_type,kwargs", [
        ("stop_limit", {"limit_price": 95000.0}),   # missing stop_price
        ("limit", {}),                              # missing limit_price
        ("bogus", {}),                              # unsupported order type
    ], ids=["stop_limit-no-stop", "limit-no-price", "unsupported-type"])
    def test_invalid_order_returns_false(self, oc, order_type, kwargs):
        """Missing required prices / an unknown order type return False, not raise."""
        assert oc.trade(side="buy", amount=1000, order_type=order_type, **kwargs) is False

    def test_api_failure_returns_false(self):
        """An exception from the broker is swallowed into False (why trade() returns a bool)."""
        client = MockTradingClient(simulate_failures=True)
        oc = AlpacaOrderClass(symbol="BTCUSD", trade_mode="notional", client=client)
        assert oc.trade(side="buy", amount=1000, order_type="market") is False

    @pytest.mark.parametrize("tif", ["gtc", "ioc"])
    def test_time_in_force_accepted(self, oc, client, tif):
        """Both GTC and IOC reach the request."""
        assert oc.trade(side="buy", amount=1000, order_type="market", time_in_force=tif) is True
        assert client.requests[-1].time_in_force.value == tif

    # --- bracket / OTO ---

    def test_stop_loss_only_creates_oto(self, oc, client):
        """stop_loss alone -> an OTO order carrying that stop."""
        assert oc.trade(side="buy", amount=1000, order_type="market", stop_loss=90000.0) is True
        req = client.requests[-1]
        assert req.order_class == OrderClass.OTO
        assert req.stop_loss.stop_price == 90000.0

    def test_take_profit_and_stop_loss_creates_bracket(self, oc, client):
        """Both TP and SL -> a BRACKET order carrying both legs."""
        assert oc.trade(side="buy", amount=1000, order_type="market",
                        take_profit=110000.0, stop_loss=90000.0) is True
        req = client.requests[-1]
        assert req.order_class == OrderClass.BRACKET
        assert req.take_profit.limit_price == 110000.0
        assert req.stop_loss.stop_price == 90000.0

    # --- notional-mode sell semantics ---

    def test_notional_sell_full_closes_ignoring_amount(self, oc, client):
        """Notional SELL does not submit a sized order: it full-closes the position and
        returns True, ignoring `amount`."""
        oc.trade(side="buy", amount=1000, order_type="market")
        assert oc.get_status()["position_status"] is not None
        assert len(client.orders) == 1

        assert oc.trade(side="sell", amount=999999, order_type="market") is True
        assert oc.get_status()["position_status"] is None    # fully closed
        assert len(client.orders) == 1                       # no sized sell order submitted

    # --- status ---

    def test_get_status_order_filled(self, oc):
        """After a market buy the order status is filled and a position exists."""
        oc.trade(side="buy", amount=1000, order_type="market")
        status = oc.get_status()

        assert isinstance(status["order_status"], OrderStatus)
        assert "filled" in str(status["order_status"].status).lower()
        assert isinstance(status["position_status"], PositionStatus)
        assert status["position_status"].qty > 0

    def test_get_clock(self, oc):
        """get_clock passes the broker clock through."""
        clock = oc.get_clock()
        assert hasattr(clock, "is_open")
        assert hasattr(clock, "next_close")
        assert hasattr(clock, "next_open")

    def test_get_open_orders_empty_after_fill(self, oc):
        """A filled market order leaves no open orders."""
        oc.trade(side="buy", amount=1000, order_type="market")
        assert oc.get_open_orders() == []

    # --- closing ---

    def test_close_position_full(self, oc):
        """close_position() flattens the position."""
        oc.trade(side="buy", amount=1000, order_type="market")

        assert oc.close_position() is True
        assert oc.get_status()["position_status"] is None

    def test_close_position_partial(self, oc):
        """close_position(qty=...) leaves the remainder open."""
        oc.trade(side="buy", amount=1000, order_type="market")
        initial_qty = oc.get_status()["position_status"].qty

        assert oc.close_position(qty=initial_qty / 2) is True
        assert oc.get_status()["position_status"].qty < initial_qty

    def test_close_all_positions(self, oc):
        """close_all_positions() reports success per symbol."""
        oc.trade(side="buy", amount=1000, order_type="market")

        results = oc.close_all_positions()
        assert results and all(v is True for v in results.values())

    # --- cancelling ---

    def test_cancel_open_orders_with_none_open(self, oc):
        """Cancelling with nothing open still succeeds."""
        assert oc.cancel_open_orders() is True

    def test_cancel_open_orders_cancels_open(self, oc, client):
        """An open order is actually cancelled.

        submit_order always fills, so the open order is injected directly.
        """
        order = MockOrder(symbol="BTCUSD", status=MockOrderStatus.NEW)
        client.orders[order.id] = order

        assert oc.cancel_open_orders() is True
        assert order.status == MockOrderStatus.CANCELED
