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

        # Mock exchange info for price precision
        client.futures_exchange_info = MagicMock(return_value={
            "symbols": [{
                "symbol": "BTCUSDT",
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                ],
            }]
        })

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
        """Test placing a limit order with price rounding."""
        success = order_executor.trade(
            side="BUY",
            quantity=0.001,
            order_type="limit",
            limit_price=49000.1234,
        )

        assert success is True

        call_kwargs = mock_client.futures_create_order.call_args[1]
        assert call_kwargs["type"] == "LIMIT"
        assert call_kwargs["price"] == 49000.1  # Rounded to 1 decimal (tick=0.10)

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
        """TP-only order must have its stopPrice rounded before submission."""
        success = order_executor.trade(
            side="BUY",
            quantity=0.001,
            order_type="market",
            take_profit=52000.1234,
        )

        assert success is True
        assert mock_client.futures_create_order.call_count >= 2
        tp_call = mock_client.futures_create_order.call_args_list[1][1]
        assert tp_call["stopPrice"] == 52000.1  # Rounded to 1 decimal

    def test_order_with_stop_loss(self, order_executor, mock_client):
        """SL-only order must have its stopPrice rounded before submission."""
        success = order_executor.trade(
            side="BUY",
            quantity=0.001,
            order_type="market",
            stop_loss=48000.5678,
        )

        assert success is True
        assert mock_client.futures_create_order.call_count >= 2
        sl_call = mock_client.futures_create_order.call_args_list[1][1]
        assert sl_call["stopPrice"] == 48000.6  # Rounded to 1 decimal

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

    @pytest.mark.parametrize("raw_tp,raw_sl,expected_tp,expected_sl", [
        (84291.4358, 82622.2122, 84291.4, 82622.2),  # BTC at ~$83k: TP +1%, SL -1%
        (50000.0, 49000.0, 50000.0, 49000.0),        # Already rounded
        (50000.15, 49999.96, 50000.2, 50000.0),      # Quantize to nearest tick
        (83456.78123, 82621.2147, 83456.8, 82621.2),  # Many decimals
    ])
    def test_bracket_order_prices_rounded_to_tick_size(self, order_executor, mock_client, raw_tp, raw_sl, expected_tp, expected_sl):
        """SL/TP prices must be quantized to exchange tick size before submission."""
        success = order_executor.trade(
            side="BUY",
            quantity=0.001,
            order_type="market",
            take_profit=raw_tp,
            stop_loss=raw_sl,
        )

        assert success is True

        # Find TP and SL calls by order type (not by index, to avoid brittleness)
        calls = mock_client.futures_create_order.call_args_list
        tp_call = next(c for c in calls if c[1].get("type") == "TAKE_PROFIT_MARKET")
        sl_call = next(c for c in calls if c[1].get("type") == "STOP_MARKET")
        assert tp_call[1]["stopPrice"] == expected_tp
        assert sl_call[1]["stopPrice"] == expected_sl

    def test_tick_size_fetched_at_init(self, order_executor):
        """Tick size should be cached from exchange info at init."""
        assert order_executor._tick_size == 0.1
        assert order_executor._tick_decimals == 1

    def test_round_price_without_precision(self, mock_client):
        """When tick size fetch fails, prices pass through unmodified."""
        from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass

        # Make exchange info fail
        mock_client.futures_exchange_info = MagicMock(side_effect=Exception("API down"))

        executor = BinanceFuturesOrderClass(
            symbol="BTCUSDT", client=mock_client,
        )
        assert executor._tick_size is None
        assert executor._round_price(82622.2122) == 82622.2122  # Unmodified

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


    def test_trade_returns_true_when_tp_fails(self, order_executor, mock_client):
        """Main order success must not be masked by SL/TP failure (stacking bug root cause)."""
        mock_client.futures_create_order = MagicMock(side_effect=[
            {"orderId": 12345, "status": "FILLED"},  # Main order succeeds
            Exception("Precision is over the maximum defined for this asset"),  # TP fails
            Exception("Precision is over the maximum defined for this asset"),  # SL fails
        ])

        success = order_executor.trade(
            side="BUY",
            quantity=0.001,
            order_type="market",
            take_profit=52000.1234,
            stop_loss=48000.5678,
        )

        # Main order succeeded, so trade() must return True
        assert success is True
        # bracket_status should reflect the failures
        assert order_executor.bracket_status["tp_placed"] is False
        assert order_executor.bracket_status["sl_placed"] is False

    def test_trade_returns_false_when_main_order_fails(self, order_executor, mock_client):
        """When the main order itself fails, trade() must return False."""
        mock_client.futures_create_order = MagicMock(
            side_effect=Exception("Insufficient margin")
        )

        success = order_executor.trade(
            side="BUY",
            quantity=0.001,
            order_type="market",
            take_profit=52000.0,
            stop_loss=48000.0,
        )

        assert success is False


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


class TestBinanceLotSizeRounding:
    """#271: binance never parsed LOT_SIZE, so every quantity went out as round(q, 3).

    Bitget, bybit and okx all fetch the real per-symbol step. Binance was the exception,
    and any symbol whose step is not exactly three decimals got a silently wrong size --
    a rejected order, or a mis-sized position. The mock in this file has carried a
    LOT_SIZE filter since before the wiring existed, and nothing read it.
    """

    @staticmethod
    def _executor(step="0.01", symbol="BTCUSDT", extra_symbols=(), with_lot_size=True):
        from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass

        filters = [{"filterType": "PRICE_FILTER", "tickSize": "0.10"}]
        if with_lot_size:
            filters.append({"filterType": "LOT_SIZE", "stepSize": step})
        symbols = [{"symbol": symbol, "filters": filters}]
        for other_symbol, other_step in extra_symbols:
            symbols.append({
                "symbol": other_symbol,
                "filters": [{"filterType": "LOT_SIZE", "stepSize": other_step}],
            })

        client = MagicMock()
        client.futures_exchange_info = MagicMock(return_value={"symbols": symbols})
        client.futures_change_leverage = MagicMock(return_value={})
        client.futures_change_margin_type = MagicMock(return_value={})
        return BinanceFuturesOrderClass(
            symbol=symbol, trade_mode="quantity", demo=True, leverage=10, client=client
        )

    @pytest.mark.parametrize("step,quantity,expected", [
        # round(q, 3) would give 0.123 -- a size the venue rejects outright.
        ("0.01", 0.12345, 0.12),
        # A whole-unit step. round(q, 3) gives 7.123, which is not a multiple of anything.
        ("1", 7.123, 7.0),
        # Finer than three decimals: round(q, 3) silently discards the tail.
        ("0.0001", 0.12345, 0.1234),
    ], ids=["coarser-than-3dp", "whole-units", "finer-than-3dp"])
    def test_quantity_is_floored_to_the_venue_step(self, step, quantity, expected):
        """Each cell is chosen so `round(quantity, 3)` gives a different, invalid answer."""
        assert self._executor(step=step)._round_quantity(quantity) == pytest.approx(expected)
        assert round(quantity, 3) != pytest.approx(expected)

    @pytest.mark.parametrize("quantity,step", [(0.29, "0.01"), (0.57, "0.01"), (0.58, "0.01")])
    def test_an_exact_multiple_is_not_shaved_by_a_whole_step(self, quantity, step):
        """`0.29 / 0.01` is 28.999999999999996, so a bare floor returns 0.28.

        207 of the first 400 exact multiples of the four common steps land just under an
        integer in binary. Without the epsilon this silently under-sizes a third of them.
        """
        assert self._executor(step=step)._round_quantity(quantity) == pytest.approx(quantity)

    def test_floor_not_nearest(self):
        """Rounding UP can ask for more than the margin covers, and the venue rejects the
        whole order. bybit floors for the same reason."""
        assert self._executor(step="0.01")._round_quantity(0.199) == pytest.approx(0.19)

    def test_other_symbols_round_to_their_own_step(self):
        """close_all_positions closes whatever the account holds, not just this symbol.

        Rounding ETHUSDT's quantity with BTCUSDT's step is the same class of bug this
        fixes, so the cache is keyed by symbol rather than holding one step.
        """
        ex = self._executor(step="0.001", extra_symbols=[("ETHUSDT", "0.01")])

        assert ex._round_quantity(1.2345) == pytest.approx(1.234)
        assert ex._round_quantity(1.2345, "ETHUSDT") == pytest.approx(1.23)

    def test_missing_lot_size_falls_back_rather_than_crashing(self):
        """A venue response without LOT_SIZE must not take down the executor; it warns and
        uses the previous hardcoded precision."""
        assert self._executor(with_lot_size=False)._round_quantity(0.12345) == pytest.approx(0.123)

    @pytest.mark.parametrize("take_profit,stop_loss", [
        (None, None),
        (60000.0, 40000.0),
    ], ids=["plain", "with-brackets"])
    def test_submitted_orders_carry_the_step_rounded_quantity(self, take_profit, stop_loss):
        """The wiring, not the helper.

        Reverting all five call sites to `round(quantity, 3)` left every other cell in this
        class green -- they exercise `_round_quantity` directly and say nothing about
        whether an order ever reaches it. That is the same shape as the LOT_SIZE mock this
        file carried for a year without a reader.

        The bracket variant is here because the TP/SL legs are separate call sites from the
        main order and were rounded separately.
        """
        ex = self._executor(step="0.01")
        ex.client.futures_create_order = MagicMock(return_value={"orderId": 1, "status": "NEW"})

        assert ex.trade(
            side="buy", quantity=0.12345, order_type="market",
            take_profit=take_profit, stop_loss=stop_loss,
        )

        quantities = [
            call.kwargs["quantity"]
            for call in ex.client.futures_create_order.call_args_list
            if "quantity" in call.kwargs
        ]
        assert quantities, "no order carried a quantity"
        for q in quantities:
            assert q == pytest.approx(0.12), f"order went out at {q}, not the 0.01 step"
