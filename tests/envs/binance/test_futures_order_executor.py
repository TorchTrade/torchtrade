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
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
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
            filters.append({"filterType": "LOT_SIZE", "stepSize": step, "minQty": step})
        symbols = [{"symbol": symbol, "filters": filters}]
        for other_symbol, other_step in extra_symbols:
            symbols.append({
                "symbol": other_symbol,
                "filters": [{"filterType": "LOT_SIZE", "stepSize": other_step, "minQty": other_step}],
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
        assert self._executor(step=step).round_quantity(quantity) == pytest.approx(expected)
        assert round(quantity, 3) != pytest.approx(expected)

    @pytest.mark.parametrize("quantity,step", [(0.29, "0.01"), (0.57, "0.01"), (0.58, "0.01")])
    def test_an_exact_multiple_is_not_shaved_by_a_whole_step(self, quantity, step):
        """`0.29 / 0.01` is 28.999999999999996, so a bare floor returns 0.28.

        207 of the first 400 exact multiples of the four common steps land just under an
        integer in binary. Without the epsilon this silently under-sizes a third of them.
        """
        assert self._executor(step=step).round_quantity(quantity) == pytest.approx(quantity)

    def test_floor_not_nearest(self):
        """Rounding UP can ask for more than the margin covers, and the venue rejects the
        whole order. bybit floors for the same reason."""
        assert self._executor(step="0.01").round_quantity(0.199) == pytest.approx(0.19)

    def test_other_symbols_round_to_their_own_step(self):
        """close_all_positions closes whatever the account holds, not just this symbol.

        Rounding ETHUSDT's quantity with BTCUSDT's step is the same class of bug this
        fixes, so the cache is keyed by symbol rather than holding one step.
        """
        ex = self._executor(step="0.001", extra_symbols=[("ETHUSDT", "0.01")])

        assert ex.round_quantity(1.2345) == pytest.approx(1.234)
        assert ex.round_quantity(1.2345, "ETHUSDT") == pytest.approx(1.23)

    def test_a_symbol_the_venue_does_not_list_refuses_to_construct(self):
        """Falling back is bit-identical to the bug being fixed.

        The venue answered and does not list this symbol, so every order would go out on
        the fallback precision -- and binance, unlike the other three, has no minQty check
        downstream to catch the result. A futures executor pointed at a spot-only symbol
        used to log two warnings and then trade against a symbol that does not exist.
        """
        with pytest.raises(ValueError, match="no LOT_SIZE"):
            self._executor(with_lot_size=False)

    def test_a_malformed_sibling_symbol_does_not_discard_the_cache(self):
        """One bad entry cost the whole parse, including the tick size, and whether it did
        depended on whether it sat before or after this symbol in the payload."""
        from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass

        client = MagicMock()
        client.futures_exchange_info = MagicMock(return_value={"symbols": [
            {"symbol": "JUNKUSDT", "filters": [{"filterType": "LOT_SIZE"}]},  # no stepSize
            {"symbol": "BTCUSDT", "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                {"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "0.01"},
            ]},
        ]})
        client.futures_change_leverage = MagicMock(return_value={})
        client.futures_change_margin_type = MagicMock(return_value={})

        ex = BinanceFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=10, client=client
        )
        assert ex._tick_size == pytest.approx(0.10)
        assert ex.round_quantity(0.12345) == pytest.approx(0.12)

    @pytest.mark.parametrize("step_str,expected", [
        ("0.001", (0.001, 3)),
        ("1", (1.0, 0)),
        ("0.00100000", (0.001, 3)),
        ("1E-8", (1e-8, 8)),
        ("1e-05", (1e-5, 5)),
    ], ids=["plain", "whole", "trailing-zeros", "sci-upper", "sci-lower"])
    def test_step_parsing_survives_scientific_notation(self, step_str, expected):
        """`"1E-8".split('.')` has no fractional part, so string surgery reported 0
        decimals and the final rounding then annihilated the quantity. okx carries a
        comment about being bitten by exactly this."""
        from torchtrade.envs.live.binance.order_executor import _step_and_decimals

        step, decimals = _step_and_decimals(step_str)
        assert step == pytest.approx(expected[0])
        assert decimals == expected[1]

    @pytest.mark.parametrize("step,quantity", [("1", 0.002), ("0.01", 0.004)],
                             ids=["whole-units", "hundredths"])
    def test_a_size_that_floors_away_is_refused_not_submitted(self, step, quantity):
        """Floored to zero, the old code submitted `quantity: 0.0` on all three legs.

        Reachable in every trade_mode -- `quantity` on any symbol whose step is coarser
        than the configured size, and `fractional`/`notional` whenever a small balance
        meets a six-figure price. The venue rejects it, but its error names the
        pre-rounding size, so both log lines point away from the cause.
        """
        ex = self._executor(step=step)
        with pytest.raises(ValueError, match="below the minimum"):
            ex._format_quantity(quantity)

    @pytest.mark.parametrize("step,quantity,expected", [
        ("1", 7.123, "7"),          # not "7.0" -- one decimal more than the venue defines
        ("0.01", 0.12345, "0.12"),
        ("0.0001", 0.12345, "0.1234"),
    ], ids=["whole-units", "hundredths", "ten-thousandths"])
    def test_quantity_is_formatted_to_the_venue_precision(self, step, quantity, expected):
        """A string, not a float. `str(7.0)` carries a decimal a symbol with
        quantityPrecision 0 does not define -- the documented shape of binance's -1111
        `Precision is over the maximum defined for this asset`. okx formats the same way
        for the same reason."""
        assert self._executor(step=step)._format_quantity(quantity) == expected

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
            assert q == "0.12", f"order went out at {q!r}, not the 0.01 step"

    def test_a_malformed_lot_size_does_not_take_the_tick_size_with_it(self):
        """Scoping the try to the SYMBOL made one bad filter abort that symbol's others.

        For the traded symbol that left `_tick_size` None, so SLTP stop prices go out at
        full float precision, draw binance -1111, and are swallowed by the non-fatal
        bracket handler -- a position open with no stop loss. LOT_SIZE is listed first
        here deliberately; the ordering is what decided whether the tick survived.
        """
        from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass

        client = MagicMock()
        client.futures_exchange_info = MagicMock(return_value={"symbols": [{
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.01"},   # minQty absent
                {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
            ],
        }]})
        client.futures_change_leverage = MagicMock(return_value={})
        client.futures_change_margin_type = MagicMock(return_value={})

        ex = BinanceFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=10, client=client
        )
        assert ex._tick_size == pytest.approx(0.10)
        assert ex._round_price(100.16) == pytest.approx(100.2)
        assert ex.round_quantity(0.12345) == pytest.approx(0.12)

    def test_a_malformed_payload_body_falls_back_rather_than_crashing(self):
        """A truncated or error-shaped body is a FAILED FETCH, not a config error.

        Reading `['symbols']` outside the try took down __init__ with a bare KeyError --
        neither the deliberate ValueError nor the fallback, and the ValueError's message
        even advertises "the payload changed shape" for a case that never reached it.
        """
        from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass

        client = MagicMock()
        client.futures_exchange_info = MagicMock(return_value={"code": -1121, "msg": "Invalid symbol"})
        client.futures_change_leverage = MagicMock(return_value={})
        client.futures_change_margin_type = MagicMock(return_value={})

        ex = BinanceFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=10, client=client
        )
        assert ex.round_quantity(0.12345) == pytest.approx(0.123)

    def test_a_minimum_larger_than_the_step_is_still_enforced(self):
        """`f['minQty']` raising skips the rest of the LOT_SIZE branch, silently.

        The step is cached on the line above, so construction succeeds and the only
        casualty is `_min_qtys` -- which leaves `minimum` degrading to `step`. A venue
        whose minQty exceeds its step (0.01 against 0.001 here) would then accept a 0.005
        order the venue rejects. Reading the field defensively keeps both.
        """
        from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass

        client = MagicMock()
        client.futures_exchange_info = MagicMock(return_value={"symbols": [{
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.01"},
            ],
        }]})
        client.futures_change_leverage = MagicMock(return_value={})
        client.futures_change_margin_type = MagicMock(return_value={})

        ex = BinanceFuturesOrderClass(
            symbol="BTCUSDT", trade_mode="quantity", demo=True, leverage=10, client=client
        )
        assert ex._min_qtys["BTCUSDT"] == pytest.approx(0.01)
        with pytest.raises(ValueError, match="below the minimum"):
            ex._format_quantity(0.005)
        assert ex._format_quantity(0.012) == "0.012"

    @pytest.mark.parametrize("residual", [0.0009, 0.0006, 0.0004],
                             ids=["just-under", "half", "quarter"])
    def test_a_sub_step_residual_is_still_closable(self, residual):
        """The minimum gate is an OPENS argument, and wiring it into closes removed a
        close that main performed.

        A reduceOnly order is clamped by the venue to the actual position, so one at the
        minimum DOES fill. Sub-step residuals are real -- partial fill of a close, ADL, a
        liquidation remnant, a venue step change -- and `binance/base.py` discards
        `close_position()`'s return on reset, so a refusal starts the episode against a
        position the env believes it closed.
        """
        ex = self._executor(step="0.001")
        assert ex._format_quantity(residual, reduce_only=True) == "0.001"
        with pytest.raises(ValueError, match="below the minimum"):
            ex._format_quantity(residual)          # opening is still refused

    @pytest.mark.parametrize("quantity,step,expected", [
        (-7.5, "1", -7.0),        # math.floor gives -8.0: MORE than the caller sized
        (-0.295, "0.01", -0.29),
        (0.295, "0.01", 0.29),
    ], ids=["negative-whole", "negative-fractional", "positive-control"])
    def test_rounding_moves_toward_zero_for_either_sign(self, quantity, step, expected):
        """`math.floor` moves a negative AWAY from zero, so the public helper returned more
        than was asked for -- the one thing its docstring promises not to do. No live call
        site passes a negative today; it is public API with no stated precondition."""
        assert self._executor(step=step).round_quantity(quantity) == pytest.approx(expected)

    @pytest.mark.parametrize("quantity", [19000.01, 19000.028, 19000.046])
    def test_a_large_exact_multiple_survives_repeated_rounding(self, quantity):
        """A fixed 1e-9 tolerance stops covering the binary error once quantity/step
        passes ~1e7, and this PR applies the rounding two or three times to one number:
        env sizing, then the delta, then _format_quantity.

        These are exact multiples of the 0.001 step. Under a fixed tolerance 19000.01
        becomes 19000.009 on the first application and 19000.008 on the second -- a step
        lost per pass. main could not do this: it floored once, then rounded to NEAREST.
        """
        ex = self._executor(step="0.001")
        once = ex.round_quantity(quantity)
        # abs=, not the default relative tolerance: pytest.approx defaults to rel=1e-6,
        # which at a quantity of 19000 is 0.019 -- nineteen times the step this is meant
        # to detect. The first version of this assertion could not fail.
        assert once == pytest.approx(quantity, abs=1e-9), "shaved a step off an exact multiple"
        assert ex.round_quantity(once) == pytest.approx(quantity, abs=1e-9), "not idempotent"
