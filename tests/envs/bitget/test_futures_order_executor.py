"""Tests for BitgetFuturesOrderClass with CCXT."""

import pytest
from unittest.mock import MagicMock, patch


class TestBitgetFuturesOrderClass:
    """Tests for BitgetFuturesOrderClass using CCXT."""

    @pytest.fixture
    def order_executor(self, mock_ccxt_client):
        """Create order executor with mock CCXT client."""
        from torchtrade.envs.live.bitget.order_executor import (
            BitgetFuturesOrderClass,
            MarginMode,
            PositionMode,
        )

        with patch('torchtrade.envs.live.bitget.order_executor.ccxt.bitget', return_value=mock_ccxt_client):
            executor = BitgetFuturesOrderClass(
                symbol="BTC/USDT:USDT",
                trade_mode="quantity",
                demo=True,
                leverage=10,
                margin_mode=MarginMode.ISOLATED,
                position_mode=PositionMode.ONE_WAY,
                api_key="test_key",
                api_secret="test_secret",
                passphrase="test_pass",
            )
            executor.client = mock_ccxt_client
            return executor

    def test_initialization(self, order_executor, mock_ccxt_client):
        """Test order executor initialization."""
        assert order_executor.symbol == "BTC/USDT:USDT"
        assert order_executor.leverage == 10
        assert order_executor.demo is True
        assert order_executor.product_type == "USDT-FUTURES"

    def test_symbol_normalization(self, mock_ccxt_client):
        """Test that symbol formats are accepted."""
        from torchtrade.envs.live.bitget.order_executor import (
            BitgetFuturesOrderClass,
        )

        with patch('torchtrade.envs.live.bitget.order_executor.ccxt.bitget', return_value=mock_ccxt_client):
            # CCXT format should work
            executor = BitgetFuturesOrderClass(
                symbol="BTC/USDT:USDT",
                trade_mode="quantity",
                api_key="test_key",
                api_secret="test_secret",
                passphrase="test_pass",
            )
            executor.client = mock_ccxt_client
            assert executor.symbol == "BTC/USDT:USDT"

    def test_product_type_demo(self, mock_ccxt_client):
        """Test that demo=True uses USDT-FUTURES product type."""
        from torchtrade.envs.live.bitget.order_executor import (
            BitgetFuturesOrderClass,
        )

        with patch('torchtrade.envs.live.bitget.order_executor.ccxt.bitget', return_value=mock_ccxt_client):
            executor = BitgetFuturesOrderClass(
                symbol="BTC/USDT:USDT",
                product_type="USDT-FUTURES",
                demo=True,
                api_key="test_key",
                api_secret="test_secret",
                passphrase="test_pass",
            )
            executor.client = mock_ccxt_client
            assert executor.product_type == "USDT-FUTURES"

    def test_market_buy_order(self, order_executor, mock_ccxt_client):
        """Test placing a market buy order."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="market",
        )

        assert success is True
        mock_ccxt_client.create_order.assert_called()

        # Check call arguments
        call_args = mock_ccxt_client.create_order.call_args
        assert call_args[1]["symbol"] == "BTC/USDT:USDT"
        assert call_args[1]["side"] == "buy"
        assert call_args[1]["type"] == "market"
        assert call_args[1]["amount"] == 0.001
        # Should include marginMode in params
        assert "marginMode" in call_args[1]["params"]

    def test_market_sell_order(self, order_executor, mock_ccxt_client):
        """Test placing a market sell order (short)."""
        success = order_executor.trade(
            side="sell",
            quantity=0.001,
            order_type="market",
        )

        assert success is True
        call_args = mock_ccxt_client.create_order.call_args
        assert call_args[1]["side"] == "sell"

    def test_limit_order(self, order_executor, mock_ccxt_client):
        """Test placing a limit order."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="limit",
            limit_price=49000.0,
        )

        assert success is True
        call_args = mock_ccxt_client.create_order.call_args
        assert call_args[1]["type"] == "limit"
        assert call_args[1]["price"] == 49000.0

    def test_limit_order_without_price_fails(self, order_executor, mock_ccxt_client):
        """Test that limit order without price is handled."""
        # Mock create_order to raise an exception for limit order without price
        mock_ccxt_client.create_order = MagicMock(side_effect=Exception("Price is required for limit orders"))

        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="limit",
            # Missing limit_price
        )
        assert success is False

    def test_stop_market_order(self, order_executor, mock_ccxt_client):
        """Test placing a stop market order."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="stop_market",
            stop_price=48000.0,
        )

        assert success is True
        # Should use create_order with stopPrice param
        assert mock_ccxt_client.create_order.called

    def test_stop_order_without_price_fails(self, order_executor):
        """Test that stop order without price returns False."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="stop_market",
            # Missing stop_price
        )
        assert success is False

    def test_bracket_order_with_tp_sl(self, order_executor, mock_ccxt_client):
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
        mock_ccxt_client.create_order_with_take_profit_and_stop_loss.assert_called_once()

    @pytest.mark.parametrize("raw_tp,raw_sl", [
        (82622.2122, 84291.4358),   # Unrounded prices from percentage calc
        (51234.5678, 48765.4321),   # Many decimal places
    ])
    def test_bracket_order_prices_rounded(self, order_executor, mock_ccxt_client, raw_tp, raw_sl):
        """SL/TP prices must be rounded via CCXT price_to_precision before submission."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="market",
            take_profit=raw_tp,
            stop_loss=raw_sl,
        )

        assert success is True
        call_kwargs = mock_ccxt_client.create_order_with_take_profit_and_stop_loss.call_args[1]
        # price_to_precision returns string, _round_price converts back to float
        assert call_kwargs["takeProfit"] == round(raw_tp, 1)
        assert call_kwargs["stopLoss"] == round(raw_sl, 1)

    def test_get_status(self, order_executor, mock_ccxt_client):
        """Test getting order and position status."""
        # Place an order first
        order_executor.trade(side="buy", quantity=0.001)

        status = order_executor.get_status()

        assert "position_status" in status
        # Position status should have attributes
        if status["position_status"]:
            assert hasattr(status["position_status"], "qty")
            assert hasattr(status["position_status"], "entry_price")

    def test_get_status_no_position(self, order_executor, mock_ccxt_client):
        """Test get_status when no position exists."""
        # Mock empty positions
        mock_ccxt_client.fetch_positions = MagicMock(return_value=[{
            "symbol": "BTC/USDT:USDT",
            "contracts": 0,
            "side": "long",
        }])

        status = order_executor.get_status()
        assert status["position_status"] is None

    def test_get_status_short_position(self, order_executor, mock_ccxt_client):
        """Test get_status with short position (negative qty)."""
        # Mock short position
        mock_ccxt_client.fetch_positions = MagicMock(return_value=[{
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

    def test_get_account_balance(self, order_executor, mock_ccxt_client):
        """Test getting account balance."""
        balance = order_executor.get_account_balance()

        assert "total_wallet_balance" in balance
        assert "available_balance" in balance
        assert "total_unrealized_profit" in balance
        assert "total_margin_balance" in balance
        assert balance["total_maintenance_margin"] is None

        assert balance["total_wallet_balance"] == 1000.0
        assert balance["available_balance"] == 900.0

    @pytest.mark.parametrize("info,expected", [
        ([
            {"marginCoin": "USDC", "assetMode": "union", "unionMm": "99"},
            {"marginCoin": "USDT", "assetMode": "union", "unionMm": "4.4"},
        ], 4.4),
        ([{"marginCoin": "USDT", "assetMode": "union", "unionMm": "0"}], 0.0),
        ([{
            "marginCoin": "USDT",
            "assetMode": "single",
            "crossedRiskRate": "0.25",
        }], None),
        ({"marginCoin": "USDT", "assetMode": "union", "unionMm": "2.5"}, 2.5),
    ], ids=["list-union", "explicit-zero", "single-unavailable", "mapping-compat"])
    def test_get_account_balance_parses_union_maintenance(
        self, order_executor, mock_ccxt_client, info, expected
    ):
        raw_balance = mock_ccxt_client.fetch_balance.return_value.copy()
        raw_balance["info"] = info
        mock_ccxt_client.fetch_balance.return_value = raw_balance

        assert order_executor.get_account_balance()["total_maintenance_margin"] == expected

    def test_get_mark_price(self, order_executor, mock_ccxt_client):
        """Test getting mark price."""
        price = order_executor.get_mark_price()
        assert price == 50100.0

    def test_close_position(self, order_executor, mock_ccxt_client):
        """Test closing a position."""
        success = order_executor.close_position()

        assert success is True
        mock_ccxt_client.create_order.assert_called()

        # Should place a close order with reduceOnly
        call_args = mock_ccxt_client.create_order.call_args
        assert call_args[1]["params"]["reduceOnly"] is True

    def test_close_position_no_position(self, order_executor, mock_ccxt_client):
        """Test closing when no position exists."""
        # Mock no position
        mock_ccxt_client.fetch_positions = MagicMock(return_value=[{
            "symbol": "BTC/USDT:USDT",
            "contracts": 0,
            "side": "long",
        }])

        success = order_executor.close_position()
        assert success is True  # Should succeed without error

    def test_cancel_open_orders(self, order_executor, mock_ccxt_client):
        """Test cancelling all open orders."""
        # Mock some open orders
        mock_ccxt_client.fetch_open_orders = MagicMock(return_value=[
            {"id": "123", "symbol": "BTC/USDT:USDT"},
            {"id": "456", "symbol": "BTC/USDT:USDT"},
        ])

        success = order_executor.cancel_open_orders()

        assert success is True
        # Should call cancel for each order
        assert mock_ccxt_client.cancel_order.call_count == 2

    def test_set_margin_mode(self, order_executor, mock_ccxt_client):
        """Test changing margin mode."""
        from torchtrade.envs.live.bitget.order_executor import MarginMode

        success = order_executor.set_margin_mode(MarginMode.CROSSED)

        assert success is True
        assert order_executor.margin_mode == MarginMode.CROSSED
        # Note: set_margin_mode may not work reliably on Bitget, but we test the call
        mock_ccxt_client.set_margin_mode.assert_called()

    def test_round_price_fallback_when_precision_unavailable(self, mock_ccxt_client):
        """When price_to_precision fails, prices must pass through unmodified."""
        from torchtrade.envs.live.bitget.order_executor import BitgetFuturesOrderClass

        # Make load_markets and price_to_precision fail
        mock_ccxt_client.load_markets = MagicMock(side_effect=Exception("Network error"))
        mock_ccxt_client.price_to_precision = MagicMock(side_effect=Exception("No market data"))

        with patch('torchtrade.envs.live.bitget.order_executor.ccxt.bitget', return_value=mock_ccxt_client):
            executor = BitgetFuturesOrderClass(
                symbol="BTC/USDT:USDT", api_key="k", api_secret="s", passphrase="p",
            )
            executor.client = mock_ccxt_client

        assert executor._round_price(82622.2122) == 82622.2122

    def test_tp_only_order_price_rounded(self, order_executor, mock_ccxt_client):
        """TP-only order must have its price rounded before submission."""
        success = order_executor.trade(
            side="buy", quantity=0.001, order_type="market",
            take_profit=51234.5678, stop_loss=None,
        )
        assert success is True
        # TP-only path uses create_order with price param
        call_args = mock_ccxt_client.create_order.call_args_list
        tp_call = call_args[-1]  # Last create_order call is the TP order
        assert tp_call[1]["price"] == 51234.6  # Rounded to 1 decimal

    def test_sl_only_order_price_rounded(self, order_executor, mock_ccxt_client):
        """SL-only order must have its stopPrice rounded before submission."""
        success = order_executor.trade(
            side="buy", quantity=0.001, order_type="market",
            take_profit=None, stop_loss=48765.4321,
        )
        assert success is True
        call_args = mock_ccxt_client.create_stop_market_order.call_args
        assert call_args[1]["stopPrice"] == 48765.4  # Rounded to 1 decimal

    def test_trade_returns_true_when_tp_only_fails(self, order_executor, mock_ccxt_client):
        """Main order success must not be masked by TP-only follow-up failure."""
        mock_ccxt_client.create_order.side_effect = [
            {"id": "main_order_123"},  # Main order succeeds
            Exception("Precision error"),  # TP follow-up fails
        ]

        success = order_executor.trade(
            side="buy", quantity=0.001, order_type="market",
            take_profit=52000.0, stop_loss=None,
        )

        assert success is True

    def test_trade_returns_true_when_sl_only_fails(self, order_executor, mock_ccxt_client):
        """Main order success must not be masked by SL-only follow-up failure."""
        mock_ccxt_client.create_stop_market_order = MagicMock(
            side_effect=Exception("Precision error")
        )

        success = order_executor.trade(
            side="buy", quantity=0.001, order_type="market",
            take_profit=None, stop_loss=48000.0,
        )

        assert success is True

    def test_trade_failure_handling(self, order_executor, mock_ccxt_client):
        """Test that trade failures are handled gracefully."""
        # Mock API failure
        mock_ccxt_client.create_order = MagicMock(side_effect=Exception("API Error"))

        success = order_executor.trade(side="buy", quantity=0.001)

        assert success is False  # Should return False, not raise

    def test_get_open_orders(self, order_executor, mock_ccxt_client):
        """Test getting open orders."""
        orders = order_executor.get_open_orders()
        assert isinstance(orders, list)

    def test_position_mode_enum(self):
        """Test PositionMode enum values."""
        from torchtrade.envs.live.bitget.order_executor import PositionMode

        assert PositionMode.ONE_WAY.value == "one_way_mode"
        assert PositionMode.HEDGE.value == "hedge_mode"

    def test_margin_mode_enum(self):
        """Test MarginMode enum values."""
        from torchtrade.envs.live.bitget.order_executor import MarginMode

        assert MarginMode.ISOLATED.value == "isolated"
        assert MarginMode.CROSSED.value == "crossed"

    def test_trade_mode_values(self):
        """Test TradeMode string literal values."""
        from torchtrade.envs.core.common import validate_trade_mode

        # Test that validation accepts valid values
        assert validate_trade_mode("fractional") == "fractional"
        assert validate_trade_mode("quantity") == "quantity"
        assert validate_trade_mode("notional") == "notional"
        assert validate_trade_mode("FRACTIONAL") == "fractional"  # Case-insensitive
        assert validate_trade_mode("QUANTITY") == "quantity"  # Case-insensitive
        assert validate_trade_mode("NOTIONAL") == "notional"  # Case-insensitive

        # Test that validation rejects invalid values
        with pytest.raises(ValueError):
            validate_trade_mode("invalid")


class TestBitgetFuturesOrderClassIntegration:
    """Integration tests that would require actual API (skipped by default)."""

    @pytest.mark.skip(reason="Requires live Bitget API connection and credentials")
    def test_live_order_placement(self):
        """Test placing a real order on Bitget testnet."""
        import os
        from torchtrade.envs.live.bitget.order_executor import (
            BitgetFuturesOrderClass,
        )

        executor = BitgetFuturesOrderClass(
            symbol="BTC/USDT:USDT",
            trade_mode="quantity",
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


class TestBitgetLotSize:
    """get_lot_size must read real lot constraints from CCXT market info, not hardcode them.

    Regression: env previously hardcoded min_qty/step=0.001, which is 10x the real
    BTC/USDT:USDT step (0.0001) and would mis-size or wrongly reject orders.
    """

    @pytest.fixture
    def order_executor(self, mock_ccxt_client):
        from torchtrade.envs.live.bitget.order_executor import BitgetFuturesOrderClass
        with patch('torchtrade.envs.live.bitget.order_executor.ccxt.bitget', return_value=mock_ccxt_client):
            executor = BitgetFuturesOrderClass(
                symbol="BTC/USDT:USDT", trade_mode="fractional",
                api_key="k", api_secret="s", passphrase="p",
            )
            executor.client = mock_ccxt_client
            return executor

    def test_reads_market_info_and_caches(self, order_executor, mock_ccxt_client):
        mock_ccxt_client.market.reset_mock()
        lot = order_executor.get_lot_size()
        # min_notional is bitget's minTradeUSDT, which CCXT normalises to limits.cost.min.
        # It was fetched with the rest of the market info and dropped, so an order under
        # the floor was submitted and rejected (#414).
        assert lot == {"min_qty": 0.0001, "qty_step": 0.0001, "min_notional": 5.0}
        order_executor.get_lot_size()  # second call served from cache
        mock_ccxt_client.market.assert_called_once()

    def test_falls_back_to_default_on_error(self, order_executor, mock_ccxt_client):
        mock_ccxt_client.market = MagicMock(side_effect=Exception("no market info"))
        assert order_executor.get_lot_size() == {"min_qty": 0.001, "qty_step": 0.001, "min_notional": 0.0}

    @pytest.mark.parametrize("amount,expected", [
        (0.00037, 0.0003),   # truncates down to the 0.0001 step (never rounds up -> margin-safe)
        (0.0003, 0.0003),    # exact multiple preserved (the old int()/str() floor could lose a step)
        (0.00125, 0.0012),   # truncates, not rounds-to-nearest
    ])
    def test_round_amount_floors_to_lot_step(self, order_executor, amount, expected):
        """_round_amount delegates to CCXT amount_to_precision (truncation), replacing the
        fragile string-parse floor that mishandled exact multiples and sci-notation steps."""
        assert order_executor._round_amount(amount) == pytest.approx(expected)

    def test_round_amount_floors_to_step_on_error(self, order_executor, mock_ccxt_client):
        """If CCXT precision fails, floor to the cached lot step (never submit a raw
        unaligned amount the exchange may reject)."""
        mock_ccxt_client.amount_to_precision = MagicMock(side_effect=Exception("boom"))
        # qty_step=0.0001 -> 0.00037 floors to 0.0003
        assert order_executor._round_amount(0.00037) == pytest.approx(0.0003)


class TestBitgetCancelReportsFailure:
    """bitget was the only venue whose cancel_open_orders could not report failure (#288)."""

    @pytest.fixture
    def order_executor(self, mock_ccxt_client):
        from torchtrade.envs.live.bitget.order_executor import (
            BitgetFuturesOrderClass, MarginMode, PositionMode,
        )
        with patch('torchtrade.envs.live.bitget.order_executor.ccxt.bitget', return_value=mock_ccxt_client):
            return BitgetFuturesOrderClass(
                symbol="BTC/USDT:USDT", trade_mode="quantity", demo=True, leverage=10,
                margin_mode=MarginMode.ISOLATED, position_mode=PositionMode.ONE_WAY,
                api_key="k", api_secret="s", passphrase="p",
            )

    @pytest.mark.parametrize("failures,expected", [
        (0, True),   # nothing failed
        (1, False),  # one of two failed -- brackets remain
        (2, False),  # everything failed
    ], ids=["all-cancelled", "one-failed", "all-failed"])
    def test_a_failed_cancel_is_reported_not_just_logged(
        self, order_executor, mock_ccxt_client, failures, expected
    ):
        """It logged each failure, then returned True regardless.

        The three other venues return False, so `_reset`'s "proceeding with potentially
        stale orders" warning could fire for them and never for bitget -- live SL/TP
        brackets left attached to a position the new episode believes is clean, with the
        one signal that would have said so hard-wired to success.
        """
        orders = [{"id": "a"}, {"id": "b"}]
        order_executor.get_open_orders = MagicMock(return_value=orders)
        calls = {"n": 0}

        def cancel(order_id, symbol):
            calls["n"] += 1
            if calls["n"] <= failures:
                raise RuntimeError("venue refused")

        mock_ccxt_client.cancel_order = MagicMock(side_effect=cancel)
        assert order_executor.cancel_open_orders() is expected
        assert calls["n"] == len(orders), "a failed cancel must not abort the remaining ones"
