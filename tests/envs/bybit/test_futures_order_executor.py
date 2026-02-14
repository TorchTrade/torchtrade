"""Tests for BybitFuturesOrderClass with pybit."""

import pytest
from unittest.mock import MagicMock


class TestBybitFuturesOrderClass:
    """Tests for BybitFuturesOrderClass using pybit."""

    @pytest.fixture
    def order_executor(self, mock_pybit_client):
        """Create order executor with mock pybit client."""
        from torchtrade.envs.live.bybit.order_executor import (
            BybitFuturesOrderClass,
            MarginMode,
            PositionMode,
        )

        executor = BybitFuturesOrderClass(
            symbol="BTCUSDT",
            trade_mode="quantity",
            demo=True,
            leverage=10,
            margin_mode=MarginMode.ISOLATED,
            position_mode=PositionMode.ONE_WAY,
            api_key="test_key",
            api_secret="test_secret",
            client=mock_pybit_client,
        )
        return executor

    @pytest.mark.parametrize("symbol,expected", [
        ("BTCUSDT", "BTCUSDT"),
        ("BTC/USDT", "BTCUSDT"),
        ("BTC/USDT:USDT", "BTCUSDT"),
    ])
    def test_symbol_normalization(self, mock_pybit_client, symbol, expected):
        """Test that symbol formats are normalized."""
        from torchtrade.envs.live.bybit.order_executor import BybitFuturesOrderClass

        executor = BybitFuturesOrderClass(
            symbol=symbol,
            client=mock_pybit_client,
        )
        assert executor.symbol == expected

    @pytest.mark.parametrize("side,expected_side", [
        ("buy", "Buy"),
        ("sell", "Sell"),
    ])
    def test_market_order(self, order_executor, mock_pybit_client, side, expected_side):
        """Test placing a market order (buy or sell)."""
        success = order_executor.trade(side=side, quantity=0.001, order_type="market")

        assert success is True
        mock_pybit_client.place_order.assert_called()
        call_kwargs = mock_pybit_client.place_order.call_args[1]
        assert call_kwargs["symbol"] == "BTCUSDT"
        assert call_kwargs["side"] == expected_side
        assert call_kwargs["orderType"] == "Market"
        assert call_kwargs["qty"] == "0.001"

    def test_bracket_order_with_tp_sl(self, order_executor, mock_pybit_client):
        """Test placing bracket order with take profit and stop loss."""
        success = order_executor.trade(
            side="buy",
            quantity=0.001,
            order_type="market",
            take_profit=51000.0,
            stop_loss=49000.0,
        )

        assert success is True
        call_kwargs = mock_pybit_client.place_order.call_args[1]
        assert call_kwargs["takeProfit"] == "51000.0"
        assert call_kwargs["stopLoss"] == "49000.0"

    def test_get_status_with_position(self, order_executor):
        """Test getting position status."""
        status = order_executor.get_status()

        assert "position_status" in status
        pos = status["position_status"]
        assert pos is not None
        assert pos.qty > 0  # Long position
        assert pos.entry_price == 50000.0
        assert pos.mark_price == 50100.0
        assert pos.leverage == 10

    def test_get_status_no_position(self, order_executor, mock_pybit_client):
        """Test get_status when no position exists."""
        mock_pybit_client.get_positions = MagicMock(return_value={
            "retCode": 0,
            "result": {"list": [{"symbol": "BTCUSDT", "size": "0", "side": "Buy"}]},
        })

        status = order_executor.get_status()
        assert status["position_status"] is None

    def test_get_status_short_position(self, order_executor, mock_short_position):
        """Test get_status with short position (negative qty)."""
        order_executor.client = mock_short_position
        status = order_executor.get_status()
        assert status["position_status"].qty < 0

    def test_get_account_balance(self, order_executor):
        """Test getting account balance."""
        balance = order_executor.get_account_balance()

        assert balance["total_wallet_balance"] == 1000.0
        assert balance["available_balance"] == 900.0
        assert "total_unrealized_profit" in balance
        assert "total_margin_balance" in balance

    @pytest.mark.parametrize("position_fixture,expected_side", [
        ("mock_pybit_client", "Sell"),      # Long position -> close with Sell
        ("mock_short_position", "Buy"),     # Short position -> close with Buy
    ])
    def test_close_position(self, order_executor, position_fixture, expected_side, request):
        """Test closing long and short positions sends correct opposite side."""
        client = request.getfixturevalue(position_fixture)
        order_executor.client = client
        success = order_executor.close_position()

        assert success is True
        call_kwargs = client.place_order.call_args[1]
        assert call_kwargs["reduceOnly"] is True
        assert call_kwargs["side"] == expected_side

    @pytest.mark.parametrize("close_side,expected_idx", [
        ("Sell", 1),  # Closing long -> positionIdx=1
        ("Buy", 2),   # Closing short -> positionIdx=2
    ])
    def test_close_position_hedge_mode(self, mock_pybit_client, close_side, expected_idx):
        """Hedge mode close_position must use correct positionIdx."""
        from torchtrade.envs.live.bybit.order_executor import (
            BybitFuturesOrderClass, PositionMode,
        )
        # Setup position fixture based on which side we're closing
        if close_side == "Sell":
            # Long position (default mock is long)
            pass
        else:
            # Short position
            mock_pybit_client.get_positions = MagicMock(return_value={
                "retCode": 0,
                "result": {"list": [{
                    "symbol": "BTCUSDT", "size": "0.001", "side": "Sell",
                    "avgPrice": "50000.0", "markPrice": "49900.0",
                    "unrealisedPnl": "0.1", "leverage": "10",
                    "tradeMode": "1", "liqPrice": "55000.0", "positionValue": "49.9",
                }]},
            })

        executor = BybitFuturesOrderClass(
            symbol="BTCUSDT",
            position_mode=PositionMode.HEDGE,
            client=mock_pybit_client,
        )
        executor.close_position()
        call_kwargs = mock_pybit_client.place_order.call_args[1]
        assert call_kwargs["positionIdx"] == expected_idx

    def test_close_position_no_position(self, order_executor, mock_pybit_client):
        """Test closing when no position exists."""
        mock_pybit_client.get_positions = MagicMock(return_value={
            "retCode": 0,
            "result": {"list": [{"symbol": "BTCUSDT", "size": "0", "side": "Buy"}]},
        })

        success = order_executor.close_position()
        assert success is True

    def test_cancel_open_orders(self, order_executor, mock_pybit_client):
        """Test cancelling all open orders."""
        success = order_executor.cancel_open_orders()

        assert success is True
        mock_pybit_client.cancel_all_orders.assert_called_once()
        call_kwargs = mock_pybit_client.cancel_all_orders.call_args[1]
        assert call_kwargs["category"] == "linear"
        assert call_kwargs["symbol"] == "BTCUSDT"

    def test_set_leverage(self, order_executor, mock_pybit_client):
        """Test changing leverage."""
        success = order_executor.set_leverage(20)

        assert success is True
        assert order_executor.leverage == 20
        call_kwargs = mock_pybit_client.set_leverage.call_args[1]
        assert call_kwargs["buyLeverage"] == "20"
        assert call_kwargs["sellLeverage"] == "20"

    def test_set_margin_mode(self, order_executor, mock_pybit_client):
        """Test changing margin mode."""
        from torchtrade.envs.live.bybit.order_executor import MarginMode

        success = order_executor.set_margin_mode(MarginMode.CROSSED)

        assert success is True
        assert order_executor.margin_mode == MarginMode.CROSSED
        mock_pybit_client.switch_margin_mode.assert_called()

    def test_trade_failure_handling(self, order_executor, mock_pybit_client):
        """Test that trade failures are handled gracefully."""
        mock_pybit_client.place_order = MagicMock(side_effect=Exception("API Error"))

        success = order_executor.trade(side="buy", quantity=0.001)
        assert success is False

    def test_margin_mode_enum_and_pybit_conversion(self):
        """Test MarginMode enum values and pybit conversion."""
        from torchtrade.envs.live.bybit.order_executor import MarginMode

        assert MarginMode.ISOLATED.value == "isolated"
        assert MarginMode.CROSSED.value == "crossed"
        assert MarginMode.ISOLATED.to_pybit() == 1
        assert MarginMode.CROSSED.to_pybit() == 0

    def test_one_way_mode_position_idx(self, order_executor, mock_pybit_client):
        """Test that one-way mode uses positionIdx=0."""
        order_executor.trade(side="buy", quantity=0.001)

        call_kwargs = mock_pybit_client.place_order.call_args[1]
        assert call_kwargs["positionIdx"] == 0

    def test_reduce_only_order(self, order_executor, mock_pybit_client):
        """Test placing a reduce-only order."""
        order_executor.trade(side="sell", quantity=0.001, reduce_only=True)

        call_kwargs = mock_pybit_client.place_order.call_args[1]
        assert call_kwargs["reduceOnly"] is True

    @pytest.mark.parametrize("side,reduce_only,expected_idx", [
        ("buy", False, 1),   # Opening long
        ("sell", False, 2),  # Opening short
        ("buy", True, 2),    # Closing short
        ("sell", True, 1),   # Closing long
    ])
    def test_hedge_mode_position_idx(self, mock_pybit_client, side, reduce_only, expected_idx):
        """Hedge mode must use correct positionIdx for open vs close trades."""
        from torchtrade.envs.live.bybit.order_executor import (
            BybitFuturesOrderClass, PositionMode,
        )
        executor = BybitFuturesOrderClass(
            symbol="BTCUSDT",
            position_mode=PositionMode.HEDGE,
            client=mock_pybit_client,
        )
        executor.trade(side=side, quantity=0.001, reduce_only=reduce_only)
        call_kwargs = mock_pybit_client.place_order.call_args[1]
        assert call_kwargs["positionIdx"] == expected_idx

    @pytest.mark.parametrize("qty,entry,mark,expected_pnl_pct", [
        (0.001, 50000, 51000, 0.02),     # Long, price up 2%
        (0.001, 50000, 49000, -0.02),    # Long, price down 2%
        (-0.001, 50000, 49000, 0.02),    # Short, price down 2% (profit)
        (-0.001, 50000, 51000, -0.02),   # Short, price up 2% (loss)
        (0.001, 0, 50000, 0.0),          # Zero entry price edge case
    ])
    def test_unrealized_pnl_pct(self, order_executor, qty, entry, mark, expected_pnl_pct):
        """Unrealized PnL % must be correct for long and short positions."""
        result = order_executor._calculate_unrealized_pnl_pct(qty, entry, mark)
        assert result == pytest.approx(expected_pnl_pct, abs=1e-6)

    @pytest.mark.parametrize("ticker_data,expected_price", [
        ({"markPrice": "50100.0"}, 50100.0),
        ({"lastPrice": "50050.0"}, 50050.0),  # Fallback to lastPrice
    ])
    def test_get_mark_price_fallback(self, order_executor, mock_pybit_client, ticker_data, expected_price):
        """Mark price should fall back to lastPrice when markPrice is missing."""
        mock_pybit_client.get_tickers = MagicMock(return_value={
            "retCode": 0, "result": {"list": [ticker_data]},
        })
        assert order_executor.get_mark_price() == expected_price

    @pytest.mark.parametrize("liq_price_value,expected", [
        ("45000.0", 45000.0),  # Normal value
        ("", 0.0),             # Empty string (Bybit returns this sometimes)
        (None, 0.0),           # None value
        ("0", 0.0),            # Zero string
    ])
    def test_get_status_liq_price_edge_cases(self, order_executor, mock_pybit_client, liq_price_value, expected):
        """liqPrice parsing must handle empty/None/normal values from Bybit."""
        position_data = {
            "symbol": "BTCUSDT", "size": "0.001", "side": "Buy",
            "avgPrice": "50000.0", "markPrice": "50100.0",
            "unrealisedPnl": "0.1", "leverage": "10",
            "tradeMode": "1", "positionValue": "50.1",
        }
        if liq_price_value is not None:
            position_data["liqPrice"] = liq_price_value
        # If None, don't include liqPrice key at all

        mock_pybit_client.get_positions = MagicMock(return_value={
            "retCode": 0, "result": {"list": [position_data]},
        })
        status = order_executor.get_status()
        assert status["position_status"].liquidation_price == expected

    def test_get_mark_price_no_data_raises(self, order_executor, mock_pybit_client):
        """Missing ticker data must raise RuntimeError."""
        mock_pybit_client.get_tickers = MagicMock(return_value={
            "retCode": 0, "result": {"list": [{}]},
        })
        with pytest.raises(RuntimeError):
            order_executor.get_mark_price()

    def test_limit_order_without_price_raises(self, order_executor):
        """Limit order without limit_price must raise ValueError."""
        with pytest.raises(ValueError, match="limit_price is required"):
            order_executor.trade(side="buy", quantity=0.001, order_type="limit")

    def test_limit_order_with_price_succeeds(self, order_executor, mock_pybit_client):
        """Limit order with limit_price must succeed."""
        success = order_executor.trade(
            side="buy", quantity=0.001, order_type="limit", limit_price=50000.0,
        )
        assert success is True
        call_kwargs = mock_pybit_client.place_order.call_args[1]
        assert call_kwargs["price"] == "50000.0"
        assert call_kwargs["orderType"] == "Limit"

    def test_get_status_hedge_mode_selects_nonzero(self, order_executor, mock_pybit_client):
        """get_status must select first non-zero position in hedge mode."""
        mock_pybit_client.get_positions = MagicMock(return_value={
            "retCode": 0,
            "result": {"list": [
                {"symbol": "BTCUSDT", "size": "0", "side": "Buy"},
                {
                    "symbol": "BTCUSDT", "size": "0.002", "side": "Sell",
                    "avgPrice": "50000.0", "markPrice": "49800.0",
                    "unrealisedPnl": "0.4", "leverage": "10",
                    "tradeMode": "1", "liqPrice": "55000.0", "positionValue": "99.6",
                },
            ]},
        })
        status = order_executor.get_status()
        assert status["position_status"] is not None
        assert status["position_status"].qty < 0  # Short

    def test_get_status_empty_positions(self, order_executor, mock_pybit_client):
        """get_status with all zero-size positions returns None."""
        mock_pybit_client.get_positions = MagicMock(return_value={
            "retCode": 0,
            "result": {"list": [
                {"symbol": "BTCUSDT", "size": "0", "side": "Buy"},
                {"symbol": "BTCUSDT", "size": "0", "side": "Sell"},
            ]},
        })
        status = order_executor.get_status()
        assert status["position_status"] is None

    def test_account_balance_fallback_to_contract(self, order_executor, mock_pybit_client):
        """get_account_balance falls back to CONTRACT when UNIFIED returns empty."""
        call_count = 0

        def mock_wallet_balance(accountType):
            nonlocal call_count
            call_count += 1
            if accountType == "UNIFIED":
                return {"retCode": 0, "result": {"list": []}}
            return {
                "retCode": 0,
                "result": {"list": [{
                    "totalEquity": "500.0",
                    "totalAvailableBalance": "400.0",
                    "totalPerpUPL": "0.0",
                    "totalMarginBalance": "500.0",
                }]},
            }

        mock_pybit_client.get_wallet_balance = MagicMock(side_effect=mock_wallet_balance)
        balance = order_executor.get_account_balance()
        assert balance["total_wallet_balance"] == 500.0
        assert call_count == 2  # Called UNIFIED then CONTRACT

    def test_account_balance_both_empty_raises(self, order_executor, mock_pybit_client):
        """get_account_balance raises when both UNIFIED and CONTRACT are empty."""
        mock_pybit_client.get_wallet_balance = MagicMock(return_value={
            "retCode": 0, "result": {"list": []},
        })
        with pytest.raises(RuntimeError, match="UNIFIED or CONTRACT"):
            order_executor.get_account_balance()

    @pytest.mark.parametrize("ret_code,ret_msg,expected", [
        (0, "OK", True),
        (110007, "Insufficient balance", False),
        (10001, "Invalid parameter", False),
    ], ids=["success", "insufficient-balance", "invalid-param"])
    def test_trade_validates_retcode(self, order_executor, mock_pybit_client, ret_code, ret_msg, expected):
        """trade() must return False when API returns non-zero retCode."""
        mock_pybit_client.place_order = MagicMock(return_value={
            "retCode": ret_code,
            "retMsg": ret_msg,
            "result": {"orderId": "123"} if ret_code == 0 else {},
        })
        result = order_executor.trade(side="buy", quantity=0.001)
        assert result is expected

    def test_close_position_requery_confirms_closed(self, order_executor, mock_pybit_client):
        """close_position must re-query to confirm when order fails."""
        # First get_positions call returns open position (for the close attempt)
        # place_order raises an exception
        # Second get_positions call returns no position (confirming it's closed)
        call_count = {"get_positions": 0}

        def mock_get_positions(**kwargs):
            call_count["get_positions"] += 1
            if call_count["get_positions"] == 1:
                return {
                    "retCode": 0,
                    "result": {"list": [{
                        "symbol": "BTCUSDT", "size": "0.001", "side": "Buy",
                        "avgPrice": "50000.0", "markPrice": "50100.0",
                        "unrealisedPnl": "0.1", "leverage": "10",
                        "tradeMode": "1", "liqPrice": "45000.0", "positionValue": "50.1",
                    }]},
                }
            # Re-query: position is now gone
            return {"retCode": 0, "result": {"list": []}}

        mock_pybit_client.get_positions = MagicMock(side_effect=mock_get_positions)
        mock_pybit_client.place_order = MagicMock(side_effect=Exception("Order failed"))

        success = order_executor.close_position()
        assert success is True
        assert call_count["get_positions"] == 2

    def test_close_position_requery_still_open(self, order_executor, mock_pybit_client):
        """close_position must return False when re-query shows position still open."""
        mock_pybit_client.place_order = MagicMock(side_effect=Exception("Order failed"))
        # get_positions always returns open position
        success = order_executor.close_position()
        assert success is False

    def test_get_lot_size_fetches_and_caches(self, order_executor, mock_pybit_client):
        """get_lot_size must fetch from API and cache the result."""
        lot_size = order_executor.get_lot_size()
        assert lot_size["min_qty"] == 0.001
        assert lot_size["qty_step"] == 0.001

        # Second call uses cache (no additional API call)
        lot_size2 = order_executor.get_lot_size()
        assert lot_size2 is lot_size
        mock_pybit_client.get_instruments_info.assert_called_once()

    def test_get_lot_size_fallback_on_failure(self, order_executor, mock_pybit_client):
        """get_lot_size must fall back to defaults if API fails."""
        order_executor._lot_size_cache = None
        mock_pybit_client.get_instruments_info = MagicMock(side_effect=Exception("API down"))
        lot_size = order_executor.get_lot_size()
        assert lot_size["min_qty"] == 0.001
        assert lot_size["qty_step"] == 0.001

    def test_close_position_hedge_both_sides(self, mock_pybit_client):
        """close_position in hedge mode must close both long and short sides."""
        from torchtrade.envs.live.bybit.order_executor import (
            BybitFuturesOrderClass, PositionMode,
        )

        mock_pybit_client.get_positions = MagicMock(return_value={
            "retCode": 0,
            "result": {"list": [
                {
                    "symbol": "BTCUSDT", "size": "0.002", "side": "Buy",
                    "avgPrice": "50000.0", "markPrice": "50100.0",
                    "unrealisedPnl": "0.2", "leverage": "10",
                    "tradeMode": "1", "liqPrice": "45000.0", "positionValue": "100.2",
                },
                {
                    "symbol": "BTCUSDT", "size": "0.001", "side": "Sell",
                    "avgPrice": "51000.0", "markPrice": "50100.0",
                    "unrealisedPnl": "0.9", "leverage": "10",
                    "tradeMode": "1", "liqPrice": "56000.0", "positionValue": "50.1",
                },
            ]},
        })

        executor = BybitFuturesOrderClass(
            symbol="BTCUSDT",
            position_mode=PositionMode.HEDGE,
            client=mock_pybit_client,
        )
        success = executor.close_position()
        assert success is True
        assert mock_pybit_client.place_order.call_count == 2

        # Verify both sides were closed with correct parameters
        calls = mock_pybit_client.place_order.call_args_list
        sides_closed = {c[1]["side"] for c in calls}
        assert sides_closed == {"Sell", "Buy"}  # Sell closes long, Buy closes short
