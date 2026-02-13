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

    def test_close_position(self, order_executor, mock_pybit_client):
        """Test closing a position."""
        success = order_executor.close_position()

        assert success is True
        call_kwargs = mock_pybit_client.place_order.call_args[1]
        assert call_kwargs["reduceOnly"] is True
        assert call_kwargs["side"] == "Sell"  # Opposite of long

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

    def test_get_account_balance_empty_raises(self, order_executor, mock_pybit_client):
        """Empty account list must raise RuntimeError."""
        mock_pybit_client.get_wallet_balance = MagicMock(return_value={
            "retCode": 0, "result": {"list": []},
        })
        with pytest.raises(RuntimeError, match="No account data"):
            order_executor.get_account_balance()

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

    def test_get_mark_price_no_data_raises(self, order_executor, mock_pybit_client):
        """Missing ticker data must raise RuntimeError."""
        mock_pybit_client.get_tickers = MagicMock(return_value={
            "retCode": 0, "result": {"list": [{}]},
        })
        with pytest.raises(RuntimeError):
            order_executor.get_mark_price()
