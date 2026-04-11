"""Tests for OKXFuturesOrderClass."""

import pytest
from unittest.mock import MagicMock


class TestOKXFuturesOrderClass:
    """Tests for OKXFuturesOrderClass."""

    @pytest.fixture
    def order_executor(self, mock_okx_trade_client, mock_okx_account_client, mock_okx_public_client):
        """Create order executor with mock OKX clients."""
        from torchtrade.envs.live.okx.order_executor import (
            OKXFuturesOrderClass,
            MarginMode,
            PositionMode,
        )

        return OKXFuturesOrderClass(
            symbol="BTC-USDT-SWAP",
            trade_mode="quantity",
            demo=True,
            leverage=10,
            margin_mode=MarginMode.ISOLATED,
            position_mode=PositionMode.NET,
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_pass",
            client=mock_okx_trade_client,
            account_client=mock_okx_account_client,
            public_client=mock_okx_public_client,
        )

    @pytest.mark.parametrize("side", ["buy", "sell"])
    def test_market_order(self, order_executor, mock_okx_trade_client, side):
        """Test placing a market order (buy or sell)."""
        success = order_executor.trade(side=side, quantity=0.001, order_type="market")

        assert success is True
        call_kwargs = mock_okx_trade_client.place_order.call_args[1]
        assert call_kwargs["instId"] == "BTC-USDT-SWAP"
        assert call_kwargs["side"] == side
        assert call_kwargs["ordType"] == "market"
        assert call_kwargs["sz"] == "0.001"

    def test_bracket_order_with_tp_sl(self, order_executor, mock_okx_trade_client):
        """Test placing bracket order with take profit and stop loss."""
        success = order_executor.trade(
            side="buy", quantity=0.001, order_type="market",
            take_profit=51000.0, stop_loss=49000.0,
        )

        assert success is True
        algo = mock_okx_trade_client.place_order.call_args[1]["attachAlgoOrds"][0]
        assert algo["tpTriggerPx"] == "51000.00"
        assert algo["slTriggerPx"] == "49000.00"

    @pytest.mark.parametrize("raw_tp,raw_sl,expected_tp,expected_sl", [
        (82622.2122, 84291.4358, "82622.21", "84291.44"),
        (51234.5678, 48765.4321, "51234.57", "48765.43"),
    ])
    def test_bracket_order_prices_rounded_to_tick(self, order_executor, mock_okx_trade_client, raw_tp, raw_sl, expected_tp, expected_sl):
        """SL/TP prices must be rounded to tick size before submission."""
        order_executor.trade(side="buy", quantity=0.001, take_profit=raw_tp, stop_loss=raw_sl)
        algo = mock_okx_trade_client.place_order.call_args[1]["attachAlgoOrds"][0]
        assert algo["tpTriggerPx"] == expected_tp
        assert algo["slTriggerPx"] == expected_sl

    def test_round_price_without_precision(self, mock_okx_trade_client, mock_okx_account_client, mock_okx_public_client):
        """When tick size fetch fails, prices pass through unmodified."""
        from torchtrade.envs.live.okx.order_executor import OKXFuturesOrderClass

        mock_okx_public_client.get_instruments = MagicMock(side_effect=Exception("API down"))
        executor = OKXFuturesOrderClass(
            symbol="BTC-USDT-SWAP",
            client=mock_okx_trade_client,
            account_client=mock_okx_account_client,
            public_client=mock_okx_public_client,
        )
        assert executor._tick_size is None
        assert executor._round_price(82622.2122) == 82622.2122

    @pytest.mark.parametrize("pos_data,expected_qty", [
        ({"pos": "0.001", "posSide": "net", "avgPx": "50000.0", "markPx": "50100.0",
          "upl": "0.1", "lever": "10", "mgnMode": "isolated", "liqPx": "45000.0",
          "notionalUsd": "50.1"}, 0.001),
        ({"pos": "0", "posSide": "net"}, None),
        ({"pos": "-0.001", "posSide": "net", "avgPx": "50000.0", "markPx": "49900.0",
          "upl": "0.1", "lever": "10", "mgnMode": "isolated", "liqPx": "55000.0",
          "notionalUsd": "49.9"}, -0.001),
    ], ids=["long", "no-position", "short"])
    def test_get_status(self, order_executor, mock_okx_account_client, pos_data, expected_qty):
        """Test get_status for long, short, and no-position scenarios."""
        mock_okx_account_client.get_positions = MagicMock(return_value={
            "code": "0", "msg": "", "data": [pos_data],
        })
        status = order_executor.get_status()
        if expected_qty is None:
            assert status["position_status"] is None
        else:
            assert status["position_status"].qty == pytest.approx(expected_qty)

    def test_get_account_balance(self, order_executor):
        """Test getting account balance."""
        balance = order_executor.get_account_balance()
        assert balance["total_wallet_balance"] == 1000.0
        assert balance["available_balance"] == 900.0

    @pytest.mark.parametrize("pos_data,expected_side", [
        ({"pos": "0.001", "posSide": "net", "avgPx": "50000.0", "markPx": "50100.0",
          "upl": "0.1", "lever": "10", "mgnMode": "isolated", "liqPx": "45000.0",
          "notionalUsd": "50.1"}, "sell"),
        ({"pos": "-0.001", "posSide": "net", "avgPx": "50000.0", "markPx": "49900.0",
          "upl": "0.1", "lever": "10", "mgnMode": "isolated", "liqPx": "55000.0",
          "notionalUsd": "49.9"}, "buy"),
        ({"pos": "0", "posSide": "net"}, None),
    ], ids=["close-long", "close-short", "no-position"])
    def test_close_position(self, order_executor, mock_okx_trade_client, mock_okx_account_client, pos_data, expected_side):
        """Test closing long, short, and no-position scenarios."""
        mock_okx_account_client.get_positions = MagicMock(return_value={
            "code": "0", "msg": "", "data": [pos_data],
        })
        success = order_executor.close_position()
        assert success is True
        if expected_side is not None:
            call_kwargs = mock_okx_trade_client.place_order.call_args[1]
            assert call_kwargs["side"] == expected_side
            assert call_kwargs["reduceOnly"] is True

    @pytest.mark.parametrize("code,expected_success,expected_leverage", [
        ("0", True, 20),
        ("51101", False, 10),
    ], ids=["success", "rejected"])
    def test_set_leverage_validates_code(self, order_executor, mock_okx_account_client, code, expected_success, expected_leverage):
        """set_leverage must validate code and only update local state on success."""
        mock_okx_account_client.set_leverage = MagicMock(return_value={
            "code": code, "msg": "", "data": [{}],
        })
        result = order_executor.set_leverage(20)
        assert result is expected_success
        assert order_executor.leverage == expected_leverage

    @pytest.mark.parametrize("code,expected_success,expect_mode_changed", [
        ("0", True, True),
        ("51101", False, False),
    ], ids=["success", "rejected"])
    def test_set_margin_mode_validates_code(self, order_executor, mock_okx_account_client, code, expected_success, expect_mode_changed):
        """set_margin_mode must validate code and only update local state on success."""
        from torchtrade.envs.live.okx.order_executor import MarginMode

        original_mode = order_executor.margin_mode
        mock_okx_account_client.set_leverage = MagicMock(return_value={
            "code": code, "msg": "", "data": [{}],
        })
        result = order_executor.set_margin_mode(MarginMode.CROSS)
        assert result is expected_success
        if expect_mode_changed:
            assert order_executor.margin_mode == MarginMode.CROSS
        else:
            assert order_executor.margin_mode == original_mode

    def test_trade_failure_handling(self, order_executor, mock_okx_trade_client):
        """Test that trade failures are handled gracefully."""
        mock_okx_trade_client.place_order = MagicMock(side_effect=Exception("API Error"))
        assert order_executor.trade(side="buy", quantity=0.001) is False

    def test_reduce_only_order(self, order_executor, mock_okx_trade_client):
        """Test placing a reduce-only order."""
        order_executor.trade(side="sell", quantity=0.001, reduce_only=True)
        assert mock_okx_trade_client.place_order.call_args[1]["reduceOnly"] is True

    @pytest.mark.parametrize("qty,entry,mark,expected_pnl_pct", [
        (0.001, 50000, 51000, 0.02),
        (0.001, 50000, 49000, -0.02),
        (-0.001, 50000, 49000, 0.02),
        (-0.001, 50000, 51000, -0.02),
        (0.001, 0, 50000, 0.0),
    ])
    def test_unrealized_pnl_pct(self, order_executor, qty, entry, mark, expected_pnl_pct):
        """Unrealized PnL % must be correct for long and short positions."""
        result = order_executor._calculate_unrealized_pnl_pct(qty, entry, mark)
        assert result == pytest.approx(expected_pnl_pct, abs=1e-6)

    def test_get_mark_price_no_data_raises(self, order_executor, mock_okx_public_client):
        """Missing mark price data must raise RuntimeError."""
        mock_okx_public_client.get_mark_price = MagicMock(return_value={
            "code": "0", "msg": "", "data": [{}],
        })
        with pytest.raises(RuntimeError):
            order_executor.get_mark_price()

    def test_limit_order_without_price_raises(self, order_executor):
        """Limit order without limit_price must raise ValueError."""
        with pytest.raises(ValueError, match="limit_price is required"):
            order_executor.trade(side="buy", quantity=0.001, order_type="limit")

    def test_limit_order_with_price_succeeds(self, order_executor, mock_okx_trade_client):
        """Limit order with limit_price must succeed."""
        success = order_executor.trade(side="buy", quantity=0.001, order_type="limit", limit_price=50000.0)
        assert success is True
        call_kwargs = mock_okx_trade_client.place_order.call_args[1]
        assert call_kwargs["px"] == "50000.00"
        assert call_kwargs["ordType"] == "limit"

    @pytest.mark.parametrize("liq_price_value,expected", [
        ("45000.0", 45000.0),
        ("", 0.0),
        (None, 0.0),
        ("0", 0.0),
    ])
    def test_get_status_liq_price_edge_cases(self, order_executor, mock_okx_account_client, liq_price_value, expected):
        """liqPx parsing must handle empty/None/normal values."""
        position_data = {
            "instId": "BTC-USDT-SWAP", "pos": "0.001", "posSide": "net",
            "avgPx": "50000.0", "markPx": "50100.0",
            "upl": "0.1", "lever": "10", "mgnMode": "isolated", "notionalUsd": "50.1",
        }
        if liq_price_value is not None:
            position_data["liqPx"] = liq_price_value
        mock_okx_account_client.get_positions = MagicMock(return_value={
            "code": "0", "msg": "", "data": [position_data],
        })
        status = order_executor.get_status()
        assert status["position_status"].liquidation_price == expected

    def test_get_status_validates_code(self, order_executor, mock_okx_account_client):
        """get_status must return position_status=None on non-zero code."""
        mock_okx_account_client.get_positions = MagicMock(return_value={
            "code": "51001", "msg": "Invalid parameter", "data": [],
        })
        assert order_executor.get_status()["position_status"] is None

    def test_account_balance_empty_raises(self, order_executor, mock_okx_account_client):
        """get_account_balance raises when data is empty."""
        mock_okx_account_client.get_account_balance = MagicMock(return_value={
            "code": "0", "msg": "", "data": [],
        })
        with pytest.raises(RuntimeError, match="No account data"):
            order_executor.get_account_balance()

    @pytest.mark.parametrize("code,msg,expected", [
        ("0", "", True),
        ("51008", "Insufficient balance", False),
        ("51001", "Invalid parameter", False),
    ], ids=["success", "insufficient-balance", "invalid-param"])
    def test_trade_validates_code(self, order_executor, mock_okx_trade_client, code, msg, expected):
        """trade() must return False when API returns non-zero code."""
        mock_okx_trade_client.place_order = MagicMock(return_value={
            "code": code, "msg": msg,
            "data": [{"ordId": "123"}] if code == "0" else [],
        })
        assert order_executor.trade(side="buy", quantity=0.001) is expected

    def test_get_lot_size_fetches_and_caches(self, order_executor, mock_okx_public_client):
        """get_lot_size must return cached data from init-time instrument fetch."""
        init_call_count = mock_okx_public_client.get_instruments.call_count
        lot_size = order_executor.get_lot_size()
        assert lot_size["min_qty"] == 0.001
        lot_size2 = order_executor.get_lot_size()
        assert lot_size2 is lot_size
        assert mock_okx_public_client.get_instruments.call_count == init_call_count

    def test_get_lot_size_fallback_on_failure(self, order_executor, mock_okx_public_client):
        """get_lot_size must fall back to defaults if API fails."""
        order_executor._lot_size_cache = None
        mock_okx_public_client.get_instruments = MagicMock(side_effect=Exception("API down"))
        lot_size = order_executor.get_lot_size()
        assert lot_size["min_qty"] == 0.001
        assert lot_size["qty_step"] == 0.001

    def test_get_lot_size_validates_code(self, order_executor, mock_okx_public_client):
        """get_lot_size must fall back to defaults on non-zero code."""
        order_executor._lot_size_cache = None
        mock_okx_public_client.get_instruments = MagicMock(return_value={
            "code": "51001", "msg": "Invalid parameter", "data": [],
        })
        lot_size = order_executor.get_lot_size()
        assert lot_size["min_qty"] == 0.001

    def test_close_position_requery_confirms_closed(self, order_executor, mock_okx_trade_client, mock_okx_account_client):
        """close_position must re-query to confirm when order fails."""
        call_count = {"get_positions": 0}

        def mock_get_positions(**kwargs):
            call_count["get_positions"] += 1
            if call_count["get_positions"] == 1:
                return {"code": "0", "msg": "", "data": [{
                    "instId": "BTC-USDT-SWAP", "pos": "0.001", "posSide": "net",
                    "avgPx": "50000.0", "markPx": "50100.0", "upl": "0.1",
                    "lever": "10", "mgnMode": "isolated", "liqPx": "45000.0",
                    "notionalUsd": "50.1",
                }]}
            return {"code": "0", "msg": "", "data": []}

        mock_okx_account_client.get_positions = MagicMock(side_effect=mock_get_positions)
        mock_okx_trade_client.place_order = MagicMock(side_effect=Exception("Order failed"))
        assert order_executor.close_position() is True
        assert call_count["get_positions"] == 2

    def test_close_position_requery_still_open(self, order_executor, mock_okx_trade_client):
        """close_position must return False when re-query shows position still open."""
        mock_okx_trade_client.place_order = MagicMock(side_effect=Exception("Order failed"))
        assert order_executor.close_position() is False

    @pytest.mark.parametrize("side,reduce_only,expected_pos_side", [
        ("buy", False, "long"),
        ("sell", False, "short"),
        ("buy", True, "short"),
        ("sell", True, "long"),
    ])
    def test_long_short_mode_position_side(self, mock_okx_trade_client, mock_okx_account_client, mock_okx_public_client, side, reduce_only, expected_pos_side):
        """Long/short mode must use correct posSide for open vs close trades."""
        from torchtrade.envs.live.okx.order_executor import OKXFuturesOrderClass, PositionMode

        executor = OKXFuturesOrderClass(
            symbol="BTC-USDT-SWAP",
            position_mode=PositionMode.LONG_SHORT,
            client=mock_okx_trade_client,
            account_client=mock_okx_account_client,
            public_client=mock_okx_public_client,
        )
        executor.trade(side=side, quantity=0.001, reduce_only=reduce_only)
        assert mock_okx_trade_client.place_order.call_args[1]["posSide"] == expected_pos_side
