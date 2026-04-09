"""Tests for replay components."""

import pytest
from torchtrade.envs.replay.order_executor import ReplayOrderExecutor


class TestReplayOrderExecutor:
    """Test ReplayOrderExecutor simulated trading."""

    @pytest.fixture
    def executor(self):
        return ReplayOrderExecutor(initial_balance=10000.0, leverage=5, transaction_fee=0.001)

    def test_initial_state(self, executor):
        """Executor starts flat with full balance."""
        status = executor.get_status()
        assert status["position_status"] is None
        balance = executor.get_account_balance()
        assert balance["total_wallet_balance"] == 10000.0

    def test_open_long_position(self, executor):
        """trade(BUY) opens a long position."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        success = executor.trade(side="BUY", quantity=0.1, order_type="market")
        assert success is True
        status = executor.get_status()
        pos = status["position_status"]
        assert pos is not None
        assert pos.qty == pytest.approx(0.1)
        assert pos.entry_price == pytest.approx(50000.0)

    def test_open_short_position(self, executor):
        """trade(SELL) opens a short position."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="SELL", quantity=0.1, order_type="market")
        status = executor.get_status()
        pos = status["position_status"]
        assert pos.qty == pytest.approx(-0.1)

    def test_close_position_with_pnl(self, executor):
        """Closing a long position updates balance with P&L."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, order_type="market")
        executor.advance_bar({"open": 50000, "high": 51000, "low": 49900, "close": 51000})
        executor.close_position()
        status = executor.get_status()
        assert status["position_status"] is None
        balance = executor.get_account_balance()
        assert balance["total_wallet_balance"] > 10000.0

    def test_get_mark_price(self, executor):
        """get_mark_price returns the latest close price."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50050})
        assert executor.get_mark_price() == 50050.0

    def test_bracket_status_set_on_trade(self, executor):
        """trade() with SL/TP sets bracket_status."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, take_profit=51000, stop_loss=49000)
        assert executor.bracket_status == {"tp_placed": True, "sl_placed": True}

    def test_cancel_open_orders_clears_brackets(self, executor):
        """cancel_open_orders clears active SL/TP brackets."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, take_profit=51000, stop_loss=49000)
        executor.cancel_open_orders()
        assert executor.sl_price == 0.0
        assert executor.tp_price == 0.0


class TestReplayOrderExecutorSLTPTriggers:
    """Test intrabar SL/TP trigger detection via advance_bar."""

    @pytest.fixture
    def executor(self):
        return ReplayOrderExecutor(initial_balance=10000.0, leverage=5, transaction_fee=0.0)

    def test_long_sl_triggers_on_low(self, executor):
        """Long SL triggers when bar low <= stop_loss price."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, stop_loss=49500, take_profit=51000)
        executor.advance_bar({"open": 50000, "high": 50200, "low": 49000, "close": 49800})
        status = executor.get_status()
        assert status["position_status"] is None

    def test_long_tp_triggers_on_high(self, executor):
        """Long TP triggers when bar high >= take_profit price."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, stop_loss=49000, take_profit=50500)
        executor.advance_bar({"open": 50000, "high": 51000, "low": 49900, "close": 50800})
        status = executor.get_status()
        assert status["position_status"] is None

    def test_short_sl_triggers_on_high(self, executor):
        """Short SL triggers when bar high >= stop_loss price."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="SELL", quantity=0.1, stop_loss=50500, take_profit=49000)
        executor.advance_bar({"open": 50000, "high": 51000, "low": 49900, "close": 50800})
        status = executor.get_status()
        assert status["position_status"] is None

    def test_short_tp_triggers_on_low(self, executor):
        """Short TP triggers when bar low <= take_profit price."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="SELL", quantity=0.1, stop_loss=51000, take_profit=49500)
        executor.advance_bar({"open": 50000, "high": 50200, "low": 49000, "close": 49800})
        status = executor.get_status()
        assert status["position_status"] is None

    def test_sl_checked_before_tp(self, executor):
        """When both SL and TP trigger on same bar, SL wins (pessimistic)."""
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, stop_loss=49500, take_profit=50500)
        executor.advance_bar({"open": 50000, "high": 51000, "low": 49000, "close": 50000})
        balance = executor.get_account_balance()
        assert balance["total_wallet_balance"] < 10000.0

    def test_no_trigger_when_flat(self, executor):
        """advance_bar with no position should not crash."""
        executor.advance_bar({"open": 50000, "high": 51000, "low": 49000, "close": 50000})
        status = executor.get_status()
        assert status["position_status"] is None

    def test_transaction_fee_applied(self):
        """Fees deducted on open and close."""
        executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5, transaction_fee=0.001)
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.trade(side="BUY", quantity=0.1, order_type="market")
        executor.advance_bar({"open": 50000, "high": 50100, "low": 49900, "close": 50000})
        executor.close_position()
        balance = executor.get_account_balance()
        assert balance["total_wallet_balance"] == pytest.approx(10000.0 - 10.0, rel=0.01)
