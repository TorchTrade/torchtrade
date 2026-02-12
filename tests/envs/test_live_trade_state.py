"""Tests for live environment trade state management.

Verifies that local position state (current_action_level, current_position,
hold_counter) stays synchronized with exchange state across success, failure,
and exception scenarios. Uses mocked trader/observer to test without exchange.

Regression tests for #189: failed trades returned executed=True, causing
permanent local state desynchronization from the exchange.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from torchtrade.envs.core.state import PositionState


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------


class MockPositionStatus:
    """Mock exchange position status."""

    def __init__(self, qty=0.0, mark_price=50000.0, notional_value=0.0,
                 entry_price=0.0, unrealized_pnl_pct=0.0, leverage=5,
                 liquidation_price=0.0):
        self.qty = qty
        self.mark_price = mark_price
        self.notional_value = notional_value
        self.entry_price = entry_price
        self.unrealized_pnl_pct = unrealized_pnl_pct
        self.leverage = leverage
        self.liquidation_price = liquidation_price


class MockTrader:
    """Mock trader that simulates exchange order execution."""

    def __init__(self, position_qty=0.0, balance=10000.0, mark_price=50000.0):
        self.position_qty = position_qty
        self.balance = balance
        self.mark_price_val = mark_price
        self.trade_should_fail = False
        self.trade_should_raise = False
        self.close_should_fail = False
        self.close_should_raise = False
        self.trades_executed = []

    def get_status(self):
        if self.position_qty != 0:
            return {
                "position_status": MockPositionStatus(
                    qty=self.position_qty,
                    mark_price=self.mark_price_val,
                )
            }
        return {"position_status": None}

    def get_mark_price(self):
        return self.mark_price_val

    def get_account_balance(self):
        return {
            "available_balance": self.balance,
            "total_wallet_balance": self.balance,
            "total_margin_balance": self.balance,
        }

    def trade(self, side, quantity, order_type="market"):
        if self.trade_should_raise:
            raise Exception("Exchange connection error")
        if self.trade_should_fail:
            return False
        self.trades_executed.append({"side": side, "quantity": quantity})
        if side in ("BUY", "buy"):
            self.position_qty += quantity
        elif side in ("SELL", "sell"):
            self.position_qty -= quantity
        return True

    def close_position(self):
        if self.close_should_raise:
            raise Exception("Close position failed: exchange error")
        if self.close_should_fail:
            return False
        self.position_qty = 0.0
        return True


# ---------------------------------------------------------------------------
# Environment factory helpers
# ---------------------------------------------------------------------------


def _make_binance_env(trader=None):
    """Create a BinanceFuturesTorchTradingEnv with mocked dependencies."""
    from torchtrade.envs.live.binance.env import (
        BinanceFuturesTorchTradingEnv,
        BinanceFuturesTradingEnvConfig,
    )
    from torchtrade.envs.core.state import HistoryTracker
    from torchtrade.envs.core.default_rewards import log_return_reward

    mock_trader = trader or MockTrader()

    config = BinanceFuturesTradingEnvConfig(
        symbol="BTCUSDT",
        demo=True,
        time_frames=["1h"],
        window_sizes=[10],
        execute_on="1h",
        leverage=5,
        action_levels=[-1.0, 0.0, 1.0],
    )

    with patch.object(BinanceFuturesTorchTradingEnv, '__init__', lambda self, *a, **kw: None):
        env = BinanceFuturesTorchTradingEnv.__new__(BinanceFuturesTorchTradingEnv)

    env.config = config
    env.trader = mock_trader
    env.action_levels = config.action_levels
    env.position = PositionState()
    env.initial_portfolio_value = 10000.0
    env.history = HistoryTracker()
    env.reward_function = log_return_reward

    return env


def _make_bitget_env(trader=None):
    """Create a BitgetFuturesTorchTradingEnv with mocked dependencies."""
    from torchtrade.envs.live.bitget.env import (
        BitgetFuturesTorchTradingEnv,
        BitgetFuturesTradingEnvConfig,
    )
    from torchtrade.envs.core.state import HistoryTracker
    from torchtrade.envs.core.default_rewards import log_return_reward

    mock_trader = trader or MockTrader()

    config = BitgetFuturesTradingEnvConfig(
        symbol="BTC/USDT:USDT",
        demo=True,
        time_frames=["1h"],
        window_sizes=[10],
        execute_on="1h",
        leverage=5,
        action_levels=[-1.0, 0.0, 1.0],
    )

    with patch.object(BitgetFuturesTorchTradingEnv, '__init__', lambda self, *a, **kw: None):
        env = BitgetFuturesTorchTradingEnv.__new__(BitgetFuturesTorchTradingEnv)

    env.config = config
    env.trader = mock_trader
    env.action_levels = config.action_levels
    env.position = PositionState()
    env.initial_portfolio_value = 10000.0
    env.history = HistoryTracker()
    env.reward_function = log_return_reward

    return env


ENV_FACTORIES = {
    "binance": _make_binance_env,
    "bitget": _make_bitget_env,
}


# ---------------------------------------------------------------------------
# Shared tests: run identical scenarios on both Binance and Bitget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exchange", ["binance", "bitget"])
class TestTradeStateSync:
    """Trade state synchronization tests that apply to both exchanges."""

    def test_successful_trade_returns_success(self, exchange):
        """Successful trade should return executed=True, success=True."""
        trader = MockTrader(position_qty=0.0)
        env = ENV_FACTORIES[exchange](trader)

        trade_info = env._execute_trade_if_needed(-1.0)

        assert trade_info["executed"] is True
        assert trade_info["success"] is True

    def test_trade_exception_returns_executed_false(self, exchange):
        """Trade that raises exception should return executed=False, success=False."""
        trader = MockTrader(position_qty=0.0)
        trader.trade_should_raise = True
        env = ENV_FACTORIES[exchange](trader)

        trade_info = env._execute_fractional_action(-1.0)

        assert trade_info["executed"] is False
        assert trade_info["success"] is False

    def test_guard_allows_retry_after_failed_trade(self, exchange):
        """After a failed trade, the guard should allow retrying the same action.

        This is the core regression test for #189: if current_action_level is
        incorrectly updated on failure, the guard permanently blocks retry.
        """
        trader = MockTrader(position_qty=0.0)
        env = ENV_FACTORIES[exchange](trader)

        # First attempt: trade raises
        trader.trade_should_raise = True
        trade_info = env._execute_trade_if_needed(-1.0)

        # Simulate _step logic: don't update state on failure
        if trade_info["executed"] and trade_info.get("success") is not False:
            env.position.current_action_level = -1.0

        assert env.position.current_action_level == 0.0

        # Second attempt: trade succeeds -- guard must NOT block
        trader.trade_should_raise = False
        trade_info2 = env._execute_trade_if_needed(-1.0)

        assert trade_info2["executed"] is True
        assert trade_info2["success"] is True

    def test_close_exception_returns_executed_false(self, exchange):
        """_handle_close_action should return executed=False if close_position raises."""
        trader = MockTrader(position_qty=0.5)
        trader.close_should_raise = True
        env = ENV_FACTORIES[exchange](trader)

        trade_info = env._handle_close_action(0.5)

        assert trade_info["executed"] is False
        assert trade_info["success"] is False

    def test_trade_returning_false_reports_failure(self, exchange):
        """When trader.trade() returns False (no exception), success should be False.

        This covers the case where the exchange rejects the order without raising.
        The _step guard `if executed and success is not False` must block state update.
        """
        trader = MockTrader(position_qty=0.0)
        trader.trade_should_fail = True
        env = ENV_FACTORIES[exchange](trader)

        trade_info = env._execute_fractional_action(-1.0)

        assert trade_info["executed"] is True
        assert trade_info["success"] is False


# ---------------------------------------------------------------------------
# Binance-specific tests
# ---------------------------------------------------------------------------


class TestBinanceDirectionSwitch:
    """Binance-specific: direction switch aborts when close fails."""

    def test_direction_switch_aborts_on_close_failure(self):
        """Direction switch should not open opposite position if close fails."""
        trader = MockTrader(position_qty=0.5, balance=10000.0)
        trader.close_should_fail = True
        env = _make_binance_env(trader)

        trade_info = env._execute_fractional_action(-1.0)

        assert trader.position_qty == 0.5
        assert len([t for t in trader.trades_executed if t["side"] == "SELL"]) == 0


# ---------------------------------------------------------------------------
# Bitget-specific tests: _handle_close_action updates current_position
# ---------------------------------------------------------------------------


class TestBitgetClosePositionState:
    """Bitget-specific: _handle_close_action conditionally updates current_position."""

    @pytest.mark.parametrize("close_should_fail,expected_position", [
        (True, 1),   # Failed close: position unchanged
        (False, 0),  # Successful close: position reset to 0
    ], ids=["close-fails", "close-succeeds"])
    def test_close_updates_position_only_on_success(self, close_should_fail, expected_position):
        """current_position should only be set to 0 when close succeeds."""
        trader = MockTrader(position_qty=0.5)
        trader.close_should_fail = close_should_fail
        env = _make_bitget_env(trader)
        env.position.current_position = 1

        env._handle_close_action(0.5)

        assert env.position.current_position == expected_position


# ---------------------------------------------------------------------------
# Regression test: the exact bug scenario from #189
# ---------------------------------------------------------------------------


class TestRegressionIssue189:
    """Regression test for #189: guard correctly blocks/allows repeated actions.

    After a successful trade, current_action_level is set (correct), and
    repeating the same action is blocked by the guard (correct). The agent
    must transition through a different action level to re-enter.
    """

    def test_successful_trade_guard_blocks_then_allows_after_transition(self):
        """After SHORT succeeds, repeated SHORT is blocked until FLAT transition."""
        trader = MockTrader(position_qty=0.0)
        env = _make_binance_env(trader)

        # Step 1: SHORT succeeds
        trade_info = env._execute_trade_if_needed(-1.0)
        if trade_info["executed"] and trade_info.get("success") is not False:
            env.position.current_action_level = -1.0

        assert env.position.current_action_level == -1.0

        # Step 2: SHORT again -- guard correctly blocks (same action level)
        trade_info2 = env._execute_trade_if_needed(-1.0)
        assert trade_info2["executed"] is False

        # Step 3: Transition to FLAT, then SHORT works
        trade_info3 = env._execute_trade_if_needed(0.0)
        if trade_info3["executed"] and trade_info3.get("success") is not False:
            env.position.current_action_level = 0.0

        trade_info4 = env._execute_trade_if_needed(-1.0)
        assert trade_info4["executed"] is True
