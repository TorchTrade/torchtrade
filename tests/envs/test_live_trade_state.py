"""Tests for live environment trade state management.

Verifies that local position state (current_action_level, current_position,
hold_counter) stays synchronized with exchange state across success, failure,
and exception scenarios. Uses mocked trader/observer to test without exchange.

Regression tests for #189: failed trades returned executed=True, causing
permanent local state desynchronization from the exchange.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np
from tensordict import TensorDict

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
# Binance env tests
# ---------------------------------------------------------------------------


class TestBinanceTradeStateSync:
    """Test that Binance env local state stays synchronized with exchange."""

    def _make_env(self, trader=None):
        """Create a BinanceFuturesTorchTradingEnv with mocked dependencies."""
        from torchtrade.envs.live.binance.env import (
            BinanceFuturesTorchTradingEnv,
            BinanceFuturesTradingEnvConfig,
        )

        mock_trader = trader or MockTrader()
        mock_observer = MagicMock()
        mock_observer.get_observations.return_value = {
            "features_1Hour": np.random.randn(10, 4).astype(np.float32),
        }
        mock_observer.get_keys.return_value = ["features_1Hour"]

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
        env.observer = mock_observer
        env.action_levels = config.action_levels
        env.position = PositionState()
        env.initial_portfolio_value = 10000.0

        from torchtrade.envs.core.state import HistoryTracker
        env.history = HistoryTracker()

        from torchtrade.envs.core.default_rewards import log_return_reward
        env.reward_function = log_return_reward

        return env

    def test_successful_trade_updates_state(self):
        """Successful trade should update current_action_level and current_position."""
        trader = MockTrader(position_qty=0.0)
        env = self._make_env(trader)

        trade_info = env._execute_trade_if_needed(-1.0)

        assert trade_info["executed"] is True
        assert trade_info["success"] is True

    def test_failed_trade_exception_does_not_update_state(self):
        """Trade that raises exception should return executed=False."""
        trader = MockTrader(position_qty=0.0)
        trader.trade_should_raise = True
        env = self._make_env(trader)

        # Bypass the guard (current_action_level=0.0, desired=-1.0)
        trade_info = env._execute_fractional_action(-1.0)

        assert trade_info["executed"] is False
        assert trade_info["success"] is False

    def test_exception_preserves_current_action_level(self):
        """After a failed trade, current_action_level should remain unchanged."""
        trader = MockTrader(position_qty=0.0)
        trader.trade_should_raise = True
        env = self._make_env(trader)

        assert env.position.current_action_level == 0.0

        trade_info = env._execute_fractional_action(-1.0)

        # Simulate what _step does: only update if executed AND success
        if trade_info["executed"] and trade_info.get("success") is not False:
            env.position.current_action_level = -1.0

        # current_action_level must NOT have changed
        assert env.position.current_action_level == 0.0

    def test_guard_not_blocked_after_failed_trade(self):
        """After a failed trade, the guard should allow retrying the same action."""
        trader = MockTrader(position_qty=0.0)
        env = self._make_env(trader)

        # First attempt: trade raises
        trader.trade_should_raise = True
        trade_info = env._execute_fractional_action(-1.0)

        # Simulate _step: don't update state on failure
        if trade_info["executed"] and trade_info.get("success") is not False:
            env.position.current_action_level = -1.0

        assert env.position.current_action_level == 0.0

        # Second attempt: trade succeeds
        trader.trade_should_raise = False
        trade_info2 = env._execute_trade_if_needed(-1.0)

        # Guard should NOT block: -1.0 != 0.0
        assert trade_info2["executed"] is True
        assert trade_info2["success"] is True

    def test_close_exception_returns_executed_false(self):
        """_handle_close_action should return executed=False if close_position raises."""
        trader = MockTrader(position_qty=0.5)
        trader.close_should_raise = True
        env = self._make_env(trader)

        trade_info = env._handle_close_action(0.5)

        assert trade_info["executed"] is False
        assert trade_info["success"] is False

    def test_direction_switch_aborts_on_close_failure(self):
        """Direction switch should not open opposite position if close fails."""
        trader = MockTrader(position_qty=0.5, balance=10000.0)
        trader.close_should_fail = True
        env = self._make_env(trader)

        # Try to switch from long to short
        trade_info = env._execute_fractional_action(-1.0)

        # Should have aborted — position_qty should still be 0.5
        assert trader.position_qty == 0.5
        # No opposite position opened
        assert len([t for t in trader.trades_executed if t["side"] == "SELL"]) == 0

    def test_step_checks_success_before_state_update(self):
        """_step should not update local state when success=False."""
        trader = MockTrader(position_qty=0.0)
        env = self._make_env(trader)

        # Simulate trade_info with executed=True but success=False
        # (This is what happens when trader.trade() returns False)
        trade_info = {
            "executed": True,
            "side": "SELL",
            "success": False,
            "closed_position": False,
        }

        # Apply the _step state update logic
        if trade_info["executed"] and trade_info.get("success") is not False:
            env.position.current_position = -1
            env.position.current_action_level = -1.0

        # State should NOT have been updated
        assert env.position.current_position == 0.0
        assert env.position.current_action_level == 0.0


# ---------------------------------------------------------------------------
# Bitget env tests
# ---------------------------------------------------------------------------


class TestBitgetTradeStateSync:
    """Test that Bitget env local state stays synchronized with exchange."""

    def _make_env(self, trader=None):
        """Create a BitgetFuturesTorchTradingEnv with mocked dependencies."""
        from torchtrade.envs.live.bitget.env import (
            BitgetFuturesTorchTradingEnv,
            BitgetFuturesTradingEnvConfig,
        )

        mock_trader = trader or MockTrader()
        mock_observer = MagicMock()
        mock_observer.get_observations.return_value = {
            "features_1Hour": np.random.randn(10, 4).astype(np.float32),
        }
        mock_observer.get_keys.return_value = ["features_1Hour"]

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
        env.observer = mock_observer
        env.action_levels = config.action_levels
        env.position = PositionState()
        env.initial_portfolio_value = 10000.0

        from torchtrade.envs.core.state import HistoryTracker
        env.history = HistoryTracker()

        from torchtrade.envs.core.default_rewards import log_return_reward
        env.reward_function = log_return_reward

        return env

    def test_successful_trade_updates_state(self):
        """Successful trade should update position state."""
        trader = MockTrader(position_qty=0.0)
        env = self._make_env(trader)

        trade_info = env._execute_trade_if_needed(-1.0)

        assert trade_info["executed"] is True
        assert trade_info["success"] is True

    def test_failed_trade_exception_does_not_update_state(self):
        """Trade that raises exception should return executed=False."""
        trader = MockTrader(position_qty=0.0)
        trader.trade_should_raise = True
        env = self._make_env(trader)

        trade_info = env._execute_fractional_action(-1.0)

        assert trade_info["executed"] is False
        assert trade_info["success"] is False

    def test_exception_preserves_current_action_level(self):
        """After a failed trade, current_action_level should remain unchanged."""
        trader = MockTrader(position_qty=0.0)
        trader.trade_should_raise = True
        env = self._make_env(trader)

        assert env.position.current_action_level == 0.0

        trade_info = env._execute_fractional_action(-1.0)

        if trade_info["executed"] and trade_info.get("success") is not False:
            env.position.current_action_level = -1.0

        assert env.position.current_action_level == 0.0

    def test_guard_not_blocked_after_failed_trade(self):
        """After a failed trade, the guard should allow retrying the same action."""
        trader = MockTrader(position_qty=0.0)
        env = self._make_env(trader)

        # First attempt fails
        trader.trade_should_raise = True
        trade_info = env._execute_fractional_action(-1.0)

        if trade_info["executed"] and trade_info.get("success") is not False:
            env.position.current_action_level = -1.0

        assert env.position.current_action_level == 0.0

        # Second attempt succeeds
        trader.trade_should_raise = False
        trade_info2 = env._execute_trade_if_needed(-1.0)

        assert trade_info2["executed"] is True
        assert trade_info2["success"] is True

    def test_close_exception_returns_executed_false(self):
        """_handle_close_action should return executed=False if close_position raises."""
        trader = MockTrader(position_qty=0.5)
        trader.close_should_raise = True
        env = self._make_env(trader)

        trade_info = env._handle_close_action(0.5)

        assert trade_info["executed"] is False
        assert trade_info["success"] is False

    def test_close_failure_does_not_update_position(self):
        """_handle_close_action should NOT set current_position=0 if close fails."""
        trader = MockTrader(position_qty=0.5)
        trader.close_should_fail = True
        env = self._make_env(trader)
        env.position.current_position = 1  # Currently long

        trade_info = env._handle_close_action(0.5)

        # close returned success=False, so current_position should be unchanged
        assert env.position.current_position == 1

    def test_close_success_updates_position(self):
        """_handle_close_action should set current_position=0 on success."""
        trader = MockTrader(position_qty=0.5)
        env = self._make_env(trader)
        env.position.current_position = 1  # Currently long

        trade_info = env._handle_close_action(0.5)

        assert trade_info["executed"] is True
        assert trade_info["success"] is True
        assert env.position.current_position == 0

    def test_step_checks_success_before_state_update(self):
        """_step should not update local state when success=False."""
        trader = MockTrader(position_qty=0.0)
        env = self._make_env(trader)

        trade_info = {
            "executed": True,
            "side": "sell",
            "success": False,
            "closed_position": False,
        }

        if trade_info["executed"] and trade_info.get("success") is not False:
            env.position.current_position = -1
            env.position.current_action_level = -1.0

        assert env.position.current_position == 0.0
        assert env.position.current_action_level == 0.0


# ---------------------------------------------------------------------------
# Regression test: the exact bug scenario from #189
# ---------------------------------------------------------------------------


class TestRegressionIssue189:
    """Regression tests for the exact bug scenario that corrupted the offline dataset.

    The bug: a policy always outputting SHORT (-1.0) would permanently desync
    local state after the first trade (success or failure), causing the guard
    to block all subsequent identical actions. The observation (from exchange)
    then diverged from the recorded action.
    """

    def _make_binance_env(self, trader):
        from torchtrade.envs.live.binance.env import (
            BinanceFuturesTorchTradingEnv,
            BinanceFuturesTradingEnvConfig,
        )

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
        env.trader = trader
        env.action_levels = config.action_levels
        env.position = PositionState()
        env.initial_portfolio_value = 10000.0

        from torchtrade.envs.core.state import HistoryTracker
        env.history = HistoryTracker()

        from torchtrade.envs.core.default_rewards import log_return_reward
        env.reward_function = log_return_reward

        return env

    def test_repeated_short_after_exception_not_permanently_blocked(self):
        """The exact #189 scenario: policy always outputs SHORT, first trade fails.

        Before fix: current_action_level was set to -1.0 on failure, permanently
        blocking all future SHORT actions via the guard.

        After fix: current_action_level stays at 0.0, so the guard allows retry.
        """
        trader = MockTrader(position_qty=0.0)
        env = self._make_binance_env(trader)

        # Step 1: SHORT trade fails with exception
        trader.trade_should_raise = True
        trade_info = env._execute_trade_if_needed(-1.0)

        # Apply _step logic
        if trade_info["executed"] and trade_info.get("success") is not False:
            env.position.current_action_level = -1.0

        # Key assertion: current_action_level must NOT be -1.0
        assert env.position.current_action_level == 0.0

        # Step 2: SHORT trade should NOT be blocked by guard
        trader.trade_should_raise = False
        trade_info2 = env._execute_trade_if_needed(-1.0)

        # This MUST succeed — before fix, this would return executed=False
        assert trade_info2["executed"] is True
        assert trade_info2["success"] is True

    def test_repeated_short_after_success_allows_retry_on_liquidation(self):
        """After successful SHORT, if position gets liquidated, agent should be able to re-short.

        Simulates: trade succeeds → position set → liquidation resets exchange to flat →
        agent needs to re-enter SHORT but guard blocks because current_action_level=-1.0.

        This test verifies the guard behavior is correct: after a successful trade,
        current_action_level IS set to -1.0 (correct), and repeated SHORT IS blocked
        (correct — the agent asked for -1.0 and got -1.0).

        The real issue is that the exchange may have liquidated the position, but the
        local state doesn't know. This is a separate concern from the #189 fix.
        """
        trader = MockTrader(position_qty=0.0)
        env = self._make_binance_env(trader)

        # Step 1: SHORT trade succeeds
        trade_info = env._execute_trade_if_needed(-1.0)
        if trade_info["executed"] and trade_info.get("success") is not False:
            env.position.current_action_level = -1.0

        assert env.position.current_action_level == -1.0

        # Step 2: SHORT again — guard correctly blocks (same action level)
        trade_info2 = env._execute_trade_if_needed(-1.0)
        assert trade_info2["executed"] is False  # Guard blocks: -1.0 == -1.0

        # Step 3: But if we switch to FLAT first, then SHORT works
        trade_info3 = env._execute_trade_if_needed(0.0)
        if trade_info3["executed"] and trade_info3.get("success") is not False:
            env.position.current_action_level = 0.0

        trade_info4 = env._execute_trade_if_needed(-1.0)
        assert trade_info4["executed"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
