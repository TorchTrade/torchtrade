"""Comprehensive tests for portfolio value and balance accounting.

Tests the fundamental accounting invariants that should hold for all trading environments:
1. Balance deduction when opening positions (spot vs futures)
2. Margin return when closing positions
3. Portfolio value calculation correctness
4. Conservation of value (no money creation from bugs)
5. History recording (initial state and position exits)

These tests would have caught the bugs fixed in commit 55dd3bf.
"""

import pytest
import pandas as pd
from torchtrade.envs.offline.sequential import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.offline.sequential_sltp import SequentialTradingEnvSLTP, SequentialTradingEnvSLTPConfig


@pytest.fixture
def constant_price_df():
    """OHLCV data with constant price for testing accounting."""
    dates = pd.date_range('2024-01-01', periods=50, freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [100.0] * 50,
        'high': [101.0] * 50,
        'low': [99.0] * 50,
        'close': [100.0] * 50,
        'volume': [1000.0] * 50,
    })
    return df


@pytest.fixture
def price_change_df():
    """OHLCV data with deliberate price changes for PnL testing."""
    dates = pd.date_range('2024-01-01', periods=10, freq='1h')
    prices = [100, 100, 110, 110, 90, 90, 100, 100, 105, 105]
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000.0] * 10,
    })
    return df


class TestSpotAccountingSequential:
    """Test spot trading (leverage=1) accounting for Sequential env."""

    @pytest.fixture
    def spot_env(self, constant_price_df):
        """Spot trading environment with no fees."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=1,  # Spot
            action_levels=[-1, 0, 1],
        )
        return SequentialTradingEnv(constant_price_df, config)

    def test_balance_deducted_when_opening_long(self, spot_env):
        """Opening long position should deduct full notional from balance (spot)."""
        td = spot_env.reset()
        initial_balance = spot_env.balance

        # Open long (action index 2 for action_level=1)
        td["action"] = 2
        td = spot_env.step(td)["next"]

        # Balance should be ~0 (spent all cash buying)
        assert spot_env.balance < 100, f"Balance should be nearly 0 after buying, got {spot_env.balance}"

        # Position should exist
        assert spot_env.position.position_size > 0, "Should have opened long position"

    def test_portfolio_value_preserved_constant_price(self, spot_env):
        """Portfolio value should stay at initial_cash with constant price (spot)."""
        td = spot_env.reset()
        initial_pv = spot_env.history.portfolio_values[-1]

        # Open long
        td["action"] = 2
        td = spot_env.step(td)["next"]
        pv_after_open = spot_env.history.portfolio_values[-1]

        # Hold
        td["action"] = 1  # action_level=0 (hold)
        td = spot_env.step(td)["next"]
        pv_after_hold = spot_env.history.portfolio_values[-1]

        # Portfolio value should be preserved (constant price, no fees)
        assert abs(pv_after_open - initial_pv) < 0.01, \
            f"PV should stay ~{initial_pv:.4f} after open, got {pv_after_open:.4f}"
        assert abs(pv_after_hold - initial_pv) < 0.01, \
            f"PV should stay ~{initial_pv:.4f} after hold, got {pv_after_hold:.4f}"

    def test_margin_returned_when_closing(self, spot_env):
        """Closing position should return margin to balance (spot)."""
        td = spot_env.reset()
        initial_balance = spot_env.balance

        # Open long
        td["action"] = 2
        td = spot_env.step(td)["next"]
        balance_after_open = spot_env.balance

        # Close (action index 0 for action_level=-1)
        td["action"] = 0
        td = spot_env.step(td)["next"]
        balance_after_close = spot_env.balance

        # Balance should be back to ~initial (no price change, no fees)
        assert abs(balance_after_close - initial_balance) < 0.01, \
            f"Balance should return to ~{initial_balance:.4f}, got {balance_after_close:.4f}"

        # Position should be flat
        assert spot_env.position.position_size == 0, "Position should be closed"

    def test_portfolio_value_formula_spot(self, price_change_df):
        """Test PV = balance + (position_size × current_price) for spot."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[1],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=1,
            action_levels=[-1, 0, 1],
        )
        env = SequentialTradingEnv(price_change_df, config)

        td = env.reset()

        # Open at price ~100
        td["action"] = 2  # Long
        td = env.step(td)["next"]

        # Check formula: PV = balance + position_value
        current_price = env.history.base_prices[-1]
        expected_pv = env.balance + (env.position.position_size * current_price)
        actual_pv = env.history.portfolio_values[-1]

        assert abs(actual_pv - expected_pv) < 0.01, \
            f"PV formula incorrect: expected {expected_pv:.4f}, got {actual_pv:.4f}"


class TestFuturesAccountingSequential:
    """Test futures trading (leverage>1) accounting for Sequential env."""

    @pytest.fixture
    def futures_env(self, constant_price_df):
        """Futures environment with 2x leverage."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=2,  # Futures
            action_levels=[-1, 0, 1],
        )
        return SequentialTradingEnv(constant_price_df, config)

    def test_balance_deducted_margin_only(self, constant_price_df):
        """Opening position should only deduct margin (notional/leverage) for futures.

        This test verifies that futures mode uses leverage correctly:
        - Spot (leverage=1): deducts full notional value
        - Futures (leverage=2): deducts only 50% (margin)
        """
        # Create futures env with 2x leverage
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=2,  # Futures
            action_levels=[-1, -0.5, 0, 0.5, 1],  # Use 0.5 to allocate only 50%
        )
        futures_env = SequentialTradingEnv(constant_price_df, config)

        td = futures_env.reset()
        initial_balance = futures_env.balance

        # Open long with 50% allocation (action index 3 for action_level=0.5)
        # With leverage=2, this will use 50% of balance as margin for 100% position
        td["action"] = 3
        td = futures_env.step(td)["next"]

        position_notional = abs(futures_env.position.position_size * futures_env.position.entry_price)
        # With action_level=0.5, we allocate 50% of balance
        # margin_required = (balance × 0.5) = 5000
        # With leverage=2, notional = margin × 2 = 10000
        expected_margin = 5000
        expected_balance = initial_balance - expected_margin

        # Balance should be reduced by margin only (not full notional)
        assert abs(futures_env.balance - expected_balance) < 50, \
            f"Balance should be ~{expected_balance:.2f} (margin deducted), got {futures_env.balance:.2f}"

        # Should have 50% balance left (used other 50% as margin)
        assert futures_env.balance > 4000, \
            f"Futures should deduct margin only (50% for action_level=0.5), balance={futures_env.balance}"

        # Verify notional is 2x the margin (leverage effect)
        assert abs(position_notional - expected_margin * 2) < 100, \
            f"Notional should be 2x margin due to leverage=2, got {position_notional}"

    def test_portfolio_value_formula_futures(self, price_change_df):
        """Test PV = free_margin + locked_margin + unrealized_pnl for futures.

        In futures mode:
        - balance = free margin (remaining cash)
        - locked_margin = (position_notional / leverage) - deducted when opening
        - unrealized_pnl = (current_price - entry_price) × position_size
        - PV = balance + locked_margin + unrealized_pnl
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[1],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=2,
            action_levels=[-1, 0, 1],
        )
        env = SequentialTradingEnv(price_change_df, config)

        td = env.reset()

        # Open at price ~100
        td["action"] = 2  # Long
        td = env.step(td)["next"]

        # Check formula: PV = balance + locked_margin + unrealized_pnl
        current_price = env.history.base_prices[-1]
        position_notional = abs(env.position.position_size * env.position.entry_price)
        locked_margin = position_notional / 2  # leverage=2
        unrealized_pnl = (current_price - env.position.entry_price) * env.position.position_size
        expected_pv = env.balance + locked_margin + unrealized_pnl
        actual_pv = env.history.portfolio_values[-1]

        assert abs(actual_pv - expected_pv) < 0.01, \
            f"PV formula incorrect: expected {expected_pv:.4f}, got {actual_pv:.4f}"


class TestSpotAccountingSLTP:
    """Test spot trading accounting for SequentialSLTP env."""

    @pytest.fixture
    def sltp_spot_env(self, constant_price_df):
        """SLTP spot environment."""
        config = SequentialTradingEnvSLTPConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=1,  # Spot
            stoploss_levels=[-0.5],  # Large to avoid trigger
            takeprofit_levels=[0.5],
            include_hold_action=True,
        )
        return SequentialTradingEnvSLTP(constant_price_df, config)

    def test_sltp_balance_deducted_spot(self, sltp_spot_env):
        """SLTP spot should deduct full notional when opening."""
        td = sltp_spot_env.reset()
        initial_balance = sltp_spot_env.balance

        # Open long with SLTP
        td["action"] = 1  # First bracket order
        td = sltp_spot_env.step(td)["next"]

        # Balance should be ~0 (full cost deducted)
        assert sltp_spot_env.balance < 100, \
            f"SLTP spot should deduct full cost, balance={sltp_spot_env.balance}"

    def test_sltp_portfolio_value_constant_price(self, sltp_spot_env):
        """SLTP portfolio value should be preserved with constant price."""
        td = sltp_spot_env.reset()
        initial_pv = sltp_spot_env.history.portfolio_values[-1]

        # Open
        td["action"] = 1
        td = sltp_spot_env.step(td)["next"]
        pv_after_open = sltp_spot_env.history.portfolio_values[-1]

        # Hold
        td["action"] = 0
        td = sltp_spot_env.step(td)["next"]
        pv_after_hold = sltp_spot_env.history.portfolio_values[-1]

        assert abs(pv_after_open - initial_pv) < 0.01, \
            f"SLTP PV should stay ~{initial_pv:.4f}, got {pv_after_open:.4f}"
        assert abs(pv_after_hold - initial_pv) < 0.01, \
            f"SLTP PV should stay ~{initial_pv:.4f}, got {pv_after_hold:.4f}"


class TestHistoryRecording:
    """Test history tracking correctness."""

    def test_initial_state_recorded(self, constant_price_df):
        """Reset should record initial state to history."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=1,
            action_levels=[-1, 0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()

        # History should have initial state
        assert len(env.history.portfolio_values) == 1, \
            "History should contain initial state after reset"
        assert env.history.portfolio_values[0] == env.initial_cash, \
            f"Initial PV should be {env.initial_cash}, got {env.history.portfolio_values[0]}"
        assert env.history.actions[0] == 0.0, \
            "Initial action should be 0 (hold)"
        assert env.history.action_types[0] == "hold", \
            "Initial action_type should be 'hold'"

    def test_position_exits_recorded_in_history(self, constant_price_df):
        """Position closes should be recorded with correct action_type.

        Spot mode (leverage=1): Uses 'buy'/'sell' terminology
        Futures mode (leverage>1): Uses 'long'/'short'/'close' terminology
        """
        # Test spot mode
        spot_config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=1,  # Spot
            action_levels=[-1, 0, 1],
        )
        spot_env = SequentialTradingEnv(constant_price_df, spot_config)

        td = spot_env.reset()

        # Open
        td["action"] = 2  # Buy
        td = spot_env.step(td)["next"]

        # Close
        td["action"] = 0  # Sell
        td = spot_env.step(td)["next"]

        # In spot mode, closing is recorded as "sell"
        assert "sell" in spot_env.history.action_types, \
            f"Spot close should be 'sell', got {spot_env.history.action_types}"

        # Test futures mode
        futures_config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=2,  # Futures
            action_levels=[-1, 0, 1],
        )
        futures_env = SequentialTradingEnv(constant_price_df, futures_config)

        td = futures_env.reset()

        # Open
        td["action"] = 2  # Long
        td = futures_env.step(td)["next"]

        # Close to flat
        td["action"] = 1  # Neutral (action_level=0)
        td = futures_env.step(td)["next"]

        # In futures mode, closing to flat is recorded as "flat"
        assert any(at in ["flat", "close"] for at in futures_env.history.action_types), \
            f"Futures close should be 'flat' or 'close', got {futures_env.history.action_types}"

    def test_sltp_triggers_recorded(self, price_change_df):
        """SL/TP triggers should be recorded as sltp_sl or sltp_tp."""
        config = SequentialTradingEnvSLTPConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[1],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=1,
            stoploss_levels=[-0.05],  # 5% SL
            takeprofit_levels=[0.05],  # 5% TP
            include_hold_action=True,
        )

        # Create data that will trigger TP
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')
        prices = [100, 100, 106, 106, 106, 106, 106, 106, 106, 106]  # +6% gain
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000.0] * 10,
        })

        env = SequentialTradingEnvSLTP(df, config)
        td = env.reset()

        # Open long
        td["action"] = 1
        td = env.step(td)["next"]

        # Hold until TP triggers
        for _ in range(5):
            if td["done"]:
                break
            td["action"] = 0
            td = env.step(td)["next"]

        # Check if TP was recorded
        has_sltp_exit = any(at in ["sltp_sl", "sltp_tp"] for at in env.history.action_types)
        assert has_sltp_exit, \
            f"SL/TP trigger should be recorded, got {env.history.action_types}"


class TestValueConservation:
    """Test that value is conserved (no money creation from bugs)."""

    def test_no_value_creation_with_fees(self, constant_price_df):
        """Portfolio value should only decrease by fees when price is constant."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[1],
            initial_cash=10000,
            transaction_fee=0.001,  # 0.1% fee
            slippage=0.0,
            leverage=1,
            action_levels=[-1, 0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        initial_pv = env.history.portfolio_values[-1]

        # Open position
        td["action"] = 2
        td = env.step(td)["next"]
        pv_after_open = env.history.portfolio_values[-1]

        # Check that PV only decreased by fee (price is constant at open)
        # Fee should be ~0.1% of notional (10000 * 0.001 = 10)
        expected_pv = initial_pv - (initial_pv * 0.001)  # Lost to fee

        # Allow small tolerance for rounding
        assert abs(pv_after_open - expected_pv) < 50, \
            f"PV should decrease by fee only, expected ~{expected_pv:.2f}, got {pv_after_open:.2f}"

        # Value should not increase magically (the bug we fixed)
        assert pv_after_open <= initial_pv, \
            "PV should not increase from opening position (no price change)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
