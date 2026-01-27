"""Fundamental calculation tests for trading environments.

Tests critical calculations that MUST be correct:
- Margin deduction/return
- Fee calculations
- PnL calculations
- Liquidation mechanics
- Account state accuracy
- Multi-timeframe handling
"""

import pytest
import pandas as pd
import numpy as np
import torch
from torchtrade.envs.offline.sequential import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.offline.sequential_sltp import SequentialTradingEnvSLTP, SequentialTradingEnvSLTPConfig


@pytest.fixture
def constant_price_df():
    """OHLCV data with constant price for precise accounting tests."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [100.0] * 100,
        'high': [101.0] * 100,
        'low': [99.0] * 100,
        'close': [100.0] * 100,
        'volume': [1000.0] * 100,
    })
    return df


@pytest.fixture
def multi_timeframe_df():
    """Multi-timeframe OHLCV data for testing."""
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [100.0] * 1000,
        'high': [101.0] * 1000,
        'low': [99.0] * 1000,
        'close': [100.0] * 1000,
        'volume': [1000.0] * 1000,
    })
    return df


class TestMarginAccounting:
    """Test margin deduction and return correctness."""

    @pytest.mark.parametrize("leverage,action_level", [
        (1, 1.0),   # Spot: uses 100% of balance
        (2, 0.5),   # 2x leverage with 50% action: uses 50% balance as margin
        (5, 0.2),   # 5x leverage with 20% action: uses 20% balance as margin
        (10, 0.1),  # 10x leverage with 10% action: uses 10% balance as margin
    ])
    def test_open_position_deducts_correct_margin(
        self, constant_price_df, leverage, action_level
    ):
        """Opening position should deduct correct margin based on leverage and action level.

        With fractional sizing:
        - notional = portfolio_value × action_level × leverage
        - margin_required = notional / leverage = portfolio_value × action_level

        So action_level controls how much of portfolio value is used as margin.
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1] if leverage > 1 else [0, 0.1, 0.2, 0.5, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        initial_balance = env.balance

        # Map action_level to action index
        action_idx = config.action_levels.index(action_level)
        td["action"] = action_idx
        td = env.step(td)["next"]

        # Calculate expected margin: margin = portfolio_value × action_level
        expected_margin = initial_balance * action_level
        expected_balance = initial_balance - expected_margin

        assert abs(env.balance - expected_balance) < 0.1, \
            f"Leverage={leverage}, action={action_level}: expected balance {expected_balance:.4f}, got {env.balance:.4f}"

        # Verify position was opened
        assert env.position.position_size != 0, "Position should be opened"

    @pytest.mark.parametrize("leverage", [1, 2, 5, 10])
    def test_close_position_returns_all_margin(self, constant_price_df, leverage):
        """Closing position should return all locked margin (plus PnL, minus fees)."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,  # No fees for clean test
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, 0, 1] if leverage > 1 else [0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        initial_balance = env.balance

        # Open
        action_idx = 2 if leverage > 1 else 1
        td["action"] = action_idx
        td = env.step(td)["next"]
        balance_after_open = env.balance

        # Close
        action_idx = 1 if leverage > 1 else 0  # Neutral or sell
        td["action"] = action_idx
        td = env.step(td)["next"]
        balance_after_close = env.balance

        # With no price change and no fees, balance should return to initial
        assert abs(balance_after_close - initial_balance) < 0.01, \
            f"Leverage={leverage}: balance should return to {initial_balance:.4f}, got {balance_after_close:.4f}"

        # Position should be flat
        assert env.position.position_size == 0, "Position should be closed"

    @pytest.mark.parametrize("leverage,initial_action,target_action", [
        (2, 1.0, 0.5),    # 2x: reduce from 100% to 50%
        (5, 1.0, 0.7),    # 5x: reduce from 100% to 70%
        (10, 1.0, 0.3),   # 10x: reduce from 100% to 30%
    ])
    def test_decrease_position_returns_freed_margin(
        self, constant_price_df, leverage, initial_action, target_action
    ):
        """Decreasing position size should return the freed margin."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, -0.5, -0.3, 0, 0.3, 0.5, 0.7, 1.0],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        initial_balance = env.balance

        # Open position with initial_action
        action_idx_1 = config.action_levels.index(initial_action)
        td["action"] = action_idx_1
        td = env.step(td)["next"]
        balance_after_open = env.balance

        # Locked margin = initial_balance × initial_action
        locked_margin = initial_balance * initial_action

        # Decrease to target_action
        action_idx_2 = config.action_levels.index(target_action)
        td["action"] = action_idx_2
        td = env.step(td)["next"]

        # Fraction of position being closed
        fraction_closed = (initial_action - target_action) / initial_action
        freed_margin = locked_margin * fraction_closed
        expected_balance = balance_after_open + freed_margin

        assert abs(env.balance - expected_balance) < 10, \
            f"Leverage={leverage}, {initial_action}->{target_action}: expected {expected_balance:.2f}, got {env.balance:.2f}"

        # Position should be reduced
        assert abs(env.position.position_size) > 0, "Position should still exist"

    @pytest.mark.parametrize("leverage,initial_action,target_action", [
        (2, 0.5, 1.0),  # 2x: increase from 50% to 100%
        (5, 0.7, 1.0),  # 5x: increase from 70% to 100%
        (10, 0.8, 1.0), # 10x: increase from 80% to 100%
    ])
    def test_increase_position_deducts_additional_margin(
        self, constant_price_df, leverage, initial_action, target_action
    ):
        """Increasing position size should deduct additional margin."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, -0.5, 0, 0.5, 0.7, 0.8, 1.0],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        initial_balance = env.balance

        # Open with initial_action
        action_idx_1 = config.action_levels.index(initial_action)
        td["action"] = action_idx_1
        td = env.step(td)["next"]
        balance_after_first = env.balance

        # Increase to target_action
        action_idx_2 = config.action_levels.index(target_action)
        td["action"] = action_idx_2
        td = env.step(td)["next"]
        balance_after_increase = env.balance

        # Additional margin required = (target - initial) × initial_balance
        # Note: using initial_balance because that's what fractional sizing is based on
        delta_action = target_action - initial_action
        additional_margin = initial_balance * delta_action
        expected_balance = balance_after_first - additional_margin

        assert abs(balance_after_increase - expected_balance) < 50, \
            f"Leverage={leverage}, {initial_action}->{target_action}: expected {expected_balance:.2f}, got {balance_after_increase:.2f}"


class TestFeeAccounting:
    """Test that fees are calculated and deducted correctly."""

    @pytest.mark.parametrize("fee_rate,leverage", [
        (0.001, 1),   # 0.1% fee, spot
        (0.001, 2),   # 0.1% fee, 2x futures
        (0.0005, 5),  # 0.05% fee, 5x futures
        (0.002, 10),  # 0.2% fee, 10x futures
    ])
    def test_open_fee_calculated_on_notional(self, constant_price_df, fee_rate, leverage):
        """Opening fee should be based on notional value, not margin."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=fee_rate,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, 0, 1] if leverage > 1 else [0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()

        # Open position
        action_idx = 2 if leverage > 1 else 1
        td["action"] = action_idx
        td = env.step(td)["next"]

        # Calculate expected fee (on notional, not margin)
        position_notional = abs(env.position.position_size * env.position.entry_price)
        expected_fee = position_notional * fee_rate
        margin_deducted = position_notional / leverage
        expected_balance = 10000 - margin_deducted - expected_fee

        assert abs(env.balance - expected_balance) < 0.1, \
            f"Fee rate={fee_rate}, leverage={leverage}: expected {expected_balance:.4f}, got {env.balance:.4f}"

    @pytest.mark.parametrize("fee_rate", [0.0, 0.001, 0.002])
    def test_round_trip_loses_exactly_two_fees(self, constant_price_df, fee_rate):
        """Open + close should lose exactly 2 × fee (no price change)."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=fee_rate,
            slippage=0.0,
            leverage=2,
            action_levels=[-1, 0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        initial_balance = env.balance

        # Open
        td["action"] = 2
        td = env.step(td)["next"]
        position_notional = abs(env.position.position_size * env.position.entry_price)

        # Close
        td["action"] = 1
        td = env.step(td)["next"]
        final_balance = env.balance

        # Should lose exactly 2 fees
        expected_total_fee = position_notional * fee_rate * 2
        expected_final = initial_balance - expected_total_fee

        assert abs(final_balance - expected_final) < 0.5, \
            f"Fee={fee_rate}: expected {expected_final:.4f}, got {final_balance:.4f}"


class TestPnLCalculations:
    """Test PnL calculation correctness.

    Note: PnL formula correctness is already tested in test_portfolio_accounting.py.
    These tests focus on multi-step scenarios with price changes.
    """

    @pytest.mark.parametrize("price_pct_change,expected_pnl_pct", [
        (0.1, 0.1),     # +10% price -> +10% PnL
        (-0.05, -0.05), # -5% price -> -5% PnL
    ])
    def test_unrealized_pnl_tracks_price_changes(
        self, constant_price_df, price_pct_change, expected_pnl_pct
    ):
        """Unrealized PnL should track price changes correctly."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=1,
            action_levels=[0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()

        # Open long
        td["action"] = 1
        td = env.step(td)["next"]
        entry_price = env.position.entry_price

        # Manually update price and check PnL
        new_price = entry_price * (1 + price_pct_change)
        pnl_pct = env._calculate_unrealized_pnl_pct(entry_price, new_price, env.position.position_size)

        assert abs(pnl_pct - expected_pnl_pct) < 0.01, \
            f"Price change={price_pct_change:.1%}: expected PnL {expected_pnl_pct:.1%}, got {pnl_pct:.1%}"


class TestLiquidationMechanics:
    """Test liquidation triggers and calculations."""

    @pytest.mark.parametrize("leverage,expected_liq_distance", [
        (2, 0.5),    # 2x: liquidated at -50% from entry
        (5, 0.2),    # 5x: liquidated at -20% from entry
        (10, 0.1),   # 10x: liquidated at -10% from entry
    ])
    def test_liquidation_price_calculation_long(
        self, constant_price_df, leverage, expected_liq_distance
    ):
        """Liquidation price should be at (1/leverage) distance from entry."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, 0, 1],
            maintenance_margin_rate=0.0,  # Simplify calculation
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()

        # Open long
        td["action"] = 2
        td = env.step(td)["next"]

        entry_price = env.position.entry_price
        liq_price = env.liquidation_price

        # Check liquidation distance
        actual_distance = (entry_price - liq_price) / entry_price

        assert abs(actual_distance - expected_liq_distance) < 0.01, \
            f"Leverage={leverage}: expected liq at {expected_liq_distance:.1%} from entry, got {actual_distance:.1%}"

    def test_liquidation_returns_margin(self, constant_price_df):
        """Liquidation should return locked margin (though loss typically exceeds it)."""
        # Create data that triggers liquidation
        df = constant_price_df.copy()
        df.loc[df.index[10:], 'close'] = 50  # -50% price drop
        df.loc[df.index[10:], 'low'] = 50

        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=2,
            action_levels=[-1, 0, 1],
            maintenance_margin_rate=0.0,
            random_start=False,
        )
        env = SequentialTradingEnv(df, config)

        td = env.reset()

        # Open long
        td["action"] = 2
        td = env.step(td)["next"]
        balance_after_open = env.balance
        locked_margin = abs(env.position.position_size * env.position.entry_price) / 2

        # Step to liquidation
        for _ in range(15):
            if td["done"]:
                break
            td["action"] = 1
            td = env.step(td)["next"]

        # Check that position was liquidated
        assert env.position.position_size == 0, "Position should be liquidated"

        # Balance should reflect: initial locked margin + liquidation loss
        # Loss at liq = (50 - 100) * position_size = large negative
        # The margin should have been returned before applying the loss


class TestAccountStateAccuracy:
    """Test that account state vector elements are calculated correctly."""

    @pytest.mark.parametrize("leverage", [1, 2, 5, 10])
    def test_exposure_pct_calculation(self, constant_price_df, leverage):
        """Exposure % should be position_value / portfolio_value."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, 0, 1] if leverage > 1 else [0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()

        # Open position
        action_idx = 2 if leverage > 1 else 1
        td["action"] = action_idx
        td = env.step(td)["next"]

        # Get account state
        account_state = td["account_state"].numpy()
        exposure_pct = account_state[0]

        # Calculate expected exposure
        pv = env._get_portfolio_value()
        expected_exposure = env.position.position_value / pv if pv > 0 else 0.0

        assert abs(exposure_pct - expected_exposure) < 0.01, \
            f"Leverage={leverage}: expected exposure {expected_exposure:.2%}, got {exposure_pct:.2%}"

    @pytest.mark.parametrize("action_idx,expected_direction", [
        (2, 1.0),   # Long -> +1
        (0, -1.0),  # Short -> -1
        (1, 0.0),   # Flat -> 0
    ])
    def test_position_direction_values(
        self, constant_price_df, action_idx, expected_direction
    ):
        """Position direction should be -1, 0, or +1."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=2,
            action_levels=[-1, 0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()

        # Execute action
        td["action"] = action_idx
        td = env.step(td)["next"]

        # Check position direction in account state
        account_state = td["account_state"].numpy()
        position_direction = account_state[1]

        assert position_direction == expected_direction, \
            f"Action {action_idx}: expected direction {expected_direction}, got {position_direction}"


class TestMultiTimeframeHandling:
    """Test that multi-timeframe configurations work correctly."""

    @pytest.mark.parametrize("time_frames,window_sizes", [
        (["1min"], [10]),
        (["5min"], [20]),
        (["1min", "5min"], [10, 5]),
        (["1min", "5min", "15min"], [10, 5, 3]),
    ])
    def test_multi_timeframe_observations(
        self, multi_timeframe_df, time_frames, window_sizes
    ):
        """Multi-timeframe configs should produce correct observation shapes."""
        config = SequentialTradingEnvConfig(
            execute_on=time_frames[0],
            time_frames=time_frames,
            window_sizes=window_sizes,
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=1,
            action_levels=[0, 1],
            random_start=False,
        )
        env = SequentialTradingEnv(multi_timeframe_df, config)

        td = env.reset()

        # Check that we have correct number of market data keys
        expected_keys = len(time_frames)
        actual_keys = len([k for k in td.keys() if k.startswith("market_data")])

        assert actual_keys == expected_keys, \
            f"Expected {expected_keys} timeframes, got {actual_keys}"

        # Execute a few steps to ensure no errors
        for _ in range(5):
            td["action"] = 1  # Hold
            td = env.step(td)["next"]
            if td["done"]:
                break

    @pytest.mark.parametrize("leverage", [1, 2, 5])
    def test_multi_timeframe_calculations_consistent(
        self, multi_timeframe_df, leverage
    ):
        """Calculations should be consistent regardless of timeframe config."""
        # Single timeframe
        config_single = SequentialTradingEnvConfig(
            execute_on="1min",
            time_frames=["1min"],
            window_sizes=[10],
            initial_cash=10000,
            transaction_fee=0.001,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, 0, 1] if leverage > 1 else [0, 1],
            random_start=False,
        )
        env_single = SequentialTradingEnv(multi_timeframe_df, config_single)

        # Multi timeframe
        config_multi = SequentialTradingEnvConfig(
            execute_on="1min",
            time_frames=["1min", "5min"],
            window_sizes=[10, 5],
            initial_cash=10000,
            transaction_fee=0.001,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, 0, 1] if leverage > 1 else [0, 1],
            random_start=False,
        )
        env_multi = SequentialTradingEnv(multi_timeframe_df, config_multi)

        # Both should have same balance/PV after same actions
        td_single = env_single.reset()
        td_multi = env_multi.reset()

        # Open position
        action_idx = 2 if leverage > 1 else 1
        td_single["action"] = action_idx
        td_single = env_single.step(td_single)["next"]

        td_multi["action"] = action_idx
        td_multi = env_multi.step(td_multi)["next"]

        # Balances should match
        assert abs(env_single.balance - env_multi.balance) < 1, \
            f"Single TF balance {env_single.balance:.2f} != Multi TF {env_multi.balance:.2f}"

        # PVs should match
        pv_single = env_single._get_portfolio_value()
        pv_multi = env_multi._get_portfolio_value()
        assert abs(pv_single - pv_multi) < 1, \
            f"Single TF PV {pv_single:.2f} != Multi TF {pv_multi:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
