"""Tests for fractional position sizing in non-SLTP environments.

TODO: Add tests for live environments (Binance, Bitget, Alpaca) with mocked APIs
    - Exchange-specific rounding and constraints (step size, min notional)
    - Query-first pattern behavior
    - Balance synchronization handling
    - Direction switching edge cases
    - Order rejection scenarios
"""

import pytest
import pandas as pd
import numpy as np
import torch

from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

# Aliases for backwards compatibility
SeqFuturesEnv = SequentialTradingEnv
SeqFuturesEnvConfig = SequentialTradingEnvConfig
SeqLongOnlyEnv = SequentialTradingEnv
SeqLongOnlyEnvConfig = SequentialTradingEnvConfig

# Test tolerance constants
POSITION_TOLERANCE = 0.001  # 0.1% tolerance for position size comparisons
BALANCE_TOLERANCE = 0.01    # 1% tolerance for balance/value comparisons
PRICE_TOLERANCE = 0.02      # 2% tolerance for price/ratio comparisons (accounts for fees)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1Min')
    close_prices = 50000 + np.cumsum(np.random.randn(1000) * 100)  # Random walk around 50k

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.randn(1000) * 10,
        'high': close_prices + np.abs(np.random.randn(1000) * 20),
        'low': close_prices - np.abs(np.random.randn(1000) * 20),
        'close': close_prices,
        'volume': np.random.randint(100, 1000, 1000)
    })
    return df


class TestFractionalActionMapping:
    """Test that action values correctly map to position sizes."""

    @pytest.mark.parametrize("action_level,action_idx,description", [
        (1.0, 4, "100% long"),
        (0.5, 3, "50% long"),
        (-0.5, 1, "50% short"),
        (-1.0, 0, "100% short"),
    ])
    def test_action_maps_to_position_size(self, sample_df, action_level, action_idx, description):
        """Action levels should map to correct position sizes (% of balance * leverage)."""
        # Shared config for all action tests
        config = SeqFuturesEnvConfig(
            action_levels=[-1.0, -0.5, 0.0, 0.5, 1.0],
            leverage=2,  # Futures mode (required for shorts)
            initial_cash=10000,
            transaction_fee=0.0,  # Disable fees for simpler calculation
            slippage=0.0,  # Disable slippage
            random_start=False,
            max_traj_length=10
        )
        env = SeqFuturesEnv(sample_df, config)

        td = env.reset()
        initial_balance = env.balance
        current_price = env._cached_base_features["close"]

        # Execute action
        td["action"] = torch.tensor([action_idx])
        td = env.step(td)["next"]

        # Calculate expected position: (balance * action_level * leverage) / price
        expected_position = (initial_balance * action_level * config.leverage) / current_price
        actual_position = env.position.position_size

        # Verify position size
        if expected_position != 0:
            assert abs(actual_position - expected_position) / abs(expected_position) < POSITION_TOLERANCE, \
                f"{description}: expected {expected_position:.4f}, got {actual_position:.4f}"
        else:
            assert actual_position == 0.0, f"{description}: expected 0.0, got {actual_position:.4f}"

        # Verify sign for short positions
        if action_level < 0:
            assert actual_position < 0, f"{description}: position should be negative (short)"

    def test_action_0_0_closes_position(self, sample_df):
        """Action 0.0 should close all positions (market neutral)."""
        config = SeqFuturesEnvConfig(
            action_levels=[-1.0, -0.5, 0.0, 0.5, 1.0],
            leverage=1,
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )
        env = SeqFuturesEnv(sample_df, config)

        td = env.reset()

        # First, open a long position
        td["action"] = torch.tensor([4])  # 1.0 = 100% long
        td = env.step(td)["next"]
        assert env.position.position_size > 0

        # Now, close with action 0.0
        td["action"] = torch.tensor([2])  # 0.0 = neutral
        td = env.step(td)["next"]

        # Position should be zero
        assert env.position.position_size == 0.0


class TestBalanceScaling:
    """Test that position sizes scale correctly with balance."""

    def test_position_scales_with_balance(self, sample_df):
        """Position size should scale proportionally with balance."""
        config = SeqFuturesEnvConfig(
            action_levels=[0.0, 0.5, 1.0],
            leverage=1,
            transaction_fee=0.0,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )

        # Test with different initial balances
        balances = [5000, 10000, 20000]
        positions = []

        for balance in balances:
            config.initial_cash = balance
            env = SeqFuturesEnv(sample_df, config)
            td = env.reset()

            current_price = env._cached_base_features["close"]

            # Action index 1 = 0.5 (50% long)
            td["action"] = torch.tensor([1])
            td = env.step(td)["next"]

            positions.append(env.position.position_size * current_price)  # Notional value

        # Positions should scale proportionally with balance
        assert abs(positions[1] / positions[0] - 2.0) < BALANCE_TOLERANCE  # 10k / 5k = 2x
        assert abs(positions[2] / positions[0] - 4.0) < BALANCE_TOLERANCE  # 20k / 5k = 4x


class TestLeverageApplication:
    """Test that leverage is correctly applied to positions."""

    @pytest.mark.parametrize("leverage_ratio,expected_multiplier", [
        (5, 5.0),
        (10, 10.0),
        (20, 20.0),
    ])
    def test_leverage_amplifies_position(self, sample_df, leverage_ratio, expected_multiplier):
        """Higher leverage should create proportionally larger positions."""
        # Config with 1x leverage (baseline)
        config_1x = SeqFuturesEnvConfig(
            action_levels=[0.0, 1.0],
            leverage=1,
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )

        # Config with specified leverage
        config_nx = SeqFuturesEnvConfig(
            action_levels=[0.0, 1.0],
            leverage=leverage_ratio,
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )

        # 1x leverage baseline
        env_1x = SeqFuturesEnv(sample_df, config_1x)
        td = env_1x.reset()
        td["action"] = torch.tensor([1])  # 100% long
        td = env_1x.step(td)["next"]
        position_1x = env_1x.position.position_size

        # Nx leverage
        env_nx = SeqFuturesEnv(sample_df, config_nx)
        td = env_nx.reset()
        td["action"] = torch.tensor([1])  # 100% long
        td = env_nx.step(td)["next"]
        position_nx = env_nx.position.position_size

        # Position with Nx leverage should be N times larger
        actual_multiplier = position_nx / position_1x
        assert abs(actual_multiplier - expected_multiplier) < BALANCE_TOLERANCE, \
            f"{leverage_ratio}x leverage: expected {expected_multiplier}x position, got {actual_multiplier:.2f}x"


class TestMarginAndFeeInteraction:
    """Test that margin calculation correctly accounts for fees with leverage."""

    @pytest.mark.parametrize("leverage,fee,min_expected,max_expected", [
        (1, 0.001, 0.98, 1.0),      # Spot: ~99% invested (accounting for fee)
        (10, 0.001, 9.7, 10.0),     # 10x: ~9.8x invested (accounting for fee)
        (20, 0.01, 16.0, 20.0),     # 20x + 1% fee: ~16.5x (margin + fee impact)
    ])
    def test_margin_calculation_with_fees(self, sample_df, leverage, fee, min_expected, max_expected):
        """Margin calculation should account for fees correctly to prevent over-leveraging.

        The formula is: margin_required = capital_allocated / (1 + leverage * fee)
        This ensures that fees don't cause the position to exceed available margin.

        The test verifies that:
        1. Position size increases with leverage
        2. Higher fees reduce effective position size
        3. System doesn't over-leverage (which could cause instant liquidation)
        """
        config = SeqFuturesEnvConfig(
            action_levels=[0.0, 1.0],
            leverage=leverage,
            transaction_fee=fee,
            initial_cash=10000,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )
        env = SeqFuturesEnv(sample_df, config)
        td = env.reset()

        # Open 100% long position
        td["action"] = torch.tensor([1])
        td = env.step(td)["next"]

        # Calculate actual notional value as multiple of initial cash
        current_price = env._cached_base_features["close"]
        position_notional = env.position.position_size * current_price
        actual_multiplier = position_notional / 10000

        # Verify position size is in expected range
        assert min_expected <= actual_multiplier <= max_expected, \
            f"Leverage {leverage}x with {fee*100:.1f}% fee: expected {min_expected:.1f}x-{max_expected:.1f}x, got {actual_multiplier:.2f}x"

        # Verify position was actually opened (not zero)
        assert actual_multiplier > 0.9, f"Position should be opened, got {actual_multiplier:.2f}x"

        env.close()


class TestDirectionSwitching:
    """Test that direction switches work correctly."""

    @pytest.mark.parametrize("first_action_idx,first_sign,second_action_idx,second_sign,description", [
        (2, 1, 0, -1, "long to short"),
        (0, -1, 2, 1, "short to long"),
    ])
    def test_direction_switch(self, sample_df, first_action_idx, first_sign, second_action_idx, second_sign, description):
        """Switching direction should close current position and open opposite position."""
        config = SeqFuturesEnvConfig(
            action_levels=[-1.0, 0.0, 1.0],
            leverage=10,  # Futures mode (leverage > 1) to allow shorts
            initial_cash=10000,
            transaction_fee=0.001,  # Small fee
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )
        env = SeqFuturesEnv(sample_df, config)

        td = env.reset()

        # Open first position
        td["action"] = torch.tensor([first_action_idx])
        td = env.step(td)["next"]
        if first_sign > 0:
            assert env.position.position_size > 0, f"{description}: first position should be long"
        else:
            assert env.position.position_size < 0, f"{description}: first position should be short"

        # Switch direction
        td["action"] = torch.tensor([second_action_idx])
        td = env.step(td)["next"]
        if second_sign > 0:
            assert env.position.position_size > 0, f"{description}: second position should be long"
        else:
            assert env.position.position_size < 0, f"{description}: second position should be short"


class TestLongOnlyFractional:
    """Test fractional position sizing for SeqLongOnlyEnv."""

    def test_long_only_default_action_levels_no_negatives(self, sample_df):
        """Long-only env action levels should not include negative values."""
        config = SeqLongOnlyEnvConfig(
            leverage=1,  # Spot mode
            action_levels=[0.0, 0.5, 1.0],  # Long-only levels
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )
        env = SeqLongOnlyEnv(sample_df, config)

        # Action levels should be non-negative
        assert all(level >= 0 for level in env.action_levels), \
            f"Long-only action_levels should be non-negative, got {env.action_levels}"

        # Should have at least close (0.0) and one positive action
        assert 0.0 in env.action_levels, "Should have action 0.0 for closing positions"
        assert any(level > 0 for level in env.action_levels), "Should have at least one positive action"

    def test_long_only_fractional_buy(self, sample_df):
        """Long-only env should correctly size buy positions."""
        config = SeqLongOnlyEnvConfig(
            action_levels=[0.0, 0.5, 1.0],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )
        env = SeqLongOnlyEnv(sample_df, config)

        td = env.reset()
        initial_balance = env.balance
        current_price = env._cached_base_features["close"]

        # Action index 1 = 0.5 (50% of cash)
        td["action"] = torch.tensor([1])
        td = env.step(td)["next"]

        expected_value = initial_balance * 0.5
        actual_value = env.position.position_size * current_price

        # Should be close (within 1% tolerance for fees/rounding)
        assert abs(actual_value - expected_value) / expected_value < BALANCE_TOLERANCE

    def test_long_only_fractional_sell(self, sample_df):
        """Long-only env should correctly close positions with action=0."""
        config = SeqLongOnlyEnvConfig(
            action_levels=[0.0, 0.5, 1.0],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )
        env = SeqLongOnlyEnv(sample_df, config)

        td = env.reset()

        # Buy first
        td["action"] = torch.tensor([2])  # 1.0 = buy all
        td = env.step(td)["next"]
        assert env.position.position_size > 0

        # Sell all with action=0
        td["action"] = torch.tensor([0])  # 0.0 = close position
        td = env.step(td)["next"]
        assert env.position.position_size == 0.0

    def test_long_only_no_shorts_allowed(self, sample_df):
        """Long-only env (leverage=1, positive actions) should only allow longs."""
        config = SeqLongOnlyEnvConfig(
            leverage=1,  # Spot mode - no shorts allowed
            action_levels=[0.0, 0.5, 1.0],  # Only non-negative actions
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )
        env = SeqLongOnlyEnv(sample_df, config)

        # Verify shorts are not allowed
        assert not env.allows_short, "Spot mode should not allow shorts"

        td = env.reset()

        # Open a long position
        td["action"] = torch.tensor([2])  # 1.0 = 100% long
        td = env.step(td)["next"]
        assert env.position.position_size > 0
        assert env.position.current_position > 0, "Should be long"

        # Close position
        td["action"] = torch.tensor([0])  # 0.0 = close
        td = env.step(td)["next"]

        # Should have closed position
        assert abs(env.position.position_size) < 0.01, "Should close position"
        assert env.position.current_position == 0, "Should be flat"


class TestPartialPositionAdjustment:
    """Test that position adjustments only trade the delta."""

    def test_reduce_position_from_100_to_50_percent(self, sample_df):
        """Reducing from 100% to 50% should only close 50% of position."""
        config = SeqFuturesEnvConfig(
            action_levels=[0.0, 0.5, 1.0],
            leverage=1,
            initial_cash=10000,
            transaction_fee=0.001,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )
        env = SeqFuturesEnv(sample_df, config)

        td = env.reset()
        initial_balance = env.balance

        # Open 100% long position
        td["action"] = torch.tensor([2])  # 1.0 = 100% long
        td = env.step(td)["next"]

        balance_after_open = env.balance
        entry_price = env.position.entry_price
        position_100 = env.position.position_size

        # Reduce to 50% long position
        td["action"] = torch.tensor([1])  # 0.5 = 50% long
        td = env.step(td)["next"]

        position_50 = env.position.position_size

        # Position should be approximately half
        assert abs(position_50 - position_100 / 2) / (position_100 / 2) < PRICE_TOLERANCE

        # Entry price should remain the same (partial close doesn't change entry)
        assert abs(env.position.entry_price - entry_price) / entry_price < POSITION_TOLERANCE

    def test_increase_position_from_50_to_100_percent(self, sample_df):
        """Increasing from 50% to 100% should add to position with weighted average entry."""
        config = SeqFuturesEnvConfig(
            action_levels=[0.0, 0.5, 1.0],
            leverage=1,
            initial_cash=10000,
            transaction_fee=0.001,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )
        env = SeqFuturesEnv(sample_df, config)

        td = env.reset()

        # Open 50% long position
        td["action"] = torch.tensor([1])  # 0.5 = 50% long
        td = env.step(td)["next"]

        position_50 = env.position.position_size
        entry_price_50 = env.position.entry_price

        # Increase to 100% long position
        td["action"] = torch.tensor([2])  # 1.0 = 100% long
        td = env.step(td)["next"]

        position_100 = env.position.position_size

        # Position should be approximately double
        assert abs(position_100 - position_50 * 2) / (position_50 * 2) < PRICE_TOLERANCE

        # Entry price should be weighted average (will be close to original if price hasn't changed much)
        # Just verify it's reasonable and not zero
        assert env.position.entry_price > 0

    def test_only_trades_difference_fees(self, sample_df):
        """Verify that only the difference is traded by checking fees."""
        config = SeqFuturesEnvConfig(
            action_levels=[0.0, 0.5, 1.0],
            leverage=1,
            initial_cash=10000,
            transaction_fee=0.001,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )
        env = SeqFuturesEnv(sample_df, config)

        td = env.reset()
        initial_balance = env.balance

        # Open 100% position
        td["action"] = torch.tensor([2])  # 1.0
        td = env.step(td)["next"]
        balance_after_100 = env.balance
        fee_for_100 = initial_balance - balance_after_100

        # Reset env
        td = env.reset()
        initial_balance_2 = env.balance

        # Open 50% position
        td["action"] = torch.tensor([1])  # 0.5
        td = env.step(td)["next"]
        balance_after_50 = env.balance
        fee_for_50 = initial_balance_2 - balance_after_50

        # Fee for 50% should be approximately half the fee for 100%
        assert abs(fee_for_50 - fee_for_100 / 2) / (fee_for_100 / 2) < PRICE_TOLERANCE


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_fraction(self, sample_df):
        """Test that very small fractions (0.01) work correctly."""
        config = SeqFuturesEnvConfig(
            action_levels=[0.0, 0.01, 1.0],
            leverage=1,
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )
        env = SeqFuturesEnv(sample_df, config)

        td = env.reset()
        initial_balance = env.balance
        current_price = env._cached_base_features["close"]

        # Action index 1 = 0.01 (1% long)
        td["action"] = torch.tensor([1])
        td = env.step(td)["next"]

        expected_position = (initial_balance * 0.01 * 1) / current_price
        actual_position = env.position.position_size

        assert abs(actual_position - expected_position) / expected_position < BALANCE_TOLERANCE

    def test_implicit_hold_does_not_trade(self, sample_df):
        """Test that selecting the same action twice does not trade (implicit hold)."""
        config = SeqFuturesEnvConfig(
            action_levels=[0.0, 0.5, 1.0],
            leverage=1,
            initial_cash=10000,
            transaction_fee=0.001,  # Enable fees to detect trades
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )
        env = SeqFuturesEnv(sample_df, config)

        td = env.reset()

        # Open position
        td["action"] = torch.tensor([2])  # 1.0 = 100% long
        td = env.step(td)["next"]
        balance_after_open = env.balance

        # Select same action again (should not trade)
        td["action"] = torch.tensor([2])  # 1.0 = 100% long again
        td = env.step(td)["next"]
        balance_after_hold = env.balance

        # Balance should be unchanged (no additional fees)
        assert balance_after_hold == balance_after_open


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
