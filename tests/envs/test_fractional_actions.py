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

    def test_action_1_0_maps_to_100_percent_long(self, sample_df):
        """Action 1.0 should create a position worth 100% of balance * leverage."""
        config = SeqFuturesEnvConfig(
            action_levels=[-1.0, -0.5, 0.0, 0.5, 1.0],
            leverage=1,
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

        # Action index 4 = 1.0 (100% long)
        td["action"] = torch.tensor([4])
        td = env.step(td)["next"]

        # Expected position: (balance * 1.0 * leverage) / price
        expected_position = (initial_balance * 1.0 * 1) / current_price
        actual_position = env.position.position_size

        # Should be very close (allowing for minor floating point differences)
        assert abs(actual_position - expected_position) / expected_position < POSITION_TOLERANCE

    def test_action_0_5_maps_to_50_percent_long(self, sample_df):
        """Action 0.5 should create a position worth 50% of balance * leverage."""
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
        initial_balance = env.balance
        current_price = env._cached_base_features["close"]

        # Action index 3 = 0.5 (50% long)
        td["action"] = torch.tensor([3])
        td = env.step(td)["next"]

        expected_position = (initial_balance * 0.5 * 1) / current_price
        actual_position = env.position.position_size

        assert abs(actual_position - expected_position) / expected_position < POSITION_TOLERANCE

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

    def test_action_negative_0_5_maps_to_50_percent_short(self, sample_df):
        """Action -0.5 should create a short position worth 50% of balance * leverage."""
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
        initial_balance = env.balance
        current_price = env._cached_base_features["close"]

        # Action index 1 = -0.5 (50% short)
        td["action"] = torch.tensor([1])
        td = env.step(td)["next"]

        expected_position = -(initial_balance * 0.5 * 1) / current_price
        actual_position = env.position.position_size

        assert abs(actual_position - expected_position) / abs(expected_position) < POSITION_TOLERANCE
        assert actual_position < 0  # Should be negative (short)

    def test_action_negative_1_0_maps_to_100_percent_short(self, sample_df):
        """Action -1.0 should create a short position worth 100% of balance * leverage."""
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
        initial_balance = env.balance
        current_price = env._cached_base_features["close"]

        # Action index 0 = -1.0 (100% short)
        td["action"] = torch.tensor([0])
        td = env.step(td)["next"]

        expected_position = -(initial_balance * 1.0 * 1) / current_price
        actual_position = env.position.position_size

        assert abs(actual_position - expected_position) / abs(expected_position) < POSITION_TOLERANCE
        assert actual_position < 0  # Should be negative (short)


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

    def test_5x_leverage_amplifies_position(self, sample_df):
        """5x leverage should create a position 5x larger than 1x leverage."""
        config_1x = SeqFuturesEnvConfig(
            action_levels=[0.0, 1.0],
            leverage=1,
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )

        config_5x = SeqFuturesEnvConfig(
            action_levels=[0.0, 1.0],
            leverage=5,
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )

        # 1x leverage
        env_1x = SeqFuturesEnv(sample_df, config_1x)
        td = env_1x.reset()
        td["action"] = torch.tensor([1])  # 100% long
        td = env_1x.step(td)["next"]
        position_1x = env_1x.position.position_size

        # 5x leverage
        env_5x = SeqFuturesEnv(sample_df, config_5x)
        td = env_5x.reset()
        td["action"] = torch.tensor([1])  # 100% long
        td = env_5x.step(td)["next"]
        position_5x = env_5x.position.position_size

        # Position with 5x leverage should be 5x larger
        assert abs(position_5x / position_1x - 5.0) < BALANCE_TOLERANCE


class TestDirectionSwitching:
    """Test that direction switches work correctly."""

    def test_long_to_short_switch(self, sample_df):
        """Switching from long to short should close long and open short."""
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

        # Go long
        td["action"] = torch.tensor([2])  # 1.0 = 100% long
        td = env.step(td)["next"]
        assert env.position.position_size > 0

        # Switch to short
        td["action"] = torch.tensor([0])  # -1.0 = 100% short
        td = env.step(td)["next"]
        assert env.position.position_size < 0  # Now short

    def test_short_to_long_switch(self, sample_df):
        """Switching from short to long should close short and open long."""
        config = SeqFuturesEnvConfig(
            action_levels=[-1.0, 0.0, 1.0],
            leverage=10,  # Futures mode (leverage > 1) to allow shorts
            initial_cash=10000,
            transaction_fee=0.001,
            slippage=0.0,
            random_start=False,
            max_traj_length=10
        )
        env = SeqFuturesEnv(sample_df, config)

        td = env.reset()

        # Go short
        td["action"] = torch.tensor([0])  # -1.0 = 100% short
        td = env.step(td)["next"]
        assert env.position.position_size < 0

        # Switch to long
        td["action"] = torch.tensor([2])  # 1.0 = 100% long
        td = env.step(td)["next"]
        assert env.position.position_size > 0  # Now long


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
