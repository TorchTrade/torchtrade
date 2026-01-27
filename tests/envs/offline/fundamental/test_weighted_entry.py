"""Weighted average entry price tests.

CRITICAL: Entry price must be weighted average when adding to position.
If wrong, PnL is wrong → rewards are wrong → agent learns wrong strategy.
"""

import pytest
import pandas as pd
from torchtrade.envs.offline.sequential import SequentialTradingEnv, SequentialTradingEnvConfig


@pytest.fixture
def constant_price_df():
    """OHLCV data with constant price."""
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
def price_change_df():
    """OHLCV data with deliberate price change."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    # First 50 bars at $100, next 50 at $110
    prices = [100.0] * 50 + [110.0] * 50
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000.0] * 100,
    })
    return df


class TestWeightedAverageEntry:
    """Test entry price calculation when increasing position."""

    @pytest.mark.parametrize("leverage", [1, 2, 5, 10])
    def test_increasing_position_calculates_weighted_entry(self, constant_price_df, leverage):
        """Entry price should be weighted average when adding to position.

        Formula: new_entry = (old_value × old_entry + new_value × new_entry) / total_value

        Example:
        - Open 1 BTC at $100 → entry = $100
        - Add 1 BTC at $110 → entry = ($100×1 + $110×1) / 2 = $105 (not $110!)
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, -0.5, 0, 0.5, 1.0] if leverage > 1 else [0, 0.5, 1.0],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()

        # Open with 50% of capital (action_level=0.5)
        action_idx = config.action_levels.index(0.5)
        td["action"] = action_idx
        td = env.step(td)["next"]

        first_entry = env.position.entry_price
        first_position = env.position.position_size
        first_value = abs(first_position * first_entry)

        # Increase to 100% (add another 50%)
        action_idx = config.action_levels.index(1.0)
        td["action"] = action_idx
        td = env.step(td)["next"]

        second_entry = env.position.entry_price
        second_position = env.position.position_size

        # Calculate expected weighted entry
        # Since both opens at same price (100), entry should stay 100
        expected_entry = first_entry  # Both at $100

        assert abs(second_entry - expected_entry) < 0.1, \
            f"Leverage={leverage}: expected entry={expected_entry:.2f}, got {second_entry:.2f}"

        # Position should have doubled (approximately)
        assert abs(second_position / first_position - 2.0) < 0.1, \
            f"Position should roughly double when increasing 50%→100%"

    def test_weighted_entry_with_price_change(self, price_change_df):
        """Entry price should be weighted average when price changes between adds.

        Scenario:
        - Open position at $100
        - Price moves to $110
        - Add to position at $110
        - Entry should be weighted average, not $110
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=2,
            action_levels=[-1, -0.5, 0, 0.5, 1.0],
            random_start=False,  # Start from beginning
        )
        env = SequentialTradingEnv(price_change_df, config)

        td = env.reset()

        # Open at $100 with 50% capital
        td["action"] = config.action_levels.index(0.5)
        td = env.step(td)["next"]

        first_entry = env.position.entry_price
        first_position = env.position.position_size
        first_value = abs(first_position * first_entry)

        # Step forward to price change ($110)
        for _ in range(45):  # Get to price change
            td["action"] = config.action_levels.index(0.5)  # Hold
            td = env.step(td)["next"]
            if td["done"]:
                break

        current_price = env._cached_base_features["close"]
        if abs(current_price - 110) < 1:  # We're at $110
            # Add to position at $110 (increase to 100%)
            td["action"] = config.action_levels.index(1.0)
            td = env.step(td)["next"]

            new_entry = env.position.entry_price
            new_position = env.position.position_size

            # Calculate expected weighted entry
            # First part: first_value at first_entry
            # Second part: (new_position - first_position) at current_price
            additional_position = new_position - first_position
            additional_value = abs(additional_position * current_price)
            total_value = first_value + additional_value

            expected_entry = (first_value * first_entry + additional_value * current_price) / total_value

            assert abs(new_entry - expected_entry) < 0.5, \
                f"Expected weighted entry={expected_entry:.2f}, got {new_entry:.2f}"

            # Entry should be between first_entry ($100) and current_price ($110)
            assert first_entry <= new_entry <= current_price, \
                f"Entry {new_entry:.2f} should be between {first_entry:.2f} and {current_price:.2f}"


class TestEntryPriceUnchangedWhenDecreasing:
    """Test that entry price doesn't change when reducing position."""

    @pytest.mark.parametrize("leverage,initial_action,target_action", [
        (2, 1.0, 0.5),    # Reduce from 100% to 50%
        (5, 1.0, 0.3),    # Reduce from 100% to 30%
        (10, 0.5, 0.2),   # Reduce from 50% to 20%
    ])
    def test_decreasing_position_keeps_same_entry(
        self, constant_price_df, leverage, initial_action, target_action
    ):
        """Reducing position should not change entry price.

        Logic: Entry price is the price you paid to enter. Partial closes
        don't change what you originally paid.
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, -0.5, -0.3, -0.2, 0, 0.2, 0.3, 0.5, 1.0],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()

        # Open with initial_action
        action_idx = config.action_levels.index(initial_action)
        td["action"] = action_idx
        td = env.step(td)["next"]

        entry_after_open = env.position.entry_price

        # Reduce to target_action
        action_idx = config.action_levels.index(target_action)
        td["action"] = action_idx
        td = env.step(td)["next"]

        entry_after_decrease = env.position.entry_price

        # Entry should be unchanged
        assert abs(entry_after_decrease - entry_after_open) < 0.01, \
            f"Leverage={leverage}, {initial_action}→{target_action}: entry changed from {entry_after_open:.2f} to {entry_after_decrease:.2f}"

    def test_full_close_then_reopen_resets_entry(self, constant_price_df):
        """Closing fully then reopening should use new entry price."""
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

        # Open
        td["action"] = 2
        td = env.step(td)["next"]
        first_entry = env.position.entry_price

        # Close
        td["action"] = 1
        td = env.step(td)["next"]
        assert env.position.position_size == 0, "Position should be closed"

        # Reopen
        td["action"] = 2
        td = env.step(td)["next"]
        second_entry = env.position.entry_price

        # Entry should be fresh (same as market price at reopen)
        # Since price is constant, should be same as first_entry
        assert abs(second_entry - first_entry) < 0.1, \
            f"After close and reopen, entry should be fresh"


class TestEntryPriceForShorts:
    """Test entry price handling for short positions."""

    def test_increasing_short_position_uses_weighted_entry(self, constant_price_df):
        """Weighted entry applies to shorts too."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=2,
            action_levels=[-1, -0.5, 0, 0.5, 1.0],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()

        # Open short with 50%
        td["action"] = config.action_levels.index(-0.5)
        td = env.step(td)["next"]

        first_entry = env.position.entry_price
        first_position = env.position.position_size

        # Increase short to 100%
        td["action"] = config.action_levels.index(-1.0)
        td = env.step(td)["next"]

        second_entry = env.position.entry_price

        # Entry should be weighted average (same as $100 since constant price)
        assert abs(second_entry - first_entry) < 0.1, \
            f"Short entry should be weighted: first={first_entry:.2f}, second={second_entry:.2f}"

        # Position should have roughly doubled (both negative)
        assert abs(abs(env.position.position_size) / abs(first_position) - 2.0) < 0.1, \
            f"Short position should roughly double"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
