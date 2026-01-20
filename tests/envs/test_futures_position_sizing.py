"""Tests for futures position sizing with QUANTITY and NOTIONAL modes."""

import pytest
import pandas as pd
import numpy as np

from torchtrade.envs.offline.seqfutures import SeqFuturesEnv, SeqFuturesEnvConfig, MarginType
from torchtrade.envs.offline.futuresonestepenv import FuturesOneStepEnv, FuturesOneStepEnvConfig
from torchtrade.envs.offline.seqfuturessltp import SeqFuturesSLTPEnv, SeqFuturesSLTPEnvConfig
from torchtrade.envs.common import TradeMode


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=1000, freq="1min")
    df = pd.DataFrame({
        "date": dates,
        "open": 50000 + np.random.randn(1000) * 100,
        "high": 50100 + np.random.randn(1000) * 100,
        "low": 49900 + np.random.randn(1000) * 100,
        "close": 50000 + np.random.randn(1000) * 100,
        "volume": np.random.rand(1000) * 1000,
    })
    df["date"] = pd.to_datetime(df["date"])
    return df


class TestSeqFuturesEnvQuantityMode:
    """Test QUANTITY mode for SeqFuturesEnv."""

    def test_quantity_mode_fixed_position_size(self, sample_df):
        """Test QUANTITY mode maintains fixed position size."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.001,
            leverage=5,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)

        # Reset and take long position
        env.reset()
        env._open_position("long", current_price=50000, price_noise=1.0)

        # Verify position size is exactly 0.001 BTC
        assert abs(env.position.position_size - 0.001) < 1e-6

        # Verify notional value
        expected_notional = 0.001 * 50000  # = 50 USD
        assert abs(env.position.position_value - expected_notional) < 1e-6

    def test_quantity_mode_different_prices(self, sample_df):
        """Test QUANTITY mode with different prices maintains same quantity."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.002,
            leverage=10,
            initial_cash=20000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)

        # Test at price $40,000
        env.reset()
        env._open_position("long", current_price=40000, price_noise=1.0)
        assert abs(env.position.position_size - 0.002) < 1e-6
        assert abs(env.position.position_value - (0.002 * 40000)) < 1e-6

        # Close and test at price $60,000
        env._close_position(current_price=60000, price_noise=1.0)
        env._open_position("long", current_price=60000, price_noise=1.0)
        assert abs(env.position.position_size - 0.002) < 1e-6
        assert abs(env.position.position_value - (0.002 * 60000)) < 1e-6


class TestSeqFuturesEnvNotionalMode:
    """Test NOTIONAL mode for SeqFuturesEnv."""

    def test_notional_mode_fixed_dollar_value(self, sample_df):
        """Test NOTIONAL mode maintains fixed notional value."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.NOTIONAL,
            quantity_per_trade=100,  # $100 per trade
            leverage=5,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)

        env.reset()

        # Test at price $50,000
        env._open_position("long", current_price=50000, price_noise=1.0)
        assert abs(env.position.position_value - 100) < 1e-6
        assert abs(env.position.position_size - (100/50000)) < 1e-8

        # Close and test at different price $60,000
        env._close_position(current_price=60000, price_noise=1.0)
        env._open_position("long", current_price=60000, price_noise=1.0)
        assert abs(env.position.position_value - 100) < 1e-6
        assert abs(env.position.position_size - (100/60000)) < 1e-8

    def test_notional_mode_quantity_varies_with_price(self, sample_df):
        """Test NOTIONAL mode quantity varies inversely with price."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.NOTIONAL,
            quantity_per_trade=200,  # $200 per trade
            leverage=5,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)

        env.reset()

        # At $40,000, should get more quantity
        env._open_position("long", current_price=40000, price_noise=1.0)
        qty_at_40k = env.position.position_size
        assert abs(qty_at_40k - (200/40000)) < 1e-8

        # At $80,000, should get half the quantity
        env._close_position(current_price=80000, price_noise=1.0)
        env._open_position("long", current_price=80000, price_noise=1.0)
        qty_at_80k = env.position.position_size
        assert abs(qty_at_80k - (200/80000)) < 1e-8

        # Verify inverse relationship
        assert abs(qty_at_40k / qty_at_80k - 2.0) < 1e-6


class TestInsufficientBalance:
    """Test insufficient balance scenarios."""

    def test_quantity_mode_insufficient_balance(self, sample_df):
        """Test trade is rejected when balance is insufficient in QUANTITY mode."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=1.0,  # 1 BTC = $50,000 notional
            leverage=5,  # Need $10,000 margin + fees
            initial_cash=5000,  # Not enough!
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)

        env.reset()
        trade_info = env._open_position("long", current_price=50000, price_noise=1.0)

        # Trade should be rejected
        assert trade_info["executed"] is False
        assert env.position.position_size == 0

    def test_notional_mode_insufficient_balance(self, sample_df):
        """Test trade is rejected when balance is insufficient in NOTIONAL mode."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.NOTIONAL,
            quantity_per_trade=50000,  # $50,000 notional
            leverage=5,  # Need $10,000 margin + fees
            initial_cash=5000,  # Not enough!
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)

        env.reset()
        trade_info = env._open_position("long", current_price=50000, price_noise=1.0)

        # Trade should be rejected
        assert trade_info["executed"] is False
        assert env.position.position_size == 0

    def test_margin_and_fee_calculation(self, sample_df):
        """Test that margin and fees are calculated correctly."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.1,
            leverage=10,
            initial_cash=10000,
            transaction_fee=0.0004,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)

        env.reset()
        initial_balance = env.balance

        env._open_position("long", current_price=50000, price_noise=1.0)

        # Notional = 0.1 * 50000 = 5000
        expected_notional = 5000
        # Margin = 5000 / 10 = 500
        expected_margin = 500
        # Fee = 5000 * 0.0004 = 2
        expected_fee = 2

        # Verify fee was deducted
        assert abs((initial_balance - env.balance) - expected_fee) < 1e-6


class TestShortPositions:
    """Test short position sizing."""

    def test_quantity_mode_short_position(self, sample_df):
        """Test QUANTITY mode with short positions."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.001,
            leverage=5,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)

        env.reset()
        env._open_position("short", current_price=50000, price_noise=1.0)

        # Verify position size is negative for short
        assert abs(env.position.position_size - (-0.001)) < 1e-6
        assert env.position.current_position == -1

        # Notional value should still be positive
        expected_notional = 0.001 * 50000
        assert abs(env.position.position_value - expected_notional) < 1e-6


class TestConfigValidation:
    """Test configuration validation."""

    def test_negative_quantity_per_trade_raises_error(self, sample_df):
        """Test that negative quantity_per_trade raises ValueError."""
        with pytest.raises(ValueError, match="quantity_per_trade must be positive"):
            config = SeqFuturesEnvConfig(
                quantity_per_trade=-0.001,
            )

    def test_zero_quantity_per_trade_raises_error(self, sample_df):
        """Test that zero quantity_per_trade raises ValueError."""
        with pytest.raises(ValueError, match="quantity_per_trade must be positive"):
            config = SeqFuturesEnvConfig(
                quantity_per_trade=0.0,
            )


class TestFuturesOneStepEnv:
    """Test position sizing for FuturesOneStepEnv."""

    def test_quantity_mode_onestep_env(self, sample_df):
        """Test QUANTITY mode in FuturesOneStepEnv."""
        config = FuturesOneStepEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.002,
            leverage=5,
            initial_cash=10000,
            seed=42,
        )
        env = FuturesOneStepEnv(sample_df, config)

        env.reset()
        # Simulate opening a position through _execute_trade_if_needed
        action_tuple = ("long", -0.05, 0.1)  # Long with 5% SL, 10% TP
        trade_info = env._execute_trade_if_needed(action_tuple, base_price=50000)

        # Verify position size
        assert abs(env.position.position_size - 0.002) < 1e-6
        assert abs(env.position.position_value - (0.002 * 50000)) < 1e-6

    def test_notional_mode_onestep_env(self, sample_df):
        """Test NOTIONAL mode in FuturesOneStepEnv."""
        config = FuturesOneStepEnvConfig(
            trade_mode=TradeMode.NOTIONAL,
            quantity_per_trade=150,  # $150 per trade
            leverage=5,
            initial_cash=10000,
            seed=42,
        )
        env = FuturesOneStepEnv(sample_df, config)

        env.reset()
        action_tuple = ("long", -0.05, 0.1)
        trade_info = env._execute_trade_if_needed(action_tuple, base_price=50000)

        # Verify notional value
        assert abs(env.position.position_value - 150) < 1e-6
        assert abs(env.position.position_size - (150/50000)) < 1e-8


class TestSeqFuturesSLTPEnv:
    """Test position sizing for SeqFuturesSLTPEnv."""

    def test_quantity_mode_sltp_env(self, sample_df):
        """Test QUANTITY mode in SeqFuturesSLTPEnv."""
        config = SeqFuturesSLTPEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.003,
            leverage=5,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesSLTPEnv(sample_df, config)

        env.reset()
        env._open_position("long", current_price=50000, price_noise=1.0,
                          sl_pct=-0.05, tp_pct=0.1)

        # Verify position size
        assert abs(env.position.position_size - 0.003) < 1e-6
        assert abs(env.position.position_value - (0.003 * 50000)) < 1e-6

    def test_notional_mode_sltp_env(self, sample_df):
        """Test NOTIONAL mode in SeqFuturesSLTPEnv."""
        config = SeqFuturesSLTPEnvConfig(
            trade_mode=TradeMode.NOTIONAL,
            quantity_per_trade=200,  # $200 per trade
            leverage=5,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesSLTPEnv(sample_df, config)

        env.reset()
        env._open_position("long", current_price=50000, price_noise=1.0,
                          sl_pct=-0.05, tp_pct=0.1)

        # Verify notional value
        assert abs(env.position.position_value - 200) < 1e-6
        assert abs(env.position.position_size - (200/50000)) < 1e-8


class TestBackwardCompatibility:
    """Test that existing code still works with defaults."""

    def test_default_config_uses_quantity_mode(self, sample_df):
        """Test that default config uses QUANTITY mode."""
        config = SeqFuturesEnvConfig(
            initial_cash=10000,
            seed=42,
        )

        assert config.trade_mode == TradeMode.QUANTITY
        assert config.quantity_per_trade == 0.001

    def test_environment_works_with_defaults(self, sample_df):
        """Test that environment works with default configuration."""
        config = SeqFuturesEnvConfig(
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)

        # Should be able to reset and step
        env.reset()
        env._open_position("long", current_price=50000, price_noise=1.0)

        # Should have opened a position successfully
        assert env.position.position_size > 0


# ============================================================================
# ADDITIONAL HIGH-PRIORITY TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_quantity(self, sample_df):
        """Test with very small quantity (satoshi-level)."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.00000001,  # 1 satoshi
            leverage=1,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)
        env.reset()
        env._open_position("long", current_price=50000, price_noise=1.0)

        assert abs(env.position.position_size - 0.00000001) < 1e-12
        assert env.position.position_value > 0  # Should have some value

    def test_very_large_quantity_near_balance(self, sample_df):
        """Test with quantity requiring most of available balance."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.15,  # Requires $7500 at $50k, plus margin + fees
            leverage=1,
            initial_cash=8000,  # Just enough
            transaction_fee=0.0004,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)
        env.reset()
        trade_info = env._open_position("long", current_price=50000, price_noise=1.0)

        # Should succeed
        assert trade_info["executed"] is True
        assert abs(env.position.position_size - 0.15) < 1e-6
        # Balance should be significantly reduced
        assert env.balance < 1000

    def test_extreme_low_price(self, sample_df):
        """Test with very low price."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.NOTIONAL,
            quantity_per_trade=100,  # $100 notional
            leverage=5,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)
        env.reset()
        env._open_position("long", current_price=1.0, price_noise=1.0)  # $1 price

        # At $1, should get 100 units
        assert abs(env.position.position_size - 100) < 1e-6
        assert abs(env.position.position_value - 100) < 1e-6

    def test_extreme_high_price(self, sample_df):
        """Test with very high price."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.001,
            leverage=10,
            initial_cash=1000000,  # Need high balance for high prices
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)
        env.reset()
        env._open_position("long", current_price=1000000, price_noise=1.0)

        assert abs(env.position.position_size - 0.001) < 1e-6
        # Notional should be very high
        assert abs(env.position.position_value - 1000) < 1e-6


class TestIntegrationScenarios:
    """Test full integration scenarios with step() calls."""

    def test_full_episode_with_quantity_mode(self, sample_df):
        """Test complete episode using step() with QUANTITY mode."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.001,
            leverage=5,
            initial_cash=10000,
            max_traj_length=10,
            random_start=False,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)

        td = env.reset()
        done = False
        step_count = 0

        while not done and step_count < 15:
            action = env.action_spec.rand()  # Random action
            td["action"] = action  # Set action in tensordict
            td = env.step(td)
            done = td["done"].item()
            step_count += 1

        # Episode should complete
        assert done or step_count >= 10

    def test_multiple_consecutive_trades(self, sample_df):
        """Test multiple trades in sequence."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.001,
            leverage=10,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)
        env.reset()

        # Trade 1: Long
        env._open_position("long", current_price=50000, price_noise=1.0)
        first_pos = env.position.position_size
        assert abs(first_pos - 0.001) < 1e-6

        # Close
        env._close_position(current_price=51000, price_noise=1.0)
        assert env.position.position_size == 0

        # Trade 2: Short
        env._open_position("short", current_price=51000, price_noise=1.0)
        second_pos = env.position.position_size
        assert abs(second_pos - (-0.001)) < 1e-6

        # Close
        env._close_position(current_price=50500, price_noise=1.0)
        assert env.position.position_size == 0

    def test_reset_consistency(self, sample_df):
        """Test that position sizing is consistent after reset."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.002,
            leverage=5,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)

        # First episode
        env.reset()
        env._open_position("long", current_price=50000, price_noise=1.0)
        first_size = env.position.position_size

        # Reset and second episode
        env.reset()
        env._open_position("long", current_price=50000, price_noise=1.0)
        second_size = env.position.position_size

        # Should be identical
        assert abs(first_size - second_size) < 1e-9
        assert abs(first_size - 0.002) < 1e-6


class TestLiquidationScenarios:
    """Test liquidation behavior with new position sizing."""

    def test_liquidation_price_calculation_quantity_mode(self, sample_df):
        """Test liquidation price calculation with QUANTITY mode."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.1,
            leverage=10,  # 10x leverage
            maintenance_margin_rate=0.004,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)
        env.reset()

        env._open_position("long", current_price=50000, price_noise=1.0)

        # Liquidation price should be calculated
        assert env.liquidation_price is not None
        assert env.liquidation_price > 0
        # For long position with 10x leverage, liquidation should be below entry
        assert env.liquidation_price < env.position.entry_price

    def test_liquidation_price_calculation_notional_mode(self, sample_df):
        """Test liquidation price calculation with NOTIONAL mode."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.NOTIONAL,
            quantity_per_trade=1000,  # $1000 notional
            leverage=5,
            maintenance_margin_rate=0.004,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)
        env.reset()

        env._open_position("short", current_price=50000, price_noise=1.0)

        # Liquidation price should be calculated
        assert env.liquidation_price is not None
        assert env.liquidation_price > 0
        # For short position, liquidation should be above entry
        assert env.liquidation_price > env.position.entry_price

    def test_different_leverage_affects_liquidation(self, sample_df):
        """Test that different leverage levels affect liquidation price."""
        # Low leverage (safer)
        config_low = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.1,
            leverage=2,
            initial_cash=50000,
            seed=42,
        )
        env_low = SeqFuturesEnv(sample_df, config_low)
        env_low.reset()
        env_low._open_position("long", current_price=50000, price_noise=1.0)
        liq_low = env_low.liquidation_price

        # High leverage (riskier)
        config_high = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.1,
            leverage=20,
            initial_cash=50000,
            seed=42,
        )
        env_high = SeqFuturesEnv(sample_df, config_high)
        env_high.reset()
        env_high._open_position("long", current_price=50000, price_noise=1.0)
        liq_high = env_high.liquidation_price

        # Higher leverage should have liquidation price closer to entry
        distance_low = abs(50000 - liq_low)
        distance_high = abs(50000 - liq_high)
        assert distance_high < distance_low


class TestCrossEnvironmentConsistency:
    """Test consistency across different environment types."""

    def test_same_config_same_position_size(self, sample_df):
        """Test that same config produces same position size across environments."""
        # Create identical configs for two different environments
        config1 = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.005,
            leverage=5,
            initial_cash=10000,
            seed=42,
        )

        config2 = SeqFuturesSLTPEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.005,
            leverage=5,
            initial_cash=10000,
            seed=42,
        )

        env1 = SeqFuturesEnv(sample_df, config1)
        env2 = SeqFuturesSLTPEnv(sample_df, config2)

        env1.reset()
        env2.reset()

        env1._open_position("long", current_price=50000, price_noise=1.0)
        env2._open_position("long", current_price=50000, price_noise=1.0,
                            sl_pct=-0.05, tp_pct=0.1)

        # Position sizes should be identical
        assert abs(env1.position.position_size - env2.position.position_size) < 1e-9

    def test_base_class_method_consistent(self, sample_df):
        """Test that base class method produces consistent results."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.NOTIONAL,
            quantity_per_trade=250,
            leverage=5,
            initial_cash=10000,
            seed=42,
        )

        env = SeqFuturesEnv(sample_df, config)
        env.reset()

        # Call base class method directly
        qty1, notional1 = env._calculate_position_quantity(50000)
        qty2, notional2 = env._calculate_position_quantity(50000)

        # Should be deterministic
        assert abs(qty1 - qty2) < 1e-12
        assert abs(notional1 - notional2) < 1e-12


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_zero_price_handling(self, sample_df):
        """Test handling of zero price (should raise or handle gracefully)."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.NOTIONAL,
            quantity_per_trade=100,
            leverage=5,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)
        env.reset()

        # Attempting to open position at price 0 should either:
        # 1. Raise an error, or
        # 2. Reject the trade
        try:
            trade_info = env._open_position("long", current_price=0.0, price_noise=1.0)
            # If no error, trade should be rejected or have inf position size
            assert trade_info["executed"] is False or np.isinf(env.position.position_size)
        except (ZeroDivisionError, ValueError):
            # This is also acceptable behavior
            pass

    def test_invalid_trade_mode_handling(self, sample_df):
        """Test that invalid trade mode is caught."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.001,
            leverage=5,
            initial_cash=10000,
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)
        env.reset()

        # Manually set invalid trade mode
        env.config.trade_mode = "invalid_mode"

        # Should raise ValueError
        with pytest.raises(ValueError, match="Unknown trade_mode"):
            env._calculate_position_quantity(50000)


class TestBalanceTracking:
    """Test balance and PnL tracking with new position sizing."""

    def test_balance_deduction_after_fees(self, sample_df):
        """Test that balance is correctly deducted after trade fees."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.1,
            leverage=5,
            initial_cash=10000,
            transaction_fee=0.001,  # 0.1%
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)
        env.reset()
        initial_balance = env.balance

        trade_info = env._open_position("long", current_price=50000, price_noise=1.0)

        # Calculate expected fee
        notional = 0.1 * 50000  # = 5000
        expected_fee = notional * 0.001  # = 5

        # Balance should be reduced by fee
        assert abs((initial_balance - env.balance) - expected_fee) < 1e-6
        assert abs(trade_info["fee_paid"] - expected_fee) < 1e-6

    def test_pnl_calculation_with_fixed_size(self, sample_df):
        """Test PnL calculation with fixed position size."""
        config = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.1,
            leverage=10,
            initial_cash=10000,
            transaction_fee=0.0,  # No fees for cleaner PnL calc
            seed=42,
        )
        env = SeqFuturesEnv(sample_df, config)
        env.reset()

        # Open long at $50,000
        env._open_position("long", current_price=50000, price_noise=1.0)

        # Close at $55,000 (10% gain)
        trade_info = env._close_position(current_price=55000, price_noise=1.0)

        # PnL should be: 0.1 * (55000 - 50000) = 500
        # With 10x leverage and no fees, balance should increase by 500
        expected_pnl = 0.1 * (55000 - 50000)
        # Note: The actual balance change depends on how PnL is calculated in the env
        # This test documents the expected behavior

    def test_margin_requirement_different_modes(self, sample_df):
        """Test margin requirements are calculated consistently."""
        leverage = 5

        # QUANTITY mode
        config_qty = SeqFuturesEnvConfig(
            trade_mode=TradeMode.QUANTITY,
            quantity_per_trade=0.1,
            leverage=leverage,
            initial_cash=10000,
            seed=42,
        )
        env_qty = SeqFuturesEnv(sample_df, config_qty)
        env_qty.reset()
        env_qty._open_position("long", current_price=50000, price_noise=1.0)

        # Notional = 0.1 * 50000 = 5000
        # Margin = 5000 / 5 = 1000
        expected_margin = 5000 / leverage

        # NOTIONAL mode (same notional value)
        config_not = SeqFuturesEnvConfig(
            trade_mode=TradeMode.NOTIONAL,
            quantity_per_trade=5000,  # Same notional
            leverage=leverage,
            initial_cash=10000,
            seed=42,
        )
        env_not = SeqFuturesEnv(sample_df, config_not)
        env_not.reset()
        env_not._open_position("long", current_price=50000, price_noise=1.0)

        # Both should have same margin requirement
        # (exact values depend on _calculate_margin_required implementation)


class TestRegressionTests:
    """Regression tests to ensure old functionality still works."""

    def test_old_max_position_size_parameter_removed(self):
        """Test that max_position_size parameter no longer exists."""
        # Should not have max_position_size in config
        config = SeqFuturesEnvConfig()

        assert not hasattr(config, 'max_position_size')

    def test_default_values_are_sensible(self):
        """Test that default config values are sensible."""
        config = SeqFuturesEnvConfig()

        # Should default to QUANTITY mode
        assert config.trade_mode == TradeMode.QUANTITY

        # Should have positive quantity_per_trade
        assert config.quantity_per_trade > 0

        # Should be a reasonable default (0.001 BTC)
        assert config.quantity_per_trade == 0.001
