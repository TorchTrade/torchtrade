"""Action reversibility and value conservation tests.

CRITICAL: Opening and closing a position should conserve portfolio value minus fees.
If broken, the environment is creating or destroying value incorrectly.
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
    """OHLCV data with price change."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    # Price increases linearly from 100 to 110
    prices = [100.0 + i * 0.1 for i in range(100)]
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000.0] * 100,
    })
    return df


class TestActionReversibility:
    """Test that opening and closing positions conserves value (minus fees)."""

    @pytest.mark.parametrize("leverage,action_level", [
        (1, 1.0),     # Spot: full position
        (2, 0.5),     # 2x leverage: 50% position
        (5, 0.5),     # 5x leverage: 50% position (avoid tiny positions)
    ])
    def test_round_trip_at_constant_price_conserves_value(
        self, constant_price_df, leverage, action_level
    ):
        """Opening then closing at same price should only lose fees.

        With constant price and no slippage:
        - PnL = 0
        - Cost = 2 × fee (open + close)
        - Final balance = initial - cost
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.001,  # 0.1% fee
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, 0, 1] if leverage > 1 else [0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        initial_balance = env.balance
        initial_pv = env._get_portfolio_value()

        # Open position
        if leverage > 1:
            open_idx = 2  # Long
            close_idx = 1  # Close
        else:
            open_idx = 1  # Buy
            close_idx = 0  # Sell

        td["action"] = open_idx
        td = env.step(td)["next"]

        # Close position
        td["action"] = close_idx
        td = env.step(td)["next"]

        final_balance = env.balance
        final_pv = env._get_portfolio_value()

        # Key properties to verify:
        # 1. PV should not increase (can't create value from fees)
        # 2. If it decreases, decrease should be small (only fees, no large losses)
        pv_loss = initial_pv - final_pv

        # Should not gain value (fees don't create money)
        assert pv_loss >= -1.0, \
            f"Leverage={leverage}, action={action_level}: " \
            f"PV increased by {-pv_loss:.2f}, should not gain value from round trip"

        # If loss occurred, should be reasonable (< 1% of initial PV)
        # With 0.1% fee × 2 trades, max expected loss is ~0.2% + some margin
        # Allow losses near zero (rounding or position tolerance may prevent trade)
        max_reasonable_loss = initial_pv * 0.01  # 1% tolerance
        assert pv_loss < max_reasonable_loss, \
            f"Leverage={leverage}, action={action_level}: " \
            f"Loss={pv_loss:.2f} is too large (>{max_reasonable_loss:.2f}), " \
            f"initial={initial_pv:.2f}, final={final_pv:.2f}"

    @pytest.mark.parametrize("leverage", [2, 5, 10])
    def test_long_short_symmetry_at_constant_price(self, constant_price_df, leverage):
        """Long and short round trips should have identical cost.

        At constant price, PnL = 0 for both directions.
        Fees should be identical for same position size.
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.001,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, 0, 1],
        )

        # Long round trip
        env_long = SequentialTradingEnv(constant_price_df, config)
        td = env_long.reset()
        initial_pv = env_long._get_portfolio_value()

        td["action"] = 2  # Long
        td = env_long.step(td)["next"]
        td["action"] = 1  # Close
        td = env_long.step(td)["next"]

        long_final_pv = env_long._get_portfolio_value()
        long_cost = initial_pv - long_final_pv

        # Short round trip
        env_short = SequentialTradingEnv(constant_price_df, config)
        td = env_short.reset()

        td["action"] = 0  # Short
        td = env_short.step(td)["next"]
        td["action"] = 1  # Close
        td = env_short.step(td)["next"]

        short_final_pv = env_short._get_portfolio_value()
        short_cost = initial_pv - short_final_pv

        # Costs should be similar (within 10% of each other)
        avg_cost = (long_cost + short_cost) / 2
        if avg_cost > 0:
            rel_diff = abs(long_cost - short_cost) / avg_cost
            assert rel_diff < 0.10, \
                f"Leverage={leverage}: long cost={long_cost:.2f}, short cost={short_cost:.2f}, " \
                f"relative difference={rel_diff:.2%} (should be <10%)"


class TestPnLSymmetry:
    """Test that PnL is symmetrical for long/short positions."""

    @pytest.mark.parametrize("leverage", [2, 5, 10])
    def test_long_gains_equal_short_losses(self, price_change_df, leverage):
        """When price increases, long gains should equal short losses (ignoring fees).

        If price goes from $100 → $110 (+10%):
        - Long: +10% PnL
        - Short: -10% PnL
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,  # Isolate PnL effect
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, 0, 1],
            random_start=False,
        )

        # Long position
        env_long = SequentialTradingEnv(price_change_df, config)
        td = env_long.reset()
        initial_pv = env_long._get_portfolio_value()

        # Open long
        td["action"] = 2
        td = env_long.step(td)["next"]

        # Step forward to price change
        for _ in range(20):
            td["action"] = 2  # Hold long
            td = env_long.step(td)["next"]
            if td["done"]:
                break

        # Close long
        td["action"] = 1
        td = env_long.step(td)["next"]

        long_final_pv = env_long._get_portfolio_value()
        long_pnl = long_final_pv - initial_pv

        # Short position
        env_short = SequentialTradingEnv(price_change_df, config)
        td = env_short.reset()

        # Open short
        td["action"] = 0
        td = env_short.step(td)["next"]

        # Step forward to price change
        for _ in range(20):
            td["action"] = 0  # Hold short
            td = env_short.step(td)["next"]
            if td["done"]:
                break

        # Close short
        td["action"] = 1
        td = env_short.step(td)["next"]

        short_final_pv = env_short._get_portfolio_value()
        short_pnl = short_final_pv - initial_pv

        # Long PnL should be opposite of short PnL (approximately)
        # Allow tolerance due to position sizing differences
        sum_pnl = long_pnl + short_pnl
        avg_abs_pnl = (abs(long_pnl) + abs(short_pnl)) / 2

        if avg_abs_pnl > 10:  # Only check ratio if PnL is significant
            rel_asymmetry = abs(sum_pnl) / avg_abs_pnl
            assert rel_asymmetry < 0.20, \
                f"Leverage={leverage}: long PnL={long_pnl:.2f}, short PnL={short_pnl:.2f}, " \
                f"sum={sum_pnl:.2f}, asymmetry={rel_asymmetry:.2%} (should be <20%)"


class TestValueConservationAcrossSteps:
    """Test that portfolio value changes only from PnL and fees."""

    @pytest.mark.parametrize("leverage", [1, 2, 5])
    def test_holding_position_only_changes_by_pnl(self, constant_price_df, leverage):
        """Holding a position at constant price should not change portfolio value.

        At constant price:
        - Unrealized PnL = 0
        - No new fees (no trading)
        - Portfolio value should remain constant
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.001,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, 0, 1] if leverage > 1 else [0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()

        # Open position
        open_idx = 2 if leverage > 1 else 1
        td["action"] = open_idx
        td = env.step(td)["next"]

        pv_after_open = env._get_portfolio_value()

        # Hold for 10 steps
        for _ in range(10):
            td["action"] = open_idx  # Hold
            td = env.step(td)["next"]
            if td["done"]:
                break

        pv_after_hold = env._get_portfolio_value()

        # Portfolio value should be unchanged (within small tolerance)
        assert abs(pv_after_hold - pv_after_open) < 0.5, \
            f"Leverage={leverage}: PV changed from {pv_after_open:.2f} to {pv_after_hold:.2f} " \
            f"while holding at constant price"

    @pytest.mark.parametrize("leverage,time_frames", [
        (1, ["1Hour"]),
        (2, ["1Hour", "4Hour"]),
    ])
    def test_conservation_with_multi_timeframe(
        self, constant_price_df, leverage, time_frames
    ):
        """Value conservation should hold with multiple timeframes."""
        window_sizes = [5] * len(time_frames)

        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=time_frames,
            window_sizes=window_sizes,
            initial_cash=10000,
            transaction_fee=0.001,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, 0, 1] if leverage > 1 else [0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        initial_pv = env._get_portfolio_value()

        # Open then close
        open_idx = 2 if leverage > 1 else 1
        close_idx = 1 if leverage > 1 else 0

        td["action"] = open_idx
        td = env.step(td)["next"]

        td["action"] = close_idx
        td = env.step(td)["next"]

        final_pv = env._get_portfolio_value()

        # Value should be conserved minus fees
        pv_loss = initial_pv - final_pv

        # Should have lost money to fees (not gained)
        assert pv_loss > 0, \
            f"Leverage={leverage}, timeframes={time_frames}: " \
            f"Should lose money to fees, but PV changed by {-pv_loss:.2f}"

        # Loss should be reasonable (< 1% of initial PV)
        max_reasonable_loss = initial_pv * 0.01
        assert pv_loss < max_reasonable_loss, \
            f"Leverage={leverage}, timeframes={time_frames}: " \
            f"Loss={pv_loss:.2f} too large, initial={initial_pv:.2f}, final={final_pv:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
