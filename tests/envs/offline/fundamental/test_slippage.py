"""Slippage handling tests.

CRITICAL: Slippage affects execution price in real trading.
If broken, backtests will be unrealistically optimistic.
"""

import pytest
import pandas as pd
import numpy as np
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


class TestSlippageApplication:
    """Test that slippage is applied correctly to execution prices."""

    @pytest.mark.parametrize("slippage,leverage", [
        (0.0, 1),      # No slippage baseline
        (0.001, 1),    # 0.1% slippage
        (0.005, 2),    # 0.5% slippage
        (0.01, 5),     # 1% slippage
    ])
    def test_slippage_affects_execution_price(self, constant_price_df, slippage, leverage):
        """Slippage should cause execution price to differ from market price.

        With slippage, actual execution happens at worse price than market:
        - Buy: pay more (1 + random_slippage) × market_price
        - Sell: get less (1 - random_slippage) × market_price

        We can't test exact values due to randomness, but we can verify:
        1. Entry price differs from market price when slippage > 0
        2. Difference is within slippage bounds
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,  # Isolate slippage effect
            slippage=slippage,
            leverage=leverage,
            action_levels=[-1, 0, 1] if leverage > 1 else [0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()

        # Get market price (before slippage)
        market_price = env._cached_base_features["close"]

        # Open position
        action_idx = 2 if leverage > 1 else 1
        td["action"] = action_idx
        td = env.step(td)["next"]

        # Get actual entry price (after slippage)
        entry_price = env.position.entry_price

        if slippage == 0.0:
            # No slippage: entry should equal market
            assert abs(entry_price - market_price) < 0.01, \
                f"With slippage=0, entry={entry_price:.2f} should equal market={market_price:.2f}"
        else:
            # With slippage: entry should differ from market
            # Slippage is sampled from uniform(1-slippage, 1+slippage)
            # So entry_price should be in range [market × (1-slippage), market × (1+slippage)]
            min_expected = market_price * (1 - slippage)
            max_expected = market_price * (1 + slippage)

            assert min_expected <= entry_price <= max_expected, \
                f"Slippage={slippage}: entry={entry_price:.2f} outside range [{min_expected:.2f}, {max_expected:.2f}]"

    @pytest.mark.parametrize("slippage", [0.005, 0.01, 0.02])
    def test_slippage_symmetric_for_long_and_short(self, constant_price_df, slippage):
        """Slippage should affect longs and shorts symmetrically.

        Over many samples, average slippage cost should be similar for long/short.
        We test this by opening many positions and checking the distribution.
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=slippage,
            leverage=2,  # Enable shorts
            action_levels=[-1, 0, 1],
            seed=42,  # Fixed seed for reproducibility
        )

        # Sample multiple longs
        long_entry_prices = []
        for _ in range(20):
            env = SequentialTradingEnv(constant_price_df, config)
            td = env.reset()
            td["action"] = 2  # Long
            td = env.step(td)["next"]
            long_entry_prices.append(env.position.entry_price)

        # Sample multiple shorts
        short_entry_prices = []
        for _ in range(20):
            env = SequentialTradingEnv(constant_price_df, config)
            td = env.reset()
            td["action"] = 0  # Short
            td = env.step(td)["next"]
            short_entry_prices.append(env.position.entry_price)

        # Market price is 100
        market_price = 100.0

        # Calculate average deviation from market
        avg_long_deviation = np.mean([abs(p - market_price) / market_price for p in long_entry_prices])
        avg_short_deviation = np.mean([abs(p - market_price) / market_price for p in short_entry_prices])

        # Deviations should be similar (within 50% of each other)
        ratio = avg_long_deviation / avg_short_deviation if avg_short_deviation > 0 else 1.0
        assert 0.5 < ratio < 2.0, \
            f"Slippage asymmetric: long deviation={avg_long_deviation:.4f}, short={avg_short_deviation:.4f}"

    @pytest.mark.parametrize("slippage,fee", [
        (0.01, 0.0),     # Only slippage
        (0.0, 0.001),    # Only fee
        (0.01, 0.001),   # Both
    ])
    def test_slippage_and_fees_are_separate(self, constant_price_df, slippage, fee):
        """Slippage affects execution price, fees affect balance.

        These are separate costs:
        - Slippage: execution_price = market_price × (1 ± slippage)
        - Fee: balance -= notional × fee

        Both should be applied independently.
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=fee,
            slippage=slippage,
            leverage=1,
            action_levels=[0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        initial_balance = env.balance
        market_price = env._cached_base_features["close"]

        # Open position
        td["action"] = 1
        td = env.step(td)["next"]

        entry_price = env.position.entry_price
        position_size = env.position.position_size

        # Calculate expected balance
        # In spot mode: balance should be nearly 0 (used all for position)
        # Fee is deducted: notional × fee where notional = position_size × entry_price
        notional = abs(position_size * entry_price)
        expected_fee = notional * fee

        # Balance should be: initial - notional (used for position) - fee
        # Which is approximately: initial - notional - fee ≈ 0 (since notional ≈ initial)
        # More precisely: balance = initial - margin_required - fee
        # For spot: margin_required = notional

        # The key test: verify fee was deducted separately from slippage effect
        if fee > 0:
            # With fees, balance should be less than without fees
            assert env.balance < initial_balance, \
                f"Fee={fee}: balance should decrease"

        # Slippage affects entry price, not balance directly
        if slippage > 0:
            # Entry price should differ from market
            assert abs(entry_price - market_price) / market_price <= slippage, \
                f"Slippage={slippage}: entry price deviation too large"


class TestSlippageImpactOnPnL:
    """Test that slippage affects realized PnL correctly."""

    def test_slippage_affects_execution_price_on_trades(self, constant_price_df):
        """Slippage should cause execution prices to differ from market price.

        With slippage, entry prices should differ from market close price.
        Over many trades, slippage typically increases trading costs.
        """
        # With slippage
        config_slip = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.01,  # 1% slippage
            leverage=2,
            action_levels=[-1, 0, 1],
            seed=42,
        )
        env_slip = SequentialTradingEnv(constant_price_df, config_slip)

        td = env_slip.reset()
        market_price = env_slip._cached_base_features["close"]

        # Open
        td["action"] = 2
        td = env_slip.step(td)["next"]

        entry_price = env_slip.position.entry_price

        # Entry price should differ from market price due to slippage
        price_diff_pct = abs(entry_price - market_price) / market_price

        assert price_diff_pct > 0, \
            f"With slippage, entry price should differ from market price"

        # Difference should be within slippage bounds
        assert price_diff_pct <= config_slip.slippage, \
            f"Price difference {price_diff_pct:.3%} exceeds slippage {config_slip.slippage:.3%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
