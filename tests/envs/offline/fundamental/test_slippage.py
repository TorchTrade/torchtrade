"""Slippage handling tests.

CRITICAL: Slippage affects execution price in real trading.
If broken, backtests will be unrealistically optimistic.

Current implementation: Random noise in range [1-slippage, 1+slippage]
These tests verify the bounds are respected and slippage is actually applied.
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


class TestSlippageBounds:
    """Test that slippage stays within configured bounds."""

    @pytest.mark.parametrize("slippage", [0.001, 0.005, 0.01, 0.02])
    def test_entry_price_within_slippage_bounds_long(self, constant_price_df, slippage):
        """Long entry price must be within [market*(1-s), market*(1+s)]."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=slippage,
            leverage=2,
            action_levels=[-1, 0, 1],
        )

        market_price = 100.0
        min_allowed = market_price * (1 - slippage)
        max_allowed = market_price * (1 + slippage)

        # Test multiple times to catch bound violations
        for i in range(50):
            env = SequentialTradingEnv(constant_price_df, config)
            td = env.reset()
            td["action"] = 2  # Long
            td = env.step(td)["next"]

            entry = env.position.entry_price
            assert min_allowed <= entry <= max_allowed, \
                f"Trial {i}: entry={entry:.4f} outside bounds [{min_allowed:.4f}, {max_allowed:.4f}]"

    @pytest.mark.parametrize("slippage", [0.001, 0.005, 0.01, 0.02])
    def test_entry_price_within_slippage_bounds_short(self, constant_price_df, slippage):
        """Short entry price must be within [market*(1-s), market*(1+s)]."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=slippage,
            leverage=2,
            action_levels=[-1, 0, 1],
        )

        market_price = 100.0
        min_allowed = market_price * (1 - slippage)
        max_allowed = market_price * (1 + slippage)

        for i in range(50):
            env = SequentialTradingEnv(constant_price_df, config)
            td = env.reset()
            td["action"] = 0  # Short
            td = env.step(td)["next"]

            entry = env.position.entry_price
            assert min_allowed <= entry <= max_allowed, \
                f"Trial {i}: entry={entry:.4f} outside bounds [{min_allowed:.4f}, {max_allowed:.4f}]"


class TestZeroSlippage:
    """Test that zero slippage means exact market price execution."""

    @pytest.mark.parametrize("leverage,action_idx", [
        (1, 1),   # Spot long
        (2, 2),   # Futures long
        (2, 0),   # Futures short
        (5, 2),   # High leverage long
    ])
    def test_zero_slippage_executes_at_market_price(self, constant_price_df, leverage, action_idx):
        """With slippage=0, entry price must EXACTLY equal market price."""
        action_levels = [-1, 0, 1] if leverage > 1 else [0, 1]
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,  # Zero slippage
            leverage=leverage,
            action_levels=action_levels,
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        market_price = env._cached_base_features["close"]

        td["action"] = action_idx
        td = env.step(td)["next"]

        entry = env.position.entry_price
        # With zero slippage, must be EXACT (not approximate)
        assert entry == market_price, \
            f"Zero slippage: entry={entry} != market={market_price}"


class TestSlippageRandomness:
    """Test that slippage introduces actual randomness."""

    def test_slippage_produces_varying_prices(self, constant_price_df):
        """Slippage should produce different entry prices across trials.

        If all entries are identical, slippage isn't being applied.
        Note: Must use seed=None to avoid torch RNG being reseeded identically.
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.01,  # 1% slippage
            leverage=2,
            action_levels=[-1, 0, 1],
            seed=None,  # Avoid reseeding torch RNG
        )

        entries = []
        for _ in range(30):
            env = SequentialTradingEnv(constant_price_df, config)
            td = env.reset()
            td["action"] = 2
            td = env.step(td)["next"]
            entries.append(env.position.entry_price)

        unique_entries = len(set(entries))
        # With random slippage, should have many unique values
        assert unique_entries > 10, \
            f"Only {unique_entries} unique entries out of 30 - slippage not random?"

    def test_slippage_distribution_covers_range(self):
        """Slippage should cover the full range, not just edges or center."""
        # Create a longer dataframe to avoid running out of data
        dates = pd.date_range('2024-01-01', periods=200, freq='1h')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 200,
            'high': [101.0] * 200,
            'low': [99.0] * 200,
            'close': [100.0] * 200,
            'volume': [1000.0] * 200,
        })

        slippage = 0.02  # 2%
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=slippage,
            leverage=2,
            action_levels=[-1, 0, 1],
            seed=None,  # Avoid reseeding torch RNG
            random_start=False,  # Avoid timestamp exhaustion
        )

        market_price = 100.0
        entries = []
        for _ in range(100):
            env = SequentialTradingEnv(df, config)
            td = env.reset()
            td["action"] = 2
            td = env.step(td)["next"]
            entries.append(env.position.entry_price)

        # Check distribution covers the range
        min_entry = min(entries)
        max_entry = max(entries)
        entry_range = max_entry - min_entry
        expected_range = market_price * slippage * 2  # Full range is 2*slippage

        # Should cover at least 50% of the possible range
        coverage = entry_range / expected_range
        assert coverage > 0.5, \
            f"Slippage only covers {coverage:.1%} of range - distribution too narrow"


class TestSlippageFeeIndependence:
    """Test that slippage and fees are independent costs."""

    @pytest.mark.parametrize("slippage,fee", [
        (0.01, 0.0),      # Only slippage
        (0.0, 0.001),     # Only fee
        (0.01, 0.001),    # Both
    ])
    def test_slippage_affects_price_fee_affects_balance(self, constant_price_df, slippage, fee):
        """Slippage changes execution price. Fee deducts from balance.

        These are INDEPENDENT:
        - entry_price = market_price * slippage_factor
        - balance -= notional * fee (calculated on slipped price)
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=fee,
            slippage=slippage,
            leverage=2,
            action_levels=[-1, 0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        initial_balance = env.balance
        market_price = env._cached_base_features["close"]

        td["action"] = 2  # Long
        td = env.step(td)["next"]

        entry_price = env.position.entry_price
        position_size = env.position.position_size

        # Verify slippage affected price (if slippage > 0)
        if slippage > 0:
            price_deviation = abs(entry_price - market_price) / market_price
            assert price_deviation <= slippage, \
                f"Price deviation {price_deviation:.4f} > slippage {slippage}"
        else:
            assert entry_price == market_price, \
                f"No slippage but price differs: {entry_price} != {market_price}"

        # Verify fee was deducted from balance (if fee > 0)
        if fee > 0:
            notional = abs(position_size * entry_price)
            expected_fee = notional * fee
            margin = notional / config.leverage
            expected_balance = initial_balance - margin - expected_fee

            assert abs(env.balance - expected_balance) < 0.01, \
                f"Balance={env.balance:.2f}, expected={expected_balance:.2f}"

    def test_fee_calculated_on_slipped_price(self, constant_price_df):
        """Fee should be calculated on the slipped execution price, not market price."""
        slippage = 0.05  # Large slippage to make difference visible
        fee = 0.001

        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=fee,
            slippage=slippage,
            leverage=2,
            action_levels=[-1, 0, 1],
        )

        # Run multiple times and verify fee is on slipped price
        for _ in range(20):
            env = SequentialTradingEnv(constant_price_df, config)
            td = env.reset()
            initial_balance = env.balance

            td["action"] = 2
            td = env.step(td)["next"]

            entry_price = env.position.entry_price
            position_size = env.position.position_size
            notional = abs(position_size * entry_price)
            margin = notional / config.leverage
            expected_fee = notional * fee

            # Balance = initial - margin - fee
            expected_balance = initial_balance - margin - expected_fee

            assert abs(env.balance - expected_balance) < 0.01, \
                f"Fee not calculated on slipped price. " \
                f"Balance={env.balance:.2f}, expected={expected_balance:.2f}"


class TestSlippageOnClose:
    """Test that slippage affects close/exit trades too."""

    def test_close_trade_has_slippage(self, constant_price_df):
        """Closing a position should also have slippage applied."""
        slippage = 0.01
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=slippage,
            leverage=2,
            action_levels=[-1, 0, 1],
            seed=None,  # Avoid reseeding torch RNG
            random_start=False,  # Avoid timestamp exhaustion
        )

        # Open and close multiple times, track PnL variation
        pnls = []
        for _ in range(30):
            env = SequentialTradingEnv(constant_price_df, config)
            td = env.reset()
            initial_pv = env._get_portfolio_value()

            # Open long
            td["action"] = 2
            td = env.step(td)["next"]

            # Close
            td["action"] = 1
            td = env.step(td)["next"]

            final_pv = env._get_portfolio_value()
            pnls.append(final_pv - initial_pv)

        # With slippage on both open AND close, PnL should vary
        unique_pnls = len(set([round(p, 2) for p in pnls]))
        assert unique_pnls > 5, \
            f"Only {unique_pnls} unique PnLs - close slippage not applied?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
