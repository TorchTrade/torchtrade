"""Value conservation tests.

CRITICAL: Portfolio value must be conserved through trading operations.
- Open/close at constant price with no fees → PV unchanged
- With fees → PV decreases by exactly fee amount
- No value should be created or destroyed except by PnL and fees

If broken, the environment is creating/destroying money incorrectly.
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
    """OHLCV data with price change from $100 to $110."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    prices = [100.0] * 20 + [110.0] * 80
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000.0] * 100,
    })
    return df


class TestPerfectConservation:
    """Test perfect value conservation with no fees/slippage."""

    @pytest.mark.parametrize("leverage,open_idx,close_idx", [
        (1, 1, 0),     # Spot: long then flat
        (2, 2, 1),     # Futures: long then flat
        (2, 0, 1),     # Futures: short then flat
        (5, 2, 1),     # High leverage long
        (10, 0, 1),    # High leverage short
    ])
    def test_round_trip_zero_fees_perfect_conservation(
        self, constant_price_df, leverage, open_idx, close_idx
    ):
        """Open→Close with zero fees at constant price: PV must be EXACTLY unchanged.

        This is the fundamental accounting test. No fees + no price change = no PV change.
        Tolerance: 0.001 (essentially zero)
        """
        action_levels = [-1, 0, 1] if leverage > 1 else [0, 1]
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,  # Zero fees
            slippage=0.0,         # Zero slippage
            leverage=leverage,
            action_levels=action_levels,
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        pv_initial = env._get_portfolio_value()

        # Open position
        td["action"] = open_idx
        td = env.step(td)["next"]
        pv_after_open = env._get_portfolio_value()

        # PV should be unchanged after open (no fees)
        assert abs(pv_after_open - pv_initial) < 0.001, \
            f"PV changed after open: {pv_initial:.4f} → {pv_after_open:.4f}"

        # Close position
        td["action"] = close_idx
        td = env.step(td)["next"]
        pv_after_close = env._get_portfolio_value()

        # PV should be unchanged after close (no fees, no price change)
        assert abs(pv_after_close - pv_initial) < 0.001, \
            f"PV not conserved: {pv_initial:.4f} → {pv_after_close:.4f}"

    @pytest.mark.parametrize("leverage", [1, 2, 5, 10])
    def test_holding_constant_price_no_pv_change(self, constant_price_df, leverage):
        """Holding position at constant price: PV must be EXACTLY unchanged.

        No price change + no trading = no PV change.
        """
        action_levels = [-1, 0, 1] if leverage > 1 else [0, 1]
        open_idx = 2 if leverage > 1 else 1

        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=leverage,
            action_levels=action_levels,
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()

        # Open position
        td["action"] = open_idx
        td = env.step(td)["next"]
        pv_after_open = env._get_portfolio_value()

        # Hold for 20 steps
        for _ in range(20):
            td["action"] = open_idx  # Hold
            td = env.step(td)["next"]
            if td["done"]:
                break

        pv_after_hold = env._get_portfolio_value()

        assert abs(pv_after_hold - pv_after_open) < 0.001, \
            f"PV changed while holding at constant price: {pv_after_open:.4f} → {pv_after_hold:.4f}"


class TestFeeAccountingExact:
    """Test that fees are deducted exactly correctly."""

    @pytest.mark.parametrize("fee,leverage", [
        (0.001, 1),    # 0.1% fee, spot
        (0.001, 2),    # 0.1% fee, futures
        (0.0005, 5),   # 0.05% fee, high leverage
        (0.002, 10),   # 0.2% fee, very high leverage
    ])
    def test_round_trip_loses_exactly_two_fees(self, constant_price_df, fee, leverage):
        """Open→Close loses exactly (open_fee + close_fee).

        fee = notional × fee_rate
        Total cost = 2 × notional × fee_rate (same notional for open and close at constant price)
        """
        action_levels = [-1, 0, 1] if leverage > 1 else [0, 1]
        open_idx = 2 if leverage > 1 else 1
        close_idx = 1 if leverage > 1 else 0

        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=fee,
            slippage=0.0,
            leverage=leverage,
            action_levels=action_levels,
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        pv_initial = env._get_portfolio_value()

        # Open position
        td["action"] = open_idx
        td = env.step(td)["next"]

        # Calculate notional for fee
        notional = abs(env.position.position_size * env.position.entry_price)
        open_fee = notional * fee

        pv_after_open = env._get_portfolio_value()
        expected_pv_after_open = pv_initial - open_fee

        assert abs(pv_after_open - expected_pv_after_open) < 0.01, \
            f"After open: PV={pv_after_open:.2f}, expected={expected_pv_after_open:.2f}"

        # Close position
        td["action"] = close_idx
        td = env.step(td)["next"]

        close_fee = notional * fee  # Same notional at constant price
        total_fees = open_fee + close_fee

        pv_after_close = env._get_portfolio_value()
        expected_pv_final = pv_initial - total_fees

        assert abs(pv_after_close - expected_pv_final) < 0.1, \
            f"After close: PV={pv_after_close:.2f}, expected={expected_pv_final:.2f}, " \
            f"fees={total_fees:.2f}"

    def test_no_fee_means_zero_cost(self, constant_price_df):
        """With fee=0, round trip should have ZERO cost."""
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
        pv_initial = env._get_portfolio_value()

        # Open
        td["action"] = 2
        td = env.step(td)["next"]

        # Close
        td["action"] = 1
        td = env.step(td)["next"]

        pv_final = env._get_portfolio_value()

        # Must be EXACTLY equal (within floating point precision)
        assert abs(pv_final - pv_initial) < 0.001, \
            f"Zero fee round trip cost: {pv_initial - pv_final:.6f}"


class TestPnLSymmetry:
    """Test that PnL is symmetric for long/short positions."""

    def test_long_profit_equals_short_loss(self, price_change_df):
        """When price rises, long profit should equal short loss (with opposite sign).

        Price $100 → $110 (+10%):
        - Long: gains ~10% of position value
        - Short: loses ~10% of position value
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=2,
            action_levels=[-1, 0, 1],
            random_start=False,
        )

        # Long position
        env_long = SequentialTradingEnv(price_change_df, config)
        td = env_long.reset()
        pv_initial = env_long._get_portfolio_value()

        td["action"] = 2  # Long
        td = env_long.step(td)["next"]

        # Step to price change
        for _ in range(25):
            td["action"] = 2  # Hold
            td = env_long.step(td)["next"]
            if td["done"]:
                break

        long_pv = env_long._get_portfolio_value()
        long_pnl = long_pv - pv_initial

        # Short position
        env_short = SequentialTradingEnv(price_change_df, config)
        td = env_short.reset()

        td["action"] = 0  # Short
        td = env_short.step(td)["next"]

        # Step to price change
        for _ in range(25):
            td["action"] = 0  # Hold
            td = env_short.step(td)["next"]
            if td["done"]:
                break

        short_pv = env_short._get_portfolio_value()
        short_pnl = short_pv - pv_initial

        # Long PnL + Short PnL should be ~0 (they're opposite)
        total = long_pnl + short_pnl
        avg_magnitude = (abs(long_pnl) + abs(short_pnl)) / 2

        if avg_magnitude > 10:  # Only check if significant PnL
            asymmetry = abs(total) / avg_magnitude
            assert asymmetry < 0.1, \
                f"PnL asymmetric: long={long_pnl:.2f}, short={short_pnl:.2f}, " \
                f"sum={total:.2f}, asymmetry={asymmetry:.1%}"


class TestAccountingEquation:
    """Test the fundamental accounting equation holds."""

    @pytest.mark.parametrize("leverage", [1, 2, 5, 10])
    def test_balance_plus_margin_plus_pnl_equals_pv(self, constant_price_df, leverage):
        """Fundamental equation: balance + locked_margin + unrealized_pnl = PV.

        This must hold at ALL times.
        """
        action_levels = [-1, 0, 1] if leverage > 1 else [0, 1]

        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.001,
            slippage=0.0,
            leverage=leverage,
            action_levels=action_levels,
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()

        # Test across various states
        actions = [2, 2, 1, 0, 0, 1] if leverage > 1 else [1, 1, 0, 1, 0, 0]

        for action in actions:
            td["action"] = action
            td = env.step(td)["next"]
            if td["done"]:
                break

            current_price = env._cached_base_features["close"]
            pv = env._get_portfolio_value(current_price)

            if env.position.position_size != 0:
                locked_margin = abs(env.position.position_size * env.position.entry_price) / env.leverage
                unrealized_pnl = env.position.position_size * (current_price - env.position.entry_price)
                calculated_pv = env.balance + locked_margin + unrealized_pnl
            else:
                calculated_pv = env.balance

            assert abs(calculated_pv - pv) < 0.01, \
                f"Accounting equation violated: " \
                f"balance({env.balance:.2f}) + margin({locked_margin if env.position.position_size != 0 else 0:.2f}) + " \
                f"pnl({unrealized_pnl if env.position.position_size != 0 else 0:.2f}) = {calculated_pv:.2f} != PV({pv:.2f})"


class TestNoValueCreation:
    """Test that value cannot be created from nothing."""

    def test_cannot_gain_from_round_trip_constant_price(self, constant_price_df):
        """Round trip at constant price can NEVER increase PV."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.001,
            slippage=0.01,
            leverage=2,
            action_levels=[-1, 0, 1],
        )

        for _ in range(20):  # Multiple trials due to slippage randomness
            env = SequentialTradingEnv(constant_price_df, config)
            td = env.reset()
            pv_initial = env._get_portfolio_value()

            # Open
            td["action"] = 2
            td = env.step(td)["next"]

            # Close
            td["action"] = 1
            td = env.step(td)["next"]

            pv_final = env._get_portfolio_value()

            # PV must NOT increase (can only stay same or decrease due to fees)
            assert pv_final <= pv_initial + 0.01, \
                f"Value created from nothing: {pv_initial:.2f} → {pv_final:.2f} (gain={pv_final - pv_initial:.2f})"

    def test_flat_position_pv_constant(self, constant_price_df):
        """Holding flat (no position) at constant price: PV must be EXACTLY constant."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.001,
            slippage=0.01,
            leverage=2,
            action_levels=[-1, 0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        pv_initial = env._get_portfolio_value()

        # Stay flat for 30 steps
        for _ in range(30):
            td["action"] = 1  # Flat
            td = env.step(td)["next"]
            if td["done"]:
                break

        pv_final = env._get_portfolio_value()

        assert abs(pv_final - pv_initial) < 0.001, \
            f"Flat position PV changed: {pv_initial:.4f} → {pv_final:.4f}"


class TestMultiTimeframeConservation:
    """Test that conservation holds with multiple timeframes."""

    @pytest.mark.parametrize("time_frames,window_sizes", [
        (["1Hour"], [5]),
        (["1Hour", "4Hour"], [5, 5]),
    ])
    def test_conservation_multi_timeframe(self, constant_price_df, time_frames, window_sizes):
        """Value conservation must hold regardless of observation timeframes."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=time_frames,
            window_sizes=window_sizes,
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=2,
            action_levels=[-1, 0, 1],
        )
        env = SequentialTradingEnv(constant_price_df, config)

        td = env.reset()
        pv_initial = env._get_portfolio_value()

        # Open
        td["action"] = 2
        td = env.step(td)["next"]

        # Close
        td["action"] = 1
        td = env.step(td)["next"]

        pv_final = env._get_portfolio_value()

        assert abs(pv_final - pv_initial) < 0.001, \
            f"Multi-timeframe conservation violated: {pv_initial:.4f} → {pv_final:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
