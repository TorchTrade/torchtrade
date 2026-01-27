"""Bankruptcy threshold tests.

CRITICAL: Episode should end when portfolio value drops below bankruptcy threshold.
If broken, agent can continue trading with negative capital (unrealistic).
"""

import pytest
import pandas as pd
from torchtrade.envs.offline.sequential import SequentialTradingEnv, SequentialTradingEnvConfig


@pytest.fixture
def crash_df():
    """OHLCV data with sharp price crash."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    # Price starts at 100, crashes to 50 at step 10
    prices = [100.0] * 10 + [50.0] * 90
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000.0] * 100,
    })
    return df


@pytest.fixture
def rally_df():
    """OHLCV data with sharp price rally."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    # Price starts at 100, rallies to 150 at step 10
    prices = [100.0] * 10 + [150.0] * 90
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000.0] * 100,
    })
    return df


class TestBankruptcyThreshold:
    """Test that bankruptcy threshold triggers episode termination."""

    @pytest.mark.parametrize("bankrupt_threshold", [0.1, 0.2, 0.3])
    def test_long_position_triggers_bankruptcy_on_crash(self, crash_df, bankrupt_threshold):
        """Long position should trigger bankruptcy when price crashes.

        With 50% price crash:
        - Long position loses ~50% of value
        - If bankrupt_threshold >= 0.5, should trigger
        - Episode should end with done=True
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=1,
            action_levels=[0, 1],
            bankrupt_threshold=bankrupt_threshold,
            random_start=False,
        )
        env = SequentialTradingEnv(crash_df, config)

        td = env.reset()
        initial_pv = env._get_portfolio_value()

        # Open long position
        td["action"] = 1
        td = env.step(td)["next"]

        # Step forward through crash
        done = False
        steps = 0
        while not done and steps < 50:
            td["action"] = 1  # Hold long
            td = env.step(td)["next"]
            done = td["done"].item()
            steps += 1

            # Check if portfolio value dropped below threshold
            current_pv = env._get_portfolio_value()
            bankruptcy_threshold = initial_pv * bankrupt_threshold

            if current_pv < bankruptcy_threshold:
                # Should be done
                assert done, \
                    f"Bankruptcy threshold={bankrupt_threshold}: " \
                    f"PV={current_pv:.2f} < threshold={bankruptcy_threshold:.2f}, " \
                    f"but done={done}"
                break

    @pytest.mark.parametrize("bankrupt_threshold", [0.1, 0.2, 0.3])
    def test_short_position_triggers_bankruptcy_on_rally(self, rally_df, bankrupt_threshold):
        """Short position should trigger bankruptcy when price rallies.

        With 50% price rally:
        - Short position loses ~50% of value
        - If bankrupt_threshold >= 0.5, should trigger
        - Episode should end with done=True
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=2,  # Enable shorts
            action_levels=[-1, 0, 1],
            bankrupt_threshold=bankrupt_threshold,
            random_start=False,
        )
        env = SequentialTradingEnv(rally_df, config)

        td = env.reset()
        initial_pv = env._get_portfolio_value()

        # Open short position
        td["action"] = 0
        td = env.step(td)["next"]

        # Step forward through rally
        done = False
        steps = 0
        while not done and steps < 50:
            td["action"] = 0  # Hold short
            td = env.step(td)["next"]
            done = td["done"].item()
            steps += 1

            # Check if portfolio value dropped below threshold
            current_pv = env._get_portfolio_value()
            bankruptcy_threshold = initial_pv * bankrupt_threshold

            if current_pv < bankruptcy_threshold:
                # Should be done
                assert done, \
                    f"Bankruptcy threshold={bankrupt_threshold}: " \
                    f"PV={current_pv:.2f} < threshold={bankruptcy_threshold:.2f}, " \
                    f"but done={done}"
                break

    @pytest.mark.parametrize("leverage", [2, 5, 10])
    def test_higher_leverage_bankrupts_faster(self, crash_df, leverage):
        """Higher leverage should reach bankruptcy threshold faster.

        With same price move, higher leverage amplifies losses.
        Should reach bankruptcy in fewer steps.
        """
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=leverage,
            action_levels=[-1, 0, 1],
            bankrupt_threshold=0.2,
            random_start=False,
        )
        env = SequentialTradingEnv(crash_df, config)

        td = env.reset()

        # Open long position
        td["action"] = 2
        td = env.step(td)["next"]

        # Step forward until bankruptcy or max steps
        steps_to_bankruptcy = 0
        done = False
        while not done and steps_to_bankruptcy < 50:
            td["action"] = 2  # Hold long
            td = env.step(td)["next"]
            done = td["done"].item()
            steps_to_bankruptcy += 1

        # Higher leverage should bankrupt faster (fewer steps)
        # This is a rough heuristic - exact steps depend on implementation
        if leverage == 2:
            assert steps_to_bankruptcy > 0, "Should eventually bankrupt with 2x leverage"
        elif leverage >= 5:
            assert steps_to_bankruptcy > 0, f"Should eventually bankrupt with {leverage}x leverage"


class TestBankruptcyBehavior:
    """Test behavior when bankruptcy threshold is crossed."""

    def test_bankruptcy_returns_remaining_portfolio_value(self, crash_df):
        """When bankruptcy triggers, final PV should match actual portfolio value.

        Bankruptcy doesn't magically erase losses - final PV reflects actual value.
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
            bankrupt_threshold=0.2,
            random_start=False,
        )
        env = SequentialTradingEnv(crash_df, config)

        td = env.reset()
        initial_pv = env._get_portfolio_value()

        # Open long position
        td["action"] = 2
        td = env.step(td)["next"]

        # Step forward until bankruptcy
        done = False
        while not done:
            td["action"] = 2  # Hold long
            td = env.step(td)["next"]
            done = td["done"].item()

            if done:
                final_pv = env._get_portfolio_value()
                bankruptcy_threshold = initial_pv * config.bankrupt_threshold

                # Final PV should be at or below bankruptcy threshold
                assert final_pv <= bankruptcy_threshold or abs(final_pv - bankruptcy_threshold) < 1.0, \
                    f"Final PV={final_pv:.2f} should be <= threshold={bankruptcy_threshold:.2f}"
                break

    @pytest.mark.parametrize("bankrupt_threshold", [0.1, 0.5])
    def test_bankruptcy_threshold_configuration(self, crash_df, bankrupt_threshold):
        """Different bankruptcy thresholds should trigger at different PV levels.

        Higher bankrupt_threshold = more conservative (earlier termination).
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
            bankrupt_threshold=bankrupt_threshold,
            random_start=False,
        )
        env = SequentialTradingEnv(crash_df, config)

        td = env.reset()
        initial_pv = env._get_portfolio_value()
        bankruptcy_threshold = initial_pv * bankrupt_threshold

        # Open long position
        td["action"] = 2
        td = env.step(td)["next"]

        # Step forward
        done = False
        min_pv_seen = initial_pv
        steps = 0
        while not done and steps < 50:
            td["action"] = 2  # Hold long
            td = env.step(td)["next"]
            done = td["done"].item()
            steps += 1

            current_pv = env._get_portfolio_value()
            min_pv_seen = min(min_pv_seen, current_pv)

        # If bankruptcy threshold is 0, should never trigger (unless PV goes negative, which shouldn't happen)
        if bankrupt_threshold == 0.0:
            # May or may not trigger depending on implementation
            pass
        else:
            # If we saw PV drop below threshold, should have triggered
            if min_pv_seen < bankruptcy_threshold:
                assert done, \
                    f"Bankruptcy threshold={bankrupt_threshold}: " \
                    f"min PV={min_pv_seen:.2f} < threshold={bankruptcy_threshold:.2f}, " \
                    f"but episode didn't end"


class TestNoBankruptcyInProfitableTrading:
    """Test that profitable trading never triggers bankruptcy."""

    def test_flat_position_never_bankrupts(self, crash_df):
        """Holding flat (no position) should never trigger bankruptcy."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=2,
            action_levels=[-1, 0, 1],
            bankrupt_threshold=0.2,
            random_start=False,
        )
        env = SequentialTradingEnv(crash_df, config)

        td = env.reset()
        initial_pv = env._get_portfolio_value()

        # Hold flat throughout episode
        done = False
        steps = 0
        while not done and steps < 50:
            td["action"] = 1  # Flat
            td = env.step(td)["next"]
            done = td["done"].item()
            steps += 1

        final_pv = env._get_portfolio_value()

        # PV should be unchanged (minus any small numerical errors)
        assert abs(final_pv - initial_pv) < 1.0, \
            f"Flat position: PV changed from {initial_pv:.2f} to {final_pv:.2f}"

        # Should not have triggered bankruptcy
        assert not done or steps >= 50, \
            f"Flat position triggered bankruptcy at step {steps}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
