"""Bankruptcy threshold tests.

CRITICAL: Episode must end when portfolio value drops below bankruptcy threshold.
If broken, agent can continue trading with insufficient capital (unrealistic).

Bankruptcy threshold = initial_pv × bankrupt_threshold
Episode terminates when current_pv < bankruptcy threshold
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
    # Price starts at 100, rallies to 200 at step 10
    prices = [100.0] * 10 + [200.0] * 90
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
def gradual_crash_df():
    """OHLCV data with gradual price decline."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    # Price declines from 100 to 20 linearly
    prices = [100.0 - i * 0.8 for i in range(100)]
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000.0] * 100,
    })
    return df


class TestBankruptcyTrigger:
    """Test that bankruptcy triggers correctly when threshold is crossed."""

    @pytest.mark.parametrize("bankrupt_threshold", [0.1, 0.2, 0.3, 0.5])
    def test_long_bankrupts_on_price_crash(self, crash_df, bankrupt_threshold):
        """Long position must trigger bankruptcy when PV drops below threshold.

        Price crash $100 → $50 = -50% loss on position.
        With leverage=1, loses 50% of PV.
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
        threshold_pv = initial_pv * bankrupt_threshold

        # Open long
        td["action"] = 1
        td = env.step(td)["next"]

        # Step until done or max steps
        done = False
        for step in range(50):
            if done:
                break
            td["action"] = 1  # Hold
            td = env.step(td)["next"]
            done = td["done"].item()

            current_pv = env._get_portfolio_value()

            # If PV dropped below threshold, MUST be done
            if current_pv < threshold_pv:
                assert done, \
                    f"Bankruptcy not triggered! PV={current_pv:.2f} < threshold={threshold_pv:.2f}, " \
                    f"but done={done} at step {step}"

    @pytest.mark.parametrize("bankrupt_threshold", [0.1, 0.2, 0.3, 0.5])
    def test_short_bankrupts_on_price_rally(self, rally_df, bankrupt_threshold):
        """Short position must trigger bankruptcy when price rallies.

        Price rally $100 → $200 = -100% loss on short position.
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
        env = SequentialTradingEnv(rally_df, config)

        td = env.reset()
        initial_pv = env._get_portfolio_value()
        threshold_pv = initial_pv * bankrupt_threshold

        # Open short
        td["action"] = 0
        td = env.step(td)["next"]

        # Step until done
        done = False
        for step in range(50):
            if done:
                break
            td["action"] = 0  # Hold short
            td = env.step(td)["next"]
            done = td["done"].item()

            current_pv = env._get_portfolio_value()

            if current_pv < threshold_pv:
                assert done, \
                    f"Short bankruptcy not triggered! PV={current_pv:.2f} < threshold={threshold_pv:.2f}"


class TestBankruptcyThresholdPrecision:
    """Test that bankruptcy triggers at the correct threshold, not before or after."""

    def test_bankruptcy_triggers_at_threshold_not_before(self, gradual_crash_df):
        """Episode should NOT end until PV actually crosses threshold."""
        bankrupt_threshold = 0.5  # 50% of initial PV
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
        env = SequentialTradingEnv(gradual_crash_df, config)

        td = env.reset()
        initial_pv = env._get_portfolio_value()
        threshold_pv = initial_pv * bankrupt_threshold

        # Open long
        td["action"] = 1
        td = env.step(td)["next"]

        # Track when bankruptcy occurs
        done = False
        last_pv_above_threshold = None

        for step in range(50):
            if done:
                break

            current_pv = env._get_portfolio_value()

            if current_pv >= threshold_pv:
                last_pv_above_threshold = current_pv

            td["action"] = 1  # Hold
            td = env.step(td)["next"]
            done = td["done"].item()

            if done:
                final_pv = env._get_portfolio_value()
                # When done, PV should be at or below threshold
                assert final_pv <= threshold_pv + 100, \
                    f"Bankruptcy triggered too early: final PV={final_pv:.2f} > threshold={threshold_pv:.2f}"


class TestNoBankruptcyWhenSafe:
    """Test that bankruptcy does NOT trigger when it shouldn't."""

    @pytest.mark.parametrize("bankrupt_threshold", [0.1, 0.3, 0.5])
    def test_flat_position_never_bankrupts(self, crash_df, bankrupt_threshold):
        """Holding no position should NEVER trigger bankruptcy.

        With no position, PV = balance = constant.
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

        # Stay flat throughout crash
        done = False
        for step in range(50):
            if done:
                # Check why done - should NOT be bankruptcy
                final_pv = env._get_portfolio_value()
                assert abs(final_pv - initial_pv) < 1.0, \
                    f"Flat position triggered bankruptcy? PV changed from {initial_pv:.2f} to {final_pv:.2f}"
                break

            td["action"] = 1  # Flat
            td = env.step(td)["next"]
            done = td["done"].item()

    def test_profitable_position_never_bankrupts(self, rally_df):
        """Profitable long position should NEVER trigger bankruptcy."""
        config = SequentialTradingEnvConfig(
            execute_on="1Hour",
            time_frames=["1Hour"],
            window_sizes=[5],
            initial_cash=10000,
            transaction_fee=0.0,
            slippage=0.0,
            leverage=1,
            action_levels=[0, 1],
            bankrupt_threshold=0.5,
            random_start=False,
        )
        env = SequentialTradingEnv(rally_df, config)

        td = env.reset()
        initial_pv = env._get_portfolio_value()

        # Long into rally = profit
        td["action"] = 1
        td = env.step(td)["next"]

        for step in range(50):
            td["action"] = 1  # Hold
            td = env.step(td)["next"]
            done = td["done"].item()

            if done:
                final_pv = env._get_portfolio_value()
                # Should have made money, not lost it
                assert final_pv >= initial_pv, \
                    f"Profitable position triggered bankruptcy? PV={final_pv:.2f} < initial={initial_pv:.2f}"
                break


class TestLeverageAndBankruptcy:
    """Test interaction between leverage and bankruptcy."""

    @pytest.mark.parametrize("leverage", [1, 2, 5, 10])
    def test_higher_leverage_faster_bankruptcy(self, crash_df, leverage):
        """Higher leverage should reach bankruptcy faster (in fewer steps).

        With 10x leverage, a 10% price move causes 100% loss.
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

        # Open long
        td["action"] = 2
        td = env.step(td)["next"]

        steps_to_bankruptcy = 0
        done = False
        while not done and steps_to_bankruptcy < 50:
            td["action"] = 2  # Hold
            td = env.step(td)["next"]
            done = td["done"].item()
            steps_to_bankruptcy += 1

        # Higher leverage should bankrupt in fewer steps
        # This is a sanity check - we don't assert exact step counts
        if leverage >= 5:
            assert steps_to_bankruptcy <= 20, \
                f"High leverage ({leverage}x) took {steps_to_bankruptcy} steps to bankrupt"


class TestBankruptcyPVValue:
    """Test that PV is correct when bankruptcy occurs."""

    def test_bankruptcy_pv_at_or_below_threshold(self, crash_df):
        """When bankruptcy triggers, PV should be at or below threshold."""
        bankrupt_threshold = 0.3
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
        threshold_pv = initial_pv * bankrupt_threshold

        # Open long
        td["action"] = 2
        td = env.step(td)["next"]

        done = False
        while not done:
            td["action"] = 2
            td = env.step(td)["next"]
            done = td["done"].item()

        final_pv = env._get_portfolio_value()

        # Final PV should be at or below threshold (allowing small tolerance for timing)
        assert final_pv <= threshold_pv + 500, \
            f"Bankruptcy PV={final_pv:.2f} above threshold={threshold_pv:.2f}"


class TestBankruptcyDisabled:
    """Test behavior when bankruptcy is effectively disabled."""

    def test_zero_threshold_allows_negative_pv(self, crash_df):
        """With bankrupt_threshold=0, episode should not end from bankruptcy.

        Note: Actual behavior depends on implementation - some envs may still
        terminate on liquidation or other conditions.
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
            bankrupt_threshold=0.0,  # Disabled
            random_start=False,
        )
        env = SequentialTradingEnv(crash_df, config)

        td = env.reset()

        # Open long
        td["action"] = 1
        td = env.step(td)["next"]

        # Should be able to continue even with big losses
        steps = 0
        done = False
        while not done and steps < 30:
            td["action"] = 1
            td = env.step(td)["next"]
            done = td["done"].item()
            steps += 1

        # With threshold=0, should have run more steps before bankruptcy
        # (or not bankrupted at all depending on implementation)
        assert steps > 5, \
            f"With threshold=0, episode ended after only {steps} steps"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
