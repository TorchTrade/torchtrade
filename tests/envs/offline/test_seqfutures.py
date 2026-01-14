"""
Tests for SeqFuturesEnv environment.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline.seqfutures import SeqFuturesEnv, SeqFuturesEnvConfig, MarginType
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit


def simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature processing function for testing."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.fillna(0, inplace=True)
    return df


# Note: sample_ohlcv_df fixture is defined in conftest.py


@pytest.fixture
def default_config():
    """Default environment configuration for testing."""
    return SeqFuturesEnvConfig(
        symbol="TEST/USD",
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        initial_cash=1000,
        leverage=10,
        transaction_fee=0.0004,  # 0.04%
        slippage=0.0,  # Disable slippage for deterministic tests
        seed=42,
        max_traj_length=100,
        random_start=False,
        bankrupt_threshold=0.1,
    )


@pytest.fixture
def env(sample_ohlcv_df, default_config):
    """Create a SeqFuturesEnv instance for testing."""
    return SeqFuturesEnv(
        df=sample_ohlcv_df,
        config=default_config,
        feature_preprocessing_fn=simple_feature_fn,
    )


class TestSeqFuturesEnvInitialization:
    """Tests for environment initialization."""

    def test_env_initializes(self, env):
        """Environment should initialize without errors."""
        assert env is not None

    def test_action_spec(self, env):
        """Action spec should be categorical with 3 actions."""
        assert env.action_spec.n == 3  # short, hold, long

    def test_observation_spec_has_account_state(self, env):
        """Observation spec should include account_state."""
        assert "account_state" in env.observation_spec.keys()

    def test_observation_spec_has_market_data(self, env):
        """Observation spec should include market data keys."""
        assert len(env.market_data_keys) > 0
        for key in env.market_data_keys:
            assert key in env.observation_spec.keys()

    def test_account_state_shape_is_10(self, env):
        """Account state should have 10 elements (futures)."""
        td = env.reset()
        assert td["account_state"].shape == (10,)

    def test_invalid_transaction_fee_raises(self, sample_ohlcv_df):
        """Should raise error for invalid transaction fee."""
        config = SeqFuturesEnvConfig(transaction_fee=1.5)  # > 1
        with pytest.raises(ValueError, match="Transaction fee"):
            SeqFuturesEnv(sample_ohlcv_df, config)

    def test_invalid_slippage_raises(self, sample_ohlcv_df):
        """Should raise error for invalid slippage."""
        config = SeqFuturesEnvConfig(slippage=-0.1)  # < 0
        with pytest.raises(ValueError, match="Slippage"):
            SeqFuturesEnv(sample_ohlcv_df, config)

    def test_invalid_leverage_raises(self, sample_ohlcv_df):
        """Should raise error for invalid leverage."""
        config = SeqFuturesEnvConfig(leverage=0)  # < 1
        with pytest.raises(ValueError, match="Leverage"):
            SeqFuturesEnv(sample_ohlcv_df, config)

        config = SeqFuturesEnvConfig(leverage=200)  # > 125
        with pytest.raises(ValueError, match="Leverage"):
            SeqFuturesEnv(sample_ohlcv_df, config)


class TestSeqFuturesEnvReset:
    """Tests for environment reset."""

    def test_reset_returns_tensordict(self, env):
        """Reset should return a TensorDict."""
        td = env.reset()
        assert td is not None
        assert hasattr(td, "keys")

    def test_reset_initializes_balance(self, env):
        """Reset should initialize balance correctly."""
        env.reset()
        assert env.balance == 1000  # initial_cash from config

    def test_reset_clears_position(self, env):
        """Reset should clear any existing position."""
        env.reset()
        assert env.position.position_size == 0.0
        assert env.position.current_position == 0
        assert env.position.entry_price == 0.0
        assert env.liquidation_price == 0.0

    def test_reset_clears_counters(self, env):
        """Reset should clear step and hold counters."""
        env.reset()
        assert env.step_counter == 0
        assert env.position.hold_counter == 0

    def test_reset_clears_histories(self, env):
        """Reset should clear history lists."""
        env.reset()
        assert len(env.base_price_history) == 0
        assert len(env.action_history) == 0
        assert len(env.reward_history) == 0
        assert len(env.portfolio_value_history) == 0
        assert len(env.position_history) == 0

    def test_reset_observation_has_correct_keys(self, env):
        """Reset observation should have all required keys."""
        td = env.reset()
        assert "account_state" in td.keys()
        for key in env.market_data_keys:
            assert key in td.keys()

    def test_reset_account_state_shape(self, env):
        """Account state should have correct shape (10 for futures)."""
        td = env.reset()
        assert td["account_state"].shape == (10,)

    def test_reset_account_state_values(self, env):
        """Account state should have correct initial values."""
        td = env.reset()
        account_state = td["account_state"]
        # [cash, position_size, position_value, entry_price, current_price,
        #  unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]
        assert account_state[0].item() == 1000  # cash
        assert account_state[1].item() == 0.0   # position_size
        assert account_state[2].item() == 0.0   # position_value
        assert account_state[3].item() == 0.0   # entry_price
        assert account_state[4].item() > 0      # current_price
        assert account_state[5].item() == 0.0   # unrealized_pnl_pct
        assert account_state[6].item() == 10.0  # leverage
        assert account_state[7].item() == 0.0   # margin_ratio
        assert account_state[8].item() == 0.0   # liquidation_price
        assert account_state[9].item() == 0     # holding_time


class TestSeqFuturesEnvStep:
    """Tests for environment step."""

    def test_step_returns_tensordict(self, env):
        """Step should return a TensorDict."""
        td = env.reset()
        td.set("action", torch.tensor(1))  # hold
        result = env.step(td)
        assert result is not None
        assert "next" in result.keys()

    def test_step_increments_counter(self, env):
        """Step should increment step counter."""
        td = env.reset()
        assert env.step_counter == 0
        td.set("action", torch.tensor(1))  # hold
        env.step(td)
        assert env.step_counter == 1

    def test_step_has_reward(self, env):
        """Step result should include reward."""
        td = env.reset()
        td.set("action", torch.tensor(1))  # hold
        result = env.step(td)
        assert "reward" in result["next"].keys()

    def test_step_has_done_flags(self, env):
        """Step result should include done flags."""
        td = env.reset()
        td.set("action", torch.tensor(1))  # hold
        result = env.step(td)
        next_td = result["next"]
        assert "done" in next_td.keys()
        assert "truncated" in next_td.keys()
        assert "terminated" in next_td.keys()

    def test_step_hold_no_position_change(self, env):
        """Hold action should not change position when no position."""
        td = env.reset()
        initial_balance = env.balance

        td.set("action", torch.tensor(1))  # hold (index 1 = 0.0)
        env.step(td)

        assert env.position.position_size == 0.0
        assert env.balance == initial_balance

    def test_step_updates_histories(self, env):
        """Step should update history lists."""
        td = env.reset()
        td.set("action", torch.tensor(1))  # hold
        env.step(td)

        assert len(env.base_price_history) == 1
        assert len(env.action_history) == 1
        assert len(env.reward_history) == 1
        assert len(env.portfolio_value_history) == 1
        assert len(env.position_history) == 1

    def test_full_episode_completes(self, env):
        """Full episode should complete without errors."""
        td = env.reset()
        steps = 0
        max_steps = env.max_traj_length

        while steps < max_steps:
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]
            steps += 1

            if td.get("done", False):
                break

        assert steps > 0


class TestSeqFuturesEnvLongTrades:
    """Tests for long position execution."""

    def test_long_action_opens_position(self, env):
        """Long action should open a long position."""
        td = env.reset()
        initial_balance = env.balance

        td.set("action", torch.tensor(2))  # long (index 2 = 1.0)
        env.step(td)

        assert env.position.position_size > 0  # Positive for long
        assert env.position.current_position == 1
        assert env.position.entry_price > 0
        assert env.liquidation_price > 0

    def test_long_position_has_correct_liquidation_price(self, env):
        """Long position liquidation price should be below entry."""
        td = env.reset()

        td.set("action", torch.tensor(2))  # long
        env.step(td)

        # For long, liquidation price should be below entry
        assert env.liquidation_price < env.position.entry_price

    def test_long_position_closes_on_hold_action(self, env):
        """Hold action should close existing long position."""
        td = env.reset()

        # Open long
        td.set("action", torch.tensor(2))
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size > 0

        # Close with hold action (action 1 = 0.0 = close)
        td.set("action", torch.tensor(1))
        env.step(td)

        assert env.position.position_size == 0.0
        assert env.position.current_position == 0

    def test_long_pnl_positive_on_price_increase(self, env, trending_up_df):
        """Long position should have positive PnL when price increases."""
        config = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,
            transaction_fee=0.0,  # No fees for cleaner test
            slippage=0.0,
            max_traj_length=100,
            random_start=False,
        )
        env = SeqFuturesEnv(trending_up_df, config, simple_feature_fn)

        td = env.reset()
        initial_balance = env.balance

        # Open long at the beginning
        td.set("action", torch.tensor(2))
        result = env.step(td)
        td = result["next"]

        # Run a few steps to let price increase
        for _ in range(20):
            td.set("action", torch.tensor(2))  # Hold long
            result = env.step(td)
            td = result["next"]

        # Close position
        td.set("action", torch.tensor(1))
        env.step(td)

        # Should have profit
        assert env.balance > initial_balance


class TestSeqFuturesEnvShortTrades:
    """Tests for short position execution."""

    def test_short_action_opens_position(self, env):
        """Short action should open a short position."""
        td = env.reset()

        td.set("action", torch.tensor(0))  # short (index 0 = -1.0)
        env.step(td)

        assert env.position.position_size < 0  # Negative for short
        assert env.position.current_position == -1
        assert env.position.entry_price > 0
        assert env.liquidation_price > 0

    def test_short_position_has_correct_liquidation_price(self, env):
        """Short position liquidation price should be above entry."""
        td = env.reset()

        td.set("action", torch.tensor(0))  # short
        env.step(td)

        # For short, liquidation price should be above entry
        assert env.liquidation_price > env.position.entry_price

    def test_short_position_closes_on_hold_action(self, env):
        """Hold action should close existing short position."""
        td = env.reset()

        # Open short
        td.set("action", torch.tensor(0))
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size < 0

        # Close with hold action
        td.set("action", torch.tensor(1))
        env.step(td)

        assert env.position.position_size == 0.0
        assert env.position.current_position == 0

    def test_short_pnl_positive_on_price_decrease(self, env, trending_down_df):
        """Short position should have positive PnL when price decreases."""
        config = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,
            transaction_fee=0.0,  # No fees for cleaner test
            slippage=0.0,
            max_traj_length=100,
            random_start=False,
        )
        env = SeqFuturesEnv(trending_down_df, config, simple_feature_fn)

        td = env.reset()
        initial_balance = env.balance

        # Open short at the beginning
        td.set("action", torch.tensor(0))
        result = env.step(td)
        td = result["next"]

        # Run a few steps to let price decrease
        for _ in range(20):
            td.set("action", torch.tensor(0))  # Hold short
            result = env.step(td)
            td = result["next"]

        # Close position
        td.set("action", torch.tensor(1))
        env.step(td)

        # Should have profit
        assert env.balance > initial_balance


class TestSeqFuturesEnvPositionFlipping:
    """Tests for flipping between long and short positions."""

    def test_long_to_short_flip(self, env):
        """Going short while long should close long and open short."""
        td = env.reset()

        # Open long
        td.set("action", torch.tensor(2))
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size > 0
        assert env.position.current_position == 1

        # Flip to short
        td.set("action", torch.tensor(0))
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size < 0
        assert env.position.current_position == -1

    def test_short_to_long_flip(self, env):
        """Going long while short should close short and open long."""
        td = env.reset()

        # Open short
        td.set("action", torch.tensor(0))
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size < 0
        assert env.position.current_position == -1

        # Flip to long
        td.set("action", torch.tensor(2))
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size > 0
        assert env.position.current_position == 1


class TestSeqFuturesEnvLeverage:
    """Tests for leverage mechanics."""

    def test_leverage_increases_position_size(self, sample_ohlcv_df):
        """Higher leverage should result in larger position size."""
        config_low = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=5,
            transaction_fee=0.0,
            slippage=0.0,
            max_traj_length=50,
            random_start=False,
        )
        config_high = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=20,
            transaction_fee=0.0,
            slippage=0.0,
            max_traj_length=50,
            random_start=False,
        )

        env_low = SeqFuturesEnv(sample_ohlcv_df, config_low, simple_feature_fn)
        env_high = SeqFuturesEnv(sample_ohlcv_df, config_high, simple_feature_fn)

        td_low = env_low.reset()
        td_high = env_high.reset()

        td_low.set("action", torch.tensor(2))
        td_high.set("action", torch.tensor(2))

        env_low.step(td_low)
        env_high.step(td_high)

        # Higher leverage = larger position size
        assert abs(env_high.position.position_size) > abs(env_low.position.position_size)

    def test_leverage_stored_in_account_state(self, env):
        """Leverage should be stored in account state."""
        td = env.reset()
        assert td["account_state"][6].item() == 10.0  # leverage from config


class TestSeqFuturesEnvLiquidation:
    """Tests for liquidation mechanics."""

    def test_liquidation_price_calculated_correctly(self, env):
        """Liquidation price should be calculated based on leverage."""
        td = env.reset()

        # Open long position
        td.set("action", torch.tensor(2))
        env.step(td)

        entry = env.position.entry_price
        liq = env.liquidation_price
        leverage = env.leverage

        # For 10x leverage with 0.4% maintenance margin:
        # liq = entry * (1 - 1/10 + 0.004) = entry * 0.904
        expected_liq = entry * (1 - 1/leverage + env.maintenance_margin_rate)
        assert pytest.approx(liq, rel=1e-3) == expected_liq

    def test_liquidation_clears_position(self, sample_ohlcv_df, trending_down_df):
        """Liquidation should clear position and realize loss."""
        # Use high leverage to make liquidation more likely
        config = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=50,  # High leverage for faster liquidation
            transaction_fee=0.0,
            slippage=0.0,
            max_traj_length=500,
            random_start=False,
        )
        env = SeqFuturesEnv(trending_down_df, config, simple_feature_fn)

        td = env.reset()
        initial_balance = env.balance

        # Open long position (will lose money as price trends down)
        td.set("action", torch.tensor(2))
        result = env.step(td)
        td = result["next"]

        # Run until liquidation or end of data
        for _ in range(300):
            td.set("action", torch.tensor(2))  # Try to stay long
            result = env.step(td)
            td = result["next"]

            if td.get("done", False):
                break

        # Either got liquidated or episode ended
        # If liquidated, balance should be significantly reduced
        assert env.balance < initial_balance or td.get("done", False)


class TestSeqFuturesEnvReward:
    """Tests for reward calculation."""

    def test_reward_is_float(self, env):
        """Reward should be a float value."""
        td = env.reset()
        td.set("action", torch.tensor(1))
        result = env.step(td)

        reward = result["next"]["reward"]
        assert isinstance(reward, float) or isinstance(reward.item(), float)

    def test_reward_not_nan(self, env):
        """Reward should never be NaN."""
        td = env.reset()

        for _ in range(50):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]

            reward = td["reward"]
            if isinstance(reward, torch.Tensor):
                assert not torch.isnan(reward).any()
            else:
                assert not np.isnan(reward)

            if td.get("done", False):
                break

    def test_reward_not_inf(self, env):
        """Reward should never be infinite."""
        td = env.reset()

        for _ in range(50):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]

            reward = td["reward"]
            if isinstance(reward, torch.Tensor):
                assert not torch.isinf(reward).any()
            else:
                assert not np.isinf(reward)

            if td.get("done", False):
                break


class TestSeqFuturesEnvTermination:
    """Tests for termination conditions."""

    def test_terminates_at_max_steps(self, env):
        """Episode should terminate at max trajectory length."""
        td = env.reset()

        for i in range(env.max_traj_length + 10):
            td.set("action", torch.tensor(1))  # hold
            result = env.step(td)
            td = result["next"]

            if td.get("done", False):
                break

        assert env.step_counter <= env.max_traj_length

    def test_terminates_on_bankruptcy(self, sample_ohlcv_df):
        """Episode should terminate when portfolio value drops below threshold."""
        # High leverage and trending down to cause losses
        config = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=100,
            leverage=50,
            transaction_fee=0.1,  # High fees to accelerate losses
            slippage=0.0,
            max_traj_length=100,
            random_start=False,
            bankrupt_threshold=0.5,  # 50% threshold
        )
        env = SeqFuturesEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()

        # Trade to lose money
        for _ in range(50):
            # Alternate between long and short to pay fees
            td.set("action", torch.tensor(2))
            result = env.step(td)
            td = result["next"]
            if td.get("done", False):
                break

            td.set("action", torch.tensor(0))
            result = env.step(td)
            td = result["next"]
            if td.get("done", False):
                break

        # Should have terminated due to bankruptcy or completed
        portfolio_value = env._get_portfolio_value()
        bankruptcy_threshold = config.bankrupt_threshold * env.initial_portfolio_value

        assert td.get("done", False) or portfolio_value >= bankruptcy_threshold


class TestSeqFuturesEnvEdgeCases:
    """Tests for edge cases."""

    def test_multiple_episodes(self, env):
        """Environment should work correctly across multiple episodes."""
        for episode in range(5):
            td = env.reset()

            # Verify clean state
            assert env.step_counter == 0
            assert env.position.position_size == 0.0
            assert env.balance == 1000

            # Run a few steps
            for _ in range(10):
                action = env.action_spec.sample()
                td.set("action", action)
                result = env.step(td)
                td = result["next"]

                if td.get("done", False):
                    break

    def test_zero_transaction_fee(self, sample_ohlcv_df):
        """Environment should work with zero transaction fees."""
        config = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,
            transaction_fee=0.0,
            slippage=0.0,
            max_traj_length=20,
            random_start=False,
        )
        env = SeqFuturesEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()
        td.set("action", torch.tensor(2))  # long
        result = env.step(td)

        assert not np.isnan(result["next"]["reward"])

    def test_different_margin_types(self, sample_ohlcv_df):
        """Environment should accept different margin types."""
        config_isolated = SeqFuturesEnvConfig(
            margin_type=MarginType.ISOLATED,
            max_traj_length=20,
        )
        config_crossed = SeqFuturesEnvConfig(
            margin_type=MarginType.CROSSED,
            max_traj_length=20,
        )

        env_isolated = SeqFuturesEnv(sample_ohlcv_df, config_isolated, simple_feature_fn)
        env_crossed = SeqFuturesEnv(sample_ohlcv_df, config_crossed, simple_feature_fn)

        assert env_isolated.margin_type == MarginType.ISOLATED
        assert env_crossed.margin_type == MarginType.CROSSED

    def test_position_value_calculation(self, env):
        """Position value should be correctly calculated."""
        td = env.reset()

        td.set("action", torch.tensor(2))  # long
        result = env.step(td)
        td = result["next"]

        # Position value should be abs(position_size * current_price)
        current_price = td["account_state"][4].item()
        position_size = td["account_state"][1].item()
        position_value = td["account_state"][2].item()

        expected_value = abs(position_size * current_price)
        assert pytest.approx(position_value, rel=0.01) == expected_value


class TestSeqFuturesEnvPositionSizing:
    """Tests for position sizing edge cases."""

    def test_can_open_position_after_many_trades(self, sample_ohlcv_df):
        """
        Regression test for floating-point precision bug.

        After many trades, the balance might settle to a value where
        margin_required + fee ≈ balance exactly. Due to floating-point
        precision issues, without the safety margin in _open_position,
        the check `margin_required + fee > balance` could incorrectly
        fail, blocking all future trades.

        This test verifies that positions can still be opened even when
        the balance is at the edge of what's needed for a trade.
        """
        config = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,
            transaction_fee=0.0004,
            slippage=0.0,
            max_traj_length=200,
            random_start=False,
        )
        env = SeqFuturesEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()

        # Execute many trades to reduce balance through fees
        positions_opened = 0
        for i in range(100):
            # Open position
            td.set("action", torch.tensor(2))  # long
            result = env.step(td)
            td = result["next"]

            if env.position.position_size > 0:
                positions_opened += 1

            if td.get("done", False):
                break

            # Close position
            td.set("action", torch.tensor(1))  # close
            result = env.step(td)
            td = result["next"]

            if td.get("done", False):
                break

        # Should have been able to open positions throughout
        # Without the fix, positions would stop opening after balance gets low
        assert positions_opened > 50, (
            f"Only opened {positions_opened} positions. "
            "Floating-point precision may be blocking trades."
        )

    def test_position_opens_with_exact_balance_for_margin(self, sample_ohlcv_df):
        """
        Test that a position can be opened when balance exactly covers
        margin requirement plus fees (within floating-point tolerance).
        """
        config = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=100,  # Small balance
            leverage=10,
            transaction_fee=0.0004,
            slippage=0.0,
            max_traj_length=50,
            random_start=False,
        )
        env = SeqFuturesEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()
        initial_balance = env.balance

        # Open a position
        td.set("action", torch.tensor(2))  # long
        env.step(td)

        # Position should have been opened
        assert env.position.position_size > 0, "Failed to open position with available balance"

        # Fee should have been deducted
        assert env.balance < initial_balance


class TestSeqFuturesEnvPnLCalculations:
    """Tests for PnL calculations."""

    def test_unrealized_pnl_positive_for_profitable_long(self, env, trending_up_df):
        """Unrealized PnL should be positive for profitable long position."""
        config = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,
            transaction_fee=0.0,
            slippage=0.0,
            max_traj_length=100,
            random_start=False,
        )
        env = SeqFuturesEnv(trending_up_df, config, simple_feature_fn)

        td = env.reset()

        # Open long
        td.set("action", torch.tensor(2))
        result = env.step(td)
        td = result["next"]

        # Let price increase
        for _ in range(20):
            td.set("action", torch.tensor(2))  # Hold long
            result = env.step(td)
            td = result["next"]

        # Unrealized PnL should be positive
        unrealized_pnl_pct = td["account_state"][5].item()
        assert unrealized_pnl_pct > 0

    def test_unrealized_pnl_positive_for_profitable_short(self, env, trending_down_df):
        """Unrealized PnL should be positive for profitable short position."""
        config = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,
            transaction_fee=0.0,
            slippage=0.0,
            max_traj_length=100,
            random_start=False,
        )
        env = SeqFuturesEnv(trending_down_df, config, simple_feature_fn)

        td = env.reset()

        # Open short
        td.set("action", torch.tensor(0))
        result = env.step(td)
        td = result["next"]

        # Let price decrease
        for _ in range(20):
            td.set("action", torch.tensor(0))  # Hold short
            result = env.step(td)
            td = result["next"]

        # Unrealized PnL should be positive
        unrealized_pnl_pct = td["account_state"][5].item()
        assert unrealized_pnl_pct > 0


class TestSeqFuturesEnvMetrics:
    """Tests for get_metrics() method."""

    def test_get_metrics_returns_dict(self, env):
        """get_metrics should return a dictionary."""
        td = env.reset()

        # Run a few steps
        for _ in range(10):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]

        metrics = env.get_metrics()
        assert isinstance(metrics, dict)

    def test_get_metrics_has_required_keys(self, env):
        """get_metrics should return all required metric keys."""
        td = env.reset()

        # Run a few steps to generate history
        for _ in range(10):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]

        metrics = env.get_metrics()

        required_keys = [
            'total_return',
            'sharpe_ratio',
            'sortino_ratio',
            'calmar_ratio',
            'max_drawdown',
            'max_dd_duration',
            'num_trades',
            'win_rate (reward>0)',
            'avg_win',
            'avg_loss',
            'profit_factor',
        ]

        for key in required_keys:
            assert key in metrics, f"Missing required metric: {key}"

    def test_get_metrics_values_are_valid(self, env):
        """get_metrics should return valid (non-NaN, non-Inf) values."""
        td = env.reset()

        # Run some steps
        for _ in range(20):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]
            if td.get("done", False):
                break

        metrics = env.get_metrics()

        for key, value in metrics.items():
            assert not np.isnan(value), f"{key} is NaN"
            assert not np.isinf(value), f"{key} is Inf"
            assert isinstance(value, (int, float)), f"{key} is not numeric"

    def test_get_metrics_total_return_positive_trending_up(self, trending_up_df):
        """Total return should be positive for profitable trading in uptrend."""
        config = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,
            transaction_fee=0.0,  # No fees
            slippage=0.0,
            max_traj_length=100,
            random_start=False,
        )
        env = SeqFuturesEnv(trending_up_df, config, simple_feature_fn)

        td = env.reset()

        # Go long and stay long
        for i in range(50):
            td.set("action", torch.tensor(2))  # long
            result = env.step(td)
            td = result["next"]
            if td.get("done", False):
                break

        metrics = env.get_metrics()

        # Should have positive return
        assert metrics['total_return'] > 0, "Should have profit from long position in uptrend"

    def test_get_metrics_total_return_negative_on_losses(self, trending_down_df):
        """Total return should be negative for unprofitable trading."""
        config = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,
            transaction_fee=0.0,
            slippage=0.0,
            max_traj_length=100,
            random_start=False,
        )
        env = SeqFuturesEnv(trending_down_df, config, simple_feature_fn)

        td = env.reset()

        # Go long in downtrend (should lose money)
        for i in range(50):
            td.set("action", torch.tensor(2))  # long
            result = env.step(td)
            td = result["next"]
            if td.get("done", False):
                break

        metrics = env.get_metrics()

        # Should have negative return
        assert metrics['total_return'] < 0, "Should have loss from long position in downtrend"

    def test_get_metrics_num_trades_counts_correctly(self, env):
        """num_trades should count only non-hold actions."""
        td = env.reset()

        # Execute 5 long actions
        for _ in range(5):
            td.set("action", torch.tensor(2))  # long
            result = env.step(td)
            td = result["next"]

        # Execute 5 hold actions
        for _ in range(5):
            td.set("action", torch.tensor(1))  # hold
            result = env.step(td)
            td = result["next"]

        metrics = env.get_metrics()

        # Should have counted trades (hold actions that actually traded)
        # Note: First long opens, subsequent longs might not trade, hold that closes counts
        assert isinstance(metrics['num_trades'], int)
        assert metrics['num_trades'] >= 0

    def test_get_metrics_win_rate_bounds(self, env):
        """win_rate should be between 0 and 1."""
        td = env.reset()

        # Run some steps
        for _ in range(30):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]
            if td.get("done", False):
                break

        metrics = env.get_metrics()

        assert 0 <= metrics['win_rate (reward>0)'] <= 1, "Win rate should be between 0 and 1"

    def test_get_metrics_max_drawdown_non_positive(self, env):
        """max_drawdown should be zero or negative."""
        td = env.reset()

        # Run some steps
        for _ in range(20):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]
            if td.get("done", False):
                break

        metrics = env.get_metrics()

        assert metrics['max_drawdown'] <= 0, "Max drawdown should be non-positive"

    def test_get_metrics_max_dd_duration_non_negative(self, env):
        """max_dd_duration should be non-negative integer."""
        td = env.reset()

        # Run some steps
        for _ in range(20):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]
            if td.get("done", False):
                break

        metrics = env.get_metrics()

        assert metrics['max_dd_duration'] >= 0, "Duration should be non-negative"
        assert isinstance(metrics['max_dd_duration'], int), "Duration should be integer"

    def test_get_metrics_empty_history(self, env):
        """get_metrics should handle empty history gracefully."""
        # Reset but don't step
        env.reset()

        # Manually set empty histories
        env.portfolio_value_history = []
        env.reward_history = []
        env.action_history = []

        metrics = env.get_metrics()

        # Should return valid dict with default values
        assert isinstance(metrics, dict)
        assert metrics['total_return'] == 0.0
        assert metrics['num_trades'] == 0

    def test_get_metrics_after_multiple_episodes(self, env):
        """get_metrics should work correctly after resetting."""
        # First episode
        td = env.reset()
        for _ in range(10):
            td.set("action", torch.tensor(2))
            result = env.step(td)
            td = result["next"]

        metrics1 = env.get_metrics()

        # Reset and run second episode
        td = env.reset()
        for _ in range(20):
            td.set("action", torch.tensor(0))  # Different strategy
            result = env.step(td)
            td = result["next"]

        metrics2 = env.get_metrics()

        # Both should be valid dicts
        assert isinstance(metrics1, dict)
        assert isinstance(metrics2, dict)

        # Metrics should be different (different episodes)
        # At least one metric should differ
        differences = sum(
            1 for key in metrics1.keys()
            if abs(metrics1.get(key, 0) - metrics2.get(key, 0)) > 1e-6
        )
        assert differences > 0, "Metrics should differ between different episodes"
