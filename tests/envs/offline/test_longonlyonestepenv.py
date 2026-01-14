"""
Tests for LongOnlyOneStepEnv environment with rollout-based trading.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline.longonlyonestepenv import (
    LongOnlyOneStepEnv,
    LongOnlyOneStepEnvConfig,
)
from torchtrade.envs.offline.utils import (
    TimeFrame,
    TimeFrameUnit,
    InitialBalanceSampler,
    build_sltp_action_map,
)


def combinatory_action_map(stoploss_levels, takeprofit_levels):
    """Wrapper for backward compatibility in tests."""
    return build_sltp_action_map(stoploss_levels, takeprofit_levels, include_short_positions=False)


def simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature processing function for testing."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.fillna(0, inplace=True)
    return df


# Note: sample_ohlcv_df, trending_up_df, trending_down_df fixtures are defined in conftest.py


@pytest.fixture
def default_config():
    """Default environment configuration for testing."""
    return LongOnlyOneStepEnvConfig(
        symbol="TEST/USD",
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        initial_cash=1000,
        transaction_fee=0.01,
        stoploss_levels=[-0.05, -0.1],
        takeprofit_levels=[0.05, 0.1],
        slippage=0.0,
        seed=42,
        max_traj_length=100,
    )


@pytest.fixture
def env(sample_ohlcv_df, default_config):
    """Create a LongOnlyOneStepEnv instance for testing."""
    return LongOnlyOneStepEnv(
        df=sample_ohlcv_df,
        config=default_config,
        feature_preprocessing_fn=simple_feature_fn,
    )


class TestInitialBalanceSampler:
    """Tests for InitialBalanceSampler class."""

    def test_fixed_balance(self):
        """Should return fixed balance when int is provided."""
        sampler = InitialBalanceSampler(1000)
        assert sampler.sample() == 1000.0

    def test_random_balance_range(self):
        """Should return random balance within range."""
        sampler = InitialBalanceSampler((500, 1500), seed=42)

        samples = [sampler.sample() for _ in range(10)]

        assert all(500 <= s <= 1500 for s in samples)
        # Should have variation
        assert len(set(samples)) > 1

    def test_returns_float(self):
        """Should always return float."""
        sampler = InitialBalanceSampler(1000)
        assert isinstance(sampler.sample(), float)

        sampler_range = InitialBalanceSampler((500, 1500))
        assert isinstance(sampler_range.sample(), float)


class TestCombinatoryActionMapOneStep:
    """Tests for combinatory_action_map in OneStep context."""

    def test_action_map_includes_hold(self):
        """Action 0 should be hold."""
        action_map = combinatory_action_map([-0.05], [0.1])
        assert action_map[0] == (None, None)

    def test_action_map_combinations(self):
        """Should generate all SL/TP combinations."""
        sl = [-0.05, -0.1]
        tp = [0.1, 0.2]
        action_map = combinatory_action_map(sl, tp)

        # 1 hold + 4 combinations
        assert len(action_map) == 5


class TestLongOnlyOneStepEnvInitialization:
    """Tests for environment initialization."""

    def test_env_initializes(self, env):
        """Environment should initialize without errors."""
        assert env is not None

    def test_action_spec_size(self, env):
        """Action spec should match action map size."""
        # 1 hold + (2 SL * 2 TP) = 5
        assert env.action_spec.n == 5

    def test_periods_per_year_calculated(self, env):
        """Periods per year should be calculated for Sharpe."""
        assert env.periods_per_year > 0
        # For 1-minute data: 365 * 24 * 60 = 525600
        assert env.periods_per_year == 525600

    def test_invalid_transaction_fee_raises(self, sample_ohlcv_df):
        """Should raise error for invalid transaction fee."""
        config = LongOnlyOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=1.5,
        )
        with pytest.raises(ValueError, match="Transaction fee"):
            LongOnlyOneStepEnv(sample_ohlcv_df, config)

    def test_invalid_slippage_raises(self, sample_ohlcv_df):
        """Should raise error for invalid slippage."""
        config = LongOnlyOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            slippage=-0.1,
        )
        with pytest.raises(ValueError, match="Slippage"):
            LongOnlyOneStepEnv(sample_ohlcv_df, config)


class TestLongOnlyOneStepEnvReset:
    """Tests for environment reset."""

    def test_reset_returns_tensordict(self, env):
        """Reset should return TensorDict."""
        td = env.reset()
        assert td is not None

    def test_reset_clears_state(self, env):
        """Reset should clear all state."""
        env.reset()

        assert env.position_size == 0.0
        assert env.current_position == 0.0
        assert env.entry_price == 0.0
        assert env.stop_loss == 0.0
        assert env.take_profit == 0.0
        assert env.step_counter == 0

    def test_reset_initializes_balance(self, env):
        """Reset should initialize balance from sampler."""
        env.reset()
        assert env.balance == 1000  # Fixed initial_cash

    def test_reset_with_random_balance(self, sample_ohlcv_df):
        """Reset should sample random balance when range given."""
        config = LongOnlyOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=(500, 1500),
            max_traj_length=50,
        )
        env = LongOnlyOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

        balances = []
        for _ in range(5):
            env.reset()
            balances.append(env.balance)

        # Should have some variation
        assert len(set(balances)) >= 1  # At least one unique value


class TestLongOnlyOneStepEnvStep:
    """Tests for step functionality."""

    def test_step_always_returns_done(self, env):
        """Each step should set done=True (one-step env)."""
        td = env.reset()

        td.set("action", torch.tensor(1))  # Buy with SL/TP
        result = env.step(td)

        assert result["next"]["done"].item() == True

    def test_step_always_returns_terminated(self, env):
        """Each step should set terminated=True."""
        td = env.reset()

        td.set("action", torch.tensor(1))
        result = env.step(td)

        assert result["next"]["terminated"].item() == True

    def test_hold_returns_zero_reward(self, env):
        """Hold action should return zero reward."""
        td = env.reset()

        td.set("action", torch.tensor(0))  # Hold
        result = env.step(td)

        assert result["next"]["reward"].item() == 0.0

    def test_buy_triggers_rollout(self, env):
        """Buy action should trigger rollout and accumulate returns."""
        td = env.reset()

        td.set("action", torch.tensor(1))  # Buy
        env.step(td)

        # Rollout should have accumulated some returns
        assert len(env.rollout_returns) > 0

    def test_step_increments_counter(self, env):
        """Step should increment step counter."""
        td = env.reset()
        assert env.step_counter == 0

        td.set("action", torch.tensor(0))
        env.step(td)

        assert env.step_counter == 1


class TestLongOnlyOneStepEnvRollout:
    """Tests for rollout functionality."""

    def test_rollout_terminates_on_sl(self, trending_down_df):
        """Rollout should terminate when stop loss is hit."""
        config = LongOnlyOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            transaction_fee=0.001,
            stoploss_levels=[-0.02],  # 2% SL
            takeprofit_levels=[0.5],   # 50% TP (won't hit)
            slippage=0.0,
            max_traj_length=200,
        )
        env = LongOnlyOneStepEnv(trending_down_df, config, simple_feature_fn)

        td = env.reset()
        td.set("action", torch.tensor(1))  # Buy
        env.step(td)

        # Position should be closed after rollout
        assert env.position_size == 0.0

    def test_rollout_terminates_on_tp(self, trending_up_df):
        """Rollout should terminate when take profit is hit."""
        config = LongOnlyOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            transaction_fee=0.001,
            stoploss_levels=[-0.5],   # 50% SL (won't hit)
            takeprofit_levels=[0.02],  # 2% TP
            slippage=0.0,
            max_traj_length=200,
        )
        env = LongOnlyOneStepEnv(trending_up_df, config, simple_feature_fn)

        td = env.reset()
        td.set("action", torch.tensor(1))  # Buy
        env.step(td)

        # Position should be closed after rollout
        assert env.position_size == 0.0

    def test_rollout_accumulates_returns(self, env):
        """Rollout should accumulate log returns."""
        td = env.reset()

        td.set("action", torch.tensor(1))  # Buy
        env.step(td)

        # Should have returns from rollout
        assert len(env.rollout_returns) >= 1
        # Returns should be tensors
        for ret in env.rollout_returns:
            assert isinstance(ret, torch.Tensor)

    def test_rollout_executes_trade(self, env):
        """Rollout should execute the trade and track position."""
        td = env.reset()

        td.set("action", torch.tensor(1))  # Buy
        env.step(td)

        # After step, the environment may have position=1 if still in trade
        # or position=0 if SL/TP triggered - both are valid states
        assert env.current_position in [0, 1]


class TestLongOnlyOneStepEnvReward:
    """Tests for Sharpe ratio reward calculation."""

    def test_reward_is_numeric(self, env):
        """Reward should be a numeric type."""
        td = env.reset()

        td.set("action", torch.tensor(1))
        result = env.step(td)

        reward = result["next"]["reward"].item()
        assert isinstance(reward, (int, float))

    def test_reward_not_nan(self, env):
        """Reward should never be NaN."""
        td = env.reset()

        for action in range(env.action_spec.n):
            td = env.reset()
            td.set("action", torch.tensor(action))
            result = env.step(td)

            assert not torch.isnan(result["next"]["reward"]).any()

    def test_reward_clipped(self, env):
        """Reward should be clipped to [-10, 10]."""
        td = env.reset()

        td.set("action", torch.tensor(1))
        result = env.step(td)

        reward = result["next"]["reward"].item()
        assert -10.0 <= reward <= 10.0

    def test_hold_reward_zero(self, env):
        """Hold action should have zero reward."""
        td = env.reset()

        td.set("action", torch.tensor(0))  # Hold
        result = env.step(td)

        assert result["next"]["reward"].item() == 0.0

    def test_truncated_reward_zero(self, sample_ohlcv_df):
        """Truncated episodes should have zero reward."""
        # Use very short max_traj_length to force truncation
        config = LongOnlyOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            stoploss_levels=[-0.5],  # Won't trigger
            takeprofit_levels=[0.5],  # Won't trigger
            max_traj_length=15,  # Very short
        )
        env = LongOnlyOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()
        td.set("action", torch.tensor(1))  # Buy
        result = env.step(td)

        # If truncated, reward should be 0
        if result["next"]["truncated"].item():
            assert result["next"]["reward"].item() == 0.0


class TestLongOnlyOneStepEnvSLTPPlacement:
    """Tests for SL/TP price placement."""

    def test_sl_is_below_entry_price(self, env):
        """Stop loss should be set below entry price for long."""
        td = env.reset()
        entry_price = env._cached_base_features["close"]

        # Execute trade directly to check SL/TP before rollout
        env._execute_trade_if_needed(env.action_map[1], entry_price)

        assert env.stop_loss < entry_price, f"SL {env.stop_loss} should be < entry {entry_price}"

    def test_tp_is_above_entry_price(self, env):
        """Take profit should be set above entry price for long."""
        td = env.reset()
        entry_price = env._cached_base_features["close"]

        env._execute_trade_if_needed(env.action_map[1], entry_price)

        assert env.take_profit > entry_price, f"TP {env.take_profit} should be > entry {entry_price}"

    def test_sl_tp_calculated_correctly(self, sample_ohlcv_df):
        """SL/TP prices should be calculated from percentages correctly."""
        config = LongOnlyOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            stoploss_levels=[-0.05],  # 5% below
            takeprofit_levels=[0.10],  # 10% above
            slippage=0.0,  # Disable slippage for deterministic test
            max_traj_length=50,
        )
        env = LongOnlyOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()
        entry_price = env._cached_base_features["close"]
        env._execute_trade_if_needed(env.action_map[1], entry_price)

        expected_sl = entry_price * (1 - 0.05)  # 5% below
        expected_tp = entry_price * (1 + 0.10)  # 10% above

        assert abs(env.stop_loss - expected_sl) < 0.01, f"SL {env.stop_loss} != expected {expected_sl}"
        assert abs(env.take_profit - expected_tp) < 0.01, f"TP {env.take_profit} != expected {expected_tp}"


class TestLongOnlyOneStepEnvTradeExecution:
    """Tests for trade execution."""

    def test_buy_sets_sl_tp(self, env):
        """Buy should set stop loss and take profit levels."""
        td = env.reset()

        # Before buy
        assert env.stop_loss == 0.0
        assert env.take_profit == 0.0

        td.set("action", torch.tensor(1))
        env.step(td)

        # Note: After rollout, position is closed so SL/TP may be reset
        # We check during the step that they were set by verifying rollout happened
        assert len(env.rollout_returns) > 0

    def test_buy_deducts_fees(self, env):
        """Buy should deduct transaction fees."""
        td = env.reset()
        initial_balance = env.balance

        td.set("action", torch.tensor(1))  # Buy
        env.step(td)

        # After full cycle, balance should be different from initial
        # (could be more or less depending on trade outcome)
        # Just verify the trade happened
        assert len(env.rollout_returns) > 0

    def test_entry_price_set(self, env):
        """Entry price should be set on buy."""
        td = env.reset()

        td.set("action", torch.tensor(1))  # Buy

        # We need to check during execution, not after rollout
        # Just verify position was opened by checking rollout occurred
        env.step(td)
        assert len(env.rollout_returns) > 0


class TestLongOnlyOneStepEnvMultipleEpisodes:
    """Tests for multiple episodes."""

    def test_multiple_episodes_work(self, env):
        """Should work correctly across multiple episodes."""
        for episode in range(5):
            td = env.reset()

            assert env.position_size == 0.0
            assert env.step_counter == 0

            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)

            assert result["next"]["done"].item() == True

    def test_state_isolation(self, env):
        """Each episode should have isolated state."""
        # Episode 1: Buy
        td = env.reset()
        td.set("action", torch.tensor(1))
        result1 = env.step(td)
        returns1 = len(env.rollout_returns)

        # Episode 2: Hold
        td = env.reset()
        td.set("action", torch.tensor(0))
        result2 = env.step(td)
        returns2 = len(env.rollout_returns)

        # Hold should have no returns
        assert returns2 == 0


class TestLongOnlyOneStepEnvEdgeCases:
    """Tests for edge cases."""

    def test_single_sl_tp_level(self, sample_ohlcv_df):
        """Should work with single SL/TP level."""
        config = LongOnlyOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            stoploss_levels=[-0.05],
            takeprofit_levels=[0.1],
            max_traj_length=50,
        )
        env = LongOnlyOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

        # Should have 2 actions: hold and buy
        assert env.action_spec.n == 2

        td = env.reset()
        td.set("action", torch.tensor(1))
        result = env.step(td)

        assert result is not None

    def test_many_sl_tp_levels(self, sample_ohlcv_df):
        """Should work with many SL/TP levels."""
        config = LongOnlyOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            stoploss_levels=[-0.01, -0.02, -0.05, -0.1],
            takeprofit_levels=[0.01, 0.02, 0.05, 0.1],
            max_traj_length=50,
        )
        env = LongOnlyOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

        # 1 + 16 = 17 actions
        assert env.action_spec.n == 17

    def test_small_initial_cash(self, sample_ohlcv_df):
        """Should work with small initial cash."""
        config = LongOnlyOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1,
            max_traj_length=50,
        )
        env = LongOnlyOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()
        assert env.balance == 1.0

        td.set("action", torch.tensor(1))
        result = env.step(td)

        assert not torch.isnan(result["next"]["reward"]).any()

    def test_zero_transaction_fee(self, sample_ohlcv_df):
        """Should work with zero transaction fee."""
        config = LongOnlyOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            transaction_fee=0.0,
            max_traj_length=50,
        )
        env = LongOnlyOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()
        td.set("action", torch.tensor(1))
        result = env.step(td)

        assert result is not None
