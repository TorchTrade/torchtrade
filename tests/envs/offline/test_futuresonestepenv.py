"""
Tests for FuturesOneStepEnv environment with rollout-based futures trading.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline.futuresonestepenv import (
    FuturesOneStepEnv,
    FuturesOneStepEnvConfig,
    MarginType,
)
from torchtrade.envs.offline.infrastructure.utils import (
    TimeFrame,
    TimeFrameUnit,
    InitialBalanceSampler,
    build_sltp_action_map,
)


def futures_onestep_action_map(stoploss_levels, takeprofit_levels):
    """Wrapper for backward compatibility in tests."""
    return build_sltp_action_map(stoploss_levels, takeprofit_levels, include_short_positions=True)


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
    return FuturesOneStepEnvConfig(
        symbol="TEST/USD",
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        initial_cash=1000,
        transaction_fee=0.0004,
        stoploss_levels=[-0.05, -0.1],
        takeprofit_levels=[0.05, 0.1],
        leverage=5,
        slippage=0.0,
        seed=42,
        max_traj_length=100,
    )


@pytest.fixture
def env(sample_ohlcv_df, default_config):
    """Create a FuturesOneStepEnv instance for testing."""
    return FuturesOneStepEnv(
        df=sample_ohlcv_df,
        config=default_config,
        feature_preprocessing_fn=simple_feature_fn,
    )


class TestFuturesOnestepActionMap:
    """Tests for futures_onestep_action_map function."""

    def test_action_map_includes_hold(self):
        """Action 0 should be hold."""
        action_map = futures_onestep_action_map([-0.05], [0.1])
        assert action_map[0] == (None, None, None)

    def test_action_map_long_combinations(self):
        """Should generate all long SL/TP combinations."""
        sl = [-0.05, -0.1]
        tp = [0.1, 0.2]
        action_map = futures_onestep_action_map(sl, tp)

        # Check long actions (2-5, since 0=HOLD, 1=CLOSE)
        long_actions = [(action_map[i][0], action_map[i][1], action_map[i][2])
                        for i in range(2, 6)]
        for side, _, _ in long_actions:
            assert side == "long"

    def test_action_map_short_combinations(self):
        """Should generate all short SL/TP combinations."""
        sl = [-0.05, -0.1]
        tp = [0.1, 0.2]
        action_map = futures_onestep_action_map(sl, tp)

        # Check short actions (6-9, since 0=HOLD, 1=CLOSE, 2-5=long)
        short_actions = [(action_map[i][0], action_map[i][1], action_map[i][2])
                         for i in range(6, 10)]
        for side, _, _ in short_actions:
            assert side == "short"

    def test_action_map_total_size(self):
        """Should have correct total actions."""
        sl = [-0.05, -0.1]
        tp = [0.1, 0.2]
        action_map = futures_onestep_action_map(sl, tp)

        # 1 hold + 1 close + 4 long + 4 short = 10
        assert len(action_map) == 10


class TestInitialBalanceSamplerFutures:
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


class TestFuturesOneStepEnvInitialization:
    """Tests for environment initialization."""

    def test_env_initializes(self, env):
        """Environment should initialize without errors."""
        assert env is not None

    def test_action_spec_size(self, env):
        """Action spec should match action map size."""
        # 1 hold + (2 SL * 2 TP) long + (2 SL * 2 TP) short = 9 (no CLOSE for OneStep)
        assert env.action_spec.n == 9

    def test_periods_per_year_calculated(self, env):
        """Periods per year should be calculated for Sharpe."""
        assert env.periods_per_year > 0
        # For 1-minute data: 365 * 24 * 60 = 525600
        assert env.periods_per_year == 525600

    def test_leverage_set(self, env):
        """Leverage should be set correctly."""
        assert env.leverage == 5

    def test_invalid_transaction_fee_raises(self, sample_ohlcv_df):
        """Should raise error for invalid transaction fee."""
        config = FuturesOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=1.5,
        )
        with pytest.raises(ValueError, match="Transaction fee"):
            FuturesOneStepEnv(sample_ohlcv_df, config)

    def test_invalid_slippage_raises(self, sample_ohlcv_df):
        """Should raise error for invalid slippage."""
        config = FuturesOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            slippage=-0.1,
        )
        with pytest.raises(ValueError, match="Slippage"):
            FuturesOneStepEnv(sample_ohlcv_df, config)

    def test_invalid_leverage_raises(self, sample_ohlcv_df):
        """Should raise error for invalid leverage."""
        config = FuturesOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            leverage=200,  # Max is 125
        )
        with pytest.raises(ValueError, match="Leverage"):
            FuturesOneStepEnv(sample_ohlcv_df, config)

    def test_account_state_shape(self, env):
        """Account state should have 10 elements."""
        td = env.reset()
        assert td["account_state"].shape == (10,)


class TestFuturesOneStepEnvReset:
    """Tests for environment reset."""

    def test_reset_returns_tensordict(self, env):
        """Reset should return TensorDict."""
        td = env.reset()
        assert td is not None

    def test_reset_clears_state(self, env):
        """Reset should clear all state."""
        env.reset()

        assert env.position.position_size == 0.0
        assert env.position.current_position == 0
        assert env.position.entry_price == 0.0
        assert env.stop_loss_price == 0.0
        assert env.take_profit_price == 0.0
        assert env.liquidation_price == 0.0
        assert env.step_counter == 0

    def test_reset_initializes_balance(self, env):
        """Reset should initialize balance from sampler."""
        env.reset()
        assert env.balance == 1000  # Fixed initial_cash

    def test_reset_with_random_balance(self, sample_ohlcv_df):
        """Reset should sample random balance when range given."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=(500, 1500),
                max_traj_length=50,
            )
            env = FuturesOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

            balances = []
            for _ in range(5):
                env.reset()
                balances.append(env.balance)

            # Should have some variation
            assert len(set(balances)) >= 1


        finally:
            env.close()
class TestFuturesOneStepEnvStep:
    """Tests for step functionality."""

    def test_step_always_returns_done(self, env):
        """Each step should set done=True (one-step env)."""
        td = env.reset()

        td.set("action", torch.tensor(2))  # Long with SL/TP (0=HOLD, 1=CLOSE, 2=first long)
        result = env.step(td)

        assert result["next"]["done"].item() == True

    def test_step_always_returns_terminated(self, env):
        """Each step should set terminated=True."""
        td = env.reset()

        td.set("action", torch.tensor(2))  # First long (0=HOLD, 1=CLOSE)
        result = env.step(td)

        assert result["next"]["terminated"].item() == True

    def test_hold_returns_zero_reward(self, env):
        """Hold action should return zero reward."""
        td = env.reset()

        td.set("action", torch.tensor(0))  # Hold
        result = env.step(td)

        assert result["next"]["reward"].item() == 0.0

    def test_long_triggers_rollout(self, env):
        """Long action should trigger rollout and accumulate returns."""
        td = env.reset()

        td.set("action", torch.tensor(2))  # Long (0=HOLD, 1=CLOSE, 2=first long)
        env.step(td)

        # Rollout should have accumulated some returns
        assert len(env.rollout_returns) > 0

    def test_short_triggers_rollout(self, env):
        """Short action should trigger rollout and accumulate returns."""
        td = env.reset()

        # Short action index = 2 + num_long_combinations (0=HOLD, 1=CLOSE)
        num_long = len(env.stoploss_levels) * len(env.takeprofit_levels)
        td.set("action", torch.tensor(2 + num_long))  # Short
        env.step(td)

        assert len(env.rollout_returns) > 0

    def test_step_increments_counter(self, env):
        """Step should increment step counter."""
        td = env.reset()
        assert env.step_counter == 0

        td.set("action", torch.tensor(0))
        env.step(td)

        assert env.step_counter == 1


class TestFuturesOneStepEnvLongPosition:
    """Tests for long position functionality."""

    def test_long_sets_positive_position(self, env):
        """Long should set positive position size."""
        td = env.reset()

        td.set("action", torch.tensor(2))  # Long (0=HOLD, 1=CLOSE, 2=first long)
        env._execute_trade_if_needed(env.action_map[2])

        assert env.position.position_size > 0
        assert env.position.current_position == 1

    def test_long_sets_sl_below_entry(self, env):
        """Long should set stop loss below entry price."""
        td = env.reset()
        entry_price = env._cached_base_features.close  # PERF: namedtuple attribute access

        td.set("action", torch.tensor(2))  # Long (0=HOLD, 1=CLOSE, 2=first long)
        env._execute_trade_if_needed(env.action_map[2])

        assert env.stop_loss_price < entry_price

    def test_long_sets_tp_above_entry(self, env):
        """Long should set take profit above entry price."""
        td = env.reset()
        entry_price = env._cached_base_features.close  # PERF: namedtuple attribute access

        td.set("action", torch.tensor(2))  # Long (0=HOLD, 1=CLOSE, 2=first long)
        env._execute_trade_if_needed(env.action_map[2])

        assert env.take_profit_price > entry_price

    def test_long_sets_liquidation_price(self, env):
        """Long should set liquidation price below entry."""
        td = env.reset()

        td.set("action", torch.tensor(2))  # Long (0=HOLD, 1=CLOSE, 2=first long)
        env._execute_trade_if_needed(env.action_map[2])

        assert env.liquidation_price > 0
        assert env.liquidation_price < env.position.entry_price


class TestFuturesOneStepEnvShortPosition:
    """Tests for short position functionality."""

    def test_short_sets_negative_position(self, env):
        """Short should set negative position size."""
        td = env.reset()

        num_long = len(env.stoploss_levels) * len(env.takeprofit_levels)
        short_action = 2 + num_long  # First short action (0=HOLD, 1=CLOSE, 2+=long)
        env._execute_trade_if_needed(env.action_map[short_action])

        assert env.position.position_size < 0
        assert env.position.current_position == -1

    def test_short_sets_sl_above_entry(self, env):
        """Short should set stop loss above entry price."""
        td = env.reset()
        entry_price = env._cached_base_features.close  # PERF: namedtuple attribute access

        num_long = len(env.stoploss_levels) * len(env.takeprofit_levels)
        short_action = 2 + num_long
        env._execute_trade_if_needed(env.action_map[short_action])

        assert env.stop_loss_price > entry_price

    def test_short_sets_tp_below_entry(self, env):
        """Short should set take profit below entry price."""
        td = env.reset()
        entry_price = env._cached_base_features.close  # PERF: namedtuple attribute access

        num_long = len(env.stoploss_levels) * len(env.takeprofit_levels)
        short_action = 2 + num_long
        env._execute_trade_if_needed(env.action_map[short_action])

        assert env.take_profit_price < entry_price

    def test_short_sets_liquidation_price(self, env):
        """Short should set liquidation price above entry."""
        td = env.reset()

        num_long = len(env.stoploss_levels) * len(env.takeprofit_levels)
        short_action = 2 + num_long
        env._execute_trade_if_needed(env.action_map[short_action])

        assert env.liquidation_price > 0
        assert env.liquidation_price > env.position.entry_price


class TestFuturesOneStepEnvRollout:
    """Tests for rollout functionality."""

    def test_rollout_terminates_on_long_sl(self, trending_down_df):
        """Rollout should terminate when long stop loss is hit."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                transaction_fee=0.0004,
                stoploss_levels=[-0.02],  # 2% SL
                takeprofit_levels=[0.5],  # 50% TP (won't hit)
                leverage=5,
                slippage=0.0,
                max_traj_length=200,
            )
            env = FuturesOneStepEnv(trending_down_df, config, simple_feature_fn)

            td = env.reset()
            td.set("action", torch.tensor(1))  # Long (0=HOLD, 1=first long) - no CLOSE for OneStep
            env.step(td)

            # Position should be closed after rollout
            assert env.position.position_size == 0.0

        finally:
            env.close()
    def test_rollout_terminates_on_long_tp(self, trending_up_df):
        """Rollout should terminate when long take profit is hit."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                transaction_fee=0.0004,
                stoploss_levels=[-0.5],  # 50% SL (won't hit)
                takeprofit_levels=[0.02],  # 2% TP
                leverage=5,
                slippage=0.0,
                max_traj_length=200,
            )
            env = FuturesOneStepEnv(trending_up_df, config, simple_feature_fn)

            td = env.reset()
            td.set("action", torch.tensor(1))  # Long (0=HOLD, 1=first long) - no CLOSE for OneStep
            env.step(td)

            # Position should be closed after rollout
            assert env.position.position_size == 0.0

        finally:
            env.close()
    def test_rollout_terminates_on_short_sl(self, trending_up_df):
        """Rollout should terminate when short stop loss is hit."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                transaction_fee=0.0004,
                stoploss_levels=[-0.02],  # 2% SL
                takeprofit_levels=[0.5],  # 50% TP (won't hit)
                leverage=5,
                slippage=0.0,
                max_traj_length=200,
            )
            env = FuturesOneStepEnv(trending_up_df, config, simple_feature_fn)

            td = env.reset()
            # Short action (0=HOLD, 1-N=long, N+1+=short) - no CLOSE for OneStep
            num_long = len(env.stoploss_levels) * len(env.takeprofit_levels)
            td.set("action", torch.tensor(1 + num_long))  # Short
            env.step(td)

            # Position should be closed after rollout
            assert env.position.position_size == 0.0

        finally:
            env.close()
    def test_rollout_terminates_on_short_tp(self, trending_down_df):
        """Rollout should terminate when short take profit is hit."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                transaction_fee=0.0004,
                stoploss_levels=[-0.5],  # 50% SL (won't hit)
                takeprofit_levels=[0.02],  # 2% TP
                leverage=5,
                slippage=0.0,
                max_traj_length=200,
            )
            env = FuturesOneStepEnv(trending_down_df, config, simple_feature_fn)

            td = env.reset()
            num_long = len(env.stoploss_levels) * len(env.takeprofit_levels)
            td.set("action", torch.tensor(1 + num_long))  # Short (0=HOLD, 1+=long) - no CLOSE for OneStep
            env.step(td)

            # Position should be closed after rollout
            assert env.position.position_size == 0.0

        finally:
            env.close()
    def test_rollout_accumulates_returns(self, env):
        """Rollout should accumulate log returns."""
        td = env.reset()

        td.set("action", torch.tensor(2))  # Long (0=HOLD, 1=CLOSE, 2=first long)
        env.step(td)

        # Should have returns from rollout
        assert len(env.rollout_returns) >= 1
        # PERF: Returns are now floats (converted to tensor once in _calculate_reward)
        for ret in env.rollout_returns:
            assert isinstance(ret, float)


class TestFuturesOneStepEnvLiquidation:
    """Tests for liquidation functionality."""

    def test_liquidation_price_calculation_long(self, env):
        """Liquidation price should be calculated correctly for long."""
        entry_price = 100.0
        position_size = 1.0  # Long

        liq_price = env._calculate_liquidation_price(entry_price, position_size)

        # For 5x leverage with 0.4% maintenance margin:
        # liq = entry * (1 - 1/5 + 0.004) = entry * 0.804
        expected = entry_price * (1 - 1/5 + 0.004)
        assert abs(liq_price - expected) < 0.01

    def test_liquidation_price_calculation_short(self, env):
        """Liquidation price should be calculated correctly for short."""
        entry_price = 100.0
        position_size = -1.0  # Short

        liq_price = env._calculate_liquidation_price(entry_price, position_size)

        # For 5x leverage with 0.4% maintenance margin:
        # liq = entry * (1 + 1/5 - 0.004) = entry * 1.196
        expected = entry_price * (1 + 1/5 - 0.004)
        assert abs(liq_price - expected) < 0.01

    def test_check_liquidation_long(self, env):
        """Should detect liquidation for long position."""
        env.reset()
        env.position.position_size = 1.0
        env.liquidation_price = 90.0

        assert env._check_liquidation(89.0) == True  # Below liquidation
        assert env._check_liquidation(91.0) == False  # Above liquidation

    def test_check_liquidation_short(self, env):
        """Should detect liquidation for short position."""
        env.reset()
        env.position.position_size = -1.0
        env.liquidation_price = 110.0

        assert env._check_liquidation(111.0) == True  # Above liquidation
        assert env._check_liquidation(109.0) == False  # Below liquidation


class TestFuturesOneStepEnvReward:
    """Tests for Sharpe ratio reward calculation."""

    def test_reward_is_numeric(self, env):
        """Reward should be a numeric type."""
        td = env.reset()

        td.set("action", torch.tensor(2))  # Long (0=HOLD, 1=CLOSE, 2=first long)
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

        td.set("action", torch.tensor(2))  # Long (0=HOLD, 1=CLOSE, 2=first long)
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
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                stoploss_levels=[-0.5],  # Won't trigger
                takeprofit_levels=[0.5],  # Won't trigger
                leverage=5,
                max_traj_length=15,  # Very short
            )
            env = FuturesOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

            td = env.reset()
            td.set("action", torch.tensor(2))  # Long (0=HOLD, 1=CLOSE, 2=first long)
            result = env.step(td)

            # If truncated, reward should be 0
            if result["next"]["truncated"].item():
                assert result["next"]["reward"].item() == 0.0


        finally:
            env.close()
class TestFuturesOneStepEnvPnLCalculation:
    """Tests for PnL calculation."""

    def test_unrealized_pnl_long_profit(self, env):
        """Unrealized PnL should be positive for long with price increase."""
        entry = 100.0
        current = 110.0
        position_size = 1.0

        pnl = env._calculate_unrealized_pnl(entry, current, position_size)
        assert pnl == 10.0  # (110 - 100) * 1

    def test_unrealized_pnl_long_loss(self, env):
        """Unrealized PnL should be negative for long with price decrease."""
        entry = 100.0
        current = 90.0
        position_size = 1.0

        pnl = env._calculate_unrealized_pnl(entry, current, position_size)
        assert pnl == -10.0  # (90 - 100) * 1

    def test_unrealized_pnl_short_profit(self, env):
        """Unrealized PnL should be positive for short with price decrease."""
        entry = 100.0
        current = 90.0
        position_size = -1.0

        pnl = env._calculate_unrealized_pnl(entry, current, position_size)
        assert pnl == 10.0  # (100 - 90) * 1

    def test_unrealized_pnl_short_loss(self, env):
        """Unrealized PnL should be negative for short with price increase."""
        entry = 100.0
        current = 110.0
        position_size = -1.0

        pnl = env._calculate_unrealized_pnl(entry, current, position_size)
        assert pnl == -10.0  # (100 - 110) * 1

    def test_unrealized_pnl_pct_long(self, env):
        """Unrealized PnL pct should be correct for long."""
        entry = 100.0
        current = 110.0
        position_size = 1.0

        pnl_pct = env._calculate_unrealized_pnl_pct(entry, current, position_size)
        assert abs(pnl_pct - 0.1) < 0.001  # 10%

    def test_unrealized_pnl_pct_short(self, env):
        """Unrealized PnL pct should be correct for short."""
        entry = 100.0
        current = 90.0
        position_size = -1.0

        pnl_pct = env._calculate_unrealized_pnl_pct(entry, current, position_size)
        assert abs(pnl_pct - 0.1) < 0.001  # 10%


class TestFuturesOneStepEnvMultipleEpisodes:
    """Tests for multiple episodes."""

    def test_multiple_episodes_work(self, env):
        """Should work correctly across multiple episodes."""
        for episode in range(5):
            td = env.reset()

            assert env.position.position_size == 0.0
            assert env.step_counter == 0

            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)

            assert result["next"]["done"].item() == True

    def test_state_isolation(self, env):
        """Each episode should have isolated state."""
        # Episode 1: Long
        td = env.reset()
        td.set("action", torch.tensor(2))  # Long (0=HOLD, 1=CLOSE, 2=first long)
        result1 = env.step(td)
        returns1 = len(env.rollout_returns)

        # Episode 2: Hold
        td = env.reset()
        td.set("action", torch.tensor(0))
        result2 = env.step(td)
        returns2 = len(env.rollout_returns)

        # Hold should have no returns
        assert returns2 == 0


class TestFuturesOneStepEnvEdgeCases:
    """Tests for edge cases."""

    def test_single_sl_tp_level(self, sample_ohlcv_df):
        """Should work with single SL/TP level."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                stoploss_levels=[-0.05],
                takeprofit_levels=[0.1],
                leverage=5,
                max_traj_length=50,
            )
            env = FuturesOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

            # Should have 3 actions: hold, 1 long, 1 short (no CLOSE for OneStep)
            assert env.action_spec.n == 3

            td = env.reset()
            td.set("action", torch.tensor(1))  # Action 1 = first long (0=HOLD)
            result = env.step(td)

            assert result is not None

        finally:
            env.close()
    def test_many_sl_tp_levels(self, sample_ohlcv_df):
        """Should work with many SL/TP levels."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                stoploss_levels=[-0.01, -0.02, -0.05, -0.1],
                takeprofit_levels=[0.01, 0.02, 0.05, 0.1],
                leverage=5,
                max_traj_length=50,
            )
            env = FuturesOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

            # 1 hold + 16 long + 16 short = 33 actions (no CLOSE for OneStep)
            assert env.action_spec.n == 33

        finally:
            env.close()
    def test_high_leverage(self, sample_ohlcv_df):
        """Should work with high leverage."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                leverage=100,
                max_traj_length=50,
            )
            env = FuturesOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

            td = env.reset()
            td.set("action", torch.tensor(2))  # Long (0=HOLD, 1=CLOSE, 2=first long)
            result = env.step(td)

            assert not torch.isnan(result["next"]["reward"]).any()

        finally:
            env.close()
    def test_small_initial_cash(self, sample_ohlcv_df):
        """Should work with small initial cash."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1,
                leverage=5,
                max_traj_length=50,
            )
            env = FuturesOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

            td = env.reset()
            assert env.balance == 1.0

            td.set("action", torch.tensor(2))  # Long (0=HOLD, 1=CLOSE, 2=first long)
            result = env.step(td)

            assert not torch.isnan(result["next"]["reward"]).any()

        finally:
            env.close()
    def test_zero_transaction_fee(self, sample_ohlcv_df):
        """Should work with zero transaction fee."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                transaction_fee=0.0,
                leverage=5,
                max_traj_length=50,
            )
            env = FuturesOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

            td = env.reset()
            td.set("action", torch.tensor(2))  # Long (0=HOLD, 1=CLOSE, 2=first long)
            result = env.step(td)

            assert result is not None


        finally:
            env.close()
class TestFuturesOneStepEnvLeverage:
    """Tests for leverage functionality."""

    def test_leverage_affects_position_size(self, sample_ohlcv_df):
        """Leverage affects liquidation risk, not position size (with QUANTITY mode)."""
        try:
            config_low = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                leverage=1,
                max_traj_length=50,
            )
            config_high = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                leverage=10,
                max_traj_length=50,
            )

            env_low = FuturesOneStepEnv(sample_ohlcv_df, config_low, simple_feature_fn)
            env_high = FuturesOneStepEnv(sample_ohlcv_df, config_high, simple_feature_fn)

            env_low.reset()
            env_high.reset()

            env_low._execute_trade_if_needed(env_low.action_map[2])  # Long (0=HOLD, 1=CLOSE, 2=first long)
            env_high._execute_trade_if_needed(env_high.action_map[2])

            # With QUANTITY mode, position size is the same regardless of leverage
            assert abs(env_high.position.position_size) == abs(env_low.position.position_size)

            # But higher leverage means liquidation price is closer to entry
            entry_price_low = env_low.position.entry_price
            entry_price_high = env_high.position.entry_price
            liq_distance_low = abs(entry_price_low - env_low.liquidation_price)
            liq_distance_high = abs(entry_price_high - env_high.liquidation_price)
            assert liq_distance_high < liq_distance_low

        finally:
            env_high.close()
            env_low.close()
    def test_leverage_affects_liquidation_distance(self, sample_ohlcv_df):
        """Higher leverage should have closer liquidation price."""
        try:
            config_low = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                leverage=2,
                max_traj_length=50,
            )
            config_high = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                leverage=20,
                max_traj_length=50,
            )

            env_low = FuturesOneStepEnv(sample_ohlcv_df, config_low, simple_feature_fn)
            env_high = FuturesOneStepEnv(sample_ohlcv_df, config_high, simple_feature_fn)

            env_low.reset()
            env_high.reset()

            env_low._execute_trade_if_needed(env_low.action_map[2])  # Long (0=HOLD, 1=CLOSE, 2=first long)
            env_high._execute_trade_if_needed(env_high.action_map[2])

            # Distance from entry to liquidation
            dist_low = abs(env_low.position.entry_price - env_low.liquidation_price) / env_low.position.entry_price
            dist_high = abs(env_high.position.entry_price - env_high.liquidation_price) / env_high.position.entry_price

            # Higher leverage should have smaller distance (closer to liquidation)
            assert dist_high < dist_low


        finally:
            env_high.close()
            env_low.close()
class TestFuturesOneStepEnvDuplicateActions:
    """Tests for duplicate action validation."""

    def test_duplicate_long_action_ignored(self, env):
        """Taking long action when already long should be ignored."""
        env.reset()

        # First long action - should execute (0=HOLD, 1=CLOSE, 2=first long)
        trade_info1 = env._execute_trade_if_needed(env.action_map[2])
        assert trade_info1["executed"] == True
        assert trade_info1["side"] == "long"
        assert env.position.current_position == 1
        entry_price_1 = env.position.entry_price
        balance_after_first = env.balance

        # Second long action - should be ignored
        trade_info2 = env._execute_trade_if_needed(env.action_map[2])
        assert trade_info2["executed"] == False
        assert trade_info2["side"] is None
        assert env.position.current_position == 1  # Still long
        assert env.position.entry_price == entry_price_1  # Entry unchanged
        assert env.balance == balance_after_first  # No additional fee
        assert trade_info2["fee_paid"] == 0.0

    def test_duplicate_short_action_ignored(self, env):
        """Taking short action when already short should be ignored."""
        env.reset()

        # First short action
        num_long = len(env.stoploss_levels) * len(env.takeprofit_levels)
        short_action = 2 + num_long
        trade_info1 = env._execute_trade_if_needed(env.action_map[short_action])
        assert trade_info1["executed"] == True
        assert trade_info1["side"] == "short"
        assert env.position.current_position == -1
        entry_price_1 = env.position.entry_price
        balance_after_first = env.balance

        # Second short action - should be ignored
        trade_info2 = env._execute_trade_if_needed(env.action_map[short_action])
        assert trade_info2["executed"] == False
        assert trade_info2["side"] is None
        assert env.position.current_position == -1  # Still short
        assert env.position.entry_price == entry_price_1  # Entry unchanged
        assert env.balance == balance_after_first  # No additional fee

    def test_different_sltp_long_when_already_long_ignored(self, env):
        """Long with different SL/TP when already long should be ignored."""
        env.reset()

        # Long with first SL/TP combination (0=HOLD, 1=CLOSE, 2=first long)
        trade_info1 = env._execute_trade_if_needed(env.action_map[2])
        assert trade_info1["executed"] == True
        entry_1 = env.position.entry_price
        sl_1 = env.stop_loss_price
        tp_1 = env.take_profit_price

        # Long with different SL/TP combination - should be ignored
        trade_info2 = env._execute_trade_if_needed(env.action_map[3])
        assert trade_info2["executed"] == False
        # SL/TP should not change
        assert env.stop_loss_price == sl_1
        assert env.take_profit_price == tp_1

    def test_hold_action_does_nothing(self, env):
        """Hold action should do nothing."""
        env.reset()
        initial_balance = env.balance

        # Hold action
        trade_info = env._execute_trade_if_needed(env.action_map[0])
        assert trade_info["executed"] == False
        assert trade_info["side"] is None
        assert env.position.current_position == 0
        assert env.position.position_size == 0.0
        assert env.balance == initial_balance  # No fee

    def test_hold_when_long_does_nothing(self, env):
        """Hold action when already long should not close position."""
        env.reset()

        # Open long position (0=HOLD, 1=CLOSE, 2=first long)
        env._execute_trade_if_needed(env.action_map[2])
        assert env.position.current_position == 1
        position_size = env.position.position_size
        balance_after_long = env.balance

        # Hold action - should not close position
        trade_info = env._execute_trade_if_needed(env.action_map[0])
        assert trade_info["executed"] == False
        assert env.position.current_position == 1  # Still long
        assert env.position.position_size == position_size  # Position unchanged
        assert env.balance == balance_after_long  # No fee

    def test_hold_when_short_does_nothing(self, env):
        """Hold action when already short should not close position."""
        env.reset()

        # Open short position
        num_long = len(env.stoploss_levels) * len(env.takeprofit_levels)
        short_action = 2 + num_long
        env._execute_trade_if_needed(env.action_map[short_action])
        assert env.position.current_position == -1
        position_size = env.position.position_size
        balance_after_short = env.balance

        # Hold action - should not close position
        trade_info = env._execute_trade_if_needed(env.action_map[0])
        assert trade_info["executed"] == False
        assert env.position.current_position == -1  # Still short
        assert env.position.position_size == position_size  # Position unchanged
        assert env.balance == balance_after_short  # No fee

    def test_history_tracks_ignored_duplicate_as_no_action(self, env):
        """History should record ignored duplicate actions as no trade (0)."""
        td = env.reset()

        # First long action (0=HOLD, 1=CLOSE, 2=first long)
        td.set("action", torch.tensor(2))
        result1 = env.step(td)
        # In one-step env, after rollout completes, the position is closed
        # So the last action in history is CLOSE (1), not the initial LONG (2)
        assert env.history.actions[-1] == 1  # CLOSE action after rollout

        # Reset for next episode
        td = env.reset()

        # Manually set position to long to simulate duplicate
        env.position.current_position = 1
        env.position.position_size = 1.0
        env.position.entry_price = 100.0

        # Take long action again (duplicate) - using direct _step flow
        old_action_count = len(env.history.actions)
        td.set("action", torch.tensor(2))  # Long action (0=HOLD, 1=CLOSE, 2=first long)
        result2 = env.step(td)

        # History should record 0 (no trade executed) for duplicate action
        assert len(env.history.actions) > old_action_count
        assert env.history.actions[-1] == 0  # No trade executed


class TestFuturesOneStepEnvIncludeHoldAction:
    """Tests for include_hold_action parameter."""

    def test_include_hold_action_true_default(self, sample_ohlcv_df):
        """Should include HOLD action by default."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                stoploss_levels=[-0.05],
                takeprofit_levels=[0.05],
                leverage=1,
                max_traj_length=50,
            )
            env = FuturesOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

            # With 1 SL and 1 TP: 1 hold + 1 long + 1 short = 3 actions (no CLOSE for OneStep)
            assert env.action_spec.n == 3
            assert env.action_map[0] == (None, None, None), "Action 0 should be HOLD"
            assert env.action_map[1][0] == "long", "Action 1 should be long trade"
            assert env.action_map[2][0] == "short", "Action 2 should be short trade"

        finally:
            env.close()
    def test_include_hold_action_true_explicit(self, sample_ohlcv_df):
        """Should include HOLD action when explicitly set to True."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                stoploss_levels=[-0.05, -0.1],
                takeprofit_levels=[0.05, 0.1],
                include_hold_action=True,
                leverage=1,
                max_traj_length=50,
            )
            env = FuturesOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

            # With 2 SL and 2 TP: 1 hold + 4 longs + 4 shorts = 9 actions (no CLOSE for OneStep)
            assert env.action_spec.n == 9
            assert env.action_map[0] == (None, None, None), "Action 0 should be HOLD"
            assert env.action_map[1][0] == "long", "Action 1 should be first long trade"
            assert env.action_map[5][0] == "short", "Action 5 should be first short trade"

        finally:
            env.close()
    def test_include_hold_action_false(self, sample_ohlcv_df):
        """Should exclude HOLD action when set to False."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                stoploss_levels=[-0.05],
                takeprofit_levels=[0.05],
                include_hold_action=False,
                leverage=1,
                max_traj_length=50,
            )
            env = FuturesOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

            # With 1 SL and 1 TP, no hold, no close: 1 long + 1 short = 2 actions
            assert env.action_spec.n == 2
            assert env.action_map[0][0] == "long", "Action 0 should be long (no hold)"
            assert env.action_map[1][0] == "short", "Action 1 should be short"

        finally:
            env.close()
    def test_include_hold_action_false_multiple_sltp(self, sample_ohlcv_df):
        """Should have correct action count without HOLD with multiple SL/TP."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                stoploss_levels=[-0.05, -0.1],
                takeprofit_levels=[0.05, 0.1],
                include_hold_action=False,
                leverage=1,
                max_traj_length=50,
            )
            env = FuturesOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

            # With 2 SL and 2 TP, no hold, no close: 4 longs + 4 shorts = 8 actions
            assert env.action_spec.n == 8
            # Verify all actions are trades (no hold, no close)
            for i in range(env.action_spec.n):
                assert env.action_map[i][0] in ["long", "short"], f"Action {i} should be long or short"

        finally:
            env.close()
    def test_env_works_without_hold_action(self, sample_ohlcv_df):
        """Environment should function correctly without HOLD action."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                stoploss_levels=[-0.05],
                takeprofit_levels=[0.05],
                include_hold_action=False,
                leverage=1,
                max_traj_length=50,
            )
            env = FuturesOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

            # Reset and step
            td = env.reset()
            assert td is not None

            # Take action 0 (which is now a trade, not hold)
            td.set("action", torch.tensor(0))
            result = env.step(td)

            assert result is not None
            assert "reward" in result["next"].keys()
            # Verify reward is numeric and not NaN
            assert not torch.isnan(result["next"]["reward"]).any()

        finally:
            env.close()
    def test_action_mapping_without_hold(self, sample_ohlcv_df):
        """Verify action mapping starts with trades when no hold (no CLOSE for OneStep)."""
        try:
            config = FuturesOneStepEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                stoploss_levels=[-0.05],
                takeprofit_levels=[0.05],
                include_hold_action=False,
                leverage=1,
                max_traj_length=50,
            )
            env = FuturesOneStepEnv(sample_ohlcv_df, config, simple_feature_fn)

            # Action 0 should be the first long position (no hold, no close for OneStep)
            assert env.action_map[0][0] == "long", "First action should be long"
            # Action 1 should be the first short position
            assert env.action_map[1][0] == "short", "Second action should be short"
        finally:
            env.close()
