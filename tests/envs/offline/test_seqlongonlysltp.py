"""
Tests for SeqLongOnlySLTPEnv environment with stop-loss/take-profit functionality.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline.seqlongonlysltp import (
    SeqLongOnlySLTPEnv,
    SeqLongOnlySLTPEnvConfig,
)
from torchtrade.envs.offline.utils import (
    TimeFrame,
    TimeFrameUnit,
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
    return SeqLongOnlySLTPEnvConfig(
        symbol="TEST/USD",
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        initial_cash=1000,
        transaction_fee=0.01,
        stoploss_levels=[-0.05, -0.1],  # 5% and 10% stop loss
        takeprofit_levels=[0.05, 0.1],   # 5% and 10% take profit
        slippage=0.0,
        seed=42,
        max_traj_length=100,
        random_start=False,
    )


@pytest.fixture
def env(sample_ohlcv_df, default_config):
    """Create a SeqLongOnlySLTPEnv instance for testing."""
    return SeqLongOnlySLTPEnv(
        df=sample_ohlcv_df,
        config=default_config,
        feature_preprocessing_fn=simple_feature_fn,
    )


class TestCombinatoryActionMap:
    """Tests for the combinatory_action_map function."""

    def test_action_map_includes_hold(self):
        """Action 0 should always be hold (None, None)."""
        action_map = combinatory_action_map([-0.05], [0.1])
        assert action_map[0] == (None, None)

    def test_action_map_size(self):
        """Action map size should be 1 + (num_sl * num_tp)."""
        sl_levels = [-0.05, -0.1]
        tp_levels = [0.05, 0.1, 0.15]
        action_map = combinatory_action_map(sl_levels, tp_levels)
        expected_size = 1 + len(sl_levels) * len(tp_levels)  # 1 hold + 6 combinations
        assert len(action_map) == expected_size

    def test_action_map_contains_all_combinations(self):
        """Action map should contain all SL/TP combinations."""
        sl_levels = [-0.05, -0.1]
        tp_levels = [0.1, 0.2]
        action_map = combinatory_action_map(sl_levels, tp_levels)

        # Get all non-hold actions
        combinations = [v for k, v in action_map.items() if k != 0]

        # Check all combinations exist
        for sl in sl_levels:
            for tp in tp_levels:
                assert (sl, tp) in combinations

    def test_action_map_single_levels(self):
        """Should work with single SL and TP level."""
        action_map = combinatory_action_map([-0.05], [0.1])
        assert len(action_map) == 2  # hold + 1 combination
        assert action_map[1] == (-0.05, 0.1)


class TestSeqLongOnlySLTPEnvInitialization:
    """Tests for environment initialization."""

    def test_env_initializes(self, env):
        """Environment should initialize without errors."""
        assert env is not None

    def test_action_spec_size(self, env):
        """Action spec should match action map size."""
        expected_size = 1 + 2 * 2  # hold + (2 SL * 2 TP)
        assert env.action_spec.n == expected_size

    def test_action_map_created(self, env):
        """Action map should be created correctly."""
        assert 0 in env.action_map
        assert env.action_map[0] == (None, None)

    def test_invalid_transaction_fee_raises(self, sample_ohlcv_df):
        """Should raise error for invalid transaction fee."""
        config = SeqLongOnlySLTPEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=1.5,
        )
        with pytest.raises(ValueError, match="Transaction fee"):
            SeqLongOnlySLTPEnv(sample_ohlcv_df, config)


class TestSeqLongOnlySLTPEnvReset:
    """Tests for environment reset."""

    def test_reset_returns_tensordict(self, env):
        """Reset should return a TensorDict."""
        td = env.reset()
        assert td is not None

    def test_reset_clears_sl_tp(self, env):
        """Reset should clear stop loss and take profit."""
        env.reset()
        assert env.stop_loss == 0.0
        assert env.take_profit == 0.0

    def test_reset_clears_position(self, env):
        """Reset should clear position."""
        env.reset()
        assert env.position.position_size == 0.0
        assert env.position.current_position == 0.0


class TestSeqLongOnlySLTPEnvBuyWithSLTP:
    """Tests for buy action with SL/TP."""

    def test_buy_sets_stop_loss(self, env):
        """Buy action should set stop loss level."""
        td = env.reset()

        # Action 1 should be first SL/TP combination
        td.set("action", torch.tensor(1))
        env.step(td)

        assert env.stop_loss > 0
        assert env.stop_loss < env.position.entry_price  # SL is below entry

    def test_buy_sets_take_profit(self, env):
        """Buy action should set take profit level."""
        td = env.reset()

        td.set("action", torch.tensor(1))
        env.step(td)

        assert env.take_profit > 0
        assert env.take_profit > env.position.entry_price  # TP is above entry

    def test_sl_tp_calculated_correctly(self, env):
        """SL/TP should be calculated as percentages of entry price."""
        td = env.reset()

        # Get the SL/TP percentages for action 1
        sl_pct, tp_pct = env.action_map[1]

        td.set("action", torch.tensor(1))
        env.step(td)

        expected_sl = env.position.entry_price * (1 + sl_pct)
        expected_tp = env.position.entry_price * (1 + tp_pct)

        assert abs(env.stop_loss - expected_sl) < 0.01
        assert abs(env.take_profit - expected_tp) < 0.01

    def test_hold_does_not_set_sl_tp(self, env):
        """Hold action should not set SL/TP."""
        td = env.reset()

        td.set("action", torch.tensor(0))  # hold
        env.step(td)

        assert env.stop_loss == 0.0
        assert env.take_profit == 0.0


class TestSeqLongOnlySLTPEnvTriggers:
    """Tests for SL/TP trigger conditions."""

    def test_position_exits_on_sl_trigger(self, trending_down_df):
        """Position should exit when stop loss is triggered."""
        config = SeqLongOnlySLTPEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            transaction_fee=0.001,
            stoploss_levels=[-0.02],  # 2% stop loss - should trigger in downtrend
            takeprofit_levels=[0.5],   # 50% TP - won't trigger
            slippage=0.0,
            max_traj_length=200,
            random_start=False,
        )
        env = SeqLongOnlySLTPEnv(trending_down_df, config, simple_feature_fn)

        td = env.reset()

        # Buy with SL/TP
        td.set("action", torch.tensor(1))
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size > 0
        initial_position = env.position.position_size

        # Continue stepping - SL should trigger
        sl_triggered = False
        for _ in range(150):
            td.set("action", torch.tensor(0))  # hold
            result = env.step(td)
            td = result["next"]

            if env.position.position_size == 0 and initial_position > 0:
                sl_triggered = True
                break

            if td.get("done", False):
                break

        assert sl_triggered, "Stop loss should have triggered in downtrend"

    def test_position_exits_on_tp_trigger(self, trending_up_df):
        """Position should exit when take profit is triggered."""
        config = SeqLongOnlySLTPEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            transaction_fee=0.001,
            stoploss_levels=[-0.5],   # 50% SL - won't trigger
            takeprofit_levels=[0.02],  # 2% TP - should trigger in uptrend
            slippage=0.0,
            max_traj_length=200,
            random_start=False,
        )
        env = SeqLongOnlySLTPEnv(trending_up_df, config, simple_feature_fn)

        td = env.reset()

        # Buy with SL/TP
        td.set("action", torch.tensor(1))
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size > 0
        initial_position = env.position.position_size

        # Continue stepping - TP should trigger
        tp_triggered = False
        for _ in range(150):
            td.set("action", torch.tensor(0))  # hold
            result = env.step(td)
            td = result["next"]

            if env.position.position_size == 0 and initial_position > 0:
                tp_triggered = True
                break

            if td.get("done", False):
                break

        assert tp_triggered, "Take profit should have triggered in uptrend"

    def test_sl_tp_cleared_after_trigger(self, trending_down_df):
        """SL/TP should be cleared after position exits."""
        config = SeqLongOnlySLTPEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            transaction_fee=0.001,
            stoploss_levels=[-0.01],  # 1% SL
            takeprofit_levels=[0.5],
            slippage=0.0,
            max_traj_length=100,
            random_start=False,
        )
        env = SeqLongOnlySLTPEnv(trending_down_df, config, simple_feature_fn)

        td = env.reset()

        # Buy
        td.set("action", torch.tensor(1))
        result = env.step(td)
        td = result["next"]

        # Wait for SL to trigger
        for _ in range(50):
            td.set("action", torch.tensor(0))
            result = env.step(td)
            td = result["next"]

            if env.position.position_size == 0:
                # Position exited - check SL/TP cleared
                assert env.position.entry_price == 0.0
                break

            if td.get("done", False):
                break


class TestSeqLongOnlySLTPEnvReward:
    """Tests for reward calculation.

    Note: These tests are intentionally general to allow reward function changes.
    We check for valid outputs (float, not NaN/Inf) rather than specific values.
    """

    def test_reward_is_valid_float(self, env):
        """Reward should be a valid floating point value."""
        td = env.reset()

        for _ in range(10):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]

            reward = td["reward"]
            assert reward.dtype in (torch.float32, torch.float64)
            assert not torch.isnan(reward).any()
            assert not torch.isinf(reward).any()

            if td.get("done", False):
                break

    def test_reward_not_nan_full_episode(self, env):
        """Reward should never be NaN throughout a full episode."""
        td = env.reset()

        for _ in range(env.max_traj_length):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]

            assert not torch.isnan(td["reward"]).any()
            assert not torch.isinf(td["reward"]).any()

            if td.get("done", False):
                break

    def test_terminal_reward_is_valid(self, env):
        """Terminal reward should be a valid float (not NaN/Inf)."""
        td = env.reset()

        # Run until episode ends
        for _ in range(env.max_traj_length):
            td.set("action", torch.tensor(0))  # hold
            result = env.step(td)
            td = result["next"]

            if td.get("done", False):
                reward = td["reward"]
                assert reward.dtype in (torch.float32, torch.float64)
                assert not torch.isnan(reward).any()
                assert not torch.isinf(reward).any()
                break


class TestSeqLongOnlySLTPEnvStep:
    """Tests for step functionality."""

    def test_cannot_buy_when_holding(self, env):
        """Should not be able to buy when already holding position."""
        td = env.reset()

        # First buy
        td.set("action", torch.tensor(1))
        result = env.step(td)
        td = result["next"]

        position_after_buy = env.position.position_size
        balance_after_buy = env.balance

        # Try to buy again with different SL/TP
        td.set("action", torch.tensor(2))
        result = env.step(td)

        # Position and balance should be unchanged (except for any SL/TP triggers)
        if env.position.position_size > 0:  # If SL/TP didn't trigger
            assert env.position.position_size == position_after_buy

    def test_hold_increments_counter(self, env):
        """Hold should increment position hold counter."""
        td = env.reset()

        # Buy first
        td.set("action", torch.tensor(1))
        result = env.step(td)
        td = result["next"]

        if env.position.position_size > 0:
            # Hold
            td.set("action", torch.tensor(0))
            result = env.step(td)
            td = result["next"]

            if env.position.position_size > 0:  # If SL/TP didn't trigger
                assert env.position.hold_counter >= 1

    def test_full_episode_completes(self, env):
        """Full episode should complete without errors."""
        td = env.reset()
        steps = 0

        while steps < env.max_traj_length:
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]
            steps += 1

            if td.get("done", False):
                break

        assert steps > 0


class TestSeqLongOnlySLTPEnvEdgeCases:
    """Tests for edge cases."""

    def test_single_sl_tp_level(self, sample_ohlcv_df):
        """Should work with single SL and TP level."""
        config = SeqLongOnlySLTPEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            stoploss_levels=[-0.05],
            takeprofit_levels=[0.1],
            max_traj_length=50,
            random_start=False,
        )
        env = SeqLongOnlySLTPEnv(sample_ohlcv_df, config, simple_feature_fn)

        # Should have 2 actions: hold and buy
        assert env.action_spec.n == 2

        td = env.reset()
        td.set("action", torch.tensor(1))
        result = env.step(td)

        assert result is not None

    def test_many_sl_tp_levels(self, sample_ohlcv_df):
        """Should work with many SL/TP levels."""
        config = SeqLongOnlySLTPEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            stoploss_levels=[-0.01, -0.02, -0.05, -0.1],
            takeprofit_levels=[0.01, 0.02, 0.05, 0.1],
            max_traj_length=50,
            random_start=False,
        )
        env = SeqLongOnlySLTPEnv(sample_ohlcv_df, config, simple_feature_fn)

        # Should have 1 + 16 = 17 actions
        assert env.action_spec.n == 17

    def test_multiple_episodes(self, env):
        """Should work correctly across multiple episodes."""
        for episode in range(3):
            td = env.reset()

            assert env.stop_loss == 0.0
            assert env.take_profit == 0.0
            assert env.position.position_size == 0.0

            for _ in range(20):
                action = env.action_spec.sample()
                td.set("action", action)
                result = env.step(td)
                td = result["next"]

                if td.get("done", False):
                    break

    def test_price_gap_triggers_sl(self, price_gap_df):
        """SL should trigger even when price gaps past the threshold."""
        config = SeqLongOnlySLTPEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            transaction_fee=0.001,
            stoploss_levels=[-0.05],  # 5% SL - price gap is 10%
            takeprofit_levels=[0.5],   # 50% TP - won't trigger
            slippage=0.0,
            max_traj_length=100,
            random_start=False,
        )
        env = SeqLongOnlySLTPEnv(price_gap_df, config, simple_feature_fn)

        td = env.reset()

        # Buy with SL/TP
        td.set("action", torch.tensor(1))
        result = env.step(td)
        td = result["next"]

        if env.position.position_size > 0:
            initial_position = env.position.position_size

            # Step through until gap-down triggers SL
            sl_triggered = False
            for _ in range(60):
                td.set("action", torch.tensor(0))  # hold
                result = env.step(td)
                td = result["next"]

                if env.position.position_size == 0 and initial_position > 0:
                    sl_triggered = True
                    break

                if td.get("done", False):
                    break

            # Either SL triggered or episode ended - both valid
            assert sl_triggered or td.get("done", False)
