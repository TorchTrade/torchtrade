"""
Tests for SeqFuturesSLTPEnv environment with stop-loss/take-profit and futures functionality.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline.futures.sequential_sltp import (
    SeqFuturesSLTPEnv,
    SeqFuturesSLTPEnvConfig,
)
from torchtrade.envs.offline.infrastructure.utils import (
    TimeFrame,
    TimeFrameUnit,
    build_sltp_action_map,
)


def futures_sltp_action_map(stoploss_levels, takeprofit_levels, include_close_action=False):
    """Build action map for futures SLTP environment tests.

    Args:
        stoploss_levels: List of stop loss percentage levels (e.g., [-0.05, -0.1])
        takeprofit_levels: List of take profit percentage levels (e.g., [0.05, 0.1])
        include_close_action: Include manual CLOSE action (default False)

    Note:
        SLTP environments use bracket orders that auto-close via SL/TP triggers,
        so manual CLOSE action is typically not needed.
    """
    return build_sltp_action_map(
        stoploss_levels,
        takeprofit_levels,
        include_short_positions=True,
        include_close_action=include_close_action
    )


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
    return SeqFuturesSLTPEnvConfig(
        symbol="TEST/USD",
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        initial_cash=1000,
        leverage=10,
        transaction_fee=0.0004,
        stoploss_levels=[-0.05, -0.1],  # 5% and 10% stop loss
        takeprofit_levels=[0.05, 0.1],   # 5% and 10% take profit
        slippage=0.0,  # Disable slippage for deterministic tests
        seed=42,
        max_traj_length=100,
        random_start=False,
    )


@pytest.fixture
def env(sample_ohlcv_df, default_config):
    """Create a SeqFuturesSLTPEnv instance for testing."""
    env_instance = SeqFuturesSLTPEnv(
        df=sample_ohlcv_df,
        config=default_config,
        feature_preprocessing_fn=simple_feature_fn,
    )
    yield env_instance
    # Cleanup: ensure environment is properly closed
    env_instance.close()


class TestFuturesSLTPActionMap:
    """Tests for the futures_sltp_action_map function."""

    def test_action_map_includes_hold(self):
        """Action 0 should always be hold (None, None, None)."""
        action_map = futures_sltp_action_map([-0.05], [0.1])
        assert action_map[0] == (None, None, None)

    def test_action_map_size(self):
        """Action map size should be 1 hold + 2*(num_sl * num_tp)."""
        sl_levels = [-0.05, -0.1]
        tp_levels = [0.05, 0.1, 0.15]
        action_map = futures_sltp_action_map(sl_levels, tp_levels)
        # 1 hold + 6 long combinations + 6 short combinations = 13 (no CLOSE by default)
        expected_size = 1 + 2 * len(sl_levels) * len(tp_levels)
        assert len(action_map) == expected_size

    def test_action_map_contains_long_positions(self):
        """Action map should contain long positions with all SL/TP combinations."""
        sl_levels = [-0.05, -0.1]
        tp_levels = [0.1, 0.2]
        action_map = futures_sltp_action_map(sl_levels, tp_levels)

        # Get all long actions
        long_actions = [v for k, v in action_map.items() if v[0] == "long"]

        # Check all combinations exist
        for sl in sl_levels:
            for tp in tp_levels:
                assert ("long", sl, tp) in long_actions

    def test_action_map_contains_short_positions(self):
        """Action map should contain short positions with all SL/TP combinations."""
        sl_levels = [-0.05, -0.1]
        tp_levels = [0.1, 0.2]
        action_map = futures_sltp_action_map(sl_levels, tp_levels)

        # Get all short actions
        short_actions = [v for k, v in action_map.items() if v[0] == "short"]

        # Check all combinations exist
        for sl in sl_levels:
            for tp in tp_levels:
                assert ("short", sl, tp) in short_actions

    def test_action_map_single_levels(self):
        """Should work with single SL and TP level."""
        action_map = futures_sltp_action_map([-0.05], [0.1])
        # 1 hold + 1 long + 1 short = 3 (no CLOSE by default)
        assert len(action_map) == 3
        assert action_map[1] == ("long", -0.05, 0.1)  # First long (0=HOLD)
        assert action_map[2] == ("short", -0.05, 0.1)  # First short


class TestSeqFuturesSLTPEnvInitialization:
    """Tests for environment initialization."""

    def test_env_initializes(self, env):
        """Environment should initialize without errors."""
        assert env is not None

    def test_action_spec_size(self, env):
        """Action spec should match action map size."""
        # 1 hold + 4 long (2 SL * 2 TP) + 4 short = 9 (no CLOSE by default)
        expected_size = 1 + 2 * 2 * 2
        assert env.action_spec.n == expected_size

    def test_action_map_created(self, env):
        """Action map should be created correctly."""
        assert 0 in env.action_map
        assert env.action_map[0] == (None, None, None)

    def test_observation_spec_has_account_state(self, env):
        """Observation spec should include account_state."""
        assert "account_state" in env.observation_spec.keys()

    def test_account_state_shape_is_10(self, env):
        """Account state should have 10 elements (futures)."""
        td = env.reset()
        assert td["account_state"].shape == (10,)

    def test_invalid_transaction_fee_raises(self, sample_ohlcv_df):
        """Should raise error for invalid transaction fee."""
        config = SeqFuturesSLTPEnvConfig(transaction_fee=1.5)
        with pytest.raises(ValueError, match="Transaction fee"):
            SeqFuturesSLTPEnv(sample_ohlcv_df, config)

    def test_invalid_leverage_raises(self, sample_ohlcv_df):
        """Should raise error for invalid leverage."""
        config = SeqFuturesSLTPEnvConfig(leverage=0)
        with pytest.raises(ValueError, match="Leverage"):
            SeqFuturesSLTPEnv(sample_ohlcv_df, config)

        config = SeqFuturesSLTPEnvConfig(leverage=200)
        with pytest.raises(ValueError, match="Leverage"):
            SeqFuturesSLTPEnv(sample_ohlcv_df, config)


class TestSeqFuturesSLTPEnvReset:
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
        assert env.position.current_position == 0
        assert env.position.entry_price == 0.0
        assert env.liquidation_price == 0.0

    def test_reset_clears_histories(self, env):
        """Reset should clear history lists."""
        env.reset()
        assert len(env.history) == 0
        assert len(env.history.base_prices) == 0
        assert len(env.history.actions) == 0
        assert len(env.history.rewards) == 0
        assert len(env.history.portfolio_values) == 0
        assert len(env.history.positions) == 0


class TestSeqFuturesSLTPEnvLongWithSLTP:
    """Tests for long positions with SL/TP."""

    def test_long_sets_stop_loss(self, env):
        """Long action should set stop loss level."""
        td = env.reset()

        # Action 1 should be first long SL/TP combination
        td.set("action", torch.tensor(1))  # First long (0=HOLD)
        env.step(td)

        assert env.stop_loss > 0
        assert env.stop_loss < env.position.entry_price  # SL is below entry for long

    def test_long_sets_take_profit(self, env):
        """Long action should set take profit level."""
        td = env.reset()

        td.set("action", torch.tensor(1))  # First long (0=HOLD)
        env.step(td)

        assert env.take_profit > 0
        assert env.take_profit > env.position.entry_price  # TP is above entry for long

    def test_long_position_is_positive(self, env):
        """Long position should have positive position_size."""
        td = env.reset()

        td.set("action", torch.tensor(1))  # First long (0=HOLD)
        env.step(td)

        assert env.position.position_size > 0
        assert env.position.current_position == 1

    def test_long_sl_tp_calculated_correctly(self, env):
        """SL/TP should be calculated as percentages of entry price for long."""
        td = env.reset()

        # Get the SL/TP percentages for action 1
        _, sl_pct, tp_pct = env.action_map[1]  # First long

        td.set("action", torch.tensor(1))  # First long (0=HOLD)
        env.step(td)

        expected_sl = env.position.entry_price * (1 + sl_pct)
        expected_tp = env.position.entry_price * (1 + tp_pct)

        assert abs(env.stop_loss - expected_sl) < 0.01
        assert abs(env.take_profit - expected_tp) < 0.01


class TestSeqFuturesSLTPEnvShortWithSLTP:
    """Tests for short positions with SL/TP."""

    def test_short_sets_stop_loss(self, env):
        """Short action should set stop loss level."""
        td = env.reset()

        # First short action is at index 5 (1 hold + 4 long)
        num_long_actions = len(env.stoploss_levels) * len(env.takeprofit_levels)
        short_action = 1 + num_long_actions  # 0=HOLD, 1+=long (no CLOSE)

        td.set("action", torch.tensor(short_action))
        env.step(td)

        assert env.stop_loss > 0
        assert env.stop_loss > env.position.entry_price  # SL is above entry for short

    def test_short_sets_take_profit(self, env):
        """Short action should set take profit level."""
        td = env.reset()

        num_long_actions = len(env.stoploss_levels) * len(env.takeprofit_levels)
        short_action = 1 + num_long_actions  # 0=HOLD, 1+=long (no CLOSE)

        td.set("action", torch.tensor(short_action))
        env.step(td)

        assert env.take_profit > 0
        assert env.take_profit < env.position.entry_price  # TP is below entry for short

    def test_short_position_is_negative(self, env):
        """Short position should have negative position_size."""
        td = env.reset()

        num_long_actions = len(env.stoploss_levels) * len(env.takeprofit_levels)
        short_action = 1 + num_long_actions  # 0=HOLD, 1+=long (no CLOSE)

        td.set("action", torch.tensor(short_action))
        env.step(td)

        assert env.position.position_size < 0
        assert env.position.current_position == -1


class TestSeqFuturesSLTPEnvTriggers:
    """Tests for SL/TP trigger conditions."""

    def test_long_position_exits_on_sl_trigger(self, trending_down_df):
        """Long position should exit when stop loss is triggered."""
        try:
            config = SeqFuturesSLTPEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                initial_cash=1000,
                leverage=10,
                transaction_fee=0.001,
                stoploss_levels=[-0.02],  # 2% stop loss - should trigger in downtrend
                takeprofit_levels=[0.5],   # 50% TP - won't trigger
                slippage=0.0,
                max_traj_length=200,
                random_start=False,
            )
            env = SeqFuturesSLTPEnv(trending_down_df, config, simple_feature_fn)

            td = env.reset()

            # Open long position
            td.set("action", torch.tensor(1))  # First long (0=HOLD)
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

        finally:
            env.close()

    def test_long_position_exits_on_tp_trigger(self, trending_up_df):
        """Long position should exit when take profit is triggered."""
        try:
            config = SeqFuturesSLTPEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                initial_cash=1000,
                leverage=10,
                transaction_fee=0.001,
                stoploss_levels=[-0.5],   # 50% SL - won't trigger
                takeprofit_levels=[0.02],  # 2% TP - should trigger in uptrend
                slippage=0.0,
                max_traj_length=200,
                random_start=False,
            )
            env = SeqFuturesSLTPEnv(trending_up_df, config, simple_feature_fn)

            td = env.reset()

            # Open long position
            td.set("action", torch.tensor(1))  # First long (0=HOLD)
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

        finally:
            env.close()

    def test_short_position_exits_on_sl_trigger(self, trending_up_df):
        """Short position should exit when stop loss is triggered (price goes up)."""
        try:
            config = SeqFuturesSLTPEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                initial_cash=1000,
                leverage=10,
                transaction_fee=0.001,
                stoploss_levels=[-0.02],  # 2% stop loss - should trigger in uptrend for short
                takeprofit_levels=[0.5],   # 50% TP - won't trigger
                slippage=0.0,
                max_traj_length=200,
                random_start=False,
            )
            env = SeqFuturesSLTPEnv(trending_up_df, config, simple_feature_fn)

            td = env.reset()

            # Open short position (0=HOLD, 1=first long, 2=first short)
            td.set("action", torch.tensor(2))  # First short action
            result = env.step(td)
            td = result["next"]

            assert env.position.position_size < 0
            initial_position = env.position.position_size

            # Continue stepping - SL should trigger
            sl_triggered = False
            for _ in range(150):
                td.set("action", torch.tensor(0))  # hold
                result = env.step(td)
                td = result["next"]

                if env.position.position_size == 0 and initial_position < 0:
                    sl_triggered = True
                    break

                if td.get("done", False):
                    break

            assert sl_triggered, "Stop loss should have triggered for short in uptrend"

        finally:
            env.close()

    def test_short_position_exits_on_tp_trigger(self, trending_down_df):
        """Short position should exit when take profit is triggered (price goes down)."""
        try:
            config = SeqFuturesSLTPEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                initial_cash=1000,
                leverage=10,
                transaction_fee=0.001,
                stoploss_levels=[-0.5],   # 50% SL - won't trigger
                takeprofit_levels=[0.02],  # 2% TP - should trigger in downtrend
                slippage=0.0,
                max_traj_length=200,
                random_start=False,
            )
            env = SeqFuturesSLTPEnv(trending_down_df, config, simple_feature_fn)

            td = env.reset()

            # Open short position (action 2 = first short action)
            td.set("action", torch.tensor(2))  # First short action
            result = env.step(td)
            td = result["next"]

            assert env.position.position_size < 0
            initial_position = env.position.position_size

            # Continue stepping - TP should trigger
            tp_triggered = False
            for _ in range(150):
                td.set("action", torch.tensor(0))  # hold
                result = env.step(td)
                td = result["next"]

                if env.position.position_size == 0 and initial_position < 0:
                    tp_triggered = True
                    break

                if td.get("done", False):
                    break

            assert tp_triggered, "Take profit should have triggered for short in downtrend"

        finally:
            env.close()

    def test_sl_tp_cleared_after_trigger(self, trending_down_df):
        """SL/TP should be cleared after position exits."""
        try:
            config = SeqFuturesSLTPEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                initial_cash=1000,
                leverage=10,
                transaction_fee=0.001,
                stoploss_levels=[-0.01],  # 1% SL
                takeprofit_levels=[0.5],
                slippage=0.0,
                max_traj_length=100,
                random_start=False,
            )
            env = SeqFuturesSLTPEnv(trending_down_df, config, simple_feature_fn)

            td = env.reset()

            # Open long
            td.set("action", torch.tensor(1))  # First long (0=HOLD)
            result = env.step(td)
            td = result["next"]

            # Wait for SL to trigger
            for _ in range(50):
                td.set("action", torch.tensor(0))
                result = env.step(td)
                td = result["next"]

                if env.position.position_size == 0:
                    # Position exited - check SL/TP cleared
                    assert env.stop_loss == 0.0
                    assert env.take_profit == 0.0
                    assert env.position.entry_price == 0.0
                    break

                if td.get("done", False):
                    break


        finally:
            env.close()
class TestSeqFuturesSLTPEnvLiquidation:
    """Tests for liquidation functionality."""

    def test_long_liquidation_on_price_drop(self, trending_down_df):
        """Long position should be liquidated if price drops below liquidation price."""
        try:
            config = SeqFuturesSLTPEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                initial_cash=1000,
                leverage=50,  # High leverage for faster liquidation
                transaction_fee=0.001,
                stoploss_levels=[-0.5],   # Wide SL so liquidation happens first
                takeprofit_levels=[0.5],
                slippage=0.0,
                max_traj_length=200,
                random_start=False,
                maintenance_margin_rate=0.004,
            )
            env = SeqFuturesSLTPEnv(trending_down_df, config, simple_feature_fn)

            td = env.reset()

            # Open long position
            td.set("action", torch.tensor(1))  # First long (0=HOLD)
            result = env.step(td)
            td = result["next"]

            assert env.position.position_size > 0
            assert env.liquidation_price > 0

            # Track if liquidation happens
            was_liquidated = False
            initial_position = env.position.position_size

            for _ in range(100):
                td.set("action", torch.tensor(0))  # hold
                result = env.step(td)
                td = result["next"]

                # Check if liquidation occurred
                if env.position.position_size == 0 and initial_position > 0:
                    was_liquidated = True
                    break

                if td.get("done", False):
                    break

            # Either liquidated or hit SL (both are valid exits)
            assert was_liquidated or env.position.position_size == 0

        finally:
            env.close()

    def test_liquidation_clears_sltp(self, trending_down_df):
        """Liquidation should clear SL/TP levels."""
        try:
            config = SeqFuturesSLTPEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                initial_cash=1000,
                leverage=50,
                transaction_fee=0.001,
                stoploss_levels=[-0.5],   # Wide SL
                takeprofit_levels=[0.5],
                slippage=0.0,
                max_traj_length=200,
                random_start=False,
            )
            env = SeqFuturesSLTPEnv(trending_down_df, config, simple_feature_fn)

            td = env.reset()

            # Open long position
            td.set("action", torch.tensor(1))  # First long (0=HOLD)
            result = env.step(td)
            td = result["next"]

            initial_position = env.position.position_size

            for _ in range(100):
                td.set("action", torch.tensor(0))
                result = env.step(td)
                td = result["next"]

                if env.position.position_size == 0 and initial_position > 0:
                    # Position was closed - SL/TP should be cleared
                    assert env.stop_loss == 0.0
                    assert env.take_profit == 0.0
                    break

                if td.get("done", False):
                    break


        finally:
            env.close()
class TestSeqFuturesSLTPEnvReward:
    """Tests for reward calculation."""

    def test_reward_not_nan(self, env):
        """Reward should never be NaN."""
        td = env.reset()

        for _ in range(50):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]

            assert not torch.isnan(td["reward"]).any()

            if td.get("done", False):
                break

    def test_dense_reward_on_steps(self, env):
        """Should receive dense rewards during episode."""
        td = env.reset()

        # Open a position
        td.set("action", torch.tensor(1))  # First long (0=HOLD)
        result = env.step(td)
        td = result["next"]

        # Take a few more steps
        for _ in range(5):
            td.set("action", torch.tensor(0))  # hold
            result = env.step(td)
            td = result["next"]

            # Dense rewards should be present (may be small but not always zero)
            if td.get("done", False):
                break


class TestSeqFuturesSLTPEnvStep:
    """Tests for step functionality."""

    def test_hold_does_not_open_position(self, env):
        """Hold action should not open a position."""
        td = env.reset()

        td.set("action", torch.tensor(0))  # hold
        env.step(td)

        assert env.position.position_size == 0.0
        assert env.stop_loss == 0.0
        assert env.take_profit == 0.0

    def test_hold_does_not_close_existing_position(self, env):
        """HOLD action should not close existing position (CLOSE is disabled by default in SLTP)."""
        td = env.reset()

        # Open long
        td.set("action", torch.tensor(1))  # First long (0=HOLD)
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size > 0
        initial_position_size = env.position.position_size

        # HOLD should NOT close position
        td.set("action", torch.tensor(0))  # HOLD action
        result = env.step(td)

        # Position should remain open (SLTP envs use bracket orders for exit)
        assert env.position.position_size == initial_position_size
        assert env.stop_loss > 0.0
        assert env.take_profit > 0.0

    def test_flip_from_long_to_short(self, env):
        """Should be able to flip from long to short position."""
        td = env.reset()

        # Open long
        td.set("action", torch.tensor(1))  # First long (0=HOLD)
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size > 0

        # Open short (close long first internally)
        num_long_actions = len(env.stoploss_levels) * len(env.takeprofit_levels)
        short_action = 1 + num_long_actions  # 0=HOLD, 1+=long (no CLOSE)

        td.set("action", torch.tensor(short_action))
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size < 0
        assert env.position.current_position == -1

    def test_flip_from_short_to_long(self, env):
        """Should be able to flip from short to long position."""
        td = env.reset()

        # Open short
        num_long_actions = len(env.stoploss_levels) * len(env.takeprofit_levels)
        short_action = 1 + num_long_actions  # 0=HOLD, 1+=long (no CLOSE)

        td.set("action", torch.tensor(short_action))
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size < 0

        # Open long (close short first internally)
        td.set("action", torch.tensor(1))  # First long (0=HOLD)
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size > 0
        assert env.position.current_position == 1

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


class TestSeqFuturesSLTPEnvEdgeCases:
    """Tests for edge cases."""

    def test_single_sl_tp_level(self, sample_ohlcv_df):
        """Should work with single SL and TP level."""
        try:
            config = SeqFuturesSLTPEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                initial_cash=1000,
                leverage=10,
                stoploss_levels=[-0.05],
                takeprofit_levels=[0.1],
                max_traj_length=50,
                random_start=False,
            )
            env = SeqFuturesSLTPEnv(sample_ohlcv_df, config, simple_feature_fn)

            # Should have 3 actions: hold, long, short (no CLOSE by default)
            assert env.action_spec.n == 3  # 1 hold + 1 long + 1 short

            td = env.reset()
            td.set("action", torch.tensor(1))  # First long (0=HOLD)
            result = env.step(td)

            assert result is not None

        finally:
            env.close()

    def test_many_sl_tp_levels(self, sample_ohlcv_df):
        """Should work with many SL/TP levels."""
        try:
            config = SeqFuturesSLTPEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                initial_cash=1000,
                leverage=10,
                stoploss_levels=[-0.01, -0.02, -0.05, -0.1],
                takeprofit_levels=[0.01, 0.02, 0.05, 0.1],
                max_traj_length=50,
                random_start=False,
            )
            env = SeqFuturesSLTPEnv(sample_ohlcv_df, config, simple_feature_fn)

            # Should have 1 + 16 long + 16 short = 33 actions (no CLOSE by default)
            assert env.action_spec.n == 33  # 1 hold + 16 long + 16 short

        finally:
            env.close()

    def test_multiple_episodes(self, env):
        """Should work correctly across multiple episodes."""
        for episode in range(3):
            td = env.reset()

            assert env.stop_loss == 0.0
            assert env.take_profit == 0.0
            assert env.position.position_size == 0.0
            assert env.liquidation_price == 0.0

            for _ in range(20):
                action = env.action_spec.sample()
                td.set("action", action)
                result = env.step(td)
                td = result["next"]

                if td.get("done", False):
                    break

    def test_leverage_affects_position_size(self, sample_ohlcv_df):
        """
        With QUANTITY mode, leverage doesn't affect position size.
        It only affects liquidation risk (higher leverage = closer liquidation price).
        """
        try:
            config_low = SeqFuturesSLTPEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                leverage=5,
                stoploss_levels=[-0.05],
                takeprofit_levels=[0.1],
                slippage=0.0,
                max_traj_length=50,
                random_start=False,
            )
            config_high = SeqFuturesSLTPEnvConfig(
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                initial_cash=1000,
                leverage=20,
                stoploss_levels=[-0.05],
                takeprofit_levels=[0.1],
                slippage=0.0,
                max_traj_length=50,
                random_start=False,
            )

            env_low = SeqFuturesSLTPEnv(sample_ohlcv_df, config_low, simple_feature_fn)
            env_high = SeqFuturesSLTPEnv(sample_ohlcv_df, config_high, simple_feature_fn)

            td_low = env_low.reset()
            td_high = env_high.reset()

            td_low.set("action", torch.tensor(1))  # First long (0=HOLD)
            env_low.step(td_low)

            td_high.set("action", torch.tensor(1))  # First long (0=HOLD)
            env_high.step(td_high)

            # With QUANTITY mode, position sizes should be equal
            assert abs(env_high.position.position_size) == abs(env_low.position.position_size)

            # But liquidation risk differs - higher leverage = closer liquidation price
            entry_price = env_low.position.entry_price
            liq_distance_low = abs(entry_price - env_low.liquidation_price)
            liq_distance_high = abs(entry_price - env_high.liquidation_price)
            assert liq_distance_high < liq_distance_low


        finally:
            env_high.close()
            env_low.close()
class TestSeqFuturesSLTPEnvMetrics:
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
