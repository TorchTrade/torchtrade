"""
Comprehensive history tracking tests for TorchTrade environments.

This test suite follows TorchRL best practices:
- Uses env.action_spec.rand() for proper action sampling
- Validates environment specs
- Tests across different devices
- Consolidates all history tracking tests in one file
"""

import pytest
import torch
import pandas as pd

from torchtrade.envs.offline import (
    SequentialTradingEnv,
    SequentialTradingEnvConfig,
    SequentialTradingEnvSLTP,
    SequentialTradingEnvSLTPConfig,
    OneStepTradingEnv,
    OneStepTradingEnvConfig,
)
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

# Import TorchRL testing utilities
try:
    from torchrl.testing import get_default_devices
    from torchrl.envs.utils import check_env_specs
    HAS_TORCHRL_TESTING = True
except ImportError:
    HAS_TORCHRL_TESTING = False

    def get_default_devices():
        """Fallback if torchrl.testing not available."""
        return [torch.device("cpu")]


def simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature processing function for testing."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.fillna(0, inplace=True)
    return df


class TestSequentialTradingEnvHistory:
    """Test history tracking for SequentialTradingEnv (spot-like, long-only)."""

    @pytest.mark.parametrize("device", get_default_devices())
    def test_action_types_recorded(self, sample_ohlcv_df, device):
        """Test that action_types are recorded in history using env.action_spec.rand()."""
        config = SequentialTradingEnvConfig(
            leverage=1,  # Spot-like (long-only)
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            max_traj_length=100,
            random_start=False,
        )
        env = SequentialTradingEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        # Validate environment specs
        if HAS_TORCHRL_TESTING:
            check_env_specs(env)

        td = env.reset()

        # Execute some steps using action_spec.rand() - TorchRL best practice
        for _ in range(10):
            # Correct: Use action_spec.rand() for proper spec-driven sampling
            action = env.action_spec.rand()
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Check history
        history = env.history.to_dict()
        assert "action_types" in history, "action_types should be in history"
        assert len(history["action_types"]) > 0, "Should have action_types recorded"

        # Check that action_types are recorded as non-empty strings
        for atype in history["action_types"]:
            assert isinstance(atype, str) and len(atype) > 0, f"Invalid action_type: {atype}"

        env.close()

    @pytest.mark.parametrize("device", get_default_devices())
    def test_binarized_actions(self, sample_ohlcv_df, device):
        """Test that actions are binarized correctly (-1, 0, 1)."""
        config = SequentialTradingEnvConfig(
            leverage=1,  # Spot-like (long-only)
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            max_traj_length=100,
            random_start=False,
        )
        env = SequentialTradingEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()

        # Execute some steps
        for _ in range(10):
            action = env.action_spec.rand()
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Check that actions are binarized (-1, 0, 1)
        history = env.history.to_dict()
        for action, atype in zip(history["actions"], history["action_types"]):
            if atype == "buy":
                assert action == 1, "Buy action should be 1"
            elif atype == "sell":
                assert action == -1, "Sell action should be -1"
            elif atype == "hold":
                assert action == 0, "Hold action should be 0"

        env.close()

    def test_rollout_history(self, sample_ohlcv_df):
        """Test history tracking with env.rollout() - TorchRL pattern."""
        config = SequentialTradingEnvConfig(
            leverage=1,
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            max_traj_length=20,  # Short for quick test
            random_start=False,
        )
        env = SequentialTradingEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        # Use rollout with random policy (TorchRL pattern)
        rollout = env.rollout(max_steps=10)

        # Verify rollout shape
        assert rollout.shape[0] <= 10, "Rollout should have max 10 steps"

        # Check history
        history = env.history.to_dict()
        assert "action_types" in history
        assert len(history["action_types"]) > 0

        env.close()


class TestSequentialTradingEnvHistoryFutures:
    """Test history tracking for SequentialTradingEnv (futures mode with leverage)."""

    @pytest.mark.parametrize("device", get_default_devices())
    def test_action_types_recorded(self, sample_ohlcv_df, device):
        """Test that action_types are recorded in history for futures trading."""
        config = SequentialTradingEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,  # Futures mode
            max_traj_length=100,
            random_start=False,
        )
        env = SequentialTradingEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        # Validate specs
        if HAS_TORCHRL_TESTING:
            check_env_specs(env)

        td = env.reset()

        # Execute some steps with spec-driven actions
        for _ in range(10):
            action = env.action_spec.rand()
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Check history
        history = env.history.to_dict()
        assert "action_types" in history, "action_types should be in history"
        assert len(history["action_types"]) > 0, "Should have action_types recorded"

        # Check that action_types are valid for futures trading
        valid_types = {"long", "short", "hold", "flat", "liquidation", "close_partial"}
        for atype in history["action_types"]:
            assert atype in valid_types, f"Invalid action_type: {atype}"

        env.close()


class TestSequentialTradingEnvSLTPHistory:
    """Test history tracking for SequentialTradingEnvSLTP (with stop-loss/take-profit)."""

    @pytest.mark.parametrize("device", get_default_devices())
    def test_action_types_recorded(self, sample_ohlcv_df, device):
        """Test that action_types are recorded for SLTP variant."""
        config = SequentialTradingEnvSLTPConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=1,  # Spot-like
            max_traj_length=100,
            random_start=False,
            stoploss_levels=[-0.05, -0.1],
            takeprofit_levels=[0.05, 0.1],
        )
        env = SequentialTradingEnvSLTP(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        # Validate specs
        if HAS_TORCHRL_TESTING:
            check_env_specs(env)

        td = env.reset()

        # Execute some steps using action_spec.rand()
        for _ in range(10):
            action = env.action_spec.rand()
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Check history
        history = env.history.to_dict()
        assert "action_types" in history, "action_types should be in history"
        assert len(history["action_types"]) > 0, "Should have action_types recorded"

        # Check that action_types are recorded as non-empty strings (including SLTP types)
        for atype in history["action_types"]:
            assert isinstance(atype, str) and len(atype) > 0, f"Invalid action_type: {atype}"

        # Verify history arrays are aligned
        assert len(history["action_types"]) == len(history["actions"]), \
            "action_types and actions should have same length"
        assert len(history["action_types"]) == len(history["base_prices"]), \
            "action_types and base_prices should have same length"

        env.close()

    @pytest.mark.parametrize("device", get_default_devices())
    def test_futures_sltp_action_types(self, sample_ohlcv_df, device):
        """Test action_types for futures SLTP variant."""
        config = SequentialTradingEnvSLTPConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,  # Futures mode
            max_traj_length=100,
            random_start=False,
            stoploss_levels=[-0.05, -0.1],
            takeprofit_levels=[0.05, 0.1],
        )
        env = SequentialTradingEnvSLTP(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()

        # Execute some steps
        for _ in range(10):
            action = env.action_spec.rand()
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Check history
        history = env.history.to_dict()
        assert "action_types" in history

        # Check that action_types include futures and SLTP types
        valid_types = {
            "long", "short", "hold", "flat", "liquidation",
            "close_partial", "close", "sltp_sl", "sltp_tp"
        }
        for atype in history["action_types"]:
            assert atype in valid_types, f"Invalid action_type: {atype}"

        # Verify history arrays are aligned
        assert len(history["action_types"]) == len(history["actions"])
        assert len(history["action_types"]) == len(history["base_prices"])

        env.close()


class TestOneStepTradingEnvHistory:
    """Test history tracking for OneStepTradingEnv (GRPO/bandit setting)."""

    @pytest.mark.parametrize("device", get_default_devices())
    def test_action_types_recorded(self, sample_ohlcv_df, device):
        """Test that action_types are recorded for OneStep variant."""
        config = OneStepTradingEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=1,  # Spot-like
            max_traj_length=100,
            random_start=False,
            stoploss_levels=[-0.05, -0.1],
            takeprofit_levels=[0.05, 0.1],
        )
        env = OneStepTradingEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        # Validate specs
        if HAS_TORCHRL_TESTING:
            check_env_specs(env)

        td = env.reset()

        # OneStep env completes in one step
        action = env.action_spec.rand()
        td = env.step(td.set("action", action))

        # Check history (OneStep envs complete in 1 step)
        history = env.history.to_dict()
        assert "action_types" in history, "action_types should be in history"
        assert len(history["action_types"]) > 0, "Should have action_types recorded"

        # Check that action_types are recorded as non-empty strings
        for atype in history["action_types"]:
            assert isinstance(atype, str) and len(atype) > 0, f"Invalid action_type: {atype}"

        # Verify history arrays are aligned
        assert len(history["action_types"]) == len(history["actions"]), \
            "action_types and actions should have same length"

        env.close()

    @pytest.mark.parametrize("device", get_default_devices())
    def test_futures_onestep_action_types(self, sample_ohlcv_df, device):
        """Test action_types for futures OneStep variant."""
        config = OneStepTradingEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,  # Futures mode
            max_traj_length=100,
            random_start=False,
            stoploss_levels=[-0.05, -0.1],
            takeprofit_levels=[0.05, 0.1],
        )
        env = OneStepTradingEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()
        action = env.action_spec.rand()
        td = env.step(td.set("action", action))

        # Check history
        history = env.history.to_dict()
        assert "action_types" in history

        # Check that action_types are valid (including special types)
        valid_types = {"long", "short", "hold", "liquidation"}
        for atype in history["action_types"]:
            assert atype in valid_types, f"Invalid action_type: {atype}"

        # Verify history arrays are aligned
        assert len(history["action_types"]) == len(history["actions"])

        env.close()


class TestHistoryReset:
    """Test that history resets properly across episodes."""

    def test_history_reset_clears_action_types(self, sample_ohlcv_df):
        """Test that history.reset() clears action_types."""
        config = SequentialTradingEnvConfig(
            leverage=1,  # Spot-like
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            max_traj_length=100,
            random_start=False,
        )
        env = SequentialTradingEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()

        # Execute some steps
        for _ in range(5):
            action = env.action_spec.rand()
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Verify history has data
        history = env.history.to_dict()
        assert len(history["action_types"]) > 0

        # Reset environment
        env.reset()

        # Check that history is cleared (except for initial state)
        history = env.history.to_dict()
        assert len(history["action_types"]) == 1, \
            "action_types should have only initial state after reset"
        assert history["action_types"][0] == "hold", \
            "Initial action_type should be 'hold'"

        env.close()

    def test_history_persists_within_episode(self, sample_ohlcv_df):
        """Test that history accumulates within a single episode."""
        config = SequentialTradingEnvConfig(
            leverage=1,
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            max_traj_length=100,
            random_start=False,
        )
        env = SequentialTradingEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()

        # Execute steps and track history growth
        history_lengths = []
        for i in range(5):
            action = env.action_spec.rand()
            td = env.step(td.set("action", action))

            history = env.history.to_dict()
            history_lengths.append(len(history["action_types"]))

            if td.get("done", False).item():
                break

        # Verify history accumulated
        assert history_lengths[-1] >= history_lengths[0], \
            "History should accumulate within episode"

        env.close()


class TestHistoryArrayConsistency:
    """Test that history arrays maintain consistency and data integrity."""

    def test_no_none_values_in_action_types(self, sample_ohlcv_df):
        """Verify no None values appear in action_types list."""
        config = SequentialTradingEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=1,
            max_traj_length=100,
            random_start=False,
        )
        env = SequentialTradingEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()

        # Execute episode (use conservative actions to avoid bankruptcy)
        for _ in range(20):
            # Hold action (index 0) - conservative
            action = torch.zeros(1, dtype=torch.long)
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Check for None values
        history = env.history.to_dict()
        assert None not in history["action_types"], \
            "action_types should not contain None"
        assert all(isinstance(atype, str) for atype in history["action_types"]), \
            "All action_types should be strings"

        env.close()

    def test_array_lengths_match(self, sample_ohlcv_df):
        """Verify all history arrays have matching lengths."""
        config = SequentialTradingEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,  # Futures mode
            max_traj_length=100,
            random_start=False,
        )
        env = SequentialTradingEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()

        # Execute episode
        for _ in range(20):
            action = env.action_spec.rand()
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Verify all arrays have same length
        history = env.history.to_dict()
        expected_length = len(history["base_prices"])

        assert len(history["actions"]) == expected_length, \
            f"actions length {len(history['actions'])} != base_prices length {expected_length}"
        assert len(history["rewards"]) == expected_length, \
            f"rewards length {len(history['rewards'])} != base_prices length {expected_length}"
        assert len(history["portfolio_values"]) == expected_length, \
            f"portfolio_values length {len(history['portfolio_values'])} != base_prices length {expected_length}"
        assert len(history["action_types"]) == expected_length, \
            f"action_types length {len(history['action_types'])} != base_prices length {expected_length}"
        assert len(history["positions"]) == expected_length, \
            f"positions length {len(history['positions'])} != base_prices length {expected_length}"

        env.close()

    def test_empty_history_after_reset(self, sample_ohlcv_df):
        """Test that history contains only initial state after reset."""
        config = SequentialTradingEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=1,
            max_traj_length=100,
            random_start=False,
        )
        env = SequentialTradingEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        env.reset()

        # History should contain only initial state right after reset
        history = env.history.to_dict()
        assert len(history["action_types"]) == 1, \
            "History should contain initial state after reset"
        assert history["action_types"][0] == "hold", "Initial action should be hold"
        assert len(history["actions"]) == 1
        assert len(history["rewards"]) == 1
        assert history["rewards"][0] == 0.0, "Initial reward should be 0"

        env.close()
