"""Tests for action_types history tracking across all environments."""

import pytest
import torch
import pandas as pd
from torchtrade.envs.offline.longonly.sequential import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.offline.futures.sequential import SeqFuturesEnv, SeqFuturesEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


def simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature processing function for testing."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.fillna(0, inplace=True)
    return df


class TestSeqLongOnlyEnvHistory:
    """Test action_types tracking for SeqLongOnlyEnv."""

    def test_action_types_recorded(self, sample_ohlcv_df):
        """Test that action_types are recorded in history."""
        config = SeqLongOnlyEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            max_traj_length=100,
            random_start=False,
        )
        env = SeqLongOnlyEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()

        # Execute some steps
        for _ in range(10):
            action = torch.randint(0, len(env.action_levels), (1,))
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Check history
        history = env.history.to_dict()
        assert "action_types" in history, "action_types should be in history"
        assert len(history["action_types"]) > 0, "Should have action_types recorded"

        # Check that action_types are valid
        valid_types = {"buy", "sell", "hold"}
        for atype in history["action_types"]:
            assert atype in valid_types, f"Invalid action_type: {atype}"

        env.close()

    def test_binarized_actions(self, sample_ohlcv_df):
        """Test that actions are binarized correctly."""
        config = SeqLongOnlyEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            max_traj_length=100,
            random_start=False,
        )
        env = SeqLongOnlyEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()

        # Execute some steps
        for _ in range(10):
            action = torch.randint(0, len(env.action_levels), (1,))
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


class TestSeqFuturesEnvHistory:
    """Test action_types tracking for SeqFuturesEnv."""

    def test_action_types_recorded(self, sample_ohlcv_df):
        """Test that action_types are recorded in history."""
        config = SeqFuturesEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,
            max_traj_length=100,
            random_start=False,
        )
        env = SeqFuturesEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()

        # Execute some steps
        for _ in range(10):
            action = torch.randint(0, len(env.action_levels), (1,))
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Check history
        history = env.history.to_dict()
        assert "action_types" in history, "action_types should be in history"
        assert len(history["action_types"]) > 0, "Should have action_types recorded"

        # Check that action_types are valid
        valid_types = {"long", "short", "hold", "flat", "liquidation", "close_partial"}
        for atype in history["action_types"]:
            assert atype in valid_types, f"Invalid action_type: {atype}"

        env.close()


class TestHistoryReset:
    """Test that history resets properly."""

    def test_history_reset_clears_action_types(self, sample_ohlcv_df):
        """Test that history.reset() clears action_types."""
        config = SeqLongOnlyEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            max_traj_length=100,
            random_start=False,
        )
        env = SeqLongOnlyEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()

        # Execute some steps
        for _ in range(5):
            action = torch.randint(0, len(env.action_levels), (1,))
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Reset environment
        env.reset()

        # Check that history is cleared
        history = env.history.to_dict()
        assert len(history["action_types"]) == 0, "action_types should be cleared after reset"

        env.close()
