"""Comprehensive action_types history tracking tests for all environment variants."""

import pytest
import torch
import pandas as pd
from torchtrade.envs.offline.longonly.sequential_sltp import SeqLongOnlySLTPEnv, SeqLongOnlySLTPEnvConfig
from torchtrade.envs.offline.futures.sequential_sltp import SeqFuturesSLTPEnv, SeqFuturesSLTPEnvConfig
from torchtrade.envs.offline.longonly.onestep import LongOnlyOneStepEnv, LongOnlyOneStepEnvConfig
from torchtrade.envs.offline.futures.onestep import FuturesOneStepEnv, FuturesOneStepEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


def simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature processing function for testing."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.fillna(0, inplace=True)
    return df


class TestSeqLongOnlySLTPEnvHistory:
    """Test action_types tracking for SeqLongOnlySLTPEnv."""

    def test_action_types_recorded(self, sample_ohlcv_df):
        """Test that action_types are recorded in history for SLTP variant."""
        config = SeqLongOnlySLTPEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            max_traj_length=100,
            random_start=False,
            stoploss_levels=[-0.05, -0.1],
            takeprofit_levels=[0.05, 0.1],
        )
        env = SeqLongOnlySLTPEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()

        # Execute some steps
        for _ in range(10):
            action = torch.randint(0, env.action_spec.space.n, (1,))
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

        # Verify history arrays are aligned
        assert len(history["action_types"]) == len(history["actions"]), \
            "action_types and actions should have same length"
        assert len(history["action_types"]) == len(history["base_prices"]), \
            "action_types and base_prices should have same length"

        env.close()


class TestSeqFuturesSLTPEnvHistory:
    """Test action_types tracking for SeqFuturesSLTPEnv."""

    def test_action_types_recorded(self, sample_ohlcv_df):
        """Test that action_types are recorded in history for futures SLTP variant."""
        config = SeqFuturesSLTPEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,
            max_traj_length=100,
            random_start=False,
            stoploss_levels=[-0.05, -0.1],
            takeprofit_levels=[0.05, 0.1],
        )
        env = SeqFuturesSLTPEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()

        # Execute some steps
        for _ in range(10):
            action = torch.randint(0, env.action_spec.space.n, (1,))
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Check history
        history = env.history.to_dict()
        assert "action_types" in history, "action_types should be in history"
        assert len(history["action_types"]) > 0, "Should have action_types recorded"

        # Check that action_types are valid (including special types)
        valid_types = {"long", "short", "hold", "flat", "liquidation", "close_partial", "close"}
        for atype in history["action_types"]:
            assert atype in valid_types, f"Invalid action_type: {atype}"

        # Verify history arrays are aligned
        assert len(history["action_types"]) == len(history["actions"]), \
            "action_types and actions should have same length"
        assert len(history["action_types"]) == len(history["base_prices"]), \
            "action_types and base_prices should have same length"

        env.close()


class TestLongOnlyOneStepEnvHistory:
    """Test action_types tracking for LongOnlyOneStepEnv."""

    def test_action_types_recorded(self, sample_ohlcv_df):
        """Test that action_types are recorded in history for OneStep variant."""
        config = LongOnlyOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            max_traj_length=100,
            random_start=False,
            stoploss_levels=[-0.05, -0.1],
            takeprofit_levels=[0.05, 0.1],
        )
        env = LongOnlyOneStepEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()
        action = torch.randint(0, env.action_spec.space.n, (1,))
        td = env.step(td.set("action", action))

        # Check history (OneStep envs complete in 1 step)
        history = env.history.to_dict()
        assert "action_types" in history, "action_types should be in history"
        assert len(history["action_types"]) > 0, "Should have action_types recorded"

        # Check that action_types are valid
        valid_types = {"buy", "sell", "hold"}
        for atype in history["action_types"]:
            assert atype in valid_types, f"Invalid action_type: {atype}"

        # Verify history arrays are aligned
        assert len(history["action_types"]) == len(history["actions"]), \
            "action_types and actions should have same length"

        env.close()


class TestFuturesOneStepEnvHistory:
    """Test action_types tracking for FuturesOneStepEnv."""

    def test_action_types_recorded(self, sample_ohlcv_df):
        """Test that action_types are recorded in history for futures OneStep variant."""
        config = FuturesOneStepEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,
            max_traj_length=100,
            random_start=False,
            stoploss_levels=[-0.05, -0.1],
            takeprofit_levels=[0.05, 0.1],
        )
        env = FuturesOneStepEnv(
            df=sample_ohlcv_df,
            config=config,
            feature_preprocessing_fn=simple_feature_fn,
        )

        td = env.reset()
        action = torch.randint(0, env.action_spec.space.n, (1,))
        td = env.step(td.set("action", action))

        # Check history (OneStep envs complete in 1 step)
        history = env.history.to_dict()
        assert "action_types" in history, "action_types should be in history"
        assert len(history["action_types"]) > 0, "Should have action_types recorded"

        # Check that action_types are valid (including special types)
        valid_types = {"long", "short", "hold", "liquidation"}
        for atype in history["action_types"]:
            assert atype in valid_types, f"Invalid action_type: {atype}"

        # Verify history arrays are aligned
        assert len(history["action_types"]) == len(history["actions"]), \
            "action_types and actions should have same length"

        env.close()


class TestHistoryArrayConsistency:
    """Test that history arrays maintain consistency."""

    def test_no_none_values_in_action_types(self, sample_ohlcv_df):
        """Verify no None values appear in action_types list."""
        from torchtrade.envs.offline.longonly.sequential import SeqLongOnlyEnv, SeqLongOnlyEnvConfig

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

        # Execute episode (may hit bankruptcy with random actions)
        try:
            for _ in range(20):
                action = torch.randint(0, len(env.action_levels), (1,))
                td = env.step(td.set("action", action))
                if td.get("done", False).item():
                    break
        except ValueError as e:
            if "bankruptcy" in str(e):
                # Expected with random actions - reset and try again with conservative actions
                td = env.reset()
                for _ in range(20):
                    action = torch.tensor([0])  # Hold action
                    td = env.step(td.set("action", action))
                    if td.get("done", False).item():
                        break
            else:
                raise

        # Check for None values
        history = env.history.to_dict()
        assert None not in history["action_types"], "action_types should not contain None"
        assert all(isinstance(atype, str) for atype in history["action_types"]), \
            "All action_types should be strings"

        env.close()

    def test_array_lengths_match(self, sample_ohlcv_df):
        """Verify all history arrays have matching lengths."""
        from torchtrade.envs.offline.futures.sequential import SeqFuturesEnv, SeqFuturesEnvConfig

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

        # Execute episode (may hit bankruptcy with random actions)
        try:
            for _ in range(20):
                action = torch.randint(0, len(env.action_levels), (1,))
                td = env.step(td.set("action", action))
                if td.get("done", False).item():
                    break
        except ValueError as e:
            if "bankruptcy" in str(e):
                # Expected with random actions - reset and try again with conservative actions
                td = env.reset()
                for _ in range(20):
                    action = torch.tensor([0])  # Hold action
                    td = env.step(td.set("action", action))
                    if td.get("done", False).item():
                        break
            else:
                raise

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
