"""Tests for render_history with action_types and backward compatibility."""

import pytest
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from torchtrade.envs.offline.sequential import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


def simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature processing function for testing."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.fillna(0, inplace=True)
    return df


class TestRenderHistoryLongOnly:
    """Test render_history for long-only environments."""

    def test_render_with_action_types(self, sample_ohlcv_df):
        """Test that render_history works with action_types."""
        config = SequentialTradingEnvConfig(
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

        # Execute some steps (handle potential bankruptcy from random actions)
        try:
            for _ in range(20):
                action = torch.randint(0, len(env.action_levels), (1,))
                td = env.step(td.set("action", action))
                if td.get("done", False).item():
                    break
        except ValueError as e:
            if "Invalid new_portfolio_value: 0.0" not in str(e):
                raise
            # Bankruptcy is fine for this test, we just need some history

        # This should not raise an error
        try:
            env.render_history()
        except Exception as e:
            pytest.fail(f"render_history raised an exception: {e}")

        env.close()

    def test_render_with_legacy_history(self, sample_ohlcv_df):
        """Test backward compatibility with history without action_types."""
        config = SequentialTradingEnvConfig(
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

        # Execute some steps (handle potential bankruptcy from random actions)
        try:
            for _ in range(20):
                action = torch.randint(0, len(env.action_levels), (1,))
                td = env.step(td.set("action", action))
                if td.get("done", False).item():
                    break
        except ValueError as e:
            if "Invalid new_portfolio_value: 0.0" not in str(e):
                raise
            # Bankruptcy is fine for this test, we just need some history

        # Manually clear action_types to simulate legacy history
        env.history.action_types.clear()

        # This should still work (fallback to action values)
        try:
            env.render_history()
        except Exception as e:
            pytest.fail(f"render_history with legacy history raised an exception: {e}")

        env.close()


class TestRenderHistoryFutures:
    """Test render_history for futures environments."""

    def test_render_with_action_types(self, sample_ohlcv_df):
        """Test that render_history works with action_types for futures."""
        config = SequentialTradingEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,
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
        for _ in range(20):
            action = torch.randint(0, len(env.action_levels), (1,))
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # This should not raise an error
        try:
            env.render_history()
        except Exception as e:
            pytest.fail(f"render_history raised an exception: {e}")

        env.close()

    def test_render_with_legacy_history(self, sample_ohlcv_df):
        """Test backward compatibility with history without action_types."""
        config = SequentialTradingEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            leverage=10,
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
        for _ in range(20):
            action = torch.randint(0, len(env.action_levels), (1,))
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Manually clear action_types to simulate legacy history
        env.history.action_types.clear()

        # This should still work (fallback to action values)
        try:
            env.render_history()
        except Exception as e:
            pytest.fail(f"render_history with legacy history raised an exception: {e}")

        env.close()
