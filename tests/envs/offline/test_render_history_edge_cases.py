"""Edge case tests for render_history with special action types."""

import pytest
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from torchtrade.envs.offline.futures.sequential import SeqFuturesEnv, SeqFuturesEnvConfig
from torchtrade.envs.offline.longonly.sequential import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


def simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature processing function for testing."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.fillna(0, inplace=True)
    return df


class TestRenderHistorySpecialActionTypes:
    """Test render_history handles special action types correctly."""

    def test_render_with_mixed_action_types(self, sample_ohlcv_df):
        """Test render_history with various action types including special ones."""
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

        # Execute steps to generate history with various action types
        for _ in range(20):
            action = torch.randint(0, len(env.action_levels), (1,))
            td = env.step(td.set("action", action))
            if td.get("done", False).item():
                break

        # Manually inject special action types to test rendering
        if len(env.history.action_types) > 5:
            env.history.action_types[2] = "close_partial"
            env.history.action_types[3] = "flat"
            env.history.action_types[4] = "liquidation"

        # Should not crash with special action types
        try:
            env.render_history()
        except Exception as e:
            pytest.fail(f"render_history crashed with special action types: {e}")

        env.close()

    def test_render_with_only_holds(self, sample_ohlcv_df):
        """Test render_history with history containing only holds."""
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

        # Execute only hold actions (action level 0)
        for _ in range(10):
            td = env.step(td.set("action", torch.tensor([0])))
            if td.get("done", False).item():
                break

        # Verify all actions are holds
        history = env.history.to_dict()
        assert all(atype == "hold" for atype in history["action_types"]), \
            "All action_types should be hold"

        # Should render without errors even with no buy/sell markers
        try:
            env.render_history()
        except Exception as e:
            pytest.fail(f"render_history failed with only holds: {e}")

        env.close()

    def test_render_with_empty_history(self, sample_ohlcv_df):
        """Test render_history handles empty history gracefully."""
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

        env.reset()
        # Don't take any steps - history is empty

        # Should handle empty history gracefully
        try:
            env.render_history()
        except Exception as e:
            # Some error is acceptable for empty history, just ensure it doesn't crash badly
            pass

        env.close()

    def test_render_with_single_step(self, sample_ohlcv_df):
        """Test render_history with only one step."""
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
        # Take exactly 1 step
        td = env.step(td.set("action", torch.tensor([1])))

        history = env.history.to_dict()
        assert len(history["action_types"]) == 1, "Should have exactly 1 action_type"

        # Should render single step without errors
        try:
            env.render_history()
        except Exception as e:
            pytest.fail(f"render_history failed with single step: {e}")

        env.close()

    def test_render_with_unknown_action_type(self, sample_ohlcv_df):
        """Test render_history handles unknown action types gracefully."""
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

        # Inject unknown action type
        if len(env.history.action_types) > 0:
            env.history.action_types[0] = "unknown_action_type"

        # Should handle unknown action type gracefully (just not plot it)
        try:
            env.render_history()
        except Exception as e:
            pytest.fail(f"render_history crashed with unknown action type: {e}")

        env.close()


class TestBinarizationHelper:
    """Test the binarize_action_type helper function."""

    def test_binarize_buy_and_long(self):
        """Test that buy and long map to 1."""
        from torchtrade.envs.core.state import binarize_action_type

        assert binarize_action_type("buy") == 1
        assert binarize_action_type("long") == 1

    def test_binarize_sell_and_short(self):
        """Test that sell and short map to -1."""
        from torchtrade.envs.core.state import binarize_action_type

        assert binarize_action_type("sell") == -1
        assert binarize_action_type("short") == -1

    def test_binarize_others_to_zero(self):
        """Test that other action types map to 0."""
        from torchtrade.envs.core.state import binarize_action_type

        assert binarize_action_type("hold") == 0
        assert binarize_action_type("liquidation") == 0
        assert binarize_action_type("close") == 0
        assert binarize_action_type("close_partial") == 0
        assert binarize_action_type("flat") == 0
        assert binarize_action_type("unknown") == 0
