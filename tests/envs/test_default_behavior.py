"""
Tests for default environment behavior (fractional position sizing + CLOSE action).

This test file validates that the new default behavior works correctly.
All environments now default to:
- include_close_action=True (for SLTP environments)

These tests ensure the default configurations work end-to-end without errors
and produce reasonable trading behavior.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline.seqfutures import SeqFuturesEnv, SeqFuturesEnvConfig
from torchtrade.envs.offline.seqfuturessltp import SeqFuturesSLTPEnv, SeqFuturesSLTPEnvConfig
from torchtrade.envs.offline.seqlongonly import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.offline.seqlongonlysltp import SeqLongOnlySLTPEnv, SeqLongOnlySLTPEnvConfig
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit


def simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature processing function for testing."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.fillna(0, inplace=True)
    return df


class TestDefaultSeqFuturesEnv:
    """Test SeqFuturesEnv with default configuration (fractional mode)."""

    def test_default_config_initializes(self, sample_ohlcv_df):
        """Environment should initialize with all default settings."""
        config = SeqFuturesEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            leverage=10,
            seed=42,
            max_traj_length=50,
            random_start=False,
        )
        # position_sizing_mode and include_close_action not specified = use defaults
        env = SeqFuturesEnv(sample_ohlcv_df, config, simple_feature_fn)

        assert env is not None
        # Default fractional levels: [-1.0, -0.5, 0.0, 0.5, 1.0]
        assert env.action_spec.n == 5

    def test_default_handles_full_episode(self, sample_ohlcv_df):
        """Default config should handle a full episode without errors."""
        config = SeqFuturesEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            leverage=5,  # Lower leverage to reduce bankruptcy risk
            transaction_fee=0.0004,
            slippage=0.0,
            seed=42,
            max_traj_length=30,
            random_start=False,
        )
        env = SeqFuturesEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()
        steps = 0

        # Use a simple strategy: cycle through actions
        actions = [2, 3, 2, 4, 2, 1, 2, 3]  # Mix of positions

        for i in range(env.max_traj_length):
            action = torch.tensor(actions[i % len(actions)])
            td.set("action", action)
            result = env.step(td)
            td = result["next"]
            steps += 1

            if td.get("done", False):
                break

        assert steps > 0, "Should complete at least one step"
        assert env.balance > 0, "Should not go bankrupt with conservative actions"

    def test_default_fractional_position_sizing_works(self, sample_ohlcv_df):
        """Fractional actions should create appropriate position sizes."""
        config = SeqFuturesEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            leverage=10,
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            max_traj_length=50,
            random_start=False,
        )
        env = SeqFuturesEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()
        initial_balance = env.balance

        # Action 4 = 1.0 (100% long)
        td.set("action", torch.tensor(4))
        result = env.step(td)
        position_size_100 = env.position.position_size

        # Should have opened a position
        assert position_size_100 > 0, "100% action should open position"

        # Reset and try 50% position
        td = env.reset()
        assert env.balance == initial_balance, "Reset should restore balance"

        # Action 3 = 0.5 (50% long)
        td.set("action", torch.tensor(3))
        result = env.step(td)
        position_size_50 = env.position.position_size

        assert position_size_50 > 0, "50% action should open position"
        # 50% should be roughly half of 100%
        assert 0.4 < (position_size_50 / position_size_100) < 0.6, \
            f"50% position should be ~half of 100%: {position_size_50} vs {position_size_100}"

    def test_default_close_action_works(self, sample_ohlcv_df):
        """CLOSE action should exit positions in fractional mode."""
        config = SeqFuturesEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            leverage=10,
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            max_traj_length=50,
            random_start=False,
        )
        env = SeqFuturesEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()

        # Open long position (action 4 = 1.0)
        td.set("action", torch.tensor(4))
        result = env.step(td)
        assert env.position.position_size > 0, "Should have position"

        td = result["next"]

        # Close position (action 2 = 0.0)
        td.set("action", torch.tensor(2))
        result = env.step(td)

        # Position should be closed
        assert abs(env.position.position_size) < 0.01, \
            f"CLOSE should exit position, got {env.position.position_size}"
        assert env.position.current_position == 0, "Position state should be neutral"

    def test_default_position_adjustments_work(self, sample_ohlcv_df):
        """Should be able to adjust position sizes in fractional mode."""
        config = SeqFuturesEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            leverage=10,
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            max_traj_length=50,
            random_start=False,
        )
        env = SeqFuturesEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()

        # Start with 50% long (action 3 = 0.5)
        td.set("action", torch.tensor(3))
        result = env.step(td)
        position_50 = env.position.position_size
        assert position_50 > 0

        td = result["next"]

        # Increase to 100% long (action 4 = 1.0)
        td.set("action", torch.tensor(4))
        result = env.step(td)
        position_100 = env.position.position_size

        # Should have increased position
        assert position_100 > position_50, \
            f"Position should increase: {position_50} -> {position_100}"
        assert 0.8 < (position_100 / position_50) < 2.5, \
            "Doubling action should roughly double position"


class TestDefaultSeqLongOnlyEnv:
    """Test SeqLongOnlyEnv with default configuration (fractional mode)."""

    def test_default_config_uses_fractional_mode(self, sample_ohlcv_df):
        """LongOnly environment should use fractional mode by default."""
        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            seed=42,
            max_traj_length=50,
            random_start=False,
        )
        env = SeqLongOnlyEnv(sample_ohlcv_df, config, simple_feature_fn)

        # Default long-only fractional levels: [0.0, 0.5, 1.0]
        # No negative values (no redundant sell actions)
        assert env.action_spec.n == 3

    def test_default_handles_long_only_positions(self, sample_ohlcv_df):
        """Fractional mode should handle long-only positions correctly."""
        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            transaction_fee=0.001,
            slippage=0.0,
            seed=42,
            max_traj_length=30,
            random_start=False,
        )
        env = SeqLongOnlyEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()

        # Action 2 = 1.0 (100% long)
        td.set("action", torch.tensor(2))
        result = env.step(td)
        position_100 = env.position.position_size

        assert position_100 > 0, "100% action should open long position"
        assert env.position.current_position > 0, "Should be long"

        td = result["next"]

        # Action 0 = 0.0 (exit) - should close position
        td.set("action", torch.tensor(0))
        result = env.step(td)

        assert abs(env.position.position_size) < 0.01, f"Should close position, got {env.position.position_size}"

        td = result["next"]

        # Action 1 = 0.5 (50% long) - should open smaller position
        td.set("action", torch.tensor(1))
        result = env.step(td)
        position_50 = env.position.position_size

        assert position_50 > 0, "50% action should open position"
        # 50% should be smaller than 100% (approximately half with some tolerance for fees/price changes)
        assert position_50 < position_100 * 0.7, \
            f"50% position should be smaller: {position_50} vs {position_100}"


class TestErrorHandlingWithDefaults:
    """Test error handling with default configurations."""

    def test_handles_very_small_balance(self, sample_ohlcv_df):
        """Should handle very small balance without crashing."""
        config = SeqFuturesEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=(1.0, 2.0),  # Fixed small balance (needs tuple with different values)
            leverage=5,
            transaction_fee=0.001,
            slippage=0.0,
            seed=42,
            max_traj_length=10,
            random_start=False,
        )
        env = SeqFuturesEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()

        # Try to open small position
        td.set("action", torch.tensor(3))  # 50% long
        result = env.step(td)

        # Should either open small position or do nothing, but not crash
        assert result is not None
        assert not torch.isnan(result["next"]["reward"]).any()

    def test_handles_position_close_when_flat(self, sample_ohlcv_df):
        """CLOSE action when flat should not cause errors."""
        config = SeqFuturesEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            leverage=10,
            seed=42,
            max_traj_length=50,
            random_start=False,
        )
        env = SeqFuturesEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()

        # Try CLOSE when already flat (action 2 = 0.0)
        td.set("action", torch.tensor(2))
        result = env.step(td)

        # Should do nothing but not error
        assert result is not None
        assert env.position.position_size == 0
        assert not torch.isnan(result["next"]["reward"]).any()

    def test_handles_rapid_position_changes(self, sample_ohlcv_df):
        """Should handle rapid position size changes without errors."""
        config = SeqFuturesEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            initial_cash=1000,
            leverage=10,
            transaction_fee=0.001,
            slippage=0.0,
            seed=42,
            max_traj_length=50,
            random_start=False,
        )
        env = SeqFuturesEnv(sample_ohlcv_df, config, simple_feature_fn)

        td = env.reset()

        # Rapidly change positions: 0 -> 50% -> 100% -> 50% -> 0 -> -50% -> 0
        actions = [2, 3, 4, 3, 2, 1, 2]

        for i, action in enumerate(actions):
            td.set("action", torch.tensor(action))
            result = env.step(td)
            td = result["next"]

            # Should not error
            assert result is not None
            assert not torch.isnan(td["reward"]).any(), f"NaN reward at step {i}"

            if td.get("done", False):
                break


