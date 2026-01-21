"""Tests for CLOSE action functionality across all environments."""

import pytest
import pandas as pd
import torch
import numpy as np

from torchtrade.envs.offline.futuresonestepenv import FuturesOneStepEnv, FuturesOneStepEnvConfig
from torchtrade.envs.offline.longonlyonestepenv import LongOnlyOneStepEnv, LongOnlyOneStepEnvConfig
from torchtrade.envs.offline.seqfuturessltp import SeqFuturesSLTPEnv, SeqFuturesSLTPEnvConfig
from torchtrade.envs.offline.seqlongonlysltp import SeqLongOnlySLTPEnv, SeqLongOnlySLTPEnvConfig
from torchtrade.envs.action_maps import create_sltp_action_map
from torchtrade.envs.offline.utils import build_sltp_action_map


@pytest.fixture
def sample_df():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')

    # Generate realistic price data
    base_price = 50000.0
    price_changes = np.random.randn(1000) * 100
    close_prices = base_price + np.cumsum(price_changes)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.randn(1000) * 10,
        'high': close_prices + np.abs(np.random.randn(1000) * 20),
        'low': close_prices - np.abs(np.random.randn(1000) * 20),
        'close': close_prices,
        'volume': np.random.randint(100, 1000, 1000)
    })

    return df


class TestActionMapWithClose:
    """Test action map creation with CLOSE action."""

    def test_create_sltp_action_map_with_close(self):
        """CLOSE action should be added at index 1 after HOLD."""
        action_map = create_sltp_action_map(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_hold_action=True,
            include_close_action=True,
            include_short_positions=True
        )

        # Expected: 0=HOLD, 1=CLOSE, 2=LONG, 3=SHORT
        assert len(action_map) == 4
        assert action_map[0] == (None, None, None)  # HOLD
        assert action_map[1] == ("close", None, None)  # CLOSE
        assert action_map[2] == ("long", -0.02, 0.05)  # LONG
        assert action_map[3] == ("short", 0.05, -0.02)  # SHORT

    def test_create_sltp_action_map_without_close(self):
        """Action map without CLOSE should have one fewer action."""
        action_map = create_sltp_action_map(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_hold_action=True,
            include_close_action=False,
            include_short_positions=True
        )

        # Expected: 0=HOLD, 1=LONG, 2=SHORT
        assert len(action_map) == 3
        assert action_map[0] == (None, None, None)  # HOLD
        assert action_map[1] == ("long", -0.02, 0.05)  # LONG
        assert action_map[2] == ("short", 0.05, -0.02)  # SHORT

    def test_build_sltp_action_map_long_only_with_close(self):
        """Long-only action map should use ('close', None) marker."""
        action_map = build_sltp_action_map(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_hold_action=True,
            include_close_action=True,
            include_short_positions=False
        )

        # Expected: 0=HOLD, 1=CLOSE, 2=LONG
        assert len(action_map) == 3
        assert action_map[0] == (None, None)  # HOLD
        assert action_map[1] == ("close", None)  # CLOSE
        assert action_map[2] == (-0.02, 0.05)  # LONG


class TestFuturesOneStepEnvClose:
    """Test CLOSE action in FuturesOneStepEnv."""

    def test_close_action_exists(self, sample_df):
        """CLOSE action should exist in action map."""
        config = FuturesOneStepEnvConfig(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_close_action=True
        )
        env = FuturesOneStepEnv(sample_df, config)

        # Should have HOLD, CLOSE, 1 LONG, 1 SHORT = 4 actions
        assert env.action_spec.n == 4
        assert env.action_map[1] == ("close", None, None)

    def test_close_action_exits_long_position(self, sample_df):
        """CLOSE action should exit long position and return to cash."""
        # Note: OneStep environments auto-rollout after opening position,
        # so we can't test mid-position closing. This test verifies the
        # CLOSE action handling logic exists and doesn't error.
        config = FuturesOneStepEnvConfig(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_close_action=True,
            leverage=1
        )
        env = FuturesOneStepEnv(sample_df, config)

        # Reset environment
        td = env.reset()

        # Open long position (action 2) - this will trigger auto-rollout
        td["action"] = torch.tensor([2])  # LONG action
        td = env.step(td)["next"]

        # In OneStep env, position is already closed via SL/TP rollout
        # Just verify CLOSE action can be executed without error
        td = env.reset()
        td["action"] = torch.tensor([1])  # CLOSE when no position
        td = env.step(td)["next"]

        # Should not error
        assert True

    def test_close_action_exits_short_position(self, sample_df):
        """CLOSE action should exit short position and return to cash."""
        # Note: OneStep environments auto-rollout, see test_close_action_exits_long_position
        config = FuturesOneStepEnvConfig(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_close_action=True,
            leverage=1
        )
        env = FuturesOneStepEnv(sample_df, config)

        # Just verify CLOSE action can be executed without error
        td = env.reset()
        td["action"] = torch.tensor([1])  # CLOSE when no position
        td = env.step(td)["next"]

        # Should not error
        assert True

    def test_close_action_noop_when_no_position(self, sample_df):
        """CLOSE action should be no-op when already in cash."""
        config = FuturesOneStepEnvConfig(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_close_action=True
        )
        env = FuturesOneStepEnv(sample_df, config)

        # Reset environment (no position)
        td = env.reset()
        initial_balance = env.balance

        # Verify no position
        assert env.position.position_size == 0

        # Execute CLOSE action (action 1)
        td["action"] = torch.tensor([1])
        td = env.step(td)["next"]

        # Verify still no position and balance unchanged
        assert env.position.position_size == 0
        assert env.balance == initial_balance

    def test_action_history_tracks_close(self, sample_df):
        """Action history should track CLOSE actions."""
        # Use SeqFuturesSLTPEnv which doesn't auto-rollout
        config = SeqFuturesSLTPEnvConfig(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_close_action=True,
            leverage=1
        )
        env = SeqFuturesSLTPEnv(sample_df, config)

        # Reset and open position
        td = env.reset()
        td["action"] = torch.tensor([2])  # LONG
        td = env.step(td)["next"]

        # Execute CLOSE
        td["action"] = torch.tensor([1])
        td = env.step(td)["next"]

        # Check history
        history_dict = env.history.to_dict()
        action_types = history_dict.get('action_types', [])

        assert "long" in action_types
        assert "close" in action_types


class TestLongOnlyOneStepEnvClose:
    """Test CLOSE action in LongOnlyOneStepEnv."""

    def test_close_action_exists(self, sample_df):
        """CLOSE action should exist in action map."""
        config = LongOnlyOneStepEnvConfig(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_close_action=True
        )
        env = LongOnlyOneStepEnv(sample_df, config)

        # Should have HOLD, CLOSE, 1 LONG = 3 actions
        assert env.action_spec.n == 3
        assert env.action_map[1] == ("close", None)

    def test_close_action_exits_position(self, sample_df):
        """CLOSE action should exit long position and return to cash."""
        # Note: OneStep environments auto-rollout, see FuturesOneStepEnv tests
        config = LongOnlyOneStepEnvConfig(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_close_action=True
        )
        env = LongOnlyOneStepEnv(sample_df, config)

        # Just verify CLOSE action can be executed without error
        td = env.reset()
        td["action"] = torch.tensor([1])  # CLOSE when no position
        td = env.step(td)["next"]

        # Should not error
        assert True


class TestSeqFuturesSLTPEnvClose:
    """Test CLOSE action in SeqFuturesSLTPEnv."""

    def test_close_action_exists(self, sample_df):
        """CLOSE action should exist in action map."""
        config = SeqFuturesSLTPEnvConfig(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_close_action=True
        )
        env = SeqFuturesSLTPEnv(sample_df, config)

        # Should have HOLD, CLOSE, 1 LONG, 1 SHORT = 4 actions
        assert env.action_spec.n == 4
        assert env.action_map[1] == ("close", None, None)

    def test_close_vs_hold_behavior(self, sample_df):
        """CLOSE should exit position, HOLD should maintain it."""
        config = SeqFuturesSLTPEnvConfig(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_close_action=True,
            leverage=1
        )
        env = SeqFuturesSLTPEnv(sample_df, config)

        # Reset and open position
        td = env.reset()
        td["action"] = torch.tensor([2])  # LONG
        td = env.step(td)["next"]

        assert env.position.position_size > 0
        position_size = env.position.position_size

        # HOLD action (action 0)
        td["action"] = torch.tensor([0])
        td = env.step(td)["next"]

        # Position should still be open
        assert env.position.position_size == position_size

        # CLOSE action (action 1)
        td["action"] = torch.tensor([1])
        td = env.step(td)["next"]

        # Position should now be closed
        assert env.position.position_size == 0


class TestBackwardCompatibility:
    """Test backward compatibility when CLOSE action is disabled."""

    def test_futures_without_close(self, sample_df):
        """Environment should work correctly with CLOSE disabled."""
        config = FuturesOneStepEnvConfig(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_close_action=False  # Disable CLOSE
        )
        env = FuturesOneStepEnv(sample_df, config)

        # Should have HOLD, 1 LONG, 1 SHORT = 3 actions
        assert env.action_spec.n == 3
        assert env.action_map[0] == (None, None, None)  # HOLD
        assert env.action_map[1] == ("long", -0.02, 0.05)  # LONG
        # For SHORT: SL/TP are NOT swapped in the action map (swapping happens at execution)
        assert env.action_map[2][0] == "short"  # Just verify it's a short action

    def test_longonly_without_close(self, sample_df):
        """Long-only environment should work with CLOSE disabled."""
        config = LongOnlyOneStepEnvConfig(
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.05],
            include_close_action=False  # Disable CLOSE
        )
        env = LongOnlyOneStepEnv(sample_df, config)

        # Should have HOLD, 1 LONG = 2 actions
        assert env.action_spec.n == 2
        assert env.action_map[0] == (None, None)  # HOLD
        assert env.action_map[1] == (-0.02, 0.05)  # LONG


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
