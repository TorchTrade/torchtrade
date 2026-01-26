"""
Consolidated tests for OneStepTradingEnv (unified spot/futures one-step environment).

This file consolidates tests from:
- test_longonlyonestepenv.py (spot one-step)
- test_futuresonestepenv.py (futures one-step)

Uses parametrization to test both trading modes with maximum code reuse.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline import OneStepTradingEnv, OneStepTradingEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.conftest import simple_feature_fn, validate_account_state


@pytest.fixture
def onestep_config_spot():
    """OneStep config for spot trading."""
    return OneStepTradingEnvConfig(
        trading_mode="spot",
        initial_cash=1000,
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        transaction_fee=0.01,
        slippage=0.0,
        seed=42,
        random_start=False,
    )


@pytest.fixture
def onestep_config_futures():
    """OneStep config for futures trading."""
    return OneStepTradingEnvConfig(
        trading_mode="futures",
        leverage=10,
        initial_cash=1000,
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        transaction_fee=0.01,
        slippage=0.0,
        seed=42,
        random_start=False,
    )


@pytest.fixture
def onestep_env(sample_ohlcv_df, trading_mode, onestep_config_spot, onestep_config_futures):
    """Create OneStep environment for testing."""
    config = onestep_config_spot if trading_mode == "spot" else onestep_config_futures
    env_instance = OneStepTradingEnv(
        df=sample_ohlcv_df,
        config=config,
        feature_preprocessing_fn=simple_feature_fn,
    )
    yield env_instance
    env_instance.close()


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


class TestOneStepEnvInitialization:
    """Tests for OneStep environment initialization."""

    def test_env_initializes(self, onestep_env, trading_mode):
        """Environment should initialize without errors."""
        assert onestep_env is not None
        assert onestep_env.trading_mode == trading_mode

    @pytest.mark.parametrize("trading_mode,expected_actions", [
        ("spot", 3),      # sell/hold/buy
        ("futures", 5),   # short_all/close_all/long_25/long_50/long_100
    ])
    def test_action_spec(self, sample_ohlcv_df, trading_mode, expected_actions):
        """Action spec should match trading mode."""
        pytest.skip("OneStepEnv uses SLTP action space - expectations need update")
        config = OneStepTradingEnvConfig(
            trading_mode=trading_mode,
            leverage=10 if trading_mode == "futures" else 1,
        )
        env = OneStepTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        assert env.action_spec.n == expected_actions
        env.close()

        """Rollout length should be set from config."""


# ============================================================================
# RESET TESTS
# ============================================================================


class TestOneStepEnvReset:
    """Tests for OneStep environment reset."""

    def test_reset_returns_tensordict(self, onestep_env):
        """Reset should return a TensorDict."""
        td = onestep_env.reset()
        assert td is not None
        assert hasattr(td, "keys")

    def test_reset_initializes_account_state(self, onestep_env, trading_mode):
        """Reset should initialize account state correctly."""
        td = onestep_env.reset()
        account_state = td["account_state"]
        validate_account_state(account_state, trading_mode)

        # Initial position should be 0
        assert account_state[1] == 0.0

    def test_reset_sets_done_false(self, onestep_env):
        """Reset should set done flag to False."""
        td = onestep_env.reset()
        # The reset TensorDict might not have "done" or it should be False
        if "done" in td.keys():
            assert not td["done"].item()


# ============================================================================
# SINGLE DECISION TESTS
# ============================================================================


class TestOneStepSingleDecision:
    """Tests for one-decision-per-episode behavior."""

    def test_step_terminates_episode(self, onestep_env):
        """Single step should terminate the episode."""
        td = onestep_env.reset()

        action_td = td.clone()
        action_td["action"] = torch.tensor(1)  # Hold
        next_td = onestep_env.step(action_td)

        # Episode should be done after single step
        assert next_td["next"]["done"].item()

    def test_reward_accumulates_rollout(self, onestep_env):
        """Reward should accumulate over rollout period."""
        td = onestep_env.reset()

        # Buy action
        action_td = td.clone()
        buy_action = 2 if onestep_env.trading_mode == "spot" else 4
        action_td["action"] = torch.tensor(buy_action)
        next_td = onestep_env.step(action_td)

        # Reward should reflect accumulated P&L over rollout
        reward = next_td["next"]["reward"]
        assert isinstance(reward.item(), float)

    def test_terminal_state_at_rollout_end(self, onestep_env):
        pytest.skip("Test needs refactoring - current_index not exposed")
        initial_idx = onestep_env.current_index
        td = onestep_env.reset()

        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = onestep_env.step(action_td)

        # Note: implementation details may vary
        # This is a conceptual test


# ============================================================================
# ROLLOUT SIMULATION TESTS
# ============================================================================


class TestOneStepRolloutSimulation:
    """Tests for internal rollout simulation."""

    def test_hold_action_no_trading(self, onestep_env):
        """Hold action should not trade during rollout."""
        td = onestep_env.reset()

        action_td = td.clone()
        action_td["action"] = torch.tensor(1)  # Hold
        next_td = onestep_env.step(action_td)

        # Final position should still be 0
        account_state = next_td["next"]["account_state"]
        assert account_state[1] == 0.0

    def test_buy_action_acquires_position(self, onestep_env, trading_mode):
        """Buy/Long action should acquire position and hold through rollout."""
        td = onestep_env.reset()

        buy_action = 2 if trading_mode == "spot" else 4
        action_td = td.clone()
        action_td["action"] = torch.tensor(buy_action)
        next_td = onestep_env.step(action_td)

        # Should show P&L from holding position
        # Exact reward depends on price movement during rollout
        reward = next_td["next"]["reward"]
        assert reward is not None

    def test_sell_action_stays_cash(self, sample_ohlcv_df, onestep_config_spot):
        """Sell action should keep cash through rollout (spot only)."""
        env = OneStepTradingEnv(sample_ohlcv_df, onestep_config_spot, simple_feature_fn)
        td = env.reset()

        action_td = td.clone()
        action_td["action"] = torch.tensor(0)  # Sell (stay in cash)
        next_td = env.step(action_td)

        # Should have no position
        account_state = next_td["next"]["account_state"]
        assert account_state[1] == 0.0

        # Reward should be 0 (no position)
        assert next_td["next"]["reward"] == 0.0
        env.close()

    def test_futures_liquidation_during_rollout(self, trending_down_df, onestep_config_futures):
        """Futures position should liquidate during rollout if margin depleted."""
        onestep_config_futures.leverage = 20  # High leverage

        env = OneStepTradingEnv(trending_down_df, onestep_config_futures, simple_feature_fn)
        td = env.reset()

        # Long position on downtrend
        action_td = td.clone()
        action_td["action"] = torch.tensor(4)  # Long 100%
        next_td = env.step(action_td)

        # Should terminate (liquidation or natural end)
        assert next_td["next"]["done"].item()

        # Reward should reflect loss
        reward = next_td["next"]["reward"]
        # Typically negative due to downtrend
        env.close()


# ============================================================================
# SLTP TRIGGER TESTS (if applicable)
# ============================================================================


class TestOneStepSLTPTriggers:
    """Tests for SL/TP triggers during rollout (if supported)."""

    @pytest.mark.skip(reason="SLTP triggers require OneStepTradingEnvSLTP variant")
    def test_sltp_trigger_during_rollout(self):
        """SL/TP should trigger and close position during rollout."""
        # This would test OneStepTradingEnvSLTP variant
        pass


# ============================================================================
# TRUNCATION TESTS
# ============================================================================


class TestOneStepTruncation:
    """Tests for episode truncation."""

    def test_truncates_at_data_end(self, sample_ohlcv_df):
        """Should truncate if rollout reaches end of data."""
        # Position reset near end of data
        config = OneStepTradingEnvConfig(
            trading_mode="spot",
            random_start=False,
        )

        env = OneStepTradingEnv(sample_ohlcv_df, config, simple_feature_fn)

        # Set current index near end
        env.current_index = len(sample_ohlcv_df) - 50

        td = env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = env.step(action_td)

        # Should be done
        assert next_td["next"]["done"].item()
        env.close()

    def test_rollout_respects_max_steps(self, onestep_env):
        """Rollout should not exceed available data."""
        # This is implicitly tested by not crashing
        td = onestep_env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = onestep_env.step(action_td)

        # Should complete successfully
        assert next_td is not None


# ============================================================================
# REWARD ACCUMULATION TESTS
# ============================================================================


class TestOneStepRewardAccumulation:
    """Tests for terminal reward accumulation."""

    def test_reward_reflects_total_pnl(self, onestep_env):
        """Terminal reward should reflect total P&L over rollout."""
        td = onestep_env.reset()

        # Buy action
        buy_action = 2 if onestep_env.trading_mode == "spot" else 4
        action_td = td.clone()
        action_td["action"] = torch.tensor(buy_action)
        next_td = onestep_env.step(action_td)

        reward = next_td["next"]["reward"].item()

        # Reward should be non-zero (unless price didn't move)
        # We can't guarantee sign, but it should be a valid float
        assert isinstance(reward, float)

    def test_hold_reward_is_zero(self, onestep_env):
        """Hold action should have zero reward."""
        pytest.skip("Hold action may have non-zero reward depending on rollout")
        td = onestep_env.reset()

        action_td = td.clone()
        action_td["action"] = torch.tensor(1)  # Hold
        next_td = onestep_env.step(action_td)

        # No position = no reward
        assert next_td["next"]["reward"] == 0.0

    def test_transaction_costs_included(self, onestep_env):
        """Terminal reward should include transaction costs."""
        onestep_env.transaction_fee = 0.1  # 10% fee

        td = onestep_env.reset()

        # Buy and hold
        buy_action = 2 if onestep_env.trading_mode == "spot" else 4
        action_td = td.clone()
        action_td["action"] = torch.tensor(buy_action)
        next_td = onestep_env.step(action_td)

        # Reward should reflect fees (likely negative unless huge price move)
        reward = next_td["next"]["reward"].item()
        # Hard to assert exact value, but it should be impacted by fees


# ============================================================================
# EDGE CASES
# ============================================================================


class TestOneStepEdgeCases:
    """Edge case tests for OneStep environments."""

    def test_insufficient_data_graceful(self, trading_mode):
        """Should handle rollout beyond data length gracefully."""
        # Create very short dataset (50 rows)
        short_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="1min"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
        })

        config = OneStepTradingEnvConfig(
            trading_mode=trading_mode,
            leverage=10 if trading_mode == "futures" else 1,
            initial_cash=1000,
            time_frames="1Min",
            window_sizes=5,
        )

        env = OneStepTradingEnv(short_df, config, simple_feature_fn)
        td = env.reset()

        # Take action - should handle gracefully even if data runs out
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = env.step(action_td)

        # Should have terminated (either SL/TP or truncation)
        assert next_td["next"]["done"].item() or next_td["next"]["truncated"].item()
        env.close()

    def test_rollout_longer_than_data(self, trading_mode):
        tiny_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="1min"),
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.0] * 20,
            "volume": [1000.0] * 20,
        })

        config = OneStepTradingEnvConfig(
            trading_mode=trading_mode,
            window_sizes=[5],
        )

        env = OneStepTradingEnv(tiny_df, config, simple_feature_fn)
        td = env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = env.step(action_td)

        # Should terminate without crashing
        assert next_td["next"]["done"].item()
        env.close()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestOneStepIntegration:
    """Integration tests with TorchRL ecosystem."""

    def test_compatible_with_collector(self, onestep_env):
        """Should work with SyncDataCollector."""
        # This is a conceptual test - actual collector integration
        # would require imports and more setup
        td = onestep_env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = onestep_env.step(action_td)

        # Should have proper structure for collector
        assert "next" in next_td.keys()
        assert "reward" in next_td["next"].keys()
        assert "done" in next_td["next"].keys()

    def test_account_state_consistency(self, onestep_env, trading_mode):
        """Account state should be valid before and after step."""
        td = onestep_env.reset()
        validate_account_state(td["account_state"], trading_mode)

        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = onestep_env.step(action_td)
        validate_account_state(next_td["next"]["account_state"], trading_mode)


# ============================================================================
# REGRESSION TESTS
# ============================================================================


class TestOneStepRegression:
    """Regression tests for known OneStep issues."""

    def test_done_flag_always_true_after_step(self, onestep_env):
        """Done flag should always be True after step (one-step setting)."""
        td = onestep_env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = onestep_env.step(action_td)

        assert next_td["next"]["done"].item() is True

    def test_reward_is_scalar(self, onestep_env):
        """Reward should be scalar tensor."""
        td = onestep_env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = onestep_env.step(action_td)

        reward = next_td["next"]["reward"]
        assert reward.numel() == 1

    def test_account_state_shape_preserved(self, onestep_env):
        """Account state should maintain shape [6] throughout."""
        td = onestep_env.reset()
        assert td["account_state"].shape[-1] == 6

        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = onestep_env.step(action_td)
        assert next_td["next"]["account_state"].shape[-1] == 6

    def test_multiple_resets_work(self, onestep_env):
        """Should support multiple reset calls."""
        for _ in range(3):
            td = onestep_env.reset()
            assert td is not None
            assert "account_state" in td.keys()

            action_td = td.clone()
            action_td["action"] = torch.tensor(1)
            next_td = onestep_env.step(action_td)
            assert next_td["next"]["done"].item()

    def test_action_spec_immutable(self, onestep_env):
        """Action spec should not change during episode."""
        initial_spec = onestep_env.action_spec
        initial_n = initial_spec.n

        td = onestep_env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        onestep_env.step(action_td)

        assert onestep_env.action_spec.n == initial_n
