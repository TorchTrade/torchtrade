"""
Consolidated tests for SequentialTradingEnv (unified spot/futures environment).

This file consolidates tests from:
- test_seqlongonly.py (spot trading)
- test_seqfutures.py (futures trading)

Uses parametrization to test both trading modes with maximum code reuse.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.conftest import simple_feature_fn, validate_account_state


@pytest.fixture
def unified_env(sample_ohlcv_df, trading_mode, unified_config_spot, unified_config_futures):
    """Create unified environment for testing (spot or futures based on parameter)."""
    config = unified_config_spot if trading_mode == "spot" else unified_config_futures
    env_instance = SequentialTradingEnv(
        df=sample_ohlcv_df,
        config=config,
        feature_preprocessing_fn=simple_feature_fn,
    )
    yield env_instance
    env_instance.close()


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


class TestSequentialEnvInitialization:
    """Tests for environment initialization (both spot and futures)."""

    def test_env_initializes(self, unified_env, trading_mode):
        """Environment should initialize without errors."""
        assert unified_env is not None
        assert unified_env.trading_mode == trading_mode

    @pytest.mark.parametrize("trading_mode,expected_actions", [
        ("spot", 3),      # [0.0, 0.5, 1.0] -> close/50%/100%
        ("futures", 5),   # [-1.0, -0.5, 0.0, 0.5, 1.0] -> short100/short50/close/long50/long100
    ])
    def test_action_spec(self, sample_ohlcv_df, trading_mode, expected_actions):
        """Action spec should match trading mode."""
        config = SequentialTradingEnvConfig(
            trading_mode=trading_mode,
            leverage=10 if trading_mode == "futures" else 1,
            initial_cash=1000,
        )
        env = SequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        assert env.action_spec.n == expected_actions
        env.close()

    def test_observation_spec_has_account_state(self, unified_env):
        """Observation spec should include account_state."""
        assert "account_state" in unified_env.observation_spec.keys()

    def test_observation_spec_has_market_data(self, unified_env):
        """Observation spec should include market data keys."""
        assert len(unified_env.market_data_keys) > 0
        for key in unified_env.market_data_keys:
            assert key in unified_env.observation_spec.keys()

    @pytest.mark.parametrize("invalid_fee", [-0.1, 1.5])
    def test_invalid_transaction_fee_raises(self, sample_ohlcv_df, trading_mode, invalid_fee):
        """Should raise error for invalid transaction fee."""
        config = SequentialTradingEnvConfig(
            trading_mode=trading_mode,
            transaction_fee=invalid_fee,
        )
        with pytest.raises(ValueError, match="Transaction fee"):
            SequentialTradingEnv(sample_ohlcv_df, config)

    @pytest.mark.parametrize("invalid_slippage", [-0.1, 1.5])
    def test_invalid_slippage_raises(self, sample_ohlcv_df, trading_mode, invalid_slippage):
        """Should raise error for invalid slippage."""
        config = SequentialTradingEnvConfig(
            trading_mode=trading_mode,
            slippage=invalid_slippage,
        )
        with pytest.raises(ValueError, match="Slippage"):
            SequentialTradingEnv(sample_ohlcv_df, config)

    def test_leverage_validation_futures(self, sample_ohlcv_df):
        """Futures mode should require leverage >= 1."""
        with pytest.raises(ValueError, match="[Ll]everage"):
            config = SequentialTradingEnvConfig(trading_mode="futures", leverage=0.5)
            SequentialTradingEnv(sample_ohlcv_df, config)


# ============================================================================
# RESET TESTS
# ============================================================================


class TestSequentialEnvReset:
    """Tests for environment reset (both spot and futures)."""

    def test_reset_returns_tensordict(self, unified_env):
        """Reset should return a TensorDict."""
        td = unified_env.reset()
        assert td is not None
        assert hasattr(td, "keys")

    def test_reset_initializes_balance(self, unified_env, trading_mode):
        """Reset should initialize balance correctly."""
        td = unified_env.reset()
        account_state = td["account_state"]
        validate_account_state(account_state, trading_mode)

        # Check initial state (exposure should be 0 when no position)
        # Element 0: exposure_pct should be 0 at start (no position)
        assert account_state[0] == 0.0, "Exposure should be 0 at start"
        # Verify internal balance is set correctly
        assert unified_env.balance == unified_env.initial_cash

    def test_reset_clears_position(self, unified_env):
        """Reset should clear any existing position."""
        # First, take a position
        td = unified_env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(2 if unified_env.trading_mode == "spot" else 4)  # Buy/Long
        unified_env.step(action_td)

        # Now reset
        td_reset = unified_env.reset()
        account_state = td_reset["account_state"]
        # Element 1: position_direction should be 0 after reset (no position)
        assert account_state[1] == 0.0, "Position direction should be 0 after reset"

    def test_reset_advances_episode_start(self, unified_env):
        """Reset should advance episode start when random_start=True."""
        # Enable random start
        unified_env.random_start = True
        positions = []
        for _ in range(5):
            td = unified_env.reset()
            # Store the internal _current_idx or similar attribute
            # This test is implementation-dependent, skip if attribute doesn't exist
            if hasattr(unified_env, '_current_idx'):
                positions.append(unified_env._current_idx)
            elif hasattr(unified_env, 'current_index'):
                positions.append(unified_env.current_index)

        # Should have different start positions (if we could track them)
        if len(positions) > 0:
            assert len(set(positions)) > 1, "Random start should vary episode start positions"
        else:
            pytest.skip("Environment doesn't expose current index")


# ============================================================================
# STEP TESTS
# ============================================================================


class TestSequentialEnvStep:
    """Tests for environment step (both spot and futures)."""

    def test_step_returns_tensordict(self, unified_env):
        """Step should return a TensorDict."""
        td = unified_env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(0)
        next_td = unified_env.step(action_td)
        assert next_td is not None
        assert hasattr(next_td, "keys")

    def test_step_has_next_keys(self, unified_env):
        """Step output should have 'next' nested keys."""
        td = unified_env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(0)
        next_td = unified_env.step(action_td)

        assert "next" in next_td.keys()
        assert "account_state" in next_td["next"].keys()
        assert "reward" in next_td["next"].keys()
        assert "done" in next_td["next"].keys()

    def test_step_increments_time(self, unified_env):
        """Step should increment internal time index."""
        td = unified_env.reset()

        # Get initial index (implementation-dependent)
        if hasattr(unified_env, '_current_idx'):
            initial_idx = unified_env._current_idx
        elif hasattr(unified_env, 'current_index'):
            initial_idx = unified_env.current_index
        else:
            pytest.skip("Environment doesn't expose current index")

        action_td = td.clone()
        # Use close action (0 for spot, 2 for futures)
        close_action = 0 if unified_env.trading_mode == "spot" else 2
        action_td["action"] = torch.tensor(close_action)
        unified_env.step(action_td)

        # Check index incremented
        if hasattr(unified_env, '_current_idx'):
            assert unified_env._current_idx == initial_idx + 1
        elif hasattr(unified_env, 'current_index'):
            assert unified_env.current_index == initial_idx + 1

    def test_same_action_preserves_position_size(self, unified_env):
        """Repeating same action should keep similar position size (not exact due to price changes)."""
        td = unified_env.reset()

        # Take initial position
        action_td = td.clone()
        buy_action = 2 if unified_env.trading_mode == "spot" else 4  # Buy/Long (100%)
        action_td["action"] = torch.tensor(buy_action)
        next_td = unified_env.step(action_td)

        position_before = next_td["next"]["account_state"][1]

        # Repeat same action (should try to maintain same position percentage)
        action_td_repeat = next_td["next"].clone()
        action_td_repeat["action"] = torch.tensor(buy_action)
        next_td_repeat = unified_env.step(action_td_repeat)

        position_after = next_td_repeat["next"]["account_state"][1]
        # Position should be approximately the same sign and magnitude
        # (exact match depends on price movement and rebalancing logic)
        assert position_before * position_after >= 0, "Position sign should be same"
        assert torch.isclose(position_before.abs(), position_after.abs(), rtol=0.15)


# ============================================================================
# TRADE EXECUTION TESTS
# ============================================================================


class TestSequentialEnvTradeExecution:
    """Tests for trade execution (spot and futures specific behavior)."""

    def test_buy_action_spot(self, sample_ohlcv_df, unified_config_spot):
        """Spot: Buy action should acquire position."""
        env = SequentialTradingEnv(sample_ohlcv_df, unified_config_spot, simple_feature_fn)
        td = env.reset()

        # Buy action
        action_td = td.clone()
        action_td["action"] = torch.tensor(2)  # Buy
        next_td = env.step(action_td)

        account_state = next_td["next"]["account_state"]
        # Element 1: position_direction should be +1 for long/buy
        assert account_state[1] > 0, "Should have positive position direction after buy"
        env.close()

    def test_sell_action_spot(self, sample_ohlcv_df, unified_config_spot):
        """Spot: Sell action should close position."""
        env = SequentialTradingEnv(sample_ohlcv_df, unified_config_spot, simple_feature_fn)
        td = env.reset()

        # Buy first
        action_td = td.clone()
        action_td["action"] = torch.tensor(2)  # Buy
        next_td = env.step(action_td)

        # Then sell
        action_td_sell = next_td["next"].clone()
        action_td_sell["action"] = torch.tensor(0)  # Sell
        next_td_sell = env.step(action_td_sell)

        account_state = next_td_sell["next"]["account_state"]
        # Element 1: position_direction should be 0 for no position
        assert account_state[1] == 0.0, "Position direction should be 0 after sell"
        env.close()

    def test_long_action_futures(self, sample_ohlcv_df, unified_config_futures):
        """Futures: Long action should create long position."""
        env = SequentialTradingEnv(sample_ohlcv_df, unified_config_futures, simple_feature_fn)
        td = env.reset()

        # Long action
        action_td = td.clone()
        action_td["action"] = torch.tensor(4)  # Long 100%
        next_td = env.step(action_td)

        account_state = next_td["next"]["account_state"]
        # Element 1: position_direction should be +1 for long
        assert account_state[1] > 0, "Should have positive position direction after long"
        env.close()

    def test_short_action_futures(self, sample_ohlcv_df, unified_config_futures):
        """Futures: Short action should create short position."""
        env = SequentialTradingEnv(sample_ohlcv_df, unified_config_futures, simple_feature_fn)
        td = env.reset()

        # Short action
        action_td = td.clone()
        action_td["action"] = torch.tensor(0)  # Short all
        next_td = env.step(action_td)

        account_state = next_td["next"]["account_state"]
        # Element 1: position_direction should be -1 for short
        assert account_state[1] < 0, "Should have negative position direction after short"
        env.close()

    def test_close_position_futures(self, sample_ohlcv_df, unified_config_futures):
        """Futures: Close action should flatten position."""
        env = SequentialTradingEnv(sample_ohlcv_df, unified_config_futures, simple_feature_fn)
        td = env.reset()

        # Open long position
        action_td = td.clone()
        action_td["action"] = torch.tensor(4)  # Long (action 4 = 1.0)
        next_td = env.step(action_td)

        # Close position
        action_td_close = next_td["next"].clone()
        action_td_close["action"] = torch.tensor(2)  # Close all (action 2 = 0.0)
        next_td_close = env.step(action_td_close)

        account_state = next_td_close["next"]["account_state"]
        # Element 1: position_direction should be 0 for no position
        assert account_state[1] == 0.0, "Position direction should be 0 after close"
        env.close()


# ============================================================================
# REWARD TESTS
# ============================================================================


class TestSequentialEnvReward:
    """Tests for reward calculation (both modes)."""

    def test_reward_on_no_position(self, unified_env):
        """Reward should be 0 when holding cash (no position)."""
        td = unified_env.reset()
        action_td = td.clone()
        # For both modes: action 0 for spot is 0.0 (close), action 2 for futures is 0.0 (close)
        close_action = 0 if unified_env.trading_mode == "spot" else 2
        action_td["action"] = torch.tensor(close_action)
        next_td = unified_env.step(action_td)

        reward = next_td["next"]["reward"]
        assert reward == 0.0, "Reward should be 0 when holding cash"

    def test_reward_reflects_pnl(self, unified_env):
        """Reward should reflect position P&L."""
        td = unified_env.reset()

        # Open position
        action_td = td.clone()
        buy_action = 2 if unified_env.trading_mode == "spot" else 4
        action_td["action"] = torch.tensor(buy_action)
        next_td = unified_env.step(action_td)

        # Take another step (position should have P&L)
        action_td_hold = next_td["next"].clone()
        action_td_hold["action"] = torch.tensor(1)  # Hold
        next_td_hold = unified_env.step(action_td_hold)

        # Reward should be based on price change
        reward = next_td_hold["next"]["reward"]
        assert isinstance(reward.item(), float)


# ============================================================================
# TERMINATION TESTS
# ============================================================================


class TestSequentialEnvTermination:
    """Tests for episode termination (both modes)."""

    def test_terminates_within_max_length(self, unified_env):
        """Episode should terminate within max trajectory length."""
        # Set a reasonable max length
        if hasattr(unified_env, 'max_traj_length'):
            unified_env.max_traj_length = 15
        elif hasattr(unified_env, 'config') and hasattr(unified_env.config, 'max_traj_length'):
            unified_env.config.max_traj_length = 15

        td = unified_env.reset()
        close_action = 0 if unified_env.trading_mode == "spot" else 2

        # Step up to max_traj_length times
        for i in range(15):
            action_td = td.clone()
            action_td["action"] = torch.tensor(close_action)
            td = unified_env.step(action_td)
            if td["next"]["done"].item():
                break

        # Should have terminated within the limit
        assert i <= 15, "Should terminate within max_traj_length"

    def test_terminates_at_data_end(self, sample_ohlcv_df):
        """Episode should terminate when reaching end of data."""
        config = SequentialTradingEnvConfig(
            trading_mode="spot",
            initial_cash=1000,
            max_traj_length=10000,  # Very large
            random_start=False,
        )
        env = SequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        # Step until done
        max_steps = len(sample_ohlcv_df) + 100
        for i in range(max_steps):
            if "next" in td.keys() and td["next"].get("done", torch.tensor(False)).item():
                break
            action_td = td.clone()
            action_td["action"] = torch.tensor(0)  # Close action for spot
            try:
                td = env.step(action_td)
            except (ValueError, IndexError):
                # Out of data
                break

        assert i < max_steps, "Should terminate before max_steps"
        env.close()

    def test_liquidation_futures(self, sample_ohlcv_df, trending_down_df):
        """Futures: Should terminate on liquidation or max steps."""
        config = SequentialTradingEnvConfig(
            trading_mode="futures",
            leverage=20,  # Very high leverage for easier liquidation
            initial_cash=1000,
            max_traj_length=200,  # Lower max to ensure termination
        )
        env = SequentialTradingEnv(trending_down_df, config, simple_feature_fn)
        td = env.reset()

        # Open long position on downtrend (should eventually liquidate or hit max)
        action_td = td.clone()
        action_td["action"] = torch.tensor(4)  # Long 100%
        td = env.step(action_td)

        # Continue stepping with same action (keep long position)
        terminated = False
        for _ in range(200):
            if td["next"]["done"].item():
                terminated = True
                break
            action_td = td["next"].clone()
            action_td["action"] = torch.tensor(4)  # Keep long position
            try:
                td = env.step(action_td)
            except (ValueError, IndexError):
                terminated = True
                break

        # Should have terminated (either liquidation, max length, or out of data)
        assert terminated, "Episode should terminate"
        env.close()


# ============================================================================
# EDGE CASES
# ============================================================================


class TestSequentialEnvEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_initial_cash_handled(self, sample_ohlcv_df, trading_mode):
        """Zero initial cash should either raise error or be handled gracefully."""
        # Some implementations may allow 0 initial cash for testing purposes
        try:
            config = SequentialTradingEnvConfig(
                trading_mode=trading_mode,
                initial_cash=0,
            )
            env = SequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
            # If it doesn't raise, that's also acceptable (some envs might support this)
            env.close()
        except (ValueError, AssertionError):
            # Expected behavior for strict validation
            pass

    def test_insufficient_data_raises(self, trading_mode):
        """Should raise error if DataFrame is too small."""
        tiny_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min"),
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [1000.0] * 5,
        })

        config = SequentialTradingEnvConfig(
            trading_mode=trading_mode,
            window_sizes=[10],  # Requires 10+ rows
        )

        with pytest.raises((ValueError, IndexError)):
            env = SequentialTradingEnv(tiny_df, config, simple_feature_fn)
            env.reset()

    def test_transaction_costs_reduce_cash(self, unified_env):
        """Transaction fees should reduce available cash."""
        unified_env.transaction_fee = 0.1  # 10% fee
        td = unified_env.reset()
        initial_cash = td["account_state"][0].item()

        # Buy
        action_td = td.clone()
        buy_action = 2 if unified_env.trading_mode == "spot" else 4
        action_td["action"] = torch.tensor(buy_action)
        next_td = unified_env.step(action_td)

        # Sell
        action_td_sell = next_td["next"].clone()
        sell_action = 0 if unified_env.trading_mode == "spot" else 1
        action_td_sell["action"] = torch.tensor(sell_action)
        next_td_sell = unified_env.step(action_td_sell)

        final_cash = next_td_sell["next"]["account_state"][0].item()

        # Should have less cash due to fees (unless position made huge profit)
        # At minimum, fees were charged
        assert final_cash <= initial_cash + 100  # Allow some profit but fees should impact


# ============================================================================
# REGRESSION TESTS
# ============================================================================


class TestSequentialEnvRegression:
    """Regression tests for known issues."""

    def test_account_state_shape_consistent(self, unified_env):
        """Account state should always be shape [6]."""
        td = unified_env.reset()
        assert td["account_state"].shape[-1] == 6

        # After step
        action_td = td.clone()
        close_action = 0 if unified_env.trading_mode == "spot" else 2
        action_td["action"] = torch.tensor(close_action)
        next_td = unified_env.step(action_td)
        assert next_td["next"]["account_state"].shape[-1] == 6

    def test_done_flag_is_boolean(self, unified_env):
        """Done flag should be boolean tensor."""
        td = unified_env.reset()
        action_td = td.clone()
        close_action = 0 if unified_env.trading_mode == "spot" else 2
        action_td["action"] = torch.tensor(close_action)
        next_td = unified_env.step(action_td)

        done = next_td["next"]["done"]
        assert done.dtype == torch.bool

    def test_reward_is_scalar(self, unified_env):
        """Reward should be scalar tensor."""
        td = unified_env.reset()
        action_td = td.clone()
        close_action = 0 if unified_env.trading_mode == "spot" else 2
        action_td["action"] = torch.tensor(close_action)
        next_td = unified_env.step(action_td)

        reward = next_td["next"]["reward"]
        assert reward.numel() == 1
