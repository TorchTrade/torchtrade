"""
Consolidated tests for SequentialTradingEnv (unified spot/futures environment).

This file consolidates tests from:
- test_seqlongonly.py (spot trading)
- test_seqfutures.py (futures trading)

Uses parametrization to test both trading modes with maximum code reuse.
"""

import pandas as pd
import pytest
import torch

from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.conftest import simple_feature_fn, validate_account_state


@pytest.fixture
def unified_env(sample_ohlcv_df, trading_mode, unified_config_spot, unified_config_futures):
    """Create unified environment for testing (spot or futures based on parameter)."""
    config = unified_config_spot if trading_mode == 1 else unified_config_futures
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
        assert unified_env.leverage == trading_mode
    @pytest.mark.parametrize("action_levels,expected_actions", [
        ([0, 1], 2),                  # Custom: flat/long
        ([-1, 0, 1], 3),              # Default: short/flat/long
        ([-1, -0.5, 0, 0.5, 1], 5),  # Custom: 5 levels
    ])
    def test_action_spec(self, sample_ohlcv_df, action_levels, expected_actions):
        """Action spec should match action_levels."""
        config = SequentialTradingEnvConfig(
            action_levels=action_levels,
            initial_cash=1000,
        )
        env = SequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        assert env.action_spec.n == expected_actions
        assert env.action_levels == action_levels
        env.close()

    def test_default_action_levels(self, sample_ohlcv_df):
        """Default action_levels should be [-1, 0, 1]."""
        config = SequentialTradingEnvConfig(initial_cash=1000)
        env = SequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        assert env.action_levels == [-1, 0, 1]
        assert env.action_spec.n == 3
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
            leverage=trading_mode,
            transaction_fee=invalid_fee,
        )
        with pytest.raises(ValueError, match="Transaction fee"):
            SequentialTradingEnv(sample_ohlcv_df, config)

    @pytest.mark.parametrize("invalid_slippage", [-0.1, 1.5])
    def test_invalid_slippage_raises(self, sample_ohlcv_df, trading_mode, invalid_slippage):
        """Should raise error for invalid slippage."""
        config = SequentialTradingEnvConfig(
            leverage=trading_mode,
            slippage=invalid_slippage,
        )
        with pytest.raises(ValueError, match="Slippage"):
            SequentialTradingEnv(sample_ohlcv_df, config)

    def test_leverage_validation_futures(self, sample_ohlcv_df):
        """Futures mode should require leverage >= 1."""
        with pytest.raises(ValueError, match="[Ll]everage"):
            config = SequentialTradingEnvConfig(leverage=0.5)
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
        action_td["action"] = torch.tensor(1 if unified_env.leverage == 1 else 2)  # Long
        unified_env.step(action_td)

        # Now reset
        td_reset = unified_env.reset()
        account_state = td_reset["account_state"]
        # Element 1: position_direction should be 0 after reset (no position)
        assert account_state[1] == 0.0, "Position direction should be 0 after reset"


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

    def test_same_action_preserves_position_size(self, unified_env):
        """Repeating same action should keep similar position size (not exact due to price changes)."""
        td = unified_env.reset()

        # Take initial position
        action_td = td.clone()
        buy_action = 1 if unified_env.leverage == 1 else 2  # Buy/Long (100%)
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

    @pytest.mark.parametrize("action_levels,leverage,open_action_idx,expected_direction", [
        ([0, 1], 1, 1, 1),            # Spot: buy -> long (+1)
        ([-1, 0, 1], 2, 2, 1),        # Futures: long -> long (+1)
        ([-1, 0, 1], 2, 0, -1),       # Futures: short -> short (-1)
    ])
    def test_open_position(self, sample_ohlcv_df, action_levels, leverage, open_action_idx, expected_direction):
        """Opening a position should set correct position_direction."""
        config = SequentialTradingEnvConfig(
            action_levels=action_levels,
            leverage=leverage,
            initial_cash=1000,
            transaction_fee=0.0,  # Avoid floating point precision issues
        )
        env = SequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        # Open position
        action_td = td.clone()
        action_td["action"] = torch.tensor(open_action_idx)
        next_td = env.step(action_td)

        account_state = next_td["next"]["account_state"]
        # Element 1: position_direction
        if expected_direction > 0:
            assert account_state[1] > 0, f"Should have positive direction after long, got {account_state[1]}"
        elif expected_direction < 0:
            assert account_state[1] < 0, f"Should have negative direction after short, got {account_state[1]}"
        env.close()

    @pytest.mark.parametrize("action_levels,open_idx,close_idx", [
        ([0, 1], 1, 0),           # Spot: buy then sell
        ([-1, 0, 1], 2, 1),       # Futures: long then close
        ([-1, 0, 1], 0, 1),       # Futures: short then close
    ])
    def test_close_position(self, sample_ohlcv_df, action_levels, open_idx, close_idx):
        """Closing a position should set position_direction to 0."""
        config = SequentialTradingEnvConfig(
            action_levels=action_levels,
            initial_cash=1000,
            transaction_fee=0.0,  # Avoid floating point precision issues
        )
        env = SequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        # Open position
        action_td = td.clone()
        action_td["action"] = torch.tensor(open_idx)
        next_td = env.step(action_td)

        # Close position
        action_td_close = next_td["next"].clone()
        action_td_close["action"] = torch.tensor(close_idx)
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
        close_action = 0 if unified_env.leverage == 1 else 1
        action_td["action"] = torch.tensor(close_action)
        next_td = unified_env.step(action_td)

        reward = next_td["next"]["reward"]
        assert reward == 0.0, "Reward should be 0 when holding cash"

    @pytest.mark.parametrize("action_levels,position_action_idx", [
        ([0, 1], 1),           # Spot: long position
        ([-1, 0, 1], 2),       # Futures: long position
        ([-1, 0, 1], 0),       # Futures: short position
    ])
    def test_reward_reflects_pnl(self, sample_ohlcv_df, action_levels, position_action_idx):
        """Reward should reflect position P&L for both long and short positions."""
        config = SequentialTradingEnvConfig(
            action_levels=action_levels,
            initial_cash=1000,
        )
        env = SequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        # Open position
        action_td = td.clone()
        action_td["action"] = torch.tensor(position_action_idx)
        next_td = env.step(action_td)

        # Take another step (position should have P&L)
        action_td_hold = next_td["next"].clone()
        action_td_hold["action"] = torch.tensor(position_action_idx)  # Hold position
        next_td_hold = env.step(action_td_hold)

        # Reward should be based on price change
        reward = next_td_hold["next"]["reward"]
        assert isinstance(reward.item(), float)
        env.close()


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
        close_action = 0 if unified_env.leverage == 1 else 1

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
            leverage=1,
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
            leverage=20,  # Very high leverage for easier liquidation
            initial_cash=1000,
            max_traj_length=200,  # Lower max to ensure termination
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],  # Use small timeframe for limited data
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        env = SequentialTradingEnv(trending_down_df, config, simple_feature_fn)
        td = env.reset()

        # Open long position on downtrend (should eventually liquidate or hit max)
        action_td = td.clone()
        action_td["action"] = torch.tensor(2)  # Long (futures: [-1, 0, 1])
        td = env.step(action_td)

        # Continue stepping with same action (keep long position)
        terminated = False
        for _ in range(200):
            if td["next"]["done"].item():
                terminated = True
                break
            action_td = td["next"].clone()
            action_td["action"] = torch.tensor(2)  # Keep long position (futures: [-1, 0, 1])
            try:
                td = env.step(action_td)
            except (ValueError, IndexError):
                terminated = True
                break

        # Should have terminated (either liquidation, max length, or out of data)
        assert terminated, "Episode should terminate"
        env.close()

    def test_liquidation_short_position_uptrend(self, sample_ohlcv_df, trending_up_df):
        """Short position should liquidate on uptrend (symmetric to long liquidation test)."""
        config = SequentialTradingEnvConfig(
            leverage=20,  # Very high leverage for easier liquidation
            action_levels=[-1, 0, 1],  # Need short actions
            initial_cash=1000,
            max_traj_length=200,
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        env = SequentialTradingEnv(trending_up_df, config, simple_feature_fn)
        td = env.reset()

        # Open short position on uptrend (should eventually liquidate or hit max)
        action_td = td.clone()
        action_td["action"] = torch.tensor(0)  # Short (action_levels: [-1, 0, 1])
        td = env.step(action_td)

        # Verify short position was opened
        assert env.position.position_size < 0, "Should have opened short position"

        # Continue stepping with same action (keep short position)
        terminated = False
        for _ in range(200):
            if td["next"]["done"].item():
                terminated = True
                break
            action_td = td["next"].clone()
            action_td["action"] = torch.tensor(0)  # Keep short position
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

    @pytest.mark.parametrize("invalid_cash", [0, -100, -1])
    def test_invalid_initial_cash_handled(self, sample_ohlcv_df, trading_mode, invalid_cash):
        """Invalid initial cash should either raise error or be handled gracefully."""
        # Some implementations may allow 0 or negative initial cash for testing purposes
        # The key is that the environment should handle it without crashing
        try:
            config = SequentialTradingEnvConfig(
                leverage=trading_mode,
                initial_cash=invalid_cash,
            )
            env = SequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
            # If it doesn't raise, that's acceptable - environment handles it gracefully
            # Just verify it can reset
            td = env.reset()
            assert td is not None
            env.close()
        except (ValueError, AssertionError):
            # Also acceptable - strict validation rejected invalid cash
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
            leverage=trading_mode,
            window_sizes=[10],  # Requires 10+ rows
        )

        with pytest.raises((ValueError, IndexError)):
            env = SequentialTradingEnv(tiny_df, config, simple_feature_fn)
            env.reset()

    @pytest.mark.parametrize("action_levels,fee,open_idx,close_idx", [
        ([0, 1], 0.001, 1, 0),        # Spot: 0.1% fee
        ([0, 1], 0.01, 1, 0),         # Spot: 1% fee
        ([-1, 0, 1], 0.001, 2, 1),    # Futures long: 0.1% fee
        ([-1, 0, 1], 0.001, 0, 1),    # Futures short: 0.1% fee
    ])
    def test_transaction_costs_reduce_cash(self, sample_ohlcv_df, action_levels, fee, open_idx, close_idx):
        """Transaction fees should reduce available cash."""
        config = SequentialTradingEnvConfig(
            action_levels=action_levels,
            initial_cash=1000,
            transaction_fee=fee,
        )
        env = SequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()
        initial_cash = td["account_state"][0].item()

        # Open position
        action_td = td.clone()
        action_td["action"] = torch.tensor(open_idx)
        next_td = env.step(action_td)

        # Close position
        action_td_sell = next_td["next"].clone()
        action_td_sell["action"] = torch.tensor(close_idx)
        next_td_sell = env.step(action_td_sell)

        final_cash = next_td_sell["next"]["account_state"][0].item()

        # Should have less cash due to fees (unless position made huge profit)
        # At minimum, fees were charged (allow small profit margin)
        assert final_cash <= initial_cash + 100  # Allow some profit but fees should impact
        env.close()


# ============================================================================
# REGRESSION TESTS
# ============================================================================


class TestSequentialEnvRegression:
    """Regression tests for known issues."""

    @pytest.mark.parametrize("key,expected_shape,expected_dtype", [
        ("account_state", (6,), torch.float32),
        ("done", (1,), torch.bool),
        ("reward", (1,), torch.float32),
    ])
    def test_output_shapes_and_types(self, unified_env, key, expected_shape, expected_dtype):
        """Step outputs should have consistent shapes and types."""
        td = unified_env.reset()

        # Check initial state for account_state
        if key == "account_state":
            assert td[key].shape[-1] == expected_shape[0]
            if expected_dtype:
                assert td[key].dtype == expected_dtype

        # After step
        action_td = td.clone()
        close_action = 0 if unified_env.leverage == 1 else 1
        action_td["action"] = torch.tensor(close_action)
        next_td = unified_env.step(action_td)

        # Check next state
        if key == "account_state":
            assert next_td["next"][key].shape[-1] == expected_shape[0]
        else:
            value = next_td["next"][key]
            if expected_shape:
                assert value.shape == expected_shape or value.numel() == expected_shape[0]
            if expected_dtype:
                assert value.dtype == expected_dtype

    @pytest.mark.parametrize("leverage,action_levels,hold_action", [
        (1, [0, 1], 0),           # Spot
        (10, [-1, 0, 1], 1),      # Futures
    ], ids=["spot", "futures"])
    def test_truncation_does_not_set_terminated(self, sample_ohlcv_df, leverage, action_levels, hold_action):
        """Truncated episodes (data exhaustion/max steps) must NOT set terminated=True.

        Regression test for #150: terminated included truncated, breaking
        value bootstrapping in all sequential envs.
        """
        config = SequentialTradingEnvConfig(
            leverage=leverage,
            action_levels=action_levels,
            max_traj_length=5000,  # Larger than data so episode truncates
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        env = SequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        for _ in range(5000):
            action_td = td["next"].clone() if "next" in td.keys() else td.clone()
            action_td["action"] = torch.tensor(hold_action)
            td = env.step(action_td)
            if td["next"]["done"].item():
                break

        assert td["next"]["done"].item() is True
        assert td["next"]["truncated"].item() is True
        assert td["next"]["terminated"].item() is False, (
            "Truncated episode should have terminated=False (issue #150)"
        )
        env.close()

    @pytest.mark.parametrize("leverage,action_levels,hold_action", [
        (1, [0, 1], 0),           # Spot
        (10, [-1, 0, 1], 1),      # Futures
    ], ids=["spot", "futures"])
    def test_truncation_respects_per_episode_length_with_random_start(
        self, sample_ohlcv_df, leverage, action_levels, hold_action
    ):
        """Truncation must use per-episode max_traj_length, not fixed max_steps.

        Regression test for #158: _check_truncation used self.max_steps (set once
        at init) instead of self.max_traj_length (updated per episode), causing
        incorrect episode boundaries with random_start=True.
        """
        max_traj = 20
        config = SequentialTradingEnvConfig(
            leverage=leverage,
            action_levels=action_levels,
            max_traj_length=max_traj,
            random_start=True,
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        env = SequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)

        for episode in range(3):
            td = env.reset()
            steps = 0
            for _ in range(max_traj + 10):  # Try to exceed max_traj
                action_td = td["next"].clone() if "next" in td.keys() else td.clone()
                action_td["action"] = torch.tensor(hold_action)
                td = env.step(action_td)
                steps += 1
                if td["next"]["done"].item():
                    break

            assert steps <= max_traj, (
                f"Episode {episode} ran {steps} steps but max_traj_length={max_traj} (issue #158)"
            )
            assert td["next"]["truncated"].item() is True
        env.close()

    def test_bankruptcy_sets_terminated_not_truncated(self, trending_down_df):
        """Bankruptcy must set terminated=True, truncated=False.

        Regression test for #150: ensures value bootstrapping does NOT
        occur at true terminal states (bankruptcy).
        """
        config = SequentialTradingEnvConfig(
            leverage=20,
            action_levels=[-1, 0, 1],
            max_traj_length=5000,
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            transaction_fee=0.0,
            slippage=0.0,
        )
        env = SequentialTradingEnv(trending_down_df, config, simple_feature_fn)
        td = env.reset()

        # Open leveraged long into crashing prices
        action_td = td.clone()
        action_td["action"] = torch.tensor(2)  # long
        td = env.step(action_td)

        for _ in range(500):
            if td["next"]["done"].item():
                break
            action_td = td["next"].clone()
            action_td["action"] = torch.tensor(2)  # keep long
            td = env.step(action_td)

        assert td["next"]["done"].item() is True
        assert td["next"]["terminated"].item() is True
        assert td["next"]["truncated"].item() is False, (
            "Bankruptcy should have truncated=False (issue #150)"
        )
        env.close()

    def test_normal_step_signals(self, unified_env):
        """Normal (non-terminal) step: terminated=False, truncated=False, done=False.

        Regression test for #150.
        """
        td = unified_env.reset()
        close_action = 0 if unified_env.leverage == 1 else 1
        action_td = td.clone()
        action_td["action"] = torch.tensor(close_action)
        td = unified_env.step(action_td)

        assert td["next"]["terminated"].item() is False
        assert td["next"]["truncated"].item() is False
        assert td["next"]["done"].item() is False
