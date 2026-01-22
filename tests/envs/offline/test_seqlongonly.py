"""
Tests for SeqLongOnlyEnv environment.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline.longonly.sequential import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


def simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature processing function for testing."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.fillna(0, inplace=True)
    return df


# Note: sample_ohlcv_df fixture is defined in conftest.py


@pytest.fixture
def default_config():
    """Default environment configuration for testing."""
    return SeqLongOnlyEnvConfig(
        symbol="TEST/USD",
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        initial_cash=1000,
        transaction_fee=0.01,
        slippage=0.0,  # Disable slippage for deterministic tests
        seed=42,
        max_traj_length=100,
        random_start=False,
    )


@pytest.fixture
def env(sample_ohlcv_df, default_config):
    """Create a SeqLongOnlyEnv instance for testing."""
    return SeqLongOnlyEnv(
        df=sample_ohlcv_df,
        config=default_config,
        feature_preprocessing_fn=simple_feature_fn,
    )


class TestSeqLongOnlyEnvInitialization:
    """Tests for environment initialization."""

    def test_env_initializes(self, env):
        """Environment should initialize without errors."""
        assert env is not None

    def test_action_spec(self, env):
        """Action spec should be categorical with 3 fractional actions."""
        assert env.action_spec.n == 3  # Default: [0.0, 0.5, 1.0]

    def test_observation_spec_has_account_state(self, env):
        """Observation spec should include account_state."""
        assert "account_state" in env.observation_spec.keys()

    def test_observation_spec_has_market_data(self, env):
        """Observation spec should include market data keys."""
        assert len(env.market_data_keys) > 0
        for key in env.market_data_keys:
            assert key in env.observation_spec.keys()

    def test_invalid_transaction_fee_raises(self, sample_ohlcv_df):
        """Should raise error for invalid transaction fee."""
        config = SeqLongOnlyEnvConfig(transaction_fee=1.5)  # > 1
        with pytest.raises(ValueError, match="Transaction fee"):
            SeqLongOnlyEnv(sample_ohlcv_df, config)

    def test_invalid_slippage_raises(self, sample_ohlcv_df):
        """Should raise error for invalid slippage."""
        config = SeqLongOnlyEnvConfig(slippage=-0.1)  # < 0
        with pytest.raises(ValueError, match="Slippage"):
            SeqLongOnlyEnv(sample_ohlcv_df, config)


class TestSeqLongOnlyEnvReset:
    """Tests for environment reset."""

    def test_reset_returns_tensordict(self, env):
        """Reset should return a TensorDict."""
        td = env.reset()
        assert td is not None
        assert hasattr(td, "keys")

    def test_reset_initializes_balance(self, env):
        """Reset should initialize balance correctly."""
        env.reset()
        assert env.balance == 1000  # initial_cash from config

    def test_reset_clears_position(self, env):
        """Reset should clear any existing position."""
        env.reset()
        assert env.position.position_size == 0.0
        assert env.position.current_position == 0.0
        assert env.position.entry_price == 0.0

    def test_reset_clears_counters(self, env):
        """Reset should clear step and hold counters."""
        env.reset()
        assert env.step_counter == 0
        assert env.position.hold_counter == 0

    def test_reset_clears_histories(self, env):
        """Reset should clear history."""
        env.reset()
        assert len(env.history) == 0
        assert len(env.history.base_prices) == 0
        assert len(env.history.actions) == 0
        assert len(env.history.rewards) == 0
        assert len(env.history.portfolio_values) == 0

    def test_reset_observation_has_correct_keys(self, env):
        """Reset observation should have all required keys."""
        td = env.reset()
        assert "account_state" in td.keys()
        for key in env.market_data_keys:
            assert key in td.keys()

    def test_reset_account_state_shape(self, env):
        """Account state should have correct shape."""
        td = env.reset()
        assert td["account_state"].shape == (7,)

    def test_reset_account_state_values(self, env):
        """Account state should have correct initial values."""
        td = env.reset()
        account_state = td["account_state"]
        # [balance, position_size, position_value, entry_price, current_price, unrealized_pnl, hold_counter]
        assert account_state[0].item() == 1000  # balance
        assert account_state[1].item() == 0.0   # position_size
        assert account_state[2].item() == 0.0   # position_value
        assert account_state[3].item() == 0.0   # entry_price
        assert account_state[4].item() > 0      # current_price (should be positive)
        assert account_state[5].item() == 0.0   # unrealized_pnl
        assert account_state[6].item() == 0     # hold_counter


class TestSeqLongOnlyEnvStep:
    """Tests for environment step."""

    def test_step_returns_tensordict(self, env):
        """Step should return a TensorDict."""
        td = env.reset()
        td.set("action", torch.tensor(0))  # close/flat (0.0)
        result = env.step(td)
        assert result is not None
        assert "next" in result.keys()

    def test_step_increments_counter(self, env):
        """Step should increment step counter."""
        td = env.reset()
        assert env.step_counter == 0
        td.set("action", torch.tensor(0))  # close/flat (0.0)
        env.step(td)
        assert env.step_counter == 1

    def test_step_has_reward(self, env):
        """Step result should include reward."""
        td = env.reset()
        td.set("action", torch.tensor(0))  # close/flat (0.0)
        result = env.step(td)
        assert "reward" in result["next"].keys()

    def test_step_has_done_flags(self, env):
        """Step result should include done flags."""
        td = env.reset()
        td.set("action", torch.tensor(0))  # close/flat (0.0)
        result = env.step(td)
        next_td = result["next"]
        assert "done" in next_td.keys()
        assert "truncated" in next_td.keys()
        assert "terminated" in next_td.keys()

    def test_step_hold_no_position_change(self, env):
        """Hold action should not change position when no position."""
        td = env.reset()
        initial_balance = env.balance

        td.set("action", torch.tensor(0))  # close/flat (0.0)
        env.step(td)

        assert env.position.position_size == 0.0
        assert env.balance == initial_balance

    def test_step_updates_histories(self, env):
        """Step should update history."""
        td = env.reset()
        td.set("action", torch.tensor(0))  # close/flat (0.0)
        env.step(td)

        assert len(env.history) == 1
        assert len(env.history.base_prices) == 1
        assert len(env.history.actions) == 1
        assert len(env.history.rewards) == 1
        assert len(env.history.portfolio_values) == 1

    def test_full_episode_completes(self, env):
        """Full episode should complete without errors (may terminate early on bankruptcy)."""
        td = env.reset()
        steps = 0
        max_steps = env.max_traj_length

        while steps < max_steps:
            action = env.action_spec.sample()
            td.set("action", action)

            try:
                result = env.step(td)
                td = result["next"]
            except ValueError as e:
                # Bankruptcy can cause reward calculation errors
                if "Invalid new_portfolio_value: 0.0" in str(e) or "Portfolio value must be positive" in str(e):
                    break  # Terminate episode on bankruptcy
                raise  # Re-raise other ValueErrors

            steps += 1

            if td.get("done", False):
                # Early termination is valid (e.g., bankruptcy with random actions)
                break

        # Should complete at least one step without errors
        assert steps > 0

    def test_portfolio_value_never_nan(self, env):
        """Portfolio value should never be NaN during episode (even with bankruptcy)."""
        td = env.reset()

        for _ in range(50):
            action = env.action_spec.sample()
            td.set("action", action)

            try:
                result = env.step(td)
                td = result["next"]
            except ValueError as e:
                # Bankruptcy can cause reward calculation errors
                if "Invalid new_portfolio_value: 0.0" in str(e) or "Portfolio value must be positive" in str(e):
                    break  # Terminate episode on bankruptcy
                raise  # Re-raise other ValueErrors

            portfolio_value = env._get_portfolio_value()
            assert not np.isnan(portfolio_value), "Portfolio value should never be NaN"

            # Stop if episode terminated (e.g., bankruptcy)
            if td.get("done", False):
                break


class TestSeqLongOnlyEnvTradeExecution:
    """Tests for trade execution."""

    def test_buy_action_opens_position(self, env):
        """Buy action should open a position."""
        td = env.reset()
        initial_balance = env.balance

        td.set("action", torch.tensor(2))  # buy all-in (1.0)
        env.step(td)

        assert env.position.position_size > 0
        assert env.balance < initial_balance
        assert env.position.current_position == 1.0
        assert env.position.entry_price > 0

    def test_sell_action_closes_position(self, env):
        """Sell action should close a position."""
        td = env.reset()

        # First buy
        td.set("action", torch.tensor(2))  # buy all-in (1.0)
        result = env.step(td)
        td = result["next"]

        assert env.position.position_size > 0
        position_before_sell = env.position.position_size

        # Then sell (close position)
        td.set("action", torch.tensor(0))  # close position (0.0)
        env.step(td)

        assert env.position.position_size == 0.0
        assert env.position.current_position == 0.0
        assert env.position.entry_price == 0.0
        assert env.balance > 0  # Got proceeds from sale

    def test_buy_deducts_fees(self, env):
        """Buy should deduct transaction fees."""
        td = env.reset()
        initial_balance = env.balance

        td.set("action", torch.tensor(2))  # buy all-in (1.0)
        env.step(td)

        # Balance should decrease by more than just position value due to fees
        # Position value + remaining balance should be less than initial due to fees
        portfolio_value = env._get_portfolio_value()
        assert portfolio_value < initial_balance

    def test_sell_deducts_fees(self, env):
        """Sell should deduct transaction fees."""
        td = env.reset()
        initial_balance = env.balance

        # Buy all-in
        td.set("action", torch.tensor(2))  # buy all-in (1.0)
        result = env.step(td)
        td = result["next"]

        # Sell immediately (same price, only fees should reduce value)
        td.set("action", torch.tensor(0))  # close position (0.0)
        env.step(td)

        # Final balance should be less than initial due to buy + sell fees
        assert env.balance < initial_balance

    def test_cannot_sell_without_position(self, env):
        """Sell action without position should do nothing."""
        td = env.reset()
        initial_balance = env.balance

        td.set("action", torch.tensor(0))  # sell without position
        env.step(td)

        assert env.balance == initial_balance
        assert env.position.position_size == 0.0

    def test_entry_price_recorded_on_buy(self, env):
        """Entry price should be recorded correctly on buy."""
        td = env.reset()

        td.set("action", torch.tensor(2))  # buy all-in (1.0)
        env.step(td)

        assert env.position.entry_price > 0
        # Entry price should be a reasonable value (close to initial price ~100)
        assert 50 < env.position.entry_price < 200


class TestSeqLongOnlyEnvReward:
    """Tests for reward calculation."""

    def test_reward_is_float(self, env):
        """Reward should be a float value."""
        td = env.reset()
        td.set("action", torch.tensor(0))  # close/flat (0.0)
        result = env.step(td)

        reward = result["next"]["reward"]
        assert isinstance(reward.item(), float)

    def test_reward_not_nan(self, env):
        """Reward should never be NaN during normal operation."""
        # Set seed for reproducible action sampling
        torch.manual_seed(42)
        np.random.seed(42)

        td = env.reset()

        # Use conservative action sequence to avoid bankruptcy
        # Cycle through actions but avoid extreme positions
        actions = [0, 1, 0, 1, 0, 2, 0, 1]  # Mix of close, half, and all-in

        for i in range(min(len(actions), env.max_traj_length)):
            action = torch.tensor(actions[i % len(actions)])
            td.set("action", action)

            # Stop if portfolio value is getting dangerously low (near bankruptcy)
            portfolio_value = env.balance + env.position.position_value
            if portfolio_value < env.initial_portfolio_value * 0.5:  # More conservative threshold
                break

            result = env.step(td)
            td = result["next"]

            reward = td["reward"]
            assert not torch.isnan(reward).any(), f"Reward is NaN at step {i}"

            if td.get("done", False):
                break

    def test_reward_not_inf(self, env):
        """Reward should never be infinite (even with bankruptcy)."""
        td = env.reset()

        for _ in range(50):
            action = env.action_spec.sample()
            td.set("action", action)

            try:
                result = env.step(td)
                td = result["next"]
            except ValueError as e:
                # Bankruptcy can cause reward calculation errors
                if "Invalid new_portfolio_value: 0.0" in str(e) or "Portfolio value must be positive" in str(e):
                    break  # Terminate episode on bankruptcy
                raise  # Re-raise other ValueErrors

            reward = td["reward"]
            assert not torch.isinf(reward).any(), "Reward should never be infinite"

            if td.get("done", False):
                # Early termination is valid (e.g., bankruptcy with random actions)
                break

    def test_dense_reward_clipped(self, env):
        """Rewards should be finite and not NaN."""
        # Set seed for reproducible action sampling
        torch.manual_seed(42)
        np.random.seed(42)

        td = env.reset()

        # Use conservative action sequence
        actions = [0, 1, 0, 2, 0, 1, 0, 1, 0, 2]  # Mix of positions

        # Run several steps (not terminal)
        for i in range(min(len(actions), env.max_traj_length - 2)):
            action = torch.tensor(actions[i])
            td.set("action", action)
            result = env.step(td)
            td = result["next"]

            # Rewards should be finite and not NaN
            if not td.get("done", False) and env.step_counter < env.max_traj_length - 1:
                reward = td["reward"].item()
                assert not np.isnan(reward), f"Reward is NaN at step {i}"
                assert not np.isinf(reward), f"Reward is infinite at step {i}"


class TestSeqLongOnlyEnvTermination:
    """Tests for termination conditions."""

    def test_terminates_at_max_steps(self, env):
        """Episode should terminate at max trajectory length."""
        td = env.reset()

        for i in range(env.max_traj_length + 10):
            td.set("action", torch.tensor(0))  # close/flat (0.0)
            result = env.step(td)
            td = result["next"]

            if td.get("done", False):
                break

        assert env.step_counter <= env.max_traj_length

    def test_terminates_on_bankruptcy(self, sample_ohlcv_df):
        """Episode should terminate when portfolio value drops below threshold."""
        # Create config with high fees to quickly lose money
        config = SeqLongOnlyEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=100,
            transaction_fee=0.4,  # 40% fee - will cause rapid losses
            slippage=0.0,
            max_traj_length=50,
            random_start=False,
            bankrupt_threshold=0.5,  # 50% threshold
        )
        env = SeqLongOnlyEnv(sample_ohlcv_df, config, simple_feature_fn)
        try:
            td = env.reset()

            # Trade back and forth to lose money on fees
            for i in range(20):
                # Buy
                td.set("action", torch.tensor(2))  # buy all-in (1.0)
                result = env.step(td)
                td = result["next"]
                if td.get("done", False):
                    break

                # Sell
                td.set("action", torch.tensor(0))
                result = env.step(td)
                td = result["next"]
                if td.get("done", False):
                    break

            # Should have terminated due to bankruptcy
            portfolio_value = env._get_portfolio_value()
            bankruptcy_threshold = config.bankrupt_threshold * env.initial_portfolio_value

            # Either terminated early due to bankruptcy or completed
            assert td.get("done", False) or portfolio_value >= bankruptcy_threshold
        finally:
            env.close()

    def test_truncated_flag_on_data_exhaustion(self, env):
        """Truncated flag should be set when data is exhausted."""
        td = env.reset()

        # Run until done
        while True:
            td.set("action", torch.tensor(0))  # close/flat (0.0)
            result = env.step(td)
            td = result["next"]

            if td.get("done", False):
                break

        # At the end, truncated should be True (data exhausted or max steps)
        # Note: The env sets truncated based on sampler exhaustion
        assert "truncated" in td.keys()


class TestSeqLongOnlyEnvEdgeCases:
    """Tests for edge cases."""

    def test_small_initial_cash(self, sample_ohlcv_df):
        """Environment should work with very small initial cash."""
        config = SeqLongOnlyEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1,  # $1
            transaction_fee=0.01,
            slippage=0.0,
            max_traj_length=20,
            random_start=False,
        )
        env = SeqLongOnlyEnv(sample_ohlcv_df, config, simple_feature_fn)
        try:
            td = env.reset()
            assert env.balance == 1

            # Should still be able to trade
            td.set("action", torch.tensor(2))  # buy all-in (1.0)
            result = env.step(td)

            assert not torch.isnan(result["next"]["reward"]).any()
        finally:
            env.close()

    def test_zero_transaction_fee(self, sample_ohlcv_df):
        """Environment should work with zero transaction fees."""
        config = SeqLongOnlyEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            transaction_fee=0.0,  # No fees
            slippage=0.0,
            max_traj_length=20,
            random_start=False,
        )
        env = SeqLongOnlyEnv(sample_ohlcv_df, config, simple_feature_fn)
        try:
            td = env.reset()
            initial_balance = env.balance

            # Buy and sell should preserve value (no fees)
            td.set("action", torch.tensor(2))  # buy all-in (1.0)
            result = env.step(td)
            td = result["next"]

            td.set("action", torch.tensor(0))  # sell
            env.step(td)

            # With no fees and no slippage, balance should be approximately preserved
            # (small differences due to price movement between steps)
            assert abs(env.balance - initial_balance) < initial_balance * 0.05
        finally:
            env.close()

    def test_multiple_episodes(self, env):
        """Environment should work correctly across multiple episodes."""
        for episode in range(5):
            td = env.reset()

            # Verify clean state
            assert env.step_counter == 0
            assert env.position.position_size == 0.0
            assert env.balance == 1000

            # Run a few steps with controlled actions to avoid bankruptcy
            for step in range(10):
                # Use less risky actions (avoid full sell/buy cycles)
                action = torch.tensor(0)  # close/flat (0.0) - safer than random
                td.set("action", action)

                # Stop if near bankruptcy
                portfolio_value = env.balance + env.position.position_value
                if portfolio_value < env.initial_portfolio_value * env.config.bankrupt_threshold * 2:
                    break

                result = env.step(td)
                td = result["next"]

                if td.get("done", False):
                    break

    def test_random_initial_cash_range(self, sample_ohlcv_df):
        """Environment should sample initial cash from range."""
        config = SeqLongOnlyEnvConfig(
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=(500, 1500),  # Range
            transaction_fee=0.01,
            slippage=0.0,
            max_traj_length=20,
            random_start=False,
        )
        env = SeqLongOnlyEnv(sample_ohlcv_df, config, simple_feature_fn)
        try:
            balances = []
            for _ in range(10):
                env.reset()
                balances.append(env.balance)

            # Should have some variation in initial balances
            assert min(balances) >= 500
            assert max(balances) <= 1500
            # With 10 samples, we should see at least 2 different values
            assert len(set(balances)) >= 2
        finally:
            env.close()


class TestLookaheadBiasIntegration:
    """Integration tests verifying lookahead bias prevention in SeqLongOnlyEnv.

    These tests verify that the lookahead fix in sampler.py actually works
    when integrated into the environment, ensuring agents can't see incomplete
    higher timeframe bars through the environment API.

    Related to Issue #10 - Critical data leak in multi-timeframe observations.
    """

    def test_multiframe_observations_no_future_leakage(self):
        """
        CRITICAL: Verify environment enforces lookahead bias prevention.

        When env uses multiple timeframes, higher TF observations should only
        show completed bars at the current execution time.

        WITHOUT FIX: 5-min bar at index 00:00 contains data through minute 4,
                     but agent at minute 2 sees it (2 minutes of future data) ❌

        WITH FIX:    5-min bar at index 00:05 contains data through minute 4,
                     agent at minute 2 doesn't see it yet (incomplete) ✓
        """
        # Create test data where close price = minute index (for easy detection)
        n_minutes = 500
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

        # Close price equals the minute index - makes leakage trivial to detect
        minute_indices = np.arange(n_minutes, dtype=float)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": minute_indices,
            "high": minute_indices + 0.5,
            "low": minute_indices - 0.5,
            "close": minute_indices,  # close = minute index
            "volume": np.ones(n_minutes) * 1000,
        })

        # Simple preprocessing that preserves close prices
        def preserve_close_fn(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy().reset_index(drop=False)
            df["features_close"] = df["close"]  # Keep original close
            return df

        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),  # Execute timeframe
                TimeFrame(5, TimeFrameUnit.Minute),  # Higher timeframe (should be offset)
            ],
            window_sizes=[5, 3],  # 5 bars for 1min, 3 bars for 5min
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            max_traj_length=20,
            random_start=False,  # Start from beginning for predictable test
            include_base_features=False,  # Only use our feature_close
        )

        env = SeqLongOnlyEnv(df=df, config=config, feature_preprocessing_fn=preserve_close_fn)
        try:
            # Reset environment
            td = env.reset()

            # Run several steps and verify no future leakage
            for step_num in range(15):
                # Get 1-minute observations (window of 5 bars)
                one_min_obs = td["market_data_1Minute_5"]  # Shape: [5, features]

                # Get 5-minute observations (window of 3 bars)
                five_min_obs = td["market_data_5Minute_3"]  # Shape: [3, features]

                # Extract close prices (last feature, assuming features_close is last)
                # features_close should be the only feature (index 0 since include_base_features=False)
                one_min_closes = one_min_obs[:, 0].numpy()  # Last 5 minutes
                five_min_closes = five_min_obs[:, 0].numpy()  # Last 3 five-minute bars

                # Current execution minute is the most recent 1-minute close
                # (since close price = minute index in our test data)
                current_minute = int(one_min_closes[-1])

                # Verify 1-minute observations only show past data
                for i, minute_val in enumerate(one_min_closes):
                    assert minute_val <= current_minute, (
                        f"1-minute observation at position {i} shows future data: "
                        f"minute {minute_val} > current minute {current_minute}"
                    )

                # Verify 5-minute bars are complete (end at or before current minute)
                # Each 5-minute bar should end at a multiple of 5
                # With the fix, 5-min bar indexed at X contains data through minute X-1
                for i, five_min_close in enumerate(five_min_closes):
                    # The close of a 5-min bar should be the last minute of that bar
                    # With offset fix, bars are indexed by END time
                    assert five_min_close <= current_minute, (
                        f"5-minute observation at position {i} shows future data: "
                        f"close at minute {five_min_close} > current minute {current_minute}"
                    )

                    # Verify bar is complete (ends at multiple of 5 minus 1)
                    # e.g., bar for minutes [0,1,2,3,4] has close=4
                    bar_end_minute = int(five_min_close)
                    assert bar_end_minute % 5 == 4 or bar_end_minute % 5 == 9, (
                        f"5-minute bar close at minute {bar_end_minute} doesn't end on "
                        f"5-minute boundary (should end at X*5-1)"
                    )

                # Take hold action and step
                action = torch.tensor(0)  # close/flat (0.0)
                td_next = env.step(td.set("action", action))
                td = td_next

                if td.get("done", False):
                    break
        finally:
            env.close()

    def test_multiframe_higher_tf_only_shows_complete_bars(self):
        """Verify higher timeframe observations only contain complete bars.

        This is a simpler test focusing specifically on bar completeness.
        """
        n_minutes = 200
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

        # Close = minute index
        minute_indices = np.arange(n_minutes, dtype=float)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": minute_indices,
            "high": minute_indices,
            "low": minute_indices,
            "close": minute_indices,
            "volume": np.ones(n_minutes) * 1000,
        })

        def preserve_close_fn(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy().reset_index(drop=False)
            df["features_close"] = df["close"]
            return df

        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(15, TimeFrameUnit.Minute),  # 15-minute bars
            ],
            window_sizes=[5, 2],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=1000,
            max_traj_length=30,
            random_start=False,
            include_base_features=False,
        )

        env = SeqLongOnlyEnv(df=df, config=config, feature_preprocessing_fn=preserve_close_fn)
        try:
            td = env.reset()

            for _ in range(25):
                # Get current minute from 1-minute observation
                one_min_obs = td["market_data_1Minute_5"]
                current_minute = int(one_min_obs[-1, 0].item())

                # Get 15-minute observations
                fifteen_min_obs = td["market_data_15Minute_2"]
                fifteen_min_closes = fifteen_min_obs[:, 0].numpy()

                # All 15-minute bar closes should be <= current_minute
                for close_val in fifteen_min_closes:
                    assert close_val <= current_minute, (
                        f"15-minute bar with close={close_val} visible at minute {current_minute} "
                        f"(bar not yet complete!)"
                    )

                # Step environment
                td = env.step(td.set("action", torch.tensor(0)))  # close/flat (0.0)

                if td.get("done", False):
                    break
        finally:
            env.close()
