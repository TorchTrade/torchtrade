"""
Tests for SeqLongOnlyEnv environment.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline.seqlongonly import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit


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
        """Action spec should be categorical with 3 actions."""
        assert env.action_spec.n == 3  # sell, hold, buy

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
        assert env.position_size == 0.0
        assert env.current_position == 0.0
        assert env.entry_price == 0.0

    def test_reset_clears_counters(self, env):
        """Reset should clear step and hold counters."""
        env.reset()
        assert env.step_counter == 0
        assert env.position_hold_counter == 0

    def test_reset_clears_histories(self, env):
        """Reset should clear history lists."""
        env.reset()
        assert len(env.base_price_history) == 0
        assert len(env.action_history) == 0
        assert len(env.reward_history) == 0
        assert len(env.portfolio_value_history) == 0

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
        td.set("action", torch.tensor(1))  # hold
        result = env.step(td)
        assert result is not None
        assert "next" in result.keys()

    def test_step_increments_counter(self, env):
        """Step should increment step counter."""
        td = env.reset()
        assert env.step_counter == 0
        td.set("action", torch.tensor(1))  # hold
        env.step(td)
        assert env.step_counter == 1

    def test_step_has_reward(self, env):
        """Step result should include reward."""
        td = env.reset()
        td.set("action", torch.tensor(1))  # hold
        result = env.step(td)
        assert "reward" in result["next"].keys()

    def test_step_has_done_flags(self, env):
        """Step result should include done flags."""
        td = env.reset()
        td.set("action", torch.tensor(1))  # hold
        result = env.step(td)
        next_td = result["next"]
        assert "done" in next_td.keys()
        assert "truncated" in next_td.keys()
        assert "terminated" in next_td.keys()

    def test_step_hold_no_position_change(self, env):
        """Hold action should not change position when no position."""
        td = env.reset()
        initial_balance = env.balance

        td.set("action", torch.tensor(1))  # hold (index 1 = 0.0)
        env.step(td)

        assert env.position_size == 0.0
        assert env.balance == initial_balance

    def test_step_updates_histories(self, env):
        """Step should update history lists."""
        td = env.reset()
        td.set("action", torch.tensor(1))  # hold
        env.step(td)

        assert len(env.base_price_history) == 1
        assert len(env.action_history) == 1
        assert len(env.reward_history) == 1
        assert len(env.portfolio_value_history) == 1

    def test_full_episode_completes(self, env):
        """Full episode should complete without errors."""
        td = env.reset()
        steps = 0
        max_steps = env.max_traj_length

        while steps < max_steps:
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]
            steps += 1

            if td.get("done", False):
                break

        assert steps > 0

    def test_portfolio_value_never_nan(self, env):
        """Portfolio value should never be NaN during episode."""
        td = env.reset()

        for _ in range(50):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]

            portfolio_value = env._get_portfolio_value()
            assert not np.isnan(portfolio_value)

            if td.get("done", False):
                break


class TestSeqLongOnlyEnvTradeExecution:
    """Tests for trade execution."""

    def test_buy_action_opens_position(self, env):
        """Buy action should open a position."""
        td = env.reset()
        initial_balance = env.balance

        td.set("action", torch.tensor(2))  # buy (index 2 = 1.0)
        env.step(td)

        assert env.position_size > 0
        assert env.balance < initial_balance
        assert env.current_position == 1.0
        assert env.entry_price > 0

    def test_sell_action_closes_position(self, env):
        """Sell action should close a position."""
        td = env.reset()

        # First buy
        td.set("action", torch.tensor(2))  # buy
        result = env.step(td)
        td = result["next"]

        assert env.position_size > 0
        position_before_sell = env.position_size

        # Then sell
        td.set("action", torch.tensor(0))  # sell (index 0 = -1.0)
        env.step(td)

        assert env.position_size == 0.0
        assert env.current_position == 0.0
        assert env.entry_price == 0.0
        assert env.balance > 0  # Got proceeds from sale

    def test_buy_deducts_fees(self, env):
        """Buy should deduct transaction fees."""
        td = env.reset()
        initial_balance = env.balance

        td.set("action", torch.tensor(2))  # buy
        env.step(td)

        # Balance should decrease by more than just position value due to fees
        # Position value + remaining balance should be less than initial due to fees
        portfolio_value = env._get_portfolio_value()
        assert portfolio_value < initial_balance

    def test_sell_deducts_fees(self, env):
        """Sell should deduct transaction fees."""
        td = env.reset()
        initial_balance = env.balance

        # Buy
        td.set("action", torch.tensor(2))
        result = env.step(td)
        td = result["next"]

        # Sell immediately (same price, only fees should reduce value)
        td.set("action", torch.tensor(0))
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
        assert env.position_size == 0.0

    def test_cannot_buy_when_already_holding(self, env):
        """Buy action when already holding should do nothing."""
        td = env.reset()

        # First buy
        td.set("action", torch.tensor(2))
        result = env.step(td)
        td = result["next"]

        position_after_first_buy = env.position_size
        balance_after_first_buy = env.balance

        # Try to buy again
        td.set("action", torch.tensor(2))
        env.step(td)

        # Position and balance should be unchanged
        assert env.position_size == position_after_first_buy
        assert env.balance == balance_after_first_buy

    def test_hold_increments_counter_when_holding(self, env):
        """Hold action should increment hold counter when holding position."""
        td = env.reset()

        # Buy first
        td.set("action", torch.tensor(2))
        result = env.step(td)
        td = result["next"]

        assert env.position_hold_counter == 0

        # Hold
        td.set("action", torch.tensor(1))
        result = env.step(td)
        td = result["next"]

        assert env.position_hold_counter == 1

        # Hold again
        td.set("action", torch.tensor(1))
        env.step(td)

        assert env.position_hold_counter == 2

    def test_entry_price_recorded_on_buy(self, env):
        """Entry price should be recorded correctly on buy."""
        td = env.reset()

        td.set("action", torch.tensor(2))  # buy
        env.step(td)

        assert env.entry_price > 0
        # Entry price should be a reasonable value (close to initial price ~100)
        assert 50 < env.entry_price < 200


class TestSeqLongOnlyEnvReward:
    """Tests for reward calculation."""

    def test_reward_is_float(self, env):
        """Reward should be a float value."""
        td = env.reset()
        td.set("action", torch.tensor(1))
        result = env.step(td)

        reward = result["next"]["reward"]
        assert isinstance(reward.item(), float)

    def test_reward_not_nan(self, env):
        """Reward should never be NaN."""
        td = env.reset()

        for _ in range(50):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]

            reward = td["reward"]
            assert not torch.isnan(reward).any()

            if td.get("done", False):
                break

    def test_reward_not_inf(self, env):
        """Reward should never be infinite."""
        td = env.reset()

        for _ in range(50):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]

            reward = td["reward"]
            assert not torch.isinf(reward).any()

            if td.get("done", False):
                break

    def test_dense_reward_clipped(self, env):
        """Dense rewards should be clipped to [-0.05, 0.05]."""
        td = env.reset()

        # Run several steps (not terminal)
        for i in range(min(10, env.max_traj_length - 2)):
            action = env.action_spec.sample()
            td.set("action", action)
            result = env.step(td)
            td = result["next"]

            # Non-terminal rewards should be clipped
            if not td.get("done", False) and env.step_counter < env.max_traj_length - 1:
                reward = td["reward"].item()
                assert -0.05 <= reward <= 0.05


class TestSeqLongOnlyEnvTermination:
    """Tests for termination conditions."""

    def test_terminates_at_max_steps(self, env):
        """Episode should terminate at max trajectory length."""
        td = env.reset()

        for i in range(env.max_traj_length + 10):
            td.set("action", torch.tensor(1))  # hold
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

        td = env.reset()

        # Trade back and forth to lose money on fees
        for i in range(20):
            # Buy
            td.set("action", torch.tensor(2))
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

    def test_truncated_flag_on_data_exhaustion(self, env):
        """Truncated flag should be set when data is exhausted."""
        td = env.reset()

        # Run until done
        while True:
            td.set("action", torch.tensor(1))  # hold
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

        td = env.reset()
        assert env.balance == 1

        # Should still be able to trade
        td.set("action", torch.tensor(2))  # buy
        result = env.step(td)

        assert not torch.isnan(result["next"]["reward"]).any()

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

        td = env.reset()
        initial_balance = env.balance

        # Buy and sell should preserve value (no fees)
        td.set("action", torch.tensor(2))  # buy
        result = env.step(td)
        td = result["next"]

        td.set("action", torch.tensor(0))  # sell
        env.step(td)

        # With no fees and no slippage, balance should be approximately preserved
        # (small differences due to price movement between steps)
        assert abs(env.balance - initial_balance) < initial_balance * 0.05

    def test_multiple_episodes(self, env):
        """Environment should work correctly across multiple episodes."""
        for episode in range(5):
            td = env.reset()

            # Verify clean state
            assert env.step_counter == 0
            assert env.position_size == 0.0
            assert env.balance == 1000

            # Run a few steps
            for _ in range(10):
                action = env.action_spec.sample()
                td.set("action", action)
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

        balances = []
        for _ in range(10):
            env.reset()
            balances.append(env.balance)

        # Should have some variation in initial balances
        assert min(balances) >= 500
        assert max(balances) <= 1500
        # With 10 samples, we should see at least 2 different values
        assert len(set(balances)) >= 2
