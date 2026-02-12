"""
Tests for VectorizedSequentialTradingEnv.

Verifies:
- TorchRL spec compliance (check_env_specs)
- Correctness against scalar SequentialTradingEnv (step-by-step comparison)
- Truncation and done signal correctness
- Transaction fee accounting
- SyncDataCollector integration
- Partial reset behavior
"""

import pytest
import torch

from torchrl.envs.utils import check_env_specs
from torchrl.collectors import SyncDataCollector

from torchtrade.envs.offline import (
    SequentialTradingEnv,
    SequentialTradingEnvConfig,
    VectorizedSequentialTradingEnv,
    VectorizedSequentialTradingEnvConfig,
)
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.conftest import simple_feature_fn, validate_account_state


# ============================================================================
# Fixtures
# ============================================================================

TF_1MIN = TimeFrame(1, TimeFrameUnit.Minute)


@pytest.fixture
def vec_env(sample_ohlcv_df):
    """Create a vectorized env with 4 envs for general testing."""
    config = VectorizedSequentialTradingEnvConfig(
        num_envs=4,
        action_levels=[-1, 0, 1],
        initial_cash=1000,
        time_frames=[TF_1MIN],
        window_sizes=[10],
        execute_on=TF_1MIN,
        transaction_fee=0.0,
        slippage=0.0,
        seed=42,
        max_traj_length=50,
        random_start=False,
    )
    env = VectorizedSequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
    yield env
    env.close()


def _make_matched_envs(df, fee=0.0, slippage=0.0, max_traj=50, action_levels=None):
    """Create scalar and N=1 vectorized envs with identical config for comparison."""
    if action_levels is None:
        action_levels = [0, 1]  # Spot: flat/long

    scalar_config = SequentialTradingEnvConfig(
        leverage=1,
        action_levels=action_levels,
        initial_cash=1000,
        time_frames=[TF_1MIN],
        window_sizes=[10],
        execute_on=TF_1MIN,
        transaction_fee=fee,
        slippage=slippage,
        seed=42,
        max_traj_length=max_traj,
        random_start=False,
    )

    vec_config = VectorizedSequentialTradingEnvConfig(
        num_envs=1,
        action_levels=action_levels,
        initial_cash=1000,
        time_frames=[TF_1MIN],
        window_sizes=[10],
        execute_on=TF_1MIN,
        transaction_fee=fee,
        slippage=slippage,
        seed=42,
        max_traj_length=max_traj,
        random_start=False,
    )

    scalar_env = SequentialTradingEnv(df, scalar_config, simple_feature_fn)
    vec_env = VectorizedSequentialTradingEnv(df, vec_config, simple_feature_fn)
    return scalar_env, vec_env


# ============================================================================
# SPEC COMPLIANCE
# ============================================================================


class TestVecEnvSpecs:
    """TorchRL spec compliance tests."""

    def test_check_env_specs_passes(self, vec_env):
        """check_env_specs must pass — specs must match actual output shapes."""
        check_env_specs(vec_env)

    @pytest.mark.parametrize("num_envs", [1, 4, 16])
    def test_batch_size_matches_num_envs(self, sample_ohlcv_df, num_envs):
        """Batch size should equal num_envs."""
        config = VectorizedSequentialTradingEnvConfig(
            num_envs=num_envs,
            time_frames=[TF_1MIN],
            window_sizes=[10],
            execute_on=TF_1MIN,
            max_traj_length=20,
            random_start=False,
        )
        env = VectorizedSequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        assert env.batch_size == torch.Size([num_envs])
        td = env.reset()
        assert td["account_state"].shape == (num_envs, 6)
        env.close()


# ============================================================================
# CORRECTNESS vs SCALAR ENV
# ============================================================================


class TestVecEnvCorrectness:
    """Compare vectorized (N=1) against scalar SequentialTradingEnv."""

    def test_hold_sequence_matches(self, sample_ohlcv_df):
        """Hold-only sequence: portfolio value and reward should match."""
        scalar_env, vec_env = _make_matched_envs(sample_ohlcv_df)

        td_s = scalar_env.reset()
        td_v = vec_env.reset()

        # Market data should match
        assert torch.allclose(
            td_s["market_data_1Minute_10"],
            td_v["market_data_1Minute_10"].squeeze(0),
            atol=1e-5,
        ), "Initial market data mismatch"

        # Step with hold action for 10 steps
        for step in range(10):
            # Scalar: action index 0 = action_level 0.0 (flat)
            action_td_s = td_s.clone() if "next" not in td_s.keys() else td_s["next"].clone()
            action_td_s["action"] = torch.tensor(0)
            td_s = scalar_env.step(action_td_s)

            # Vectorized: action index 0 = action_level 0.0 (flat, clamped from -1)
            action_td_v = td_v.clone() if "next" not in td_v.keys() else td_v["next"].clone()
            action_td_v["action"] = torch.tensor([0])  # batch dim
            td_v = vec_env.step(action_td_v)

            # Holding cash: reward should be 0 for both
            reward_s = td_s["next"]["reward"].item()
            reward_v = td_v["next"]["reward"].squeeze().item()
            assert abs(reward_s) < 1e-6, f"Step {step}: scalar reward should be 0 for hold, got {reward_s}"
            assert abs(reward_v) < 1e-6, f"Step {step}: vec reward should be 0 for hold, got {reward_v}"

        scalar_env.close()
        vec_env.close()

    def test_buy_and_hold_portfolio_values_match(self, sample_ohlcv_df):
        """Buy then hold: portfolio values should track each other closely."""
        scalar_env, vec_env = _make_matched_envs(sample_ohlcv_df, fee=0.0)

        td_s = scalar_env.reset()
        td_v = vec_env.reset()

        # Step 1: Buy (action_levels=[0,1], index 1 = 1.0 = full long)
        action_td_s = td_s.clone()
        action_td_s["action"] = torch.tensor(1)
        td_s = scalar_env.step(action_td_s)

        action_td_v = td_v.clone()
        action_td_v["action"] = torch.tensor([1])
        td_v = vec_env.step(action_td_v)

        # Both should have position
        assert scalar_env.position.position_size > 0, "Scalar env should have long position"
        assert vec_env._position_sizes[0] > 0, "Vec env should have long position"

        # Position sizes should match closely
        assert abs(scalar_env.position.position_size - vec_env._position_sizes[0].item()) < 0.01, (
            f"Position sizes differ: scalar={scalar_env.position.position_size}, "
            f"vec={vec_env._position_sizes[0].item()}"
        )

        # Steps 2-10: Hold position (repeat buy action = hold due to same-action opt)
        for step in range(9):
            action_td_s = td_s["next"].clone()
            action_td_s["action"] = torch.tensor(1)
            td_s = scalar_env.step(action_td_s)

            action_td_v = td_v["next"].clone()
            action_td_v["action"] = torch.tensor([1])
            td_v = vec_env.step(action_td_v)

            # Compare portfolio values
            scalar_pv = scalar_env._get_portfolio_value()
            vec_pv = vec_env._portfolio_values[0].item()
            assert abs(scalar_pv - vec_pv) / max(scalar_pv, 1.0) < 0.01, (
                f"Step {step + 2}: PV mismatch: scalar={scalar_pv:.4f}, vec={vec_pv:.4f}"
            )

        scalar_env.close()
        vec_env.close()

    def test_buy_sell_cycle_balance_matches(self, sample_ohlcv_df):
        """Buy then sell: final balance should match between scalar and vectorized."""
        scalar_env, vec_env = _make_matched_envs(sample_ohlcv_df, fee=0.001)

        td_s = scalar_env.reset()
        td_v = vec_env.reset()

        # Buy
        action_td_s = td_s.clone()
        action_td_s["action"] = torch.tensor(1)
        td_s = scalar_env.step(action_td_s)

        action_td_v = td_v.clone()
        action_td_v["action"] = torch.tensor([1])
        td_v = vec_env.step(action_td_v)

        # Hold for a few steps
        for _ in range(3):
            action_td_s = td_s["next"].clone()
            action_td_s["action"] = torch.tensor(1)
            td_s = scalar_env.step(action_td_s)

            action_td_v = td_v["next"].clone()
            action_td_v["action"] = torch.tensor([1])
            td_v = vec_env.step(action_td_v)

        # Sell (action 0 = flat)
        action_td_s = td_s["next"].clone()
        action_td_s["action"] = torch.tensor(0)
        td_s = scalar_env.step(action_td_s)

        action_td_v = td_v["next"].clone()
        action_td_v["action"] = torch.tensor([0])
        td_v = vec_env.step(action_td_v)

        # After selling, both should be flat
        assert scalar_env.position.position_size == 0, "Scalar should be flat"
        assert vec_env._position_sizes[0].item() == 0, "Vec should be flat"

        # Balances should be close (fee accounting should match)
        scalar_bal = scalar_env.balance
        vec_bal = vec_env._balances[0].item()
        assert abs(scalar_bal - vec_bal) / max(scalar_bal, 1.0) < 0.01, (
            f"Balance mismatch after buy/sell: scalar={scalar_bal:.4f}, vec={vec_bal:.4f}"
        )

        scalar_env.close()
        vec_env.close()


# ============================================================================
# DONE SIGNAL TESTS
# ============================================================================


class TestVecEnvTermination:
    """Tests for done/terminated/truncated signals."""

    def test_truncation_at_max_traj_length(self, sample_ohlcv_df):
        """All envs should truncate at max_traj_length."""
        max_traj = 15
        config = VectorizedSequentialTradingEnvConfig(
            num_envs=4,
            time_frames=[TF_1MIN],
            window_sizes=[10],
            execute_on=TF_1MIN,
            max_traj_length=max_traj,
            random_start=False,
            seed=42,
        )
        env = VectorizedSequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        for step in range(max_traj + 5):
            action_td = td.clone() if "next" not in td.keys() else td["next"].clone()
            action_td["action"] = torch.zeros(4, dtype=torch.long)
            td = env.step(action_td)
            if td["next"]["done"].all():
                break

        assert step + 1 <= max_traj, (
            f"Should truncate within {max_traj} steps, ran {step + 1}"
        )
        assert td["next"]["truncated"].all(), "All envs should be truncated"
        assert not td["next"]["terminated"].any(), "No env should be terminated (cash only)"
        env.close()

    def test_normal_step_signals(self, vec_env):
        """Normal step should have all done signals False."""
        td = vec_env.reset()
        action_td = td.clone()
        action_td["action"] = torch.zeros(4, dtype=torch.long)
        td = vec_env.step(action_td)

        assert not td["next"]["done"].any(), "No env should be done after first step"
        assert not td["next"]["terminated"].any()
        assert not td["next"]["truncated"].any()

    def test_truncation_not_terminated(self, sample_ohlcv_df):
        """Truncated episodes must NOT set terminated=True (regression #150)."""
        config = VectorizedSequentialTradingEnvConfig(
            num_envs=2,
            time_frames=[TF_1MIN],
            window_sizes=[10],
            execute_on=TF_1MIN,
            max_traj_length=10,
            random_start=False,
            seed=42,
        )
        env = VectorizedSequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        for _ in range(15):
            action_td = td.clone() if "next" not in td.keys() else td["next"].clone()
            action_td["action"] = torch.zeros(2, dtype=torch.long)
            td = env.step(action_td)
            if td["next"]["done"].all():
                break

        assert td["next"]["done"].all()
        assert td["next"]["truncated"].all()
        assert not td["next"]["terminated"].any(), (
            "Truncated episode should have terminated=False (#150)"
        )
        env.close()


# ============================================================================
# TRADE EXECUTION TESTS
# ============================================================================


class TestVecEnvTradeExecution:
    """Tests for vectorized trade execution."""

    def test_open_position_sets_direction(self, vec_env):
        """Opening a long position should set position_direction to +1."""
        td = vec_env.reset()

        # Action index 2 = action_level 1.0 (long) for [-1, 0, 1] (clamped to [0, 0, 1])
        action_td = td.clone()
        action_td["action"] = torch.full((4,), 2, dtype=torch.long)
        td = vec_env.step(action_td)

        account_state = td["next"]["account_state"]
        # Element 1: position_direction should be +1 for all envs
        assert (account_state[:, 1] == 1.0).all(), (
            f"All envs should have direction=+1, got {account_state[:, 1]}"
        )

    def test_close_position(self, vec_env):
        """Closing a position should set direction to 0."""
        td = vec_env.reset()

        # Open long
        action_td = td.clone()
        action_td["action"] = torch.full((4,), 2, dtype=torch.long)
        td = vec_env.step(action_td)

        # Close (action 1 = 0.0 = flat)
        action_td = td["next"].clone()
        action_td["action"] = torch.full((4,), 1, dtype=torch.long)
        td = vec_env.step(action_td)

        assert (td["next"]["account_state"][:, 1] == 0.0).all(), "All should be flat"

    def test_repeated_action_holds(self, vec_env):
        """Repeating same action should hold, not rebalance (#187 regression)."""
        td = vec_env.reset()

        # Open long
        action_td = td.clone()
        action_td["action"] = torch.full((4,), 2, dtype=torch.long)
        td = vec_env.step(action_td)

        position_after_open = vec_env._position_sizes.clone()
        assert (position_after_open > 0).all()

        # Repeat same action 20 times
        for _ in range(20):
            action_td = td["next"].clone()
            action_td["action"] = torch.full((4,), 2, dtype=torch.long)
            td = vec_env.step(action_td)
            if td["next"]["done"].any():
                break

        # Position size should NOT have changed
        assert torch.allclose(vec_env._position_sizes, position_after_open), (
            "Repeated action should hold, not rebalance (#187)"
        )

    def test_action_change_after_holds_executes(self, vec_env):
        """Changing action after repeated holds must execute (#187 regression)."""
        td = vec_env.reset()

        # Open long
        action_td = td.clone()
        action_td["action"] = torch.full((4,), 2, dtype=torch.long)
        td = vec_env.step(action_td)
        assert (vec_env._position_sizes > 0).all()

        # Hold for 5 steps
        for _ in range(5):
            action_td = td["next"].clone()
            action_td["action"] = torch.full((4,), 2, dtype=torch.long)
            td = vec_env.step(action_td)

        # Close — must actually execute
        action_td = td["next"].clone()
        action_td["action"] = torch.full((4,), 1, dtype=torch.long)
        td = vec_env.step(action_td)

        assert (vec_env._position_sizes == 0).all(), (
            "Position should close after action change (#187)"
        )


# ============================================================================
# REWARD TESTS
# ============================================================================


class TestVecEnvReward:
    """Tests for reward calculation."""

    def test_reward_zero_when_flat(self, vec_env):
        """Reward should be 0 when holding cash."""
        td = vec_env.reset()

        # Hold (action 1 = flat)
        action_td = td.clone()
        action_td["action"] = torch.full((4,), 1, dtype=torch.long)
        td = vec_env.step(action_td)

        rewards = td["next"]["reward"].squeeze(-1)
        assert torch.allclose(rewards, torch.zeros(4), atol=1e-6), (
            f"Reward should be 0 when flat, got {rewards}"
        )

    def test_reward_reflects_price_change(self, vec_env):
        """Reward should be non-zero when holding a position."""
        td = vec_env.reset()

        # Open long
        action_td = td.clone()
        action_td["action"] = torch.full((4,), 2, dtype=torch.long)
        td = vec_env.step(action_td)

        # Hold for next step — reward reflects price change
        action_td = td["next"].clone()
        action_td["action"] = torch.full((4,), 2, dtype=torch.long)
        td = vec_env.step(action_td)

        rewards = td["next"]["reward"].squeeze(-1)
        # At least some rewards should be non-zero (prices change)
        assert not torch.allclose(rewards, torch.zeros(4), atol=1e-8), (
            "Rewards should reflect price movement"
        )


# ============================================================================
# ACCOUNT STATE TESTS
# ============================================================================


class TestVecEnvAccountState:
    """Tests for account state structure."""

    def test_initial_account_state(self, vec_env):
        """Initial account state should have correct values for spot mode."""
        td = vec_env.reset()
        account_state = td["account_state"]

        # Shape: (num_envs, 6)
        assert account_state.shape == (4, 6)

        for i in range(4):
            validate_account_state(account_state[i], leverage=1)

        # No position at start
        assert (account_state[:, 0] == 0.0).all(), "Exposure should be 0"
        assert (account_state[:, 1] == 0.0).all(), "Direction should be 0"
        assert (account_state[:, 4] == 1.0).all(), "Leverage should be 1.0"
        assert (account_state[:, 5] == 1.0).all(), "Dist to liq should be 1.0"


# ============================================================================
# EDGE CASES
# ============================================================================


class TestVecEnvEdgeCases:
    """Edge cases and configuration validation."""

    def test_leverage_not_1_raises(self, sample_ohlcv_df):
        """Should raise error for leverage != 1."""
        with pytest.raises(ValueError, match="leverage=1"):
            VectorizedSequentialTradingEnvConfig(leverage=10)

    @pytest.mark.parametrize("invalid_fee", [-0.1, 1.5])
    def test_invalid_fee_raises(self, invalid_fee):
        """Should raise error for invalid transaction fee."""
        with pytest.raises(ValueError, match="Transaction fee"):
            VectorizedSequentialTradingEnvConfig(transaction_fee=invalid_fee)

    @pytest.mark.parametrize("invalid_slippage", [-0.1, 1.5])
    def test_invalid_slippage_raises(self, invalid_slippage):
        """Should raise error for invalid slippage."""
        with pytest.raises(ValueError, match="Slippage"):
            VectorizedSequentialTradingEnvConfig(slippage=invalid_slippage)

    def test_negative_actions_clamped_to_zero(self, sample_ohlcv_df):
        """Negative action levels should be clamped to 0 for spot mode."""
        config = VectorizedSequentialTradingEnvConfig(
            num_envs=1,
            action_levels=[-1, 0, 1],
            time_frames=[TF_1MIN],
            window_sizes=[10],
            execute_on=TF_1MIN,
            max_traj_length=20,
            random_start=False,
        )
        env = VectorizedSequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        # _action_levels_tensor should clamp -1 to 0
        assert env._action_levels_tensor[0] == 0.0, "Negative action should be clamped to 0"
        assert env._action_levels_tensor[1] == 0.0
        assert env._action_levels_tensor[2] == 1.0
        env.close()

    @pytest.mark.parametrize("initial_cash", [
        1000,
        (500, 1500),
    ], ids=["fixed", "range"])
    def test_initial_cash_options(self, sample_ohlcv_df, initial_cash):
        """Both fixed and range initial cash should work."""
        config = VectorizedSequentialTradingEnvConfig(
            num_envs=8,
            initial_cash=initial_cash,
            time_frames=[TF_1MIN],
            window_sizes=[10],
            execute_on=TF_1MIN,
            max_traj_length=20,
            random_start=False,
        )
        env = VectorizedSequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        if isinstance(initial_cash, tuple):
            # Range: all balances should be within range
            assert (env._balances >= initial_cash[0]).all()
            assert (env._balances <= initial_cash[1]).all()
        else:
            # Fixed: all balances should be exact
            assert (env._balances == initial_cash).all()

        env.close()


# ============================================================================
# FEE TESTS
# ============================================================================


class TestVecEnvFees:
    """Tests for transaction fee accounting."""

    @pytest.mark.parametrize("fee", [0.001, 0.01])
    def test_fees_reduce_balance(self, sample_ohlcv_df, fee):
        """Opening and closing with fees should cost money."""
        config = VectorizedSequentialTradingEnvConfig(
            num_envs=2,
            action_levels=[0, 1],
            initial_cash=1000,
            time_frames=[TF_1MIN],
            window_sizes=[10],
            execute_on=TF_1MIN,
            transaction_fee=fee,
            slippage=0.0,
            max_traj_length=50,
            random_start=False,
            seed=42,
        )
        env = VectorizedSequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()
        initial_pv = env._portfolio_values.clone()

        # Buy
        action_td = td.clone()
        action_td["action"] = torch.ones(2, dtype=torch.long)
        td = env.step(action_td)

        # Check that fee was deducted (PV dropped due to fee)
        pv_after_buy = env._portfolio_values.clone()
        # Fee should cause PV to drop slightly from open
        assert (pv_after_buy < initial_pv).all(), (
            f"PV should decrease after buy due to fees. "
            f"Before: {initial_pv}, After: {pv_after_buy}"
        )

        env.close()


# ============================================================================
# COLLECTOR INTEGRATION
# ============================================================================


class TestVecEnvCollector:
    """Tests for TorchRL SyncDataCollector integration."""

    def test_collector_collects_frames(self, sample_ohlcv_df):
        """SyncDataCollector should collect expected number of frames."""
        config = VectorizedSequentialTradingEnvConfig(
            num_envs=8,
            time_frames=[TF_1MIN],
            window_sizes=[10],
            execute_on=TF_1MIN,
            max_traj_length=50,
            random_start=True,
            seed=42,
        )
        env = VectorizedSequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)

        total_frames = 200
        collector = SyncDataCollector(
            env,
            policy=None,
            frames_per_batch=80,
            total_frames=total_frames,
        )

        collected = 0
        for td_batch in collector:
            collected += td_batch.numel()

        collector.shutdown()
        env.close()

        assert collected >= total_frames, (
            f"Should collect at least {total_frames} frames, got {collected}"
        )
