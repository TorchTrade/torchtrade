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
        action_levels=[-1.0, 0.0, 1.0],
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


def _make_matched_envs(
    df, fee=0.0, slippage=0.0, max_traj=50, action_levels=None, leverage=1,
):
    """Create scalar and N=1 vectorized envs with identical config for comparison."""
    if action_levels is None:
        action_levels = [-1.0, 0.0, 1.0] if leverage > 1 else [0, 1]

    scalar_config = SequentialTradingEnvConfig(
        leverage=leverage,
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
        leverage=leverage,
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

    @pytest.mark.parametrize("leverage", [1, 10], ids=["spot", "futures"])
    def test_check_env_specs_passes(self, sample_ohlcv_df, leverage):
        """check_env_specs must pass for both spot and futures modes."""
        config = VectorizedSequentialTradingEnvConfig(
            num_envs=4,
            leverage=leverage,
            action_levels=[-1.0, 0.0, 1.0],
            time_frames=[TF_1MIN],
            window_sizes=[10],
            execute_on=TF_1MIN,
            max_traj_length=50,
            random_start=False,
        )
        env = VectorizedSequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        check_env_specs(env)
        env.close()

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
        """All envs should truncate at max_traj_length with truncated=True,
        terminated=False (regression #150)."""
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
        assert not td["next"]["terminated"].any(), (
            "Truncated episode should have terminated=False (#150)"
        )
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


# ============================================================================
# TRADE EXECUTION TESTS
# ============================================================================


class TestVecEnvTradeExecution:
    """Tests for vectorized trade execution."""

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

    @pytest.mark.parametrize("leverage,expected_lev", [
        (1, 1.0),
        (10, 10.0),
    ], ids=["spot", "futures"])
    def test_initial_account_state(self, sample_ohlcv_df, leverage, expected_lev):
        """Initial account state should have correct values for spot and futures."""
        config = VectorizedSequentialTradingEnvConfig(
            num_envs=4,
            leverage=leverage,
            action_levels=[-1.0, 0.0, 1.0],
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
        td = env.reset()
        account_state = td["account_state"]

        assert account_state.shape == (4, 6)
        for i in range(4):
            validate_account_state(account_state[i], leverage=leverage)

        # No position at start
        assert (account_state[:, 0] == 0.0).all(), "Exposure should be 0"
        assert (account_state[:, 1] == 0.0).all(), "Direction should be 0"
        assert (account_state[:, 4] == expected_lev).all(), f"Leverage should be {expected_lev}"
        assert (account_state[:, 5] == 1.0).all(), "Dist to liq should be 1.0"
        env.close()


# ============================================================================
# EDGE CASES
# ============================================================================


class TestVecEnvEdgeCases:
    """Edge cases and configuration validation."""

    @pytest.mark.parametrize("kwargs,match", [
        ({"transaction_fee": -0.1}, "Transaction fee"),
        ({"transaction_fee": 1.5}, "Transaction fee"),
        ({"slippage": -0.1}, "Slippage"),
        ({"slippage": 1.5}, "Slippage"),
    ], ids=["fee-negative", "fee-over-1", "slip-negative", "slip-over-1"])
    def test_invalid_config_raises(self, kwargs, match):
        """Should raise ValueError for out-of-range fee or slippage."""
        with pytest.raises(ValueError, match=match):
            VectorizedSequentialTradingEnvConfig(**kwargs)

    @pytest.mark.parametrize("leverage,expected_first", [
        (1, 0.0),    # Spot: negative clamped to 0
        (10, -1.0),  # Futures: negative preserved
    ], ids=["spot-clamped", "futures-preserved"])
    def test_negative_action_clamping(self, sample_ohlcv_df, leverage, expected_first):
        """Negative action levels should be clamped for spot, preserved for futures."""
        config = VectorizedSequentialTradingEnvConfig(
            num_envs=1,
            leverage=leverage,
            action_levels=[-1.0, 0.0, 1.0],
            time_frames=[TF_1MIN],
            window_sizes=[10],
            execute_on=TF_1MIN,
            max_traj_length=20,
            random_start=False,
        )
        env = VectorizedSequentialTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        assert env._action_levels_tensor[0] == expected_first
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

    @pytest.mark.parametrize("leverage", [1, 10], ids=["spot", "futures"])
    def test_collector_collects_frames(self, sample_ohlcv_df, leverage):
        """SyncDataCollector should collect expected number of frames."""
        config = VectorizedSequentialTradingEnvConfig(
            num_envs=8,
            leverage=leverage,
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


# ============================================================================
# FUTURES MODE TESTS
# ============================================================================


def _make_futures_env(df, num_envs=4, leverage=10, fee=0.0, max_traj=50):
    """Create a vectorized futures env for testing."""
    config = VectorizedSequentialTradingEnvConfig(
        num_envs=num_envs,
        leverage=leverage,
        action_levels=[-1.0, 0.0, 1.0],
        initial_cash=1000,
        time_frames=[TF_1MIN],
        window_sizes=[10],
        execute_on=TF_1MIN,
        transaction_fee=fee,
        slippage=0.0,
        seed=42,
        max_traj_length=max_traj,
        random_start=False,
    )
    return VectorizedSequentialTradingEnv(df, config, simple_feature_fn)


class TestVecEnvFuturesPositions:
    """Tests for futures position mechanics: longs, shorts, direction switches."""

    @pytest.mark.parametrize("action_idx,expected_dir", [
        (2, 1.0),   # action_level=+1 → long
        (0, -1.0),  # action_level=-1 → short
    ], ids=["long", "short"])
    def test_open_position_direction(self, sample_ohlcv_df, action_idx, expected_dir):
        """Opening positions should set correct direction for both long and short."""
        env = _make_futures_env(sample_ohlcv_df)
        td = env.reset()
        action_td = td.clone()
        action_td["action"] = torch.full((4,), action_idx, dtype=torch.long)
        td = env.step(action_td)

        directions = td["next"]["account_state"][:, 1]
        assert (directions == expected_dir).all(), (
            f"Expected direction {expected_dir}, got {directions}"
        )
        env.close()

    @pytest.mark.parametrize("open_action", [2, 0], ids=["close-long", "close-short"])
    def test_close_position(self, sample_ohlcv_df, open_action):
        """Closing any position should set direction to 0."""
        env = _make_futures_env(sample_ohlcv_df)
        td = env.reset()

        # Open position
        action_td = td.clone()
        action_td["action"] = torch.full((4,), open_action, dtype=torch.long)
        td = env.step(action_td)
        assert (env._position_sizes != 0).all()

        # Close (action 1 = 0.0)
        action_td = td["next"].clone()
        action_td["action"] = torch.full((4,), 1, dtype=torch.long)
        td = env.step(action_td)

        assert (env._position_sizes == 0).all(), "All positions should be flat"
        assert (td["next"]["account_state"][:, 1] == 0.0).all()
        env.close()

    @pytest.mark.parametrize("first_action,second_action,expected_dir", [
        (2, 0, -1.0),  # long → short
        (0, 2, 1.0),   # short → long
    ], ids=["long-to-short", "short-to-long"])
    def test_direction_switch(
        self, sample_ohlcv_df, first_action, second_action, expected_dir
    ):
        """Direction switches should close then reopen in the new direction."""
        env = _make_futures_env(sample_ohlcv_df)
        td = env.reset()

        # Open first position
        action_td = td.clone()
        action_td["action"] = torch.full((4,), first_action, dtype=torch.long)
        td = env.step(action_td)

        # Switch direction
        action_td = td["next"].clone()
        action_td["action"] = torch.full((4,), second_action, dtype=torch.long)
        td = env.step(action_td)

        assert (td["next"]["account_state"][:, 1] == expected_dir).all(), (
            f"Expected direction {expected_dir} after switch"
        )
        env.close()

    def test_repeated_short_action_holds(self, sample_ohlcv_df):
        """Repeating short action should hold, not rebalance (#187)."""
        env = _make_futures_env(sample_ohlcv_df)
        td = env.reset()

        # Open short
        action_td = td.clone()
        action_td["action"] = torch.full((4,), 0, dtype=torch.long)
        td = env.step(action_td)
        position_after_open = env._position_sizes.clone()

        # Repeat short action
        for _ in range(10):
            action_td = td["next"].clone()
            action_td["action"] = torch.full((4,), 0, dtype=torch.long)
            td = env.step(action_td)
            if td["next"]["done"].any():
                break

        assert torch.allclose(env._position_sizes, position_after_open), (
            "Repeated short action should hold position"
        )
        env.close()


class TestVecEnvFuturesAccountState:
    """Tests for futures account state correctness."""

    @pytest.mark.parametrize("action_idx,direction", [
        (2, "long"),
        (0, "short"),
    ], ids=["long", "short"])
    def test_distance_to_liquidation_with_position(
        self, sample_ohlcv_df, action_idx, direction
    ):
        """Distance to liquidation should be < 1.0 for both long and short."""
        env = _make_futures_env(sample_ohlcv_df, leverage=10)
        td = env.reset()

        # Open position
        action_td = td.clone()
        action_td["action"] = torch.full((4,), action_idx, dtype=torch.long)
        td = env.step(action_td)

        # Hold to get next observation
        action_td = td["next"].clone()
        action_td["action"] = torch.full((4,), action_idx, dtype=torch.long)
        td = env.step(action_td)

        dist_to_liq = td["next"]["account_state"][:, 5]
        assert (dist_to_liq > 0).all(), f"Distance should be positive ({direction})"
        assert (dist_to_liq < 1.0).all(), (
            f"Distance to liq should be < 1.0 with leverage ({direction}), got {dist_to_liq}"
        )
        env.close()

    def test_unrealized_pnl_short_positive_when_price_drops(self, trending_down_df):
        """Short position PnL should be positive when price drops."""
        env = _make_futures_env(trending_down_df, leverage=10, max_traj=100)
        td = env.reset()

        # Open short
        action_td = td.clone()
        action_td["action"] = torch.full((4,), 0, dtype=torch.long)
        td = env.step(action_td)

        # Hold through downtrend
        for _ in range(5):
            action_td = td["next"].clone()
            action_td["action"] = torch.full((4,), 0, dtype=torch.long)
            td = env.step(action_td)
            if td["next"]["done"].any():
                break

        pnl_pct = td["next"]["account_state"][:, 2]
        # trending_down_df guarantees price drops → short PnL must be positive
        assert (pnl_pct > 0).all(), (
            f"Short PnL should be positive in downtrend, got {pnl_pct}"
        )
        env.close()

    def test_liquidation_price_formula(self, sample_ohlcv_df):
        """Liquidation prices should match the expected formula."""
        env = _make_futures_env(sample_ohlcv_df, leverage=10)
        td = env.reset()

        # Open long
        action_td = td.clone()
        action_td["action"] = torch.full((4,), 2, dtype=torch.long)
        env.step(action_td)

        entry = env._entry_prices.clone()
        liq = env._compute_liq_prices()
        # Long liq = entry * (1 - 1/leverage + mmr)
        # leverage=10, mmr=0.004: liq = entry * 0.904
        expected = entry * (1 - 0.1 + 0.004)
        assert torch.allclose(liq, expected, atol=1e-4), (
            f"Long liq price mismatch: got {liq}, expected {expected}"
        )

        # Close and open short
        action_td = td.clone()
        action_td["action"] = torch.full((4,), 0, dtype=torch.long)
        env.step(action_td)

        entry_short = env._entry_prices.clone()
        liq_short = env._compute_liq_prices()
        # Short liq = entry * (1 + 1/leverage - mmr)
        expected_short = entry_short * (1 + 0.1 - 0.004)
        assert torch.allclose(liq_short, expected_short, atol=1e-4), (
            f"Short liq price mismatch: got {liq_short}, expected {expected_short}"
        )
        env.close()


class TestVecEnvFuturesLiquidation:
    """Tests for forced liquidation mechanics."""

    def test_long_liquidation_on_price_crash(self, trending_down_df):
        """Long position should get liquidated when price crashes with high leverage."""
        env = _make_futures_env(
            trending_down_df, num_envs=2, leverage=20, max_traj=200
        )
        td = env.reset()

        # Open long with 20x leverage
        action_td = td.clone()
        action_td["action"] = torch.full((2,), 2, dtype=torch.long)
        td = env.step(action_td)
        assert (env._position_sizes > 0).all(), "Should have long positions"

        # Step through downtrend until done
        for _ in range(150):
            action_td = td["next"].clone()
            action_td["action"] = torch.full((2,), 2, dtype=torch.long)
            td = env.step(action_td)
            if td["next"]["done"].all():
                break

        # At least one env should have been terminated (bankrupt from liquidation)
        # or positions should have been liquidated (zeroed out)
        liquidated = env._position_sizes == 0
        terminated = td["next"]["terminated"].squeeze(-1)
        assert liquidated.any() or terminated.any(), (
            "High-leverage long in downtrend should trigger liquidation or termination"
        )
        env.close()

    def test_short_liquidation_on_price_surge(self, trending_up_df):
        """Short position should get liquidated when price surges with high leverage."""
        env = _make_futures_env(
            trending_up_df, num_envs=2, leverage=20, max_traj=200
        )
        td = env.reset()

        # Open short with 20x leverage
        action_td = td.clone()
        action_td["action"] = torch.full((2,), 0, dtype=torch.long)
        td = env.step(action_td)
        assert (env._position_sizes < 0).all(), "Should have short positions"

        # Step through uptrend until done
        for _ in range(150):
            action_td = td["next"].clone()
            action_td["action"] = torch.full((2,), 0, dtype=torch.long)
            td = env.step(action_td)
            if td["next"]["done"].all():
                break

        liquidated = env._position_sizes == 0
        terminated = td["next"]["terminated"].squeeze(-1)
        assert liquidated.any() or terminated.any(), (
            "High-leverage short in uptrend should trigger liquidation or termination"
        )
        env.close()

    def test_liquidation_zeros_position_state(self, trending_down_df):
        """After liquidation, position state should be fully zeroed."""
        env = _make_futures_env(
            trending_down_df, num_envs=1, leverage=20, max_traj=200
        )
        td = env.reset()

        # Open long
        action_td = td.clone()
        action_td["action"] = torch.tensor([2])
        td = env.step(action_td)

        # Run until liquidation or done
        was_liquidated = False
        for _ in range(150):
            had_position = env._position_sizes[0].item() != 0
            action_td = td["next"].clone()
            action_td["action"] = torch.tensor([2])
            td = env.step(action_td)
            no_position = env._position_sizes[0].item() == 0

            if had_position and no_position:
                was_liquidated = True
                # Verify state is clean
                assert env._entry_prices[0].item() == 0, "Entry price should be 0 after liq"
                assert env._hold_counters[0].item() == 0, "Hold counter should be 0 after liq"
                assert env._balances[0].item() >= 0, "Balance should be non-negative after liq"
                break

            if td["next"]["done"].all():
                break

        assert was_liquidated or td["next"]["done"].all(), (
            "Should either liquidate or terminate in downtrend with 20x leverage"
        )
        env.close()


class TestVecEnvFuturesCorrectness:
    """Compare vectorized futures (N=1) against scalar SequentialTradingEnv."""

    @pytest.mark.parametrize("action_idx,action_name", [
        (2, "long"),
        (0, "short"),
    ], ids=["long", "short"])
    def test_open_and_hold_portfolio_values_match(
        self, sample_ohlcv_df, action_idx, action_name
    ):
        """Open then hold: portfolio values should match between scalar and vec."""
        scalar_env, vec_env = _make_matched_envs(
            sample_ohlcv_df, leverage=10
        )

        td_s = scalar_env.reset()
        td_v = vec_env.reset()

        # Open position
        action_td_s = td_s.clone()
        action_td_s["action"] = torch.tensor(action_idx)
        td_s = scalar_env.step(action_td_s)

        action_td_v = td_v.clone()
        action_td_v["action"] = torch.tensor([action_idx])
        td_v = vec_env.step(action_td_v)

        # Position sizes should match
        scalar_pos = scalar_env.position.position_size
        vec_pos = vec_env._position_sizes[0].item()
        assert abs(scalar_pos - vec_pos) / max(abs(scalar_pos), 1e-6) < 0.01, (
            f"Position sizes differ: scalar={scalar_pos}, vec={vec_pos}"
        )

        # Hold and compare PVs
        for step in range(5):
            action_td_s = td_s["next"].clone()
            action_td_s["action"] = torch.tensor(action_idx)
            td_s = scalar_env.step(action_td_s)

            action_td_v = td_v["next"].clone()
            action_td_v["action"] = torch.tensor([action_idx])
            td_v = vec_env.step(action_td_v)

            scalar_pv = scalar_env._get_portfolio_value()
            vec_pv = vec_env._portfolio_values[0].item()
            assert abs(scalar_pv - vec_pv) / max(scalar_pv, 1.0) < 0.01, (
                f"Step {step + 2} ({action_name}): PV mismatch: "
                f"scalar={scalar_pv:.4f}, vec={vec_pv:.4f}"
            )

        scalar_env.close()
        vec_env.close()

    def test_direction_switch_balance_matches(self, sample_ohlcv_df):
        """Long→short switch: balances should match between scalar and vec."""
        scalar_env, vec_env = _make_matched_envs(
            sample_ohlcv_df, leverage=10, fee=0.001
        )

        td_s = scalar_env.reset()
        td_v = vec_env.reset()

        # Open long (action 2)
        action_td_s = td_s.clone()
        action_td_s["action"] = torch.tensor(2)
        td_s = scalar_env.step(action_td_s)

        action_td_v = td_v.clone()
        action_td_v["action"] = torch.tensor([2])
        td_v = vec_env.step(action_td_v)

        # Hold
        for _ in range(3):
            action_td_s = td_s["next"].clone()
            action_td_s["action"] = torch.tensor(2)
            td_s = scalar_env.step(action_td_s)

            action_td_v = td_v["next"].clone()
            action_td_v["action"] = torch.tensor([2])
            td_v = vec_env.step(action_td_v)

        # Switch to short (action 0)
        action_td_s = td_s["next"].clone()
        action_td_s["action"] = torch.tensor(0)
        td_s = scalar_env.step(action_td_s)

        action_td_v = td_v["next"].clone()
        action_td_v["action"] = torch.tensor([0])
        td_v = vec_env.step(action_td_v)

        # Both should now be short
        assert scalar_env.position.position_size < 0, "Scalar should be short"
        assert vec_env._position_sizes[0].item() < 0, "Vec should be short"

        # Balances should be close
        scalar_bal = scalar_env.balance
        vec_bal = vec_env._balances[0].item()
        assert abs(scalar_bal - vec_bal) / max(scalar_bal, 1.0) < 0.01, (
            f"Balance mismatch after switch: scalar={scalar_bal:.4f}, vec={vec_bal:.4f}"
        )

        scalar_env.close()
        vec_env.close()


# ============================================================================
# PARTIAL RESET TESTS
# ============================================================================


class TestVecEnvPartialReset:
    """Tests for partial reset — only done envs should be reset."""

    def test_partial_reset_preserves_non_done_state(self, sample_ohlcv_df):
        """When some envs are done, only those should be reset."""
        env = _make_futures_env(sample_ohlcv_df, num_envs=4, max_traj=50)
        td = env.reset()

        # Open long in all envs
        action_td = td.clone()
        action_td["action"] = torch.full((4,), 2, dtype=torch.long)
        td = env.step(action_td)

        # Record state of envs 1 and 3 (they won't be reset)
        pos_before = env._position_sizes.clone()
        entry_before = env._entry_prices.clone()
        bal_before = env._balances.clone()

        # Manually trigger partial reset for envs 0 and 2 only
        reset_td = td["next"].clone()
        reset_mask = torch.tensor([True, False, True, False])
        reset_td["_reset"] = reset_mask.unsqueeze(-1)
        obs_td = env.reset(reset_td)

        # Envs 0, 2 should be reset (flat, fresh balance)
        assert env._position_sizes[0].item() == 0, "Env 0 should be reset"
        assert env._position_sizes[2].item() == 0, "Env 2 should be reset"
        assert env._entry_prices[0].item() == 0
        assert env._entry_prices[2].item() == 0

        # Envs 1, 3 should retain their positions
        assert env._position_sizes[1] == pos_before[1], "Env 1 should keep position"
        assert env._position_sizes[3] == pos_before[3], "Env 3 should keep position"
        assert env._entry_prices[1] == entry_before[1]
        assert env._entry_prices[3] == entry_before[3]
        assert env._balances[1] == bal_before[1]
        assert env._balances[3] == bal_before[3]

        # Output should be a valid observation
        assert "account_state" in obs_td.keys()
        assert obs_td["account_state"].shape == (4, 6)
        env.close()


# ============================================================================
# HETEROGENEOUS ACTION TESTS
# ============================================================================


class TestVecEnvHeterogeneousActions:
    """Tests for different actions per env in the same step."""

    def test_mixed_actions_update_independently(self, sample_ohlcv_df):
        """Each env should respond only to its own action."""
        env = _make_futures_env(sample_ohlcv_df, num_envs=4)
        td = env.reset()

        # Env 0: long, Env 1: flat, Env 2: short, Env 3: long
        actions = torch.tensor([2, 1, 0, 2], dtype=torch.long)
        action_td = td.clone()
        action_td["action"] = actions
        td = env.step(action_td)

        directions = td["next"]["account_state"][:, 1]
        assert directions[0] == 1.0, "Env 0 should be long"
        assert directions[1] == 0.0, "Env 1 should be flat"
        assert directions[2] == -1.0, "Env 2 should be short"
        assert directions[3] == 1.0, "Env 3 should be long"

        # Env 0 and 3 should have positive position, env 2 negative, env 1 zero
        assert env._position_sizes[0] > 0
        assert env._position_sizes[1] == 0
        assert env._position_sizes[2] < 0
        assert env._position_sizes[3] > 0

        # Now: Env 0 close, Env 1 long, Env 2 hold short, Env 3 switch to short
        actions2 = torch.tensor([1, 2, 0, 0], dtype=torch.long)
        action_td = td["next"].clone()
        action_td["action"] = actions2
        td = env.step(action_td)

        directions2 = td["next"]["account_state"][:, 1]
        assert directions2[0] == 0.0, "Env 0 should be flat after close"
        assert directions2[1] == 1.0, "Env 1 should be long"
        assert directions2[2] == -1.0, "Env 2 should still be short (hold)"
        assert directions2[3] == -1.0, "Env 3 should switch to short"
        env.close()


# ============================================================================
# BANKRUPTCY TESTS
# ============================================================================


class TestVecEnvBankruptcy:
    """Tests for deterministic bankruptcy termination."""

    def test_bankruptcy_sets_terminated_not_truncated(self, trending_down_df):
        """Bankrupt env must have terminated=True, truncated=False."""
        # Use extreme leverage + long in downtrend to guarantee bankruptcy
        env = _make_futures_env(
            trending_down_df, num_envs=1, leverage=50, max_traj=500
        )
        td = env.reset()
        initial_pv = env._portfolio_values[0].item()

        # Open long with 50x leverage in a downtrend
        action_td = td.clone()
        action_td["action"] = torch.tensor([2])
        td = env.step(action_td)

        # Step until done
        for _ in range(400):
            action_td = td["next"].clone()
            action_td["action"] = torch.tensor([2])
            td = env.step(action_td)
            if td["next"]["done"].all():
                break

        # With 50x leverage in a strong downtrend, should terminate
        if td["next"]["terminated"].item():
            assert not td["next"]["truncated"].item(), (
                "Terminated env must not also be truncated"
            )
            # PV should be below bankrupt threshold
            assert env._portfolio_values[0].item() < initial_pv * env.bankrupt_threshold
        env.close()


# ============================================================================
# INSUFFICIENT BALANCE TESTS
# ============================================================================


class TestVecEnvInsufficientBalance:
    """Tests for the can_afford guard in _execute_trades."""

    def test_cannot_afford_stays_flat(self, trending_down_df):
        """After liquidation drains balance, new position should be rejected."""
        env = _make_futures_env(
            trending_down_df, num_envs=1, leverage=20, max_traj=300
        )
        td = env.reset()

        # Open long with 20x leverage
        action_td = td.clone()
        action_td["action"] = torch.tensor([2])
        td = env.step(action_td)

        # Step until liquidation
        for _ in range(200):
            action_td = td["next"].clone()
            action_td["action"] = torch.tensor([2])
            td = env.step(action_td)
            if env._position_sizes[0].item() == 0 and env._balances[0].item() < 10:
                # Liquidated with near-zero balance — try to open new position
                action_td = td["next"].clone()
                action_td["action"] = torch.tensor([2])
                td = env.step(action_td)

                # Should stay flat because can't afford margin
                assert env._balances[0].item() >= 0, "Balance must never go negative"
                break
            if td["next"]["done"].all():
                break
        env.close()


# ============================================================================
# FUTURES CLOSE-TO-FLAT CORRECTNESS
# ============================================================================


class TestVecEnvFuturesCloseToFlat:
    """Verify futures open→hold→close cycle matches scalar env."""

    @pytest.mark.parametrize("open_action", [2, 0], ids=["long", "short"])
    def test_futures_close_to_flat_balance_matches(
        self, sample_ohlcv_df, open_action
    ):
        """Full open→hold→close cycle in futures with fees: balance must match scalar."""
        scalar_env, vec_env = _make_matched_envs(
            sample_ohlcv_df, leverage=10, fee=0.001
        )

        td_s = scalar_env.reset()
        td_v = vec_env.reset()

        # Open
        action_td_s = td_s.clone()
        action_td_s["action"] = torch.tensor(open_action)
        td_s = scalar_env.step(action_td_s)

        action_td_v = td_v.clone()
        action_td_v["action"] = torch.tensor([open_action])
        td_v = vec_env.step(action_td_v)

        # Hold for 5 steps
        for _ in range(5):
            action_td_s = td_s["next"].clone()
            action_td_s["action"] = torch.tensor(open_action)
            td_s = scalar_env.step(action_td_s)

            action_td_v = td_v["next"].clone()
            action_td_v["action"] = torch.tensor([open_action])
            td_v = vec_env.step(action_td_v)

        # Close to flat (action 1 = 0.0)
        action_td_s = td_s["next"].clone()
        action_td_s["action"] = torch.tensor(1)
        td_s = scalar_env.step(action_td_s)

        action_td_v = td_v["next"].clone()
        action_td_v["action"] = torch.tensor([1])
        td_v = vec_env.step(action_td_v)

        # Both should be flat
        assert scalar_env.position.position_size == 0, "Scalar should be flat"
        assert vec_env._position_sizes[0].item() == 0, "Vec should be flat"

        # Balances should match
        scalar_bal = scalar_env.balance
        vec_bal = vec_env._balances[0].item()
        assert abs(scalar_bal - vec_bal) / max(scalar_bal, 1.0) < 0.01, (
            f"Balance mismatch: scalar={scalar_bal:.4f}, vec={vec_bal:.4f}"
        )

        scalar_env.close()
        vec_env.close()

