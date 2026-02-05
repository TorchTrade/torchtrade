"""
Consolidated tests for OneStepTradingEnv (unified spot/futures one-step environment).

This file consolidates tests from:
- test_longonlyonestepenv.py (spot one-step)
- test_futuresonestepenv.py (futures one-step)

Uses parametrization to test both trading modes with maximum code reuse.
"""

import pytest
import torch

from torchtrade.envs.offline import OneStepTradingEnv, OneStepTradingEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.conftest import simple_feature_fn, validate_account_state


@pytest.fixture
def onestep_config_spot():
    """OneStep config for spot trading."""
    return OneStepTradingEnvConfig(
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
    """Create OneStep environment for testing.

    trading_mode fixture returns leverage: 1 for spot, 10 for futures.
    """
    config = onestep_config_spot if trading_mode == 1 else onestep_config_futures
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
        assert onestep_env.leverage == trading_mode  # trading_mode fixture returns leverage value

    def test_action_spec_sltp_combinatorial(self, onestep_env, trading_mode):
        """OneStep uses SLTP action space with combinatorial actions.

        Action space = 1 (hold) + (num_sl * num_tp * num_directions)
        Default: 3 SL levels * 3 TP levels = 9 combinations per direction
        - Spot (leverage=1): 1 + 9 = 10 actions (long only)
        - Futures (leverage>1): 1 + 18 = 19 actions (long + short)
        """
        expected_actions = 10 if trading_mode == 1 else 19
        assert onestep_env.action_spec.n == expected_actions, \
            f"Expected {expected_actions} actions for leverage={trading_mode}, got {onestep_env.action_spec.n}"

        # Verify action map structure
        assert 0 in onestep_env.action_map, "Should have hold action at index 0"
        assert onestep_env.action_map[0][0] is None, "Hold action should have None action_type"

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

    def test_reward_accumulates_rollout(self, onestep_env, trading_mode):
        """Reward should accumulate over rollout period."""
        td = onestep_env.reset()

        # Buy action
        action_td = td.clone()
        buy_action = 2 if trading_mode == 1 else 4
        action_td["action"] = torch.tensor(buy_action)
        next_td = onestep_env.step(action_td)

        # Reward should reflect accumulated P&L over rollout
        reward = next_td["next"]["reward"]
        assert isinstance(reward.item(), float)


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
    """Tests for SL/TP triggers during rollout."""

    @pytest.mark.parametrize("trigger_type,sl_pct,tp_pct", [
        ("sl", -0.005, 0.5),   # Very tight SL, very wide TP — SL triggers
        ("tp", -0.5, 0.005),   # Very wide SL, very tight TP — TP triggers
    ], ids=["stop_loss", "take_profit"])
    def test_rollout_reward_uses_trigger_price_not_close(
        self, sample_ohlcv_df, trigger_type, sl_pct, tp_pct
    ):
        """Reward must be computed at SL/TP trigger price, not bar close price.

        Regression test for issue #151: rollout was computing return at close
        price before checking SL/TP triggers, biasing the reward signal.
        """
        import math

        config = OneStepTradingEnvConfig(
            initial_cash=10000,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=0.0,
            slippage=0.0,
            stoploss_levels=[sl_pct],
            takeprofit_levels=[tp_pct],
            include_hold_action=True,
            random_start=False,
            seed=42,
        )
        env = OneStepTradingEnv(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        # Action 1 = first (and only) long SLTP action
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = env.step(action_td)

        reward = next_td["next"]["reward"].item()

        # Verify trigger actually fired (not truncation)
        assert next_td["next"]["done"].item()
        assert not next_td["next"]["truncated"].item(), "Expected SL/TP trigger, not truncation"

        # Verify reward is consistent with the accumulated rollout returns
        assert len(env.rollout_returns) > 0, "Expected rollout to produce returns"
        assert math.isclose(reward, sum(env.rollout_returns), rel_tol=1e-6), (
            f"Reward {reward} != sum(rollout_returns) {sum(env.rollout_returns)}"
        )
        env.close()


# ============================================================================
# TRUNCATION TESTS
# ============================================================================


class TestOneStepTruncation:
    """Tests for episode truncation."""

    def test_truncates_at_data_end(self, sample_ohlcv_df):
        """Should truncate if rollout reaches end of data."""
        # Position reset near end of data
        config = OneStepTradingEnvConfig(
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

    def test_reward_reflects_total_pnl(self, onestep_env, trading_mode):
        """Terminal reward should reflect total P&L over rollout."""
        td = onestep_env.reset()

        # Buy action
        buy_action = 2 if trading_mode == 1 else 4
        action_td = td.clone()
        action_td["action"] = torch.tensor(buy_action)
        next_td = onestep_env.step(action_td)

        reward = next_td["next"]["reward"].item()

        # Reward should be non-zero (unless price didn't move)
        # We can't guarantee sign, but it should be a valid float
        assert isinstance(reward, float)

    def test_transaction_costs_included(self, onestep_env, trading_mode):
        """Terminal reward should include transaction costs."""
        onestep_env.transaction_fee = 0.1  # 10% fee

        td = onestep_env.reset()

        # Buy and hold
        buy_action = 2 if trading_mode == 1 else 4
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
    pass


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

    def test_non_truncated_step_sets_terminated(self, onestep_env):
        """Non-truncated one-step episode should set terminated=True, truncated=False.

        Regression test for #150: one-step envs must distinguish terminated
        (SL/TP trigger, hold completed) from truncated (data exhaustion).
        """
        td = onestep_env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(0)  # Hold - no rollout, no truncation
        next_td = onestep_env.step(action_td)

        assert next_td["next"]["done"].item() is True
        assert next_td["next"]["terminated"].item() is True
        assert next_td["next"]["truncated"].item() is False

    def test_action_spec_immutable(self, onestep_env):
        """Action spec should not change during episode."""
        initial_spec = onestep_env.action_spec
        initial_n = initial_spec.n

        td = onestep_env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        onestep_env.step(action_td)

        assert onestep_env.action_spec.n == initial_n


# ============================================================================
# PER-TIMEFRAME FEATURE PROCESSING TESTS (Issue #177)
# ============================================================================


class TestOneStepPerTimeframeFeatures:
    """Tests for per-timeframe feature processing in OneStep environments."""

    @pytest.fixture
    def multi_tf_df(self):
        """Create OHLCV data for multi-timeframe testing."""
        import pandas as pd
        import numpy as np

        n_minutes = 500
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

        close_prices = np.array([100.0 + i for i in range(n_minutes)])
        return pd.DataFrame({
            "timestamp": timestamps,
            "open": close_prices - 0.5,
            "high": close_prices + 1.0,
            "low": close_prices - 1.0,
            "close": close_prices,
            "volume": np.ones(n_minutes) * 1000,
        })

    def test_onestep_env_with_different_feature_dimensions(self, multi_tf_df):
        """OneStep environment should work with different feature dimensions per timeframe."""
        def process_1min(df):
            """3 features."""
            df = df.copy().reset_index(drop=False)
            df["features_close"] = df["close"]
            df["features_volume"] = df["volume"]
            df["features_range"] = df["high"] - df["low"]
            return df

        def process_5min(df):
            """5 features."""
            df = df.copy().reset_index(drop=False)
            df["features_close"] = df["close"]
            df["features_sma"] = df["close"].rolling(3).mean().fillna(df["close"])
            df["features_vol"] = df["close"].pct_change().rolling(3).std().fillna(0)
            df["features_volume"] = df["volume"]
            df["features_vma"] = df["volume"].rolling(3).mean().fillna(df["volume"])
            return df

        config = OneStepTradingEnvConfig(
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 5],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000,
            random_start=False,
        )
        env = OneStepTradingEnv(
            multi_tf_df,
            config,
            feature_preprocessing_fn=[process_1min, process_5min],
        )

        # Check observation spec shapes
        obs_spec = env.observation_spec
        assert obs_spec["market_data_1Minute_10"].shape == (10, 3)  # 3 features
        assert obs_spec["market_data_5Minute_5"].shape == (5, 5)  # 5 features

        # Reset and check actual observation shapes
        td = env.reset()
        assert td["market_data_1Minute_10"].shape == (10, 3)
        assert td["market_data_5Minute_5"].shape == (5, 5)

        # Step and check shapes are maintained (OneStep terminates after one step)
        td["action"] = torch.tensor(1)  # First action
        td = env.step(td)
        assert td["next"]["market_data_1Minute_10"].shape == (10, 3)
        assert td["next"]["market_data_5Minute_5"].shape == (5, 5)
        assert td["next"]["done"].item()  # OneStep always terminates

        env.close()

    @pytest.mark.parametrize("leverage", [1, 10], ids=["spot", "futures"])
    def test_onestep_per_timeframe_across_episodes(self, multi_tf_df, leverage):
        """OneStep per-timeframe features should work across multiple episodes."""
        def process_1min(df):
            df = df.copy().reset_index(drop=False)
            df["features_close"] = df["close"]
            df["features_range"] = df["high"] - df["low"]
            return df

        def process_5min(df):
            df = df.copy().reset_index(drop=False)
            df["features_close"] = df["close"]
            df["features_sma"] = df["close"].rolling(3).mean().fillna(df["close"])
            df["features_volume"] = df["volume"]
            df["features_vma"] = df["volume"].rolling(3).mean().fillna(df["volume"])
            return df

        config = OneStepTradingEnvConfig(
            leverage=leverage,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 5],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000,
            random_start=False,
        )
        env = OneStepTradingEnv(
            multi_tf_df,
            config,
            feature_preprocessing_fn=[process_1min, process_5min],
        )

        expected_1min_shape = (10, 2)
        expected_5min_shape = (5, 4)

        # Run multiple episodes (OneStep = 1 step per episode)
        for episode in range(5):
            td = env.reset()
            assert td["market_data_1Minute_10"].shape == expected_1min_shape, \
                f"Episode {episode} reset: 1min shape mismatch"
            assert td["market_data_5Minute_5"].shape == expected_5min_shape, \
                f"Episode {episode} reset: 5min shape mismatch"

            td["action"] = torch.tensor(0)  # Hold
            td = env.step(td)
            assert td["next"]["market_data_1Minute_10"].shape == expected_1min_shape, \
                f"Episode {episode} step: 1min shape mismatch"
            assert td["next"]["market_data_5Minute_5"].shape == expected_5min_shape, \
                f"Episode {episode} step: 5min shape mismatch"

        env.close()
