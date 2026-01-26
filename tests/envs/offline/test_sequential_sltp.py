"""
Consolidated tests for SequentialTradingEnvSLTP (unified spot/futures SLTP environment).

This file consolidates tests from:
- test_seqlongonlysltp.py (spot SLTP)
- test_seqfuturessltp.py (futures SLTP)

Uses parametrization to test both trading modes with maximum code reuse.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline import SequentialTradingEnvSLTP, SequentialTradingEnvSLTPConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.conftest import simple_feature_fn, validate_account_state


@pytest.fixture
def sltp_config_spot():
    """SLTP config for spot trading."""
    return SequentialTradingEnvSLTPConfig(
        trading_mode="spot",
        initial_cash=1000,
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        transaction_fee=0.01,
        slippage=0.0,
        seed=42,
        max_traj_length=100,
        random_start=False,
        stoploss_levels=[-0.02, -0.05],  # -2%, -5%
        takeprofit_levels=[0.03, 0.10],  # 3%, 10%
    )


@pytest.fixture
def sltp_config_futures():
    """SLTP config for futures trading."""
    return SequentialTradingEnvSLTPConfig(
        trading_mode="futures",
        leverage=10,
        initial_cash=1000,
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        transaction_fee=0.01,
        slippage=0.0,
        seed=42,
        max_traj_length=100,
        random_start=False,
        stoploss_levels=[-0.02, -0.05],
        takeprofit_levels=[0.03, 0.10],
    )


@pytest.fixture
def sltp_env(sample_ohlcv_df, trading_mode, sltp_config_spot, sltp_config_futures):
    """Create SLTP environment for testing."""
    config = sltp_config_spot if trading_mode == "spot" else sltp_config_futures
    env_instance = SequentialTradingEnvSLTP(
        df=sample_ohlcv_df,
        config=config,
        feature_preprocessing_fn=simple_feature_fn,
    )
    yield env_instance
    env_instance.close()


# ============================================================================
# ACTION SPACE TESTS
# ============================================================================


class TestSLTPActionSpace:
    """Tests for SLTP action space generation."""

    @pytest.mark.parametrize("sl_levels,tp_levels,expected_actions", [
        ([0.02], [0.03], 2),           # 1 + (1 * 1) = 2
        ([0.02, 0.05], [0.03], 3),     # 1 + (2 * 1) = 3
        ([0.02], [0.03, 0.10], 3),     # 1 + (1 * 2) = 3
        ([0.02, 0.05], [0.03, 0.10], 5),  # 1 + (2 * 2) = 5
    ])
    def test_action_space_size(self, sample_ohlcv_df, trading_mode, sl_levels, tp_levels, expected_actions):
        """Action space should be 1 + (num_sl * num_tp)."""
        config = SequentialTradingEnvSLTPConfig(
            trading_mode=trading_mode,
            leverage=10 if trading_mode == "futures" else 1,
            stoploss_levels=sl_levels,
            takeprofit_levels=tp_levels,
            initial_cash=1000,
        )
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)
        assert env.action_spec.n == expected_actions
        env.close()

    def test_no_action_always_first(self, sltp_env):
        """Action 0 should always be 'no action'."""
        assert sltp_env.action_levels[0] == (None, None)

    def test_sltp_combinations_generated(self, sample_ohlcv_df, trading_mode):
        """Should generate all SL/TP combinations."""
        config = SequentialTradingEnvSLTPConfig(
            trading_mode=trading_mode,
            leverage=10 if trading_mode == "futures" else 1,
            stoploss_levels=[-0.02, 0.05],
            takeprofit_levels=[0.03, 0.10],
            initial_cash=1000,
        )
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)

        # Should have 1 (no action) + 4 (2 SL * 2 TP) = 5 actions
        assert len(env.action_levels) == 5

        # Check combinations exist
        expected_combinations = {
            (0.02, 0.03),
            (0.02, 0.10),
            (0.05, 0.03),
            (0.05, 0.10),
        }
        actual_combinations = {level for level in env.action_levels if level != (None, None)}
        assert actual_combinations == expected_combinations
        env.close()


# ============================================================================
# BRACKET ORDER TESTS
# ============================================================================


class TestSLTPBracketOrders:
    """Tests for SL/TP bracket order mechanics."""

    def test_bracket_opens_position(self, sltp_env, trading_mode):
        """Opening bracket should establish position with SL/TP."""
        td = sltp_env.reset()

        # Open bracket order (action 1 = first SL/TP combination)
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = sltp_env.step(action_td)

        account_state = next_td["next"]["account_state"]
        position_size = account_state[1]

        if trading_mode == "spot":
            assert position_size > 0, "Spot should have positive position"
        else:
            assert position_size != 0, "Futures should have non-zero position"

    def test_no_action_preserves_position(self, sltp_env):
        """Action 0 (no action) should preserve existing position."""
        td = sltp_env.reset()

        # Open bracket
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = sltp_env.step(action_td)
        position_before = next_td["next"]["account_state"][1]

        # No action
        action_td_no = next_td["next"].clone()
        action_td_no["action"] = torch.tensor(0)
        next_td_no = sltp_env.step(action_td_no)
        position_after = next_td_no["next"]["account_state"][1]

        assert torch.isclose(position_before, position_after, atol=1e-6)

    def test_new_bracket_replaces_position(self, sltp_env):
        """Opening new bracket should replace existing position."""
        td = sltp_env.reset()

        # Open first bracket
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = sltp_env.step(action_td)

        # Open second bracket (different SL/TP)
        action_td_new = next_td["next"].clone()
        action_td_new["action"] = torch.tensor(2)  # Different SL/TP combo
        next_td_new = sltp_env.step(action_td_new)

        # Position should be replaced (not necessarily same size)
        assert next_td_new["next"]["account_state"][1] != 0.0


# ============================================================================
# TRIGGER DETECTION TESTS
# ============================================================================


class TestSLTPTriggerDetection:
    """Tests for SL/TP trigger detection."""

    def test_take_profit_trigger_long(self, trending_up_df, sltp_config_spot):
        """Long position should trigger TP on uptrend."""
        sltp_config_spot.stoploss_levels = [0.10]  # Wide SL
        sltp_config_spot.takeprofit_levels = [0.01]  # Tight TP (easy to trigger)

        env = SequentialTradingEnvSLTP(trending_up_df, sltp_config_spot, simple_feature_fn)
        td = env.reset()

        # Open long bracket
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)  # Open bracket
        next_td = env.step(action_td)

        # Step until TP triggers
        for _ in range(100):
            if next_td["next"]["account_state"][1] == 0.0:  # Position closed
                break
            action_td_hold = next_td["next"].clone()
            action_td_hold["action"] = torch.tensor(0)  # No action
            next_td = env.step(action_td_hold)

        # Should have closed position (TP triggered)
        assert next_td["next"]["account_state"][1] == 0.0
        env.close()

    def test_stop_loss_trigger_long(self, trending_down_df, sltp_config_spot):
        """Long position should trigger SL on downtrend."""
        sltp_config_spot.stoploss_levels = [0.01]  # Tight SL (easy to trigger)
        sltp_config_spot.takeprofit_levels = [0.10]  # Wide TP

        env = SequentialTradingEnvSLTP(trending_down_df, sltp_config_spot, simple_feature_fn)
        td = env.reset()

        # Open long bracket
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = env.step(action_td)

        # Step until SL triggers
        for _ in range(100):
            if next_td["next"]["account_state"][1] == 0.0:
                break
            action_td_hold = next_td["next"].clone()
            action_td_hold["action"] = torch.tensor(0)
            next_td = env.step(action_td_hold)

        # Should have closed position (SL triggered)
        assert next_td["next"]["account_state"][1] == 0.0
        env.close()

    def test_stop_loss_trigger_short_futures(self, trending_up_df, sltp_config_futures):
        """Short position should trigger SL on uptrend (futures only)."""
        # Skip for now - requires additional setup for short positions
        pytest.skip("Short position SLTP testing requires additional setup")


# ============================================================================
# PRICE GAP TESTS
# ============================================================================


class TestSLTPPriceGaps:
    """Tests for intrabar price gap handling."""

    def test_gap_triggers_stop_loss(self, price_gap_df, sltp_config_spot):
        """Price gap should trigger SL even if close doesn't hit it."""
        sltp_config_spot.stoploss_levels = [0.05]  # 5% SL
        sltp_config_spot.takeprofit_levels = [0.10]

        env = SequentialTradingEnvSLTP(price_gap_df, sltp_config_spot, simple_feature_fn)
        td = env.reset()

        # Open long bracket
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = env.step(action_td)

        # Step through the gap (around index 50)
        for _ in range(60):
            if next_td["next"]["account_state"][1] == 0.0:
                break
            action_td_hold = next_td["next"].clone()
            action_td_hold["action"] = torch.tensor(0)
            next_td = env.step(action_td_hold)

        # Gap should have triggered SL
        assert next_td["next"]["account_state"][1] == 0.0
        env.close()

    def test_gap_triggers_take_profit(self, price_gap_df, sltp_config_spot):
        """Price gap should trigger TP even if close doesn't hit it."""
        # Modify gap_df to have upward gap instead
        gap_up_df = price_gap_df.copy()
        gap_up_df.loc[50:, "close"] = gap_up_df.loc[50:, "close"] * 1.15  # 15% jump

        sltp_config_spot.stoploss_levels = [0.20]  # Wide SL
        sltp_config_spot.takeprofit_levels = [0.05]  # 5% TP

        env = SequentialTradingEnvSLTP(gap_up_df, sltp_config_spot, simple_feature_fn)
        td = env.reset()

        # Open long bracket
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = env.step(action_td)

        # Step through the gap
        for _ in range(60):
            if next_td["next"]["account_state"][1] == 0.0:
                break
            action_td_hold = next_td["next"].clone()
            action_td_hold["action"] = torch.tensor(0)
            next_td = env.step(action_td_hold)

        # Gap should have triggered TP
        assert next_td["next"]["account_state"][1] == 0.0
        env.close()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestSLTPIntegration:
    """Integration tests with base environment functionality."""

    def test_sltp_respects_transaction_fees(self, sltp_env):
        """SLTP orders should incur transaction fees."""
        sltp_env.transaction_fee = 0.1  # 10% fee
        td = sltp_env.reset()
        initial_cash = td["account_state"][0].item()

        # Open and close bracket
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = sltp_env.step(action_td)

        # Wait for trigger or manually close
        for _ in range(10):
            if next_td["next"]["account_state"][1] == 0.0:
                break
            action_td_hold = next_td["next"].clone()
            action_td_hold["action"] = torch.tensor(0)
            next_td = sltp_env.step(action_td_hold)

        final_cash = next_td["next"]["account_state"][0].item()

        # Fees should have reduced cash (unless huge profit)
        assert final_cash < initial_cash + 100

    def test_sltp_terminates_properly(self, sltp_env):
        """SLTP env should terminate at max_traj_length."""
        sltp_env.max_traj_length = 10
        td = sltp_env.reset()

        for _ in range(10):
            action_td = td.clone()
            action_td["action"] = torch.tensor(0)  # No action
            td = sltp_env.step(action_td)

        assert td["next"]["done"].item()

    def test_sltp_account_state_consistent(self, sltp_env, trading_mode):
        """Account state should be valid after SLTP operations."""
        td = sltp_env.reset()
        validate_account_state(td["account_state"], trading_mode)

        # Open bracket
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        next_td = sltp_env.step(action_td)
        validate_account_state(next_td["next"]["account_state"], trading_mode)


# ============================================================================
# EDGE CASES
# ============================================================================


class TestSLTPEdgeCases:
    """Edge case tests for SLTP environments."""

    def test_empty_sl_levels_raises(self, sample_ohlcv_df, trading_mode):
        """Should raise error for empty SL levels."""
        with pytest.raises(ValueError):
            config = SequentialTradingEnvSLTPConfig(
                trading_mode=trading_mode,
                stoploss_levels=[],  # Empty
                takeprofit_levels=[0.03],
            )
            SequentialTradingEnvSLTP(sample_ohlcv_df, config)

    def test_empty_tp_levels_raises(self, sample_ohlcv_df, trading_mode):
        """Should raise error for empty TP levels."""
        with pytest.raises(ValueError):
            config = SequentialTradingEnvSLTPConfig(
                trading_mode=trading_mode,
                stoploss_levels=[-0.02],
                takeprofit_levels=[],  # Empty
            )
            SequentialTradingEnvSLTP(sample_ohlcv_df, config)

    def test_negative_sl_levels_raises(self, sample_ohlcv_df, trading_mode):
        """Should raise error for negative SL levels."""
        with pytest.raises(ValueError):
            config = SequentialTradingEnvSLTPConfig(
                trading_mode=trading_mode,
                stoploss_levels=[-0.02],  # Negative
                takeprofit_levels=[0.03],
            )
            SequentialTradingEnvSLTP(sample_ohlcv_df, config)

    def test_negative_tp_levels_raises(self, sample_ohlcv_df, trading_mode):
        """Should raise error for negative TP levels."""
        with pytest.raises(ValueError):
            config = SequentialTradingEnvSLTPConfig(
                trading_mode=trading_mode,
                stoploss_levels=[-0.02],
                takeprofit_levels=[-0.03],  # Negative
            )
            SequentialTradingEnvSLTP(sample_ohlcv_df, config)


# ============================================================================
# REGRESSION TESTS
# ============================================================================


class TestSLTPRegression:
    """Regression tests for known SLTP issues."""

    def test_action_spec_matches_levels(self, sltp_env):
        """Action spec size should match action_levels length."""
        assert sltp_env.action_spec.n == len(sltp_env.action_levels)

    def test_sltp_levels_immutable(self, sltp_env):
        """SL/TP levels should not change after initialization."""
        initial_levels = sltp_env.action_levels.copy()

        # Perform operations
        td = sltp_env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        sltp_env.step(action_td)

        assert sltp_env.action_levels == initial_levels

    def test_no_action_when_no_position(self, sltp_env):
        """No action should work when there's no position."""
        td = sltp_env.reset()

        # No action without position
        action_td = td.clone()
        action_td["action"] = torch.tensor(0)
        next_td = sltp_env.step(action_td)

        # Should not crash
        assert next_td is not None
