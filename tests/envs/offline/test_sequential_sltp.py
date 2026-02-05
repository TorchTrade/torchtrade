"""
Consolidated tests for SequentialTradingEnvSLTP (unified spot/futures SLTP environment).

This file consolidates tests from:
- test_seqlongonlysltp.py (spot SLTP)
- test_seqfuturessltp.py (futures SLTP)

Uses parametrization to test both trading modes with maximum code reuse.
"""

import pytest
import torch

from torchtrade.envs.offline import SequentialTradingEnvSLTP, SequentialTradingEnvSLTPConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.conftest import simple_feature_fn, validate_account_state


@pytest.fixture
def sltp_config_spot():
    """SLTP config for spot trading."""
    return SequentialTradingEnvSLTPConfig(
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
    """Create SLTP environment for testing.

    trading_mode fixture returns leverage: 1 for spot, 10 for futures.
    """
    config = sltp_config_spot if trading_mode == 1 else sltp_config_futures
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
        ([-0.02], [0.03], 2),           # 1 + (1 * 1) = 2
        ([-0.02, -0.05], [0.03], 3),     # 1 + (2 * 1) = 3
        ([-0.02], [0.03, 0.10], 3),     # 1 + (1 * 2) = 3
        ([-0.02, -0.05], [0.03, 0.10], 5),  # 1 + (2 * 2) = 5
    ])
    def test_action_space_size(self, sample_ohlcv_df, trading_mode, sl_levels, tp_levels, expected_actions):
        """Action space should be 1 + (num_sl * num_tp)."""
        config = SequentialTradingEnvSLTPConfig(
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
        assert sltp_env.action_map[0] == (None, None, None)  # (action_type, sl, tp)

    def test_sltp_combinations_generated(self, sample_ohlcv_df, trading_mode):
        """Should generate all SL/TP combinations."""
        config = SequentialTradingEnvSLTPConfig(
            leverage=trading_mode,  # trading_mode fixture returns leverage: 1 or 10
            stoploss_levels=[-0.02, -0.05],
            takeprofit_levels=[0.03, 0.10],
            initial_cash=1000,
        )
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)

        # Should have 1 (no action) + 4 (2 SL * 2 TP) long actions
        # Futures mode adds 4 more short actions
        expected_size = 1 + 4 if trading_mode == 1 else 1 + 4 + 4
        assert len(env.action_map) == expected_size

        # Check long combinations exist
        expected_long_combinations = {
            ("long", -0.02, 0.03),
            ("long", -0.02, 0.10),
            ("long", -0.05, 0.03),
            ("long", -0.05, 0.10),
        }
        actual_long_combinations = {
            v for v in env.action_map.values() if v[0] == "long"
        }
        assert actual_long_combinations == expected_long_combinations

        # Check short combinations have swapped SL/TP (issue #149)
        if trading_mode != 1:  # futures
            expected_short_combinations = {
                ("short", 0.03, -0.02),   # tp_pct -> sl_pct, sl_pct -> tp_pct
                ("short", 0.10, -0.02),
                ("short", 0.03, -0.05),
                ("short", 0.10, -0.05),
            }
            actual_short_combinations = {
                v for v in env.action_map.values() if v[0] == "short"
            }
            assert actual_short_combinations == expected_short_combinations
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
        sltp_config_spot.stoploss_levels = [-0.10]  # Wide SL
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
        sltp_config_spot.stoploss_levels = [-0.01]  # Tight SL (easy to trigger)
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

    def test_stop_loss_trigger_short_futures(self, trending_up_df):
        """Short position should trigger SL on uptrend (futures only)."""
        config = SequentialTradingEnvSLTPConfig(
            leverage=10,
            initial_cash=10000,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            max_traj_length=400,
            random_start=False,
            stoploss_levels=[-0.01],  # Tight SL
            takeprofit_levels=[0.10],  # Wide TP
        )
        env = SequentialTradingEnvSLTP(trending_up_df, config, simple_feature_fn)
        td = env.reset()

        # Find short action
        short_idx = next(i for i, v in env.action_map.items() if v[0] == "short")

        # Open short bracket
        action_td = td.clone()
        action_td["action"] = torch.tensor(short_idx)
        next_td = env.step(action_td)

        # Verify SL is above entry (correct after fix)
        assert env.stop_loss > env.position.entry_price

        # Step until SL triggers (price going up = bad for shorts)
        for _ in range(300):
            if next_td["next"]["account_state"][1] == 0.0:
                break
            action_td_hold = next_td["next"].clone()
            action_td_hold["action"] = torch.tensor(0)
            next_td = env.step(action_td_hold)

        assert next_td["next"]["account_state"][1] == 0.0, "SL should have triggered on uptrend"
        env.close()


# ============================================================================
# PRICE GAP TESTS
# ============================================================================


class TestSLTPPriceGaps:
    """Tests for intrabar price gap handling."""

    def test_gap_triggers_stop_loss(self, price_gap_df, sltp_config_spot):
        """Price gap should trigger SL even if close doesn't hit it."""
        sltp_config_spot.stoploss_levels = [-0.05]  # 5% SL
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

    def test_negative_sl_levels_raises(self, sample_ohlcv_df, trading_mode):
        """Should raise error for positive SL levels (must be negative)."""
        with pytest.raises(ValueError):
            config = SequentialTradingEnvSLTPConfig(
                    stoploss_levels=[0.02],  # Positive (invalid)
                takeprofit_levels=[0.03],
            )
            SequentialTradingEnvSLTP(sample_ohlcv_df, config)

    def test_negative_tp_levels_raises(self, sample_ohlcv_df, trading_mode):
        """Should raise error for negative TP levels."""
        with pytest.raises(ValueError):
            config = SequentialTradingEnvSLTPConfig(
                    stoploss_levels=[-0.02],
                takeprofit_levels=[-0.03],  # Negative
            )
            SequentialTradingEnvSLTP(sample_ohlcv_df, config)


# ============================================================================
# REGRESSION TESTS
# ============================================================================


class TestSLTPRegression:
    """Regression tests for known SLTP issues."""

    def test_short_bracket_prices_not_inverted(self, sample_ohlcv_df):
        """Short positions must have SL above entry and TP below entry (issue #149)."""
        config = SequentialTradingEnvSLTPConfig(
            leverage=10,
            initial_cash=10000,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            max_traj_length=100,
            random_start=False,
            stoploss_levels=[-0.025],
            takeprofit_levels=[0.05],
        )
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        # Find the short action in the action map
        short_idx = next(i for i, v in env.action_map.items() if v[0] == "short")

        # Open short position
        action_td = td.clone()
        action_td["action"] = torch.tensor(short_idx)
        env.step(action_td)

        entry_price = env.position.entry_price
        assert entry_price > 0, "Position should be opened"

        # For shorts: SL must be ABOVE entry, TP must be BELOW entry
        assert env.stop_loss > entry_price, (
            f"Short SL ({env.stop_loss}) must be above entry ({entry_price})"
        )
        assert env.take_profit < entry_price, (
            f"Short TP ({env.take_profit}) must be below entry ({entry_price})"
        )
        env.close()

    def test_short_action_map_matches_live_env(self, sample_ohlcv_df):
        """Offline short action map must match live env convention (issue #149)."""
        from torchtrade.envs.utils.action_maps import create_sltp_action_map

        sl_levels = [-0.025, -0.05]
        tp_levels = [0.05, 0.1]

        # Live env action map
        live_map = create_sltp_action_map(
            sl_levels, tp_levels, include_short_positions=True,
            include_hold_action=True, include_close_action=False,
        )

        # Offline env action map
        config = SequentialTradingEnvSLTPConfig(
            leverage=10,
            initial_cash=10000,
            stoploss_levels=sl_levels,
            takeprofit_levels=tp_levels,
        )
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)

        # Compare short actions between live and offline
        live_shorts = {v for v in live_map.values() if v[0] == "short"}
        offline_shorts = {v for v in env.action_map.values() if v[0] == "short"}
        assert live_shorts == offline_shorts, (
            f"Offline shorts {offline_shorts} != live shorts {live_shorts}"
        )
        env.close()

    def test_no_action_when_no_position(self, sltp_env):
        """No action should work when there's no position."""
        td = sltp_env.reset()

        # No action without position
        action_td = td.clone()
        action_td["action"] = torch.tensor(0)
        next_td = sltp_env.step(action_td)

        # Should not crash
        assert next_td is not None

    @pytest.mark.parametrize("leverage", [1, 10], ids=["spot", "futures"])
    def test_truncation_does_not_set_terminated(self, sample_ohlcv_df, leverage):
        """Truncated episodes (data exhaustion) must NOT set terminated=True.

        Regression test for #150.
        """
        config = SequentialTradingEnvSLTPConfig(
            leverage=leverage,
            max_traj_length=5000,
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        for _ in range(5000):
            action_td = td["next"].clone() if "next" in td.keys() else td.clone()
            action_td["action"] = torch.tensor(0)  # No action
            td = env.step(action_td)
            if td["next"]["done"].item():
                break

        assert td["next"]["done"].item() is True
        assert td["next"]["truncated"].item() is True
        assert td["next"]["terminated"].item() is False, (
            "Truncated episode should have terminated=False (issue #150)"
        )
        env.close()
