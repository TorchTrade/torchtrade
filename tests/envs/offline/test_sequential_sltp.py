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

    @pytest.mark.parametrize("sl_levels,tp_levels", [
        ([-0.02], [0.03]),
        ([-0.02, -0.05], [0.03]),
        ([-0.02], [0.03, 0.10]),
        ([-0.02, -0.05], [0.03, 0.10]),
    ])
    def test_action_space_size(self, sample_ohlcv_df, trading_mode, sl_levels, tp_levels):
        """Action space: 1 + combos (spot) or 1 + 2*combos (futures with shorts)."""
        config = SequentialTradingEnvSLTPConfig(
            leverage=trading_mode,
            stoploss_levels=sl_levels,
            takeprofit_levels=tp_levels,
            initial_cash=1000,
        )
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)
        combos = len(sl_levels) * len(tp_levels)
        sides = 2 if trading_mode > 1 else 1  # long + short for futures
        expected = 1 + combos * sides
        assert env.action_spec.n == expected
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

        if trading_mode == 1:  # spot
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

    @pytest.mark.parametrize("bars_after_entry", [1, 2], ids=["next-bar", "two-bars-later"])
    @pytest.mark.parametrize(
        "leverage,sl_pct,tp_pct,wick_low,wick_high,fee,"
        "expected_exit,expected_balance,expected_terminated", [
            # The fee rows pin exact values rather than an inequality: a double-charged
            # fee is also "cheaper than free". On the next-bar arm both fees land in a
            # single step, which happens nowhere else in the codebase.
            (1, -0.025, 0.50, 80.0, None, 0.0, "sltp_sl", 9750.0, False),
            (1, -0.025, 0.50, 80.0, None, 0.001, "sltp_sl", 9730.519481, False),
            (10, -0.50, 0.50, 88.0, None, 0.0, "liquidation", 400.0, True),
            (10, -0.50, 0.50, 88.0, None, 0.001, "liquidation", 306.534653, True),
            # The fix repeats the trigger->fill-price mapping, so pin the take-profit
            # side too: a wrong fill here is invisible to every other assertion.
            (1, -0.50, 0.025, None, 120.0, 0.0, "sltp_tp", 10250.0, False),
        ], ids=["stop-loss", "stop-loss-fee", "liquidation", "liquidation-fee", "take-profit"])
    def test_exit_fires_on_the_first_bar_the_position_is_exposed_to(
        self, bars_after_entry, leverage, sl_pct, tp_pct, wick_low, wick_high, fee,
        expected_exit, expected_balance, expected_terminated
    ):
        """The outcome must not depend on how soon after entry the crash lands (#268).

        _step advanced the sampler and checked triggers against bar N+1 *before* opening
        the position at bar N's close, so a position opened one bar before a crash had
        its first check on bar N+2 and rode straight through it. Entering two bars early
        exited correctly, which is what made the two paths disagree on identical data.
        """
        import numpy as np
        import pandas as pd

        n = 50
        crash_bar = 20
        prices = np.full(n, 100.0)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
            "open": prices.copy(),
            "high": prices.copy(),
            "low": prices.copy(),
            "close": prices.copy(),
            "volume": np.ones(n) * 1000,
        })
        if wick_low is not None:
            df.loc[crash_bar, "low"] = wick_low
        else:
            df.loc[crash_bar, "high"] = wick_high

        config = SequentialTradingEnvSLTPConfig(
            leverage=leverage,
            initial_cash=10000,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=fee,
            slippage=0.0,
            seed=42,
            random_start=False,
            stoploss_levels=[sl_pct],
            takeprofit_levels=[tp_pct],
        )
        env = SequentialTradingEnvSLTP(df, config, simple_feature_fn)
        td = env.reset()

        # reset caches bar 10 (window_sizes=[10]), so step k executes at bar 10+k and
        # advances onto bar 11+k. The crash bar is therefore reached during exit_step
        # whatever bar the position was opened on — that sameness is the point.
        exit_step = crash_bar - 10 - 1
        open_step = crash_bar - bars_after_entry - 10
        long_idx = next(i for i, v in env.action_map.items() if v[0] == "long")
        for step in range(exit_step + 1):
            action_td = (td if step == 0 else td["next"]).clone()
            action_td["action"] = torch.tensor(long_idx if step == open_step else 0)
            td = env.step(action_td)

        assert env.position.position_size == 0, (
            f"a wick {bars_after_entry} bar(s) after entry must close the "
            f"position, but position_size={env.position.position_size}"
        )
        assert env.history.action_types[-1] == expected_exit
        # An entry-bar exit records only the exit. HistoryTracker has one row per step
        # and the exit is what determines the ending state, matching what already
        # happens when a previously-held position triggers. It is now reachable one bar
        # earlier, so trade counts drawn from action_types under-count entries -- pinned
        # here so that stays a decision rather than a surprise.
        assert env.history.action_types.count("long") == (0 if bars_after_entry == 1 else 1)
        assert env.balance == pytest.approx(expected_balance)

        # The exit must be applied BEFORE the observation, reward and termination are
        # built. Nothing else pins that: with the exit moved after them the internal
        # balance is still right, so every other assertion here passes while the policy
        # is handed an open levered position on a step the env already closed.
        account_state = td["next"]["account_state"]
        assert account_state[0].item() == pytest.approx(0.0), "exposure_pct must read flat"
        assert account_state[1].item() == pytest.approx(0.0), "direction must read flat"
        assert account_state[3].item() == pytest.approx(0.0), "holding_time must read flat"
        reward = td["next"]["reward"].item()
        if expected_balance < 10000:
            assert reward < 0, "a losing round trip must produce a negative reward"
        else:
            assert reward > 0, "a winning round trip must produce a positive reward"
        assert bool(td["next"]["terminated"].item()) == expected_terminated
        env.close()

    @pytest.mark.parametrize("leverage,stoploss_pct,wick_low,expected_exit,expected_balance", [
        (1, -0.02, 95.0, "sltp_sl", 9800.0),      # SL at 98, no liquidation at 1x
        (10, -0.50, 88.0, "liquidation", 400.0),  # SL at 50 untouched; liquidation at 90.4
    ], ids=["stop-loss", "liquidation"])
    def test_exit_fires_on_a_mid_bar_wick_when_execute_on_is_coarse(
        self, leverage, stoploss_pct, wick_low, expected_exit, expected_balance
    ):
        """A wick inside the execution bar must close the position (issue #267).

        The sampler built execution bars with .last(), so high/low collapsed onto the
        final sub-bar and any excursion earlier in the bar was invisible. Both exits
        that read that range are covered here, because the sampler feeds them both.

        The wick sits mid-bar, not on the last sub-bar, so it is only visible once the
        bar aggregates its sub-bars.
        """
        import numpy as np
        import pandas as pd

        n = 240
        prices = np.full(n, 100.0)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
            "open": prices.copy(),
            "high": prices.copy(),
            "low": prices.copy(),
            "close": prices.copy(),
            "volume": np.ones(n) * 1000,
        })
        # Minute 183 sits inside the 03:00 bar (minutes 180-194), which is the first
        # bar the position is exposed to, and is not that bar's final sub-bar.
        df.loc[183, "low"] = wick_low

        config = SequentialTradingEnvSLTPConfig(
            leverage=leverage,
            initial_cash=10000,
            execute_on=TimeFrame(15, TimeFrameUnit.Minute),
            time_frames=[TimeFrame(15, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            random_start=False,
            stoploss_levels=[stoploss_pct],
            takeprofit_levels=[0.50],
        )
        env = SequentialTradingEnvSLTP(df, config, simple_feature_fn)
        td = env.reset()

        long_idx = next(i for i, v in env.action_map.items() if v[0] == "long")
        open_td = td.clone()
        open_td["action"] = torch.tensor(long_idx)
        td_next = env.step(open_td)
        assert env.position.position_size > 0, "long should be open before the wick bar"

        hold_td = td_next["next"].clone()
        hold_td["action"] = torch.tensor(0)
        env.step(hold_td)

        # Pinned so the two ways this can break report themselves, rather than surfacing
        # as an unexplained still-open position below.
        assert env._cached_base_features["low"] == pytest.approx(wick_low), (
            "the bar the position is checked against must report the wick: either the "
            "step landed on a different bar, or the bar did not aggregate its sub-bars"
        )
        assert env.position.position_size == 0, (
            f"wick to {wick_low} inside the execution bar must close the position, "
            f"but position_size={env.position.position_size}"
        )
        # Without this the parametrize labels live only in prose: any exit satisfies the
        # assertion above, so a change making SL leverage-scaled would silently turn the
        # liquidation case into a second stop-loss case.
        assert env.history.action_types[-1] == expected_exit
        assert env.balance == pytest.approx(expected_balance)
        env.close()

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

    def test_sl_triggers_on_next_bar_not_delayed(self, sample_ohlcv_df):
        """SL must trigger on the new bar fetched in _step(), not the stale cached bar.

        Regression test for off-by-one timing bug where _step() checked SL/TP
        against the stale bar (already observed by the agent) instead of the
        new bar (first bar the position is actually exposed to).

        Timing (window_size=10):
        - Reset: scaffold fetches bar 10, caches it.
        - Step 1: cached_price=bar10_close=100. Advance to bar 11 (benign).
          Agent opens long at 100 → SL=98, TP=110. Position now exists.
        - Step 2: cached_price=bar11_close. Advance to bar 12 (low=90).
          SL/TP checked on bar 12 → SL triggers. Position MUST close.

        Before the fix, _step() checked SL/TP on the stale bar (bar 11) and
        only advanced the sampler at the end, so bar 12's crash was missed
        until step 3.
        """
        import numpy as np
        import pandas as pd

        n = 50
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
        prices = np.full(n, 100.0)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices.copy(),
            "high": prices + 1.0,
            "low": prices.copy(),
            "close": prices.copy(),
            "volume": np.ones(n) * 1000,
        })
        # Bar 12: intrabar crash to 90 (well below SL of 98)
        df.loc[12, "low"] = 90.0
        df.loc[12, "close"] = 95.0

        config = SequentialTradingEnvSLTPConfig(
            leverage=1,
            initial_cash=1000,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            max_traj_length=100,
            random_start=False,
            stoploss_levels=[-0.02],   # SL at -2% → price 98.0
            takeprofit_levels=[0.10],  # TP at +10% → price 110.0
        )
        env = SequentialTradingEnvSLTP(df, config, simple_feature_fn)
        td = env.reset()

        long_idx = next(i for i, v in env.action_map.items() if v[0] == "long")

        # Step 1: Open long (executes at bar 10's close = 100)
        action_td = td.clone()
        action_td["action"] = torch.tensor(long_idx)
        td_next = env.step(action_td)

        assert env.position.position_size > 0, "Position should be open"
        assert abs(env.position.entry_price - 100.0) < 0.01
        assert env.stop_loss > 0, "SL should be set"

        # Step 2: Hold — sampler advances to bar 12 (low=90, below SL=98)
        hold_td = td_next["next"].clone()
        hold_td["action"] = torch.tensor(0)
        env.step(hold_td)

        # Position MUST be closed by SL on bar 12
        assert env.position.position_size == 0, (
            f"Position should be closed by SL on the new bar, "
            f"but position_size={env.position.position_size}"
        )
        assert env.stop_loss == 0.0, "SL should be reset after trigger"
        assert env.take_profit == 0.0, "TP should be reset after trigger"

        env.close()

    def test_tp_triggers_on_next_bar_not_delayed(self, sample_ohlcv_df):
        """TP must trigger on the new bar fetched in _step(), not the stale cached bar.

        Mirror of test_sl_triggers_on_next_bar_not_delayed for take-profit.

        Timing (window_size=10):
        - Reset: scaffold fetches bar 10, caches it.
        - Step 1: cached_price=bar10_close=100. Advance to bar 11 (benign).
          Agent opens long at 100 → SL=98, TP=105. Position now exists.
        - Step 2: cached_price=bar11_close. Advance to bar 12 (high=110).
          SL/TP checked on bar 12 → TP triggers. Position MUST close.
        """
        import numpy as np
        import pandas as pd

        n = 50
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
        prices = np.full(n, 100.0)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices.copy(),
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices.copy(),
            "volume": np.ones(n) * 1000,
        })
        # Bar 12: intrabar spike to 110 (above TP of 105)
        df.loc[12, "high"] = 110.0
        df.loc[12, "close"] = 106.0

        config = SequentialTradingEnvSLTPConfig(
            leverage=1,
            initial_cash=1000,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            max_traj_length=100,
            random_start=False,
            stoploss_levels=[-0.10],   # SL at -10% → price 90.0 (won't trigger)
            takeprofit_levels=[0.05],  # TP at +5% → price 105.0
        )
        env = SequentialTradingEnvSLTP(df, config, simple_feature_fn)
        td = env.reset()

        long_idx = next(i for i, v in env.action_map.items() if v[0] == "long")

        # Step 1: Open long (executes at bar 10's close = 100)
        action_td = td.clone()
        action_td["action"] = torch.tensor(long_idx)
        td_next = env.step(action_td)

        assert env.position.position_size > 0, "Position should be open"
        assert env.take_profit > 0, "TP should be set"

        # Step 2: Hold — sampler advances to bar 12 (high=110, above TP=105)
        hold_td = td_next["next"].clone()
        hold_td["action"] = torch.tensor(0)
        env.step(hold_td)

        # Position MUST be closed by TP on bar 12
        assert env.position.position_size == 0, (
            f"Position should be closed by TP on the new bar, "
            f"but position_size={env.position.position_size}"
        )
        assert env.stop_loss == 0.0, "SL should be reset after trigger"
        assert env.take_profit == 0.0, "TP should be reset after trigger"

        env.close()

    def test_liquidation_triggers_on_next_bar_futures(self, sample_ohlcv_df):
        """Liquidation must trigger on the new bar fetched in _step() (futures).

        Timing (window_size=10, leverage=10):
        - Reset: scaffold fetches bar 10, caches it.
        - Step 1: cached_price=bar10_close=100. Advance to bar 11 (benign).
          Agent opens long at 100 with 10x leverage.
          Liquidation price = 100 * (1 - 1/10 + 0.004) = 90.4.
        - Step 2: cached_price=bar11_close. Advance to bar 12 (close=85).
          Liquidation checked on bar 12 → triggers (85 < 90.4).
        """
        import numpy as np
        import pandas as pd

        n = 50
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
        prices = np.full(n, 100.0)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices.copy(),
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices.copy(),
            "volume": np.ones(n) * 1000,
        })
        # Bar 12: intrabar wick to 85 (below liq=90.4) but close recovers to 95, so
        # a close-based check would miss it. This test only asserts the position
        # closed, which both exits satisfy -- which of them fired is pinned in
        # TestLiquidationTakesPriorityOverBrackets.
        df.loc[12, "low"] = 85.0
        df.loc[12, "close"] = 95.0

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
            stoploss_levels=[-0.05],   # SL 95 is inside the wick too: this bar
                                       # breaches both exits
            takeprofit_levels=[0.10],
        )
        env = SequentialTradingEnvSLTP(df, config, simple_feature_fn)
        td = env.reset()

        long_idx = next(i for i, v in env.action_map.items() if v[0] == "long")

        # Step 1: Open long with 10x leverage
        action_td = td.clone()
        action_td["action"] = torch.tensor(long_idx)
        td_next = env.step(action_td)

        assert env.position.position_size > 0, "Position should be open"
        liq_price = env.liquidation_price
        assert 90.0 < liq_price < 91.0, f"Liquidation price should be ~90.4, got {liq_price}"

        # Step 2: Hold — sampler advances to bar 12 (low=85, close=95; wick below liq=90.4)
        hold_td = td_next["next"].clone()
        hold_td["action"] = torch.tensor(0)
        td_next2 = env.step(hold_td)

        # Position MUST be liquidated on bar 12
        assert env.position.position_size == 0, (
            f"Position should be liquidated on the new bar, "
            f"but position_size={env.position.position_size}"
        )

        env.close()

    def test_check_env_specs_passes(self, sltp_env):
        """check_env_specs must pass — specs must match actual output shapes."""
        from torchrl.envs.utils import check_env_specs
        check_env_specs(sltp_env)


# ============================================================================
# SAMPLER EXHAUSTION TESTS (Issue #204)
# ============================================================================


class TestSLTPSamplerExhaustion:
    """Verify SLTP _step gracefully terminates when the sampler is exhausted."""

    @pytest.mark.parametrize("leverage", [1, 10], ids=["spot", "futures"])
    def test_step_after_sampler_exhausted_does_not_crash(self, short_ohlcv_df, leverage):
        """Stepping past the sampler's last timestamp must return done=True,
        not raise ValueError (issue #204)."""
        config = SequentialTradingEnvSLTPConfig(
            leverage=leverage,
            initial_cash=1000,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            max_traj_length=9999,
            random_start=False,
            stoploss_levels=[-0.01],
            takeprofit_levels=[0.02],
        )
        env = SequentialTradingEnvSLTP(short_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        # Action 0 = hold (no position) in SLTP envs
        hold_action = 0

        for _ in range(200):
            action_td = td["next"].clone() if "next" in td.keys() else td.clone()
            action_td["action"] = torch.tensor(hold_action)
            td = env.step(action_td)
            if td["next"]["done"].item():
                break

        assert td["next"]["done"].item()
        assert td["next"]["truncated"].item()

        # The critical test: one MORE step must not crash
        extra_td = td["next"].clone()
        extra_td["action"] = torch.tensor(hold_action)
        result = env.step(extra_td)

        assert result["next"]["done"].item()
        assert result["next"]["truncated"].item()
        assert not result["next"]["terminated"].item()

        env.close()


# ============================================================================
# POSITION SIZING TESTS
# ============================================================================


class TestSLTPPositionSizing:
    """Test trade_mode-aware position sizing for SLTP environments."""

    @pytest.fixture
    def base_config_kwargs(self):
        return dict(
            initial_cash=10000,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            max_traj_length=100,
            random_start=False,
            leverage=10,
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.03],
        )

    def _open_long(self, env):
        """Helper: reset env and open a long position (action 1)."""
        td = env.reset()
        action_td = td.clone()
        action_td["action"] = torch.tensor(1)
        env.step(action_td)

    def test_fractional_position_sizing(self, sample_ohlcv_df, base_config_kwargs):
        """Fractional mode: position_fraction=0.1 should use 10% of portfolio."""
        config = SequentialTradingEnvSLTPConfig(
            **base_config_kwargs,
            trade_mode="fractional",
            position_fraction=0.1,
        )
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)
        self._open_long(env)
        fractional_size = abs(env.position.position_size)

        # Compare with all-in
        config_allin = SequentialTradingEnvSLTPConfig(
            **base_config_kwargs,
            trade_mode="fractional",
            position_fraction=1.0,
        )
        env_allin = SequentialTradingEnvSLTP(sample_ohlcv_df, config_allin, simple_feature_fn)
        self._open_long(env_allin)
        allin_size = abs(env_allin.position.position_size)

        assert fractional_size == pytest.approx(allin_size * 0.1, rel=0.05)

    def test_notional_position_sizing(self, sample_ohlcv_df, base_config_kwargs):
        """Notional mode: quantity_per_trade=500 should buy $500 worth."""
        config = SequentialTradingEnvSLTPConfig(
            **base_config_kwargs,
            trade_mode="notional",
            quantity_per_trade=500.0,
        )
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)
        self._open_long(env)

        price = env.position.entry_price
        expected_size = 500.0 / price
        assert abs(env.position.position_size) == pytest.approx(expected_size, rel=0.02)

    def test_quantity_position_sizing(self, sample_ohlcv_df, base_config_kwargs):
        """Quantity mode: quantity_per_trade=0.05 should buy exactly 0.05 units."""
        config = SequentialTradingEnvSLTPConfig(
            **base_config_kwargs,
            trade_mode="quantity",
            quantity_per_trade=0.05,
        )
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)
        self._open_long(env)

        assert abs(env.position.position_size) == pytest.approx(0.05, rel=1e-6)

    @pytest.mark.parametrize("trade_mode,quantity_per_trade", [
        ("notional", 500.0),
        ("quantity", 0.05),
    ], ids=["notional-short", "quantity-short"])
    def test_short_position_direction(self, sample_ohlcv_df, base_config_kwargs, trade_mode, quantity_per_trade):
        """Notional and quantity modes must open short positions with negative size."""
        config = SequentialTradingEnvSLTPConfig(
            **base_config_kwargs,
            trade_mode=trade_mode,
            quantity_per_trade=quantity_per_trade,
        )
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()
        # Last action in action map is a short (with leverage=10, shorts are enabled)
        short_action = len(env.action_map) - 1
        action_td = td.clone()
        action_td["action"] = torch.tensor(short_action)
        env.step(action_td)
        assert env.position.position_size < 0

    def test_default_fractional_is_backward_compatible(self, sample_ohlcv_df, base_config_kwargs):
        """Default config (fractional, position_fraction=1.0) matches old all-in behavior."""
        config = SequentialTradingEnvSLTPConfig(**base_config_kwargs)
        assert config.trade_mode == "fractional"
        assert config.position_fraction == 1.0

        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)
        self._open_long(env)
        assert env.position.position_size != 0


class TestLockPositionUntilSLTP:
    """Test lock_position_until_sltp config option."""

    @pytest.fixture
    def locked_config(self):
        return SequentialTradingEnvSLTPConfig(
            initial_cash=10000,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            max_traj_length=100,
            random_start=False,
            leverage=10,
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.03],
            lock_position_until_sltp=True,
        )

    def test_locked_position_ignores_switch_action(self, sample_ohlcv_df, locked_config):
        """With lock=True, a short action while long should be ignored (position held)."""
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, locked_config, simple_feature_fn)
        td = env.reset()

        # Open long
        long_idx = next(i for i, v in env.action_map.items() if v[0] == "long")
        action_td = td.clone()
        action_td["action"] = torch.tensor(long_idx)
        td_next = env.step(action_td)["next"]
        assert env.position.position_size > 0
        initial_size = env.position.position_size

        # Try to switch to short — should be ignored
        short_idx = next(i for i, v in env.action_map.items() if v[0] == "short")
        action_td = td_next.clone()
        action_td["action"] = torch.tensor(short_idx)
        env.step(action_td)

        # Position should still be the same long
        assert env.position.position_size == pytest.approx(initial_size, rel=1e-6)

    def test_locked_position_ignores_close_action(self, sample_ohlcv_df):
        """With lock=True and include_close_action=True, close action is ignored."""
        config = SequentialTradingEnvSLTPConfig(
            initial_cash=10000,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            max_traj_length=100,
            random_start=False,
            leverage=10,
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.03],
            lock_position_until_sltp=True,
            include_close_action=True,
        )
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        # Open long
        long_idx = next(i for i, v in env.action_map.items() if v[0] == "long")
        action_td = td.clone()
        action_td["action"] = torch.tensor(long_idx)
        td_next = env.step(action_td)["next"]
        assert env.position.position_size > 0
        initial_size = env.position.position_size

        # Try close action — should be ignored
        close_idx = next(i for i, v in env.action_map.items() if v[0] == "close")
        action_td = td_next.clone()
        action_td["action"] = torch.tensor(close_idx)
        env.step(action_td)

        assert env.position.position_size == pytest.approx(initial_size, rel=1e-6)

    def test_unlocked_allows_switch(self, sample_ohlcv_df):
        """With lock=False (default), switching positions works normally."""
        config = SequentialTradingEnvSLTPConfig(
            initial_cash=10000,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            max_traj_length=100,
            random_start=False,
            leverage=10,
            stoploss_levels=[-0.02],
            takeprofit_levels=[0.03],
            lock_position_until_sltp=False,
        )
        env = SequentialTradingEnvSLTP(sample_ohlcv_df, config, simple_feature_fn)
        td = env.reset()

        # Open long
        long_idx = next(i for i, v in env.action_map.items() if v[0] == "long")
        action_td = td.clone()
        action_td["action"] = torch.tensor(long_idx)
        td_next = env.step(action_td)["next"]
        assert env.position.position_size > 0

        # Switch to short — should work
        short_idx = next(i for i, v in env.action_map.items() if v[0] == "short")
        action_td = td_next.clone()
        action_td["action"] = torch.tensor(short_idx)
        env.step(action_td)
        assert env.position.position_size < 0

    def test_sltp_still_fires_when_locked(self):
        """With lock=True, SL/TP triggers must still close the position."""
        import numpy as np
        import pandas as pd

        n = 50
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
        prices = np.full(n, 100.0)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices.copy(),
            "high": prices + 1.0,
            "low": prices.copy(),
            "close": prices.copy(),
            "volume": np.ones(n) * 1000,
        })
        # Bar 12: intrabar crash to 90 (below SL of 98)
        df.loc[12, "low"] = 90.0
        df.loc[12, "close"] = 95.0

        config = SequentialTradingEnvSLTPConfig(
            leverage=1,
            initial_cash=1000,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            transaction_fee=0.0,
            slippage=0.0,
            seed=42,
            max_traj_length=100,
            random_start=False,
            stoploss_levels=[-0.02],   # SL at -2% → price 98.0
            takeprofit_levels=[0.10],  # TP at +10% → price 110.0
            lock_position_until_sltp=True,
        )
        env = SequentialTradingEnvSLTP(df, config, simple_feature_fn)
        td = env.reset()

        long_idx = next(i for i, v in env.action_map.items() if v[0] == "long")

        # Step 1: Open long at bar 10's close = 100
        action_td = td.clone()
        action_td["action"] = torch.tensor(long_idx)
        td_next = env.step(action_td)
        assert env.position.position_size > 0

        # Step 2: Hold — bar 12 has low=90, SL=98 should trigger
        hold_td = td_next["next"].clone()
        hold_td["action"] = torch.tensor(0)
        env.step(hold_td)

        # Position MUST be closed by SL despite lock=True
        assert env.position.position_size == 0, (
            "SL must still fire when lock_position_until_sltp=True"
        )
        env.close()

    def test_locked_default_is_false(self):
        """Default lock_position_until_sltp should be False for backward compat."""
        config = SequentialTradingEnvSLTPConfig()
        assert config.lock_position_until_sltp is False


HOLD = (None, None, None)
CLOSE = ("close", None, None)
LONG_TIGHT = ("long", -0.025, 0.025)
SHORT_TIGHT = ("short", 0.025, -0.025)


def _run_sltp(actions, *, leverage, sl_levels, tp_levels, wick_high=None,
              wick_low=None, include_close=False):
    """Run `actions` through a flat 100.0 series carrying one wick at bar 20.

    reset() caches bar 10, so action k executes at bar 10+k and is exposed to bar
    11+k. Each caller states which index it acts on and what that means for its own
    scenario.
    """
    import numpy as np
    import pandas as pd

    n = 60
    flat = np.full(n, 100.0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": flat.copy(), "high": flat.copy(), "low": flat.copy(),
        "close": flat.copy(), "volume": np.ones(n) * 1000,
    })
    if wick_high is not None:
        df.loc[20, "high"] = wick_high
    if wick_low is not None:
        df.loc[20, "low"] = wick_low

    config = SequentialTradingEnvSLTPConfig(
        leverage=leverage, initial_cash=10000,
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10], transaction_fee=0.0, slippage=0.0,
        seed=42, random_start=False,
        stoploss_levels=sl_levels, takeprofit_levels=tp_levels,
        include_close_action=include_close,
    )
    env = SequentialTradingEnvSLTP(df, config, simple_feature_fn)
    # Resolved from the map, not hardcoded: enabling the close action shifts every
    # index, which would silently turn these into a run of holds.
    index_of = {spec: idx for idx, spec in env.action_map.items()}
    td = env.reset()
    for step, action in enumerate(actions):
        action_td = (td if step == 0 else td["next"]).clone()
        action_td["action"] = torch.tensor(index_of[action])
        td = env.step(action_td)
    return env


class TestAgentActionPrecedesNextBarTrigger:
    """The agent's order fills at close(N); bar N+1 unfolds after it (#292).

    _step checked the position held *coming into* the step against bar N+1 before
    running the agent's action, so a held position whose bracket or liquidation fired
    on N+1 discarded that action outright. The account ended the step in a position it
    had asked to leave, at a price it never chose.

    Balances are pinned, not just action types: booking the agent's close leg at the
    bracket price rather than the market close it selected is the same class of error
    and leaves every action_type assertion green.
    """

    def test_switch_is_not_cancelled_by_the_incoming_positions_take_profit(self):
        """Incoming long's TP sits inside bar 20; the agent flips to short on bar 19.

        The long is closed by the agent at close(19)=100, so its TP must never fire.
        The short opens at 100 and bar 20's spike stops IT out at 102.5 instead --
        a 2.5% adverse move at 2x on the full portfolio, so 10000 -> 9500.
        """
        env = _run_sltp(
            # reset caches bar 10, so step k executes at bar 10+k: open at step 5 (bar 15),
            # switch at step 9 (bar 19), and bar 20 is the step-9 exposure bar.
            [HOLD] * 5 + [LONG_TIGHT] + [HOLD] * 3 + [SHORT_TIGHT] + [HOLD] * 2,
            leverage=2, sl_levels=[-0.025, -0.50], tp_levels=[0.025, 0.50], wick_high=120.0,
        )
        # sltp_sl, not sltp_tp: the bar stopped out the SHORT the agent asked for,
        # rather than taking profit on the long it had just closed.
        assert env.history.action_types[-3] == "sltp_sl"
        assert env.position.position_size == 0
        assert env.balance == pytest.approx(9500.0)

    def test_liquidation_does_not_cancel_the_agents_close(self):
        """Same defect, liquidation instead of a bracket -- and here it is total.

        A 10x long liquidates around 90.4 and bar 20 wicks to 88. The agent closes at
        close(19)=100, so the position is gone before that bar and the account keeps its
        capital; previously the pre-action liquidation check wiped it to near zero.
        """
        env = _run_sltp(
            [HOLD] * 5 + [("long", -0.50, 0.50)] + [HOLD] * 3 + [CLOSE] + [HOLD] * 2,
            leverage=10, sl_levels=[-0.50], tp_levels=[0.50], wick_low=88.0,
            include_close=True,
        )
        assert "liquidation" not in env.history.action_types, (
            "the agent closed at close(19); bar 20 must not liquidate a position it had left"
        )
        assert env.position.position_size == 0
        assert env.balance == pytest.approx(10000.0), "a flat round trip must not move the balance"

    def test_agent_close_beats_a_bracket_firing_on_the_same_bar(self):
        """The reorder applies to close actions, not only switches.

        The agent closes at close(19)=100 for a flat round trip; the long's TP inside
        bar 20 must not book a profit it did not ask for.
        """
        env = _run_sltp(
            [HOLD] * 5 + [LONG_TIGHT] + [HOLD] * 3 + [CLOSE] + [HOLD] * 2,
            leverage=2, sl_levels=[-0.025, -0.50], tp_levels=[0.025, 0.50],
            wick_high=120.0, include_close=True,
        )
        # Recorded as a close, not sltp_tp: reverting the reorder books the bracket
        # instead, which both changes this label and moves the balance below.
        assert env.history.action_types[-3] == "close"
        assert env.position.position_size == 0
        assert env.balance == pytest.approx(10000.0), "a flat round trip must not move the balance"


class TestLiquidationTakesPriorityOverBrackets:
    """When one bar breaches both a bracket and the liquidation price, the liquidation
    must win (#298).

    Nothing held this. A double-breach bar has existed since
    TestSLTPRegression::test_liquidation_triggers_on_next_bar_futures, but no test
    asserted anything that *distinguishes* the two exits, so swapping the checks left
    the whole offline suite green. All four quadrants of direction x trigger are
    covered because each is invisible to the other three: a reorder scoped to one of
    them passes every test the others pin.

    What this pins is the env's documented pessimistic convention, not a claim about
    physical fill order; #300 tracks whether the rule should be conditional. Either
    way this class pins today's behaviour.
    """

    # The action tuple is spelled out rather than derived: create_sltp_action_map
    # stores a short as ("short", tp, sl) -- the pair swapped, not negated -- so a
    # derived tuple raises KeyError rather than opening the wrong position. What CAN
    # drift silently is the levels: move a bracket outside the wick and the row still
    # liquidates, still balances 400, and no longer tests a double breach. The
    # armed-bracket assertion is what pins that.
    @pytest.mark.parametrize(
        "action,stoploss_levels,takeprofit_levels,wick_low,wick_high,unwanted,armed", [
            # 10x long: liquidation 90.4, SL 95, low 88 breaches both.
            (("long", -0.05, 0.50), [-0.05], [0.50], 88.0, None, "sltp_sl",
             ("stop_loss", 95.0)),
            # 10x short: liquidation 109.6, SL 105, high 112 breaches both. The short
            # branches of _check_liquidation, _check_sltp_trigger and
            # _execute_liquidation are all separate code from the long ones.
            (("short", 0.05, -0.50), [-0.50], [0.05], None, 112.0, "sltp_sl",
             ("stop_loss", 105.0)),
            # 10x long: liquidation 90.4 on the low, TP 150 on the high. The SL is put
            # at 50, outside the bar, so the documented SL-before-TP bias cannot mask
            # the TP.
            (("long", -0.50, 0.50), [-0.50], [0.50], 88.0, 155.0, "sltp_tp",
             ("take_profit", 150.0)),
            # 10x short: liquidation 109.6 on the high, TP 90 on the low, SL 150
            # outside the bar. Only a mutation scoped to BOTH short and tp reaches
            # this one, which is why it is the quadrant that stayed hidden longest.
            (("short", 0.50, -0.10), [-0.10], [0.50], 88.0, 112.0, "sltp_tp",
             ("take_profit", 90.0)),
        ],
        ids=["long-stop", "short-stop", "long-take-profit", "short-take-profit"],
    )
    def test_a_bar_breaching_both_liquidates_rather_than_closing_at_the_bracket(
        self, action, stoploss_levels, takeprofit_levels, wick_low, wick_high,
        unwanted, armed
    ):
        """All four liquidate 1000 units at a liquidation price 9.6 from entry, so all
        four balance 400 -- one number, because the liquidation arm is agnostic to both
        direction and bracket type. The discrimination is in the counterfactuals, which
        are row-specific: the stop would leave 5000, the long TP 60000, the short TP
        20000.

        The label is what fires today; the balance is the backstop that survives a
        future exit type reusing it.
        """
        env = _run_sltp(
            [HOLD] * 5 + [action] + [HOLD] * 4,
            leverage=10, sl_levels=stoploss_levels, tp_levels=takeprofit_levels,
            wick_low=wick_low, wick_high=wick_high,
        )
        assert "liquidation" in env.history.action_types, (
            f"the {unwanted} pre-empted the liquidation on a bar that breached both"
        )
        # Liquidation deliberately leaves the brackets armed, so they can still be read
        # back: this pins that the bracket really did sit inside the wick.
        attr, price = armed
        assert getattr(env, attr) == pytest.approx(price)
        assert env.balance == pytest.approx(400.0), (
            "closed at the liquidation price; anything else means the bracket won"
        )
