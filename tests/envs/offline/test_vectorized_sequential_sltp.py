"""
Tests for VectorizedSequentialTradingEnvSLTP.

Verifies:
- Action space size (spot/futures × different SL/TP level counts)
- Config validation (positive SL raises, negative TP raises)
- TorchRL spec compliance (check_env_specs)
- SL/TP trigger mechanics
- Partial reset preserves SL/TP state
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchrl.envs.utils import check_env_specs

from torchtrade.envs.offline import (
    VectorizedSequentialTradingEnvSLTP,
    VectorizedSequentialTradingEnvSLTPConfig,
)
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.conftest import simple_feature_fn

TF_1MIN = TimeFrame(1, TimeFrameUnit.Minute)


def _make_sltp_vec_env(df, leverage=1, num_envs=4, fee=0.0, sl_levels=None, tp_levels=None, max_traj=50):
    """Create a vectorized SLTP env for testing."""
    if sl_levels is None:
        sl_levels = [-0.025, -0.05]
    if tp_levels is None:
        tp_levels = [0.05, 0.1]

    config = VectorizedSequentialTradingEnvSLTPConfig(
        num_envs=num_envs,
        leverage=leverage,
        stoploss_levels=sl_levels,
        takeprofit_levels=tp_levels,
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
    return VectorizedSequentialTradingEnvSLTP(df, config, simple_feature_fn)


# ============================================================================
# ACTION SPACE TESTS
# ============================================================================


class TestVecSLTPActionSpace:
    """Tests for SLTP action space generation."""

    @pytest.mark.parametrize("leverage,sl_count,tp_count", [
        (1, 1, 1),
        (1, 2, 2),
        (1, 3, 3),
        (10, 1, 1),
        (10, 2, 2),
        (10, 3, 3),
    ], ids=[
        "spot-1x1", "spot-2x2", "spot-3x3",
        "futures-1x1", "futures-2x2", "futures-3x3",
    ])
    def test_action_space_size(self, sample_ohlcv_df, leverage, sl_count, tp_count):
        """Action space: 1 + combos (spot) or 1 + 2*combos (futures)."""
        sl_levels = [-0.01 * (i + 1) for i in range(sl_count)]
        tp_levels = [0.01 * (i + 1) for i in range(tp_count)]

        env = _make_sltp_vec_env(
            sample_ohlcv_df, leverage=leverage, num_envs=2,
            sl_levels=sl_levels, tp_levels=tp_levels,
        )
        combos = sl_count * tp_count
        sides = 2 if leverage > 1 else 1
        expected = 1 + combos * sides
        assert env.action_spec.n == expected
        env.close()


# ============================================================================
# CONFIG VALIDATION
# ============================================================================


class TestVecSLTPConfigValidation:
    """Config validation tests."""

    @pytest.mark.parametrize("sl,tp,match", [
        ([0.05], None, "Stop-loss"),
        (None, [-0.05], "Take-profit"),
        ([0.0], None, "Stop-loss"),
        (None, [0.0], "Take-profit"),
    ], ids=["positive-sl", "negative-tp", "zero-sl", "zero-tp"])
    def test_invalid_levels_raises(self, sl, tp, match):
        """Invalid SL/TP levels must raise ValueError."""
        kwargs = {}
        if sl is not None:
            kwargs["stoploss_levels"] = sl
        if tp is not None:
            kwargs["takeprofit_levels"] = tp
        with pytest.raises(ValueError, match=match):
            VectorizedSequentialTradingEnvSLTPConfig(**kwargs)


# ============================================================================
# SPEC COMPLIANCE
# ============================================================================


class TestVecSLTPSpecs:
    """TorchRL spec compliance tests."""

    @pytest.mark.parametrize("leverage", [1, 10], ids=["spot", "futures"])
    def test_check_env_specs_passes(self, sample_ohlcv_df, leverage):
        """check_env_specs must pass for both spot and futures modes."""
        env = _make_sltp_vec_env(sample_ohlcv_df, leverage=leverage, num_envs=4)
        check_env_specs(env)
        env.close()


# ============================================================================
# SL/TP TRIGGER TESTS
# ============================================================================


class TestVecSLTPTriggers:
    """Tests for SL/TP trigger mechanics."""

    @pytest.mark.parametrize("trigger_type,sl,tp", [
        ("sl", [-0.005], [0.1]),
        ("tp", [-0.1], [0.005]),
    ], ids=["stop-loss", "take-profit"])
    def test_trigger_clears_position(self, trending_down_df, trending_up_df, trigger_type, sl, tp):
        """SL/TP trigger should close position and clear bracket prices."""
        # SL triggers in downtrend, TP triggers in uptrend
        df = trending_down_df if trigger_type == "sl" else trending_up_df
        env = _make_sltp_vec_env(
            df, leverage=1, num_envs=2,
            sl_levels=sl, tp_levels=tp, max_traj=100,
        )
        td = env.reset()

        # Open long
        action_td = td.clone()
        action_td["action"] = torch.ones(2, dtype=torch.long)
        td = env.step(action_td)
        assert (env._position_sizes > 0).all()
        assert (env._sl_prices > 0).all()

        # Hold until trigger fires
        for _ in range(50):
            action_td = td["next"].clone()
            action_td["action"] = torch.zeros(2, dtype=torch.long)
            td = env.step(action_td)
            if (env._position_sizes == 0).all():
                break

        assert (env._position_sizes == 0).all(), f"{trigger_type.upper()} should close positions"
        assert (env._sl_prices == 0).all(), "SL price should be cleared"
        assert (env._tp_prices == 0).all(), "TP price should be cleared"
        env.close()


# ============================================================================
# PARTIAL RESET TESTS
# ============================================================================


class TestVecSLTPPartialReset:
    """Tests for partial reset preserving SL/TP state."""

    def test_partial_reset_clears_sltp_for_reset_envs(self, sample_ohlcv_df):
        """Only reset envs should have SL/TP cleared."""
        env = _make_sltp_vec_env(sample_ohlcv_df, leverage=10, num_envs=4)
        td = env.reset()

        # Open positions in all envs
        action_td = td.clone()
        action_td["action"] = torch.ones(4, dtype=torch.long)
        td = env.step(action_td)

        sl_before = env._sl_prices.clone()
        tp_before = env._tp_prices.clone()
        assert (sl_before > 0).all(), "All envs should have SL set"

        # Partial reset envs 0 and 2
        reset_td = td["next"].clone()
        reset_mask = torch.tensor([True, False, True, False])
        reset_td["_reset"] = reset_mask.unsqueeze(-1)
        env.reset(reset_td)

        # Reset envs should have SL/TP cleared
        assert env._sl_prices[0].item() == 0.0
        assert env._sl_prices[2].item() == 0.0
        assert env._tp_prices[0].item() == 0.0
        assert env._tp_prices[2].item() == 0.0

        # Non-reset envs should keep SL/TP
        assert env._sl_prices[1] == sl_before[1]
        assert env._sl_prices[3] == sl_before[3]
        assert env._tp_prices[1] == tp_before[1]
        assert env._tp_prices[3] == tp_before[3]
        env.close()


# ============================================================================
# SAME DIRECTION HOLD TESTS
# ============================================================================


class TestVecSLTPSameDirectionHold:
    """Tests for same-direction hold behavior."""

    def test_repeated_long_holds(self, sample_ohlcv_df):
        """Repeating long action should hold, not reopen."""
        env = _make_sltp_vec_env(sample_ohlcv_df, leverage=10, num_envs=2)
        td = env.reset()

        # Open long
        action_td = td.clone()
        action_td["action"] = torch.ones(2, dtype=torch.long)
        td = env.step(action_td)
        pos_after_open = env._position_sizes.clone()
        sl_after_open = env._sl_prices.clone()

        # Repeat long — should hold
        for _ in range(5):
            action_td = td["next"].clone()
            action_td["action"] = torch.ones(2, dtype=torch.long)
            td = env.step(action_td)

        assert torch.allclose(env._position_sizes, pos_after_open)
        assert torch.allclose(env._sl_prices, sl_after_open)
        assert (env._hold_counters > 0).all()
        env.close()


# ============================================================================
# SL-BEFORE-TP PRIORITY TEST
# ============================================================================


class TestVecSLTPPriority:
    """SL must fire before TP when both could trigger on the same bar."""

    def test_sl_wins_over_tp_same_bar(self):
        """When a bar spans both SL and TP, SL should win (pessimistic bias)."""
        n = 30
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")
        # Steady price at 100 for first 12 bars (window=10 + 1 open + 1 trigger)
        close = np.full(n, 100.0)
        high = np.full(n, 100.5)
        low = np.full(n, 99.5)
        open_ = np.full(n, 100.0)

        # Bar 12 (after window): wide-range candle that spans both SL and TP
        # With SL=-2% (sl_price=98) and TP=+2% (tp_price=102):
        # low=97 < 98 (SL triggers) AND high=103 > 102 (TP would trigger)
        wide_bar = 12
        low[wide_bar] = 97.0
        high[wide_bar] = 103.0

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_, "high": high, "low": low,
            "close": close, "volume": np.ones(n) * 1000,
        })

        env = _make_sltp_vec_env(
            df, leverage=1, num_envs=1,
            sl_levels=[-0.02], tp_levels=[0.02], max_traj=20,
        )
        td = env.reset()

        # Open long at bar 10 (close=100)
        action_td = td.clone()
        action_td["action"] = torch.ones(1, dtype=torch.long)
        td = env.step(action_td)
        entry = env._entry_prices[0].item()
        sl_price = env._sl_prices[0].item()

        # Hold — next bar is the wide-range candle
        action_td = td["next"].clone()
        action_td["action"] = torch.zeros(1, dtype=torch.long)
        td = env.step(action_td)

        # Position should be closed (trigger fired)
        assert env._position_sizes[0].item() == 0, "Position should be closed"

        # Balance should reflect SL close (loss), not TP close (gain)
        # SL close: balance = 1000 + position_size * (sl_price - entry)
        # Since SL fires, balance < initial_cash (loss)
        assert env._balances[0].item() < 1000.0, (
            "SL should have won over TP — balance should reflect a loss"
        )
        env.close()
