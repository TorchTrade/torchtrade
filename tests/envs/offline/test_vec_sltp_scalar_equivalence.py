"""
Exhaustive equivalence test: SequentialTradingEnvSLTP == VectorizedSequentialTradingEnvSLTP(num_envs=1).

Runs both environments with identical configs and action sequences,
comparing ALL observable state at every step. Any divergence is a bug.
"""

import pytest
import torch

from torchtrade.envs.offline import (
    SequentialTradingEnvSLTP,
    SequentialTradingEnvSLTPConfig,
    VectorizedSequentialTradingEnvSLTP,
    VectorizedSequentialTradingEnvSLTPConfig,
)
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.conftest import simple_feature_fn

TF_1MIN = TimeFrame(1, TimeFrameUnit.Minute)


def _make_sltp_pair(
    df,
    leverage=1,
    fee=0.0,
    sl_levels=None,
    tp_levels=None,
    max_traj=40,
    include_hold=True,
    include_close=False,
):
    """Create matched scalar and N=1 vectorized SLTP envs."""
    if sl_levels is None:
        sl_levels = [-0.025, -0.05]
    if tp_levels is None:
        tp_levels = [0.05, 0.1]

    scalar = SequentialTradingEnvSLTP(
        df,
        SequentialTradingEnvSLTPConfig(
            leverage=leverage,
            stoploss_levels=sl_levels,
            takeprofit_levels=tp_levels,
            include_hold_action=include_hold,
            include_close_action=include_close,
            initial_cash=1000,
            time_frames=[TF_1MIN],
            window_sizes=[10],
            execute_on=TF_1MIN,
            transaction_fee=fee,
            slippage=0.0,
            seed=42,
            max_traj_length=max_traj,
            random_start=False,
        ),
        simple_feature_fn,
    )
    vec = VectorizedSequentialTradingEnvSLTP(
        df,
        VectorizedSequentialTradingEnvSLTPConfig(
            num_envs=1,
            leverage=leverage,
            stoploss_levels=sl_levels,
            takeprofit_levels=tp_levels,
            include_hold_action=include_hold,
            include_close_action=include_close,
            initial_cash=1000,
            time_frames=[TF_1MIN],
            window_sizes=[10],
            execute_on=TF_1MIN,
            transaction_fee=fee,
            slippage=0.0,
            seed=42,
            max_traj_length=max_traj,
            random_start=False,
        ),
        simple_feature_fn,
    )
    return scalar, vec


def _compare_sltp_state(scalar, vec, step, label, atol=5e-4, rtol=1e-3):
    """Compare all observable state between scalar and vectorized SLTP envs.

    Returns list of (field, scalar_val, vec_val) mismatches.
    """
    mismatches = []

    def check(field, s_val, v_val, atol=atol, rtol=rtol):
        diff = abs(s_val - v_val)
        tol = atol + rtol * max(abs(s_val), abs(v_val))
        if diff > tol:
            mismatches.append((field, s_val, v_val, diff))

    check("balance", scalar.balance, vec._balances[0].item())
    check("position_size", scalar.position.position_size, vec._position_sizes[0].item())
    check("entry_price", scalar.position.entry_price, vec._entry_prices[0].item())
    check("hold_counter", scalar.position.hold_counter, vec._hold_counters[0].item(), atol=0)

    s_pv = scalar._get_portfolio_value()
    v_pv = vec._portfolio_values[0].item()
    check("portfolio_value", s_pv, v_pv)

    check("stop_loss", scalar.stop_loss, vec._sl_prices[0].item())
    check("take_profit", scalar.take_profit, vec._tp_prices[0].item())

    return mismatches


def _run_sltp_sequence(
    df,
    action_indices,
    leverage=1,
    fee=0.0,
    sl_levels=None,
    tp_levels=None,
    max_traj=40,
    label="",
    include_hold=True,
    include_close=False,
):
    """Run a sequence of actions through both envs and compare at every step."""
    scalar, vec = _make_sltp_pair(
        df,
        leverage=leverage,
        fee=fee,
        sl_levels=sl_levels,
        tp_levels=tp_levels,
        max_traj=max_traj,
        include_hold=include_hold,
        include_close=include_close,
    )
    all_mismatches = []

    td_s = scalar.reset()
    td_v = vec.reset()

    atol, rtol = 5e-4, 1e-3

    # Compare initial state
    mismatches = _compare_sltp_state(scalar, vec, 0, label)
    for field, s_val, v_val, diff in mismatches:
        all_mismatches.append(
            f"[{label}] Step 0 RESET {field}: scalar={s_val:.6f} vec={v_val:.6f} diff={diff:.6f}"
        )

    for step, action_idx in enumerate(action_indices):
        # Step scalar
        action_td_s = td_s.clone() if "next" not in td_s.keys() else td_s["next"].clone()
        action_td_s["action"] = torch.tensor(action_idx)
        td_s = scalar.step(action_td_s)

        # Step vectorized
        action_td_v = td_v.clone() if "next" not in td_v.keys() else td_v["next"].clone()
        action_td_v["action"] = torch.tensor([action_idx])
        td_v = vec.step(action_td_v)

        # Compare rewards
        r_s = td_s["next"]["reward"].item()
        r_v = td_v["next"]["reward"].squeeze().item()
        r_diff = abs(r_s - r_v)
        if r_diff > atol + rtol * max(abs(r_s), abs(r_v)):
            all_mismatches.append(
                f"[{label}] Step {step+1} reward: scalar={r_s:.6f} vec={r_v:.6f} diff={r_diff:.6f}"
            )

        # Compare done signals
        for sig in ["done", "terminated", "truncated"]:
            s_sig = td_s["next"][sig].item()
            v_sig = td_v["next"][sig].item()
            if s_sig != v_sig:
                all_mismatches.append(
                    f"[{label}] Step {step+1} {sig}: scalar={s_sig} vec={v_sig}"
                )

        # Compare account state
        as_s = td_s["next"]["account_state"]
        as_v = td_v["next"]["account_state"].squeeze(0)
        as_names = [
            "exposure_pct", "position_direction", "unrealized_pnl_pct",
            "holding_time", "leverage", "distance_to_liq",
        ]
        for i, name in enumerate(as_names):
            s_val = as_s[i].item()
            v_val = as_v[i].item()
            diff = abs(s_val - v_val)
            if diff > atol + rtol * max(abs(s_val), abs(v_val)):
                all_mismatches.append(
                    f"[{label}] Step {step+1} account_state[{name}]: scalar={s_val:.6f} vec={v_val:.6f} diff={diff:.6f}"
                )

        # Compare internal state
        mismatches = _compare_sltp_state(scalar, vec, step + 1, label)
        for field, s_val, v_val, diff in mismatches:
            all_mismatches.append(
                f"[{label}] Step {step+1} {field}: scalar={s_val:.6f} vec={v_val:.6f} diff={diff:.6f}"
            )

        if td_s["next"]["done"].item() or td_v["next"]["done"].item():
            break

    scalar.close()
    vec.close()
    return all_mismatches


# ============================================================================
# SPOT MODE EQUIVALENCE
# ============================================================================


class TestSLTPScalarVecEquivalenceSpot:
    """Spot mode SLTP equivalence (leverage=1)."""

    def test_hold_only(self, sample_ohlcv_df):
        """All-hold: both envs should remain flat."""
        actions = [0] * 30
        mismatches = _run_sltp_sequence(
            sample_ohlcv_df, actions, label="sltp-spot-hold"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_open_bracket_and_hold(self, sample_ohlcv_df):
        """Open long bracket then hold — PV should track identically."""
        actions = [1] + [0] * 25
        mismatches = _run_sltp_sequence(
            sample_ohlcv_df, actions, label="sltp-spot-open-hold"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_different_brackets(self, sample_ohlcv_df):
        """Open different SL/TP combinations across steps."""
        # With 2 SL x 2 TP = 4 combos + hold = 5 actions (indices 0-4)
        actions = [1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0]
        mismatches = _run_sltp_sequence(
            sample_ohlcv_df, actions, label="sltp-spot-diff-brackets"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_with_fees(self, sample_ohlcv_df):
        """Bracket operations with transaction fees."""
        actions = [1, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0]
        mismatches = _run_sltp_sequence(
            sample_ohlcv_df, actions, fee=0.001, label="sltp-spot-fees"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_sl_trigger_downtrend(self, trending_down_df):
        """Long position SL should trigger identically in downtrend."""
        actions = [1] + [0] * 30
        mismatches = _run_sltp_sequence(
            trending_down_df, actions,
            sl_levels=[-0.01], tp_levels=[0.1],
            label="sltp-spot-sl-downtrend"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_tp_trigger_uptrend(self, trending_up_df):
        """Long position TP should trigger identically in uptrend."""
        actions = [1] + [0] * 30
        mismatches = _run_sltp_sequence(
            trending_up_df, actions,
            sl_levels=[-0.1], tp_levels=[0.01],
            label="sltp-spot-tp-uptrend"
        )
        assert not mismatches, "\n".join(mismatches)


# ============================================================================
# FUTURES MODE EQUIVALENCE
# ============================================================================


class TestSLTPScalarVecEquivalenceFutures:
    """Futures mode SLTP equivalence (leverage>1)."""

    def test_long_bracket(self, sample_ohlcv_df):
        """Open long bracket and hold in futures."""
        actions = [1] + [0] * 25
        mismatches = _run_sltp_sequence(
            sample_ohlcv_df, actions, leverage=10,
            label="sltp-futures-long-bracket"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_short_bracket(self, sample_ohlcv_df):
        """Open short bracket and hold in futures."""
        # With 2 SL x 2 TP: indices 0=hold, 1-4=long, 5-8=short
        actions = [5] + [0] * 25
        mismatches = _run_sltp_sequence(
            sample_ohlcv_df, actions, leverage=10,
            label="sltp-futures-short-bracket"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_long_sl_downtrend(self, trending_down_df):
        """Long SL in downtrend — should trigger identically."""
        actions = [1] + [0] * 30
        mismatches = _run_sltp_sequence(
            trending_down_df, actions, leverage=10,
            sl_levels=[-0.01], tp_levels=[0.1],
            label="sltp-futures-long-sl"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_short_sl_uptrend(self, trending_up_df):
        """Short SL in uptrend — should trigger identically."""
        # With 1 SL x 1 TP + futures: 0=hold, 1=long, 2=short
        actions = [2] + [0] * 30
        mismatches = _run_sltp_sequence(
            trending_up_df, actions, leverage=10,
            sl_levels=[-0.01], tp_levels=[0.1],
            label="sltp-futures-short-sl"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_long_tp_uptrend(self, trending_up_df):
        """Long TP in uptrend — should trigger identically."""
        actions = [1] + [0] * 30
        mismatches = _run_sltp_sequence(
            trending_up_df, actions, leverage=10,
            sl_levels=[-0.1], tp_levels=[0.01],
            label="sltp-futures-long-tp"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_short_tp_downtrend(self, trending_down_df):
        """Short TP in downtrend — should trigger identically."""
        actions = [2] + [0] * 30
        mismatches = _run_sltp_sequence(
            trending_down_df, actions, leverage=10,
            sl_levels=[-0.1], tp_levels=[0.01],
            label="sltp-futures-short-tp"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_direction_switch(self, sample_ohlcv_df):
        """Long to short direction switch."""
        # 0=hold, 1=long, 2=short (with 1 SL x 1 TP)
        actions = [1, 0, 0, 2, 0, 0, 1, 0, 0]
        mismatches = _run_sltp_sequence(
            sample_ohlcv_df, actions, leverage=10,
            sl_levels=[-0.05], tp_levels=[0.1],
            label="sltp-futures-switch"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_with_fees(self, sample_ohlcv_df):
        """Futures bracket operations with fees."""
        actions = [1, 0, 0, 5, 0, 0, 0, 1, 0]
        mismatches = _run_sltp_sequence(
            sample_ohlcv_df, actions, leverage=10, fee=0.001,
            label="sltp-futures-fees"
        )
        assert not mismatches, "\n".join(mismatches)


# ============================================================================
# LEVERAGE LEVELS
# ============================================================================


class TestSLTPScalarVecEquivalenceLeverages:
    """Verify equivalence across different leverage levels."""

    @pytest.mark.parametrize("leverage", [2, 5, 10, 25], ids=lambda l: f"lev{l}")
    @pytest.mark.parametrize("direction", ["long", "short"], ids=["long", "short"])
    def test_bracket_at_leverage(self, sample_ohlcv_df, leverage, direction):
        """Open bracket at various leverage levels and directions."""
        # 0=hold, 1=long, 2=short (with 1 SL x 1 TP)
        open_action = 1 if direction == "long" else 2
        actions = [open_action] + [0] * 11
        mismatches = _run_sltp_sequence(
            sample_ohlcv_df, actions, leverage=leverage, fee=0.001,
            sl_levels=[-0.05], tp_levels=[0.1],
            label=f"sltp-{direction}-lev{leverage}"
        )
        assert not mismatches, "\n".join(mismatches)


# ============================================================================
# CLOSE ACTION EQUIVALENCE
# ============================================================================


class TestSLTPScalarVecEquivalenceCloseAction:
    """Verify equivalence with include_close_action=True."""

    @pytest.mark.parametrize("leverage", [1, 10], ids=["spot", "futures"])
    def test_close_action(self, sample_ohlcv_df, leverage):
        """Close action should behave identically in both envs."""
        # With include_close=True, 1 SL x 1 TP:
        # spot: 0=hold, 1=close, 2=long (3 actions)
        # futures: 0=hold, 1=close, 2=long, 3=short (4 actions)
        actions = [2, 0, 0, 1, 0, 2, 0, 0, 1, 0]
        mismatches = _run_sltp_sequence(
            sample_ohlcv_df, actions, leverage=leverage,
            sl_levels=[-0.05], tp_levels=[0.1],
            include_close=True,
            label=f"sltp-close-action-lev{leverage}"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_close_on_flat(self, sample_ohlcv_df):
        """Close action on flat position should be a no-op."""
        # Close without ever opening — should be equivalent to hold
        actions = [1, 0, 1, 0]  # close=1 when include_close=True
        mismatches = _run_sltp_sequence(
            sample_ohlcv_df, actions, leverage=10,
            sl_levels=[-0.05], tp_levels=[0.1],
            include_close=True,
            label="sltp-close-on-flat"
        )
        assert not mismatches, "\n".join(mismatches)


# ============================================================================
# TRENDING MARKET EQUIVALENCE
# ============================================================================


class TestSLTPScalarVecEquivalenceTrending:
    """Verify equivalence in trending markets with tight SL/TP triggers."""

    def test_reopen_after_trigger(self, trending_up_df):
        """After TP trigger, reopen and hold — state should match."""
        actions = [1] + [0] * 10 + [1] + [0] * 10
        mismatches = _run_sltp_sequence(
            trending_up_df, actions, leverage=10,
            sl_levels=[-0.1], tp_levels=[0.005],
            max_traj=40,
            label="sltp-trending-reopen"
        )
        assert not mismatches, "\n".join(mismatches)


# ============================================================================
# LIQUIDATION EQUIVALENCE (FUTURES)
# ============================================================================


class TestSLTPScalarVecEquivalenceLiquidation:
    """Verify liquidation behavior matches between scalar and vectorized SLTP envs."""

    @pytest.mark.parametrize("direction,open_action", [
        ("long", 1), ("short", 2),
    ], ids=["long", "short"])
    def test_liquidation_equivalence(self, direction, open_action, trending_down_df, trending_up_df):
        """Both envs should liquidate at the same step with the same balance."""
        df = trending_down_df if direction == "long" else trending_up_df
        actions = [open_action] + [0] * 150
        mismatches = _run_sltp_sequence(
            df, actions, leverage=20, fee=0.001,
            sl_levels=[-0.1], tp_levels=[0.2],
            max_traj=200,
            label=f"sltp-liquidation-{direction}"
        )
        assert not mismatches, "\n".join(mismatches)
