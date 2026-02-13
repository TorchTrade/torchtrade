"""
Exhaustive equivalence test: SequentialTradingEnv == VectorizedSequentialTradingEnv(num_envs=1).

Runs both environments with identical configs and action sequences,
comparing ALL observable state at every step. Any divergence is a bug.
"""

import pytest
import torch

from torchtrade.envs.offline import (
    SequentialTradingEnv,
    SequentialTradingEnvConfig,
    VectorizedSequentialTradingEnv,
    VectorizedSequentialTradingEnvConfig,
)
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from tests.conftest import simple_feature_fn

TF_1MIN = TimeFrame(1, TimeFrameUnit.Minute)


def _make_pair(df, leverage=1, fee=0.0, action_levels=None, max_traj=40):
    """Create matched scalar and N=1 vectorized envs."""
    if action_levels is None:
        action_levels = [-1.0, 0.0, 1.0] if leverage > 1 else [0.0, 1.0]

    scalar = SequentialTradingEnv(
        df,
        SequentialTradingEnvConfig(
            leverage=leverage,
            action_levels=action_levels,
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
    vec = VectorizedSequentialTradingEnv(
        df,
        VectorizedSequentialTradingEnvConfig(
            num_envs=1,
            leverage=leverage,
            action_levels=action_levels,
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


def _compare_state(scalar, vec, step, label, atol=5e-4, rtol=1e-3):
    """Compare all observable state between scalar and vectorized envs.

    Uses numpy-style allclose: |s - v| <= atol + rtol * max(|s|, |v|).
    atol=5e-4 accommodates float32 (vec) vs float64 (scalar) precision differences
    (~1.2e-4 ULP at balance=1000).

    Returns list of (field, scalar_val, vec_val) mismatches.
    """
    mismatches = []

    def check(field, s_val, v_val, atol=atol, rtol=rtol):
        diff = abs(s_val - v_val)
        tol = atol + rtol * max(abs(s_val), abs(v_val))
        if diff > tol:
            mismatches.append((field, s_val, v_val, diff))

    # Balance
    check("balance", scalar.balance, vec._balances[0].item())

    # Position size
    check("position_size", scalar.position.position_size, vec._position_sizes[0].item())

    # Entry price
    check("entry_price", scalar.position.entry_price, vec._entry_prices[0].item())

    # Hold counter
    check("hold_counter", scalar.position.hold_counter, vec._hold_counters[0].item(), atol=0)

    # Portfolio value (use the cached value from vec, compute for scalar)
    s_pv = scalar._get_portfolio_value()
    v_pv = vec._portfolio_values[0].item()
    check("portfolio_value", s_pv, v_pv)

    return mismatches


def _run_sequence(df, action_indices, leverage=1, fee=0.0, action_levels=None, max_traj=40, label=""):
    """Run a sequence of actions through both envs and compare at every step.

    Returns list of all mismatches found across all steps.
    """
    scalar, vec = _make_pair(df, leverage=leverage, fee=fee, action_levels=action_levels, max_traj=max_traj)
    all_mismatches = []

    td_s = scalar.reset()
    td_v = vec.reset()

    # Compare initial state
    mismatches = _compare_state(scalar, vec, 0, label)
    for field, s_val, v_val, diff in mismatches:
        all_mismatches.append(f"[{label}] Step 0 RESET {field}: scalar={s_val:.6f} vec={v_val:.6f} diff={diff:.6f}")

    for step, action_idx in enumerate(action_indices):
        # Step scalar
        action_td_s = td_s.clone() if "next" not in td_s.keys() else td_s["next"].clone()
        action_td_s["action"] = torch.tensor(action_idx)
        td_s = scalar.step(action_td_s)

        # Step vectorized
        action_td_v = td_v.clone() if "next" not in td_v.keys() else td_v["next"].clone()
        action_td_v["action"] = torch.tensor([action_idx])
        td_v = vec.step(action_td_v)

        # Tolerances: float32 (vec) vs float64 (scalar) precision
        atol, rtol = 5e-4, 1e-3

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

        # Compare market data observations
        for key in td_s["next"].keys():
            if not str(key).startswith("market_data_"):
                continue
            md_s = td_s["next"][key]
            md_v = td_v["next"][key].squeeze(0)
            if not torch.allclose(md_s, md_v, atol=atol, rtol=rtol):
                max_diff = (md_s - md_v).abs().max().item()
                all_mismatches.append(
                    f"[{label}] Step {step+1} {key}: max_diff={max_diff:.6f}"
                )

        # Compare internal state
        mismatches = _compare_state(scalar, vec, step + 1, label)
        for field, s_val, v_val, diff in mismatches:
            all_mismatches.append(
                f"[{label}] Step {step+1} {field}: scalar={s_val:.6f} vec={v_val:.6f} diff={diff:.6f}"
            )

        # Stop if either is done
        if td_s["next"]["done"].item() or td_v["next"]["done"].item():
            break

    scalar.close()
    vec.close()
    return all_mismatches


# ============================================================================
# SPOT MODE EQUIVALENCE
# ============================================================================


class TestScalarVecEquivalenceSpot:
    """Exhaustive spot mode equivalence (leverage=1)."""

    def test_hold_only(self, sample_ohlcv_df):
        """All-hold sequence: both envs should produce identical zeros."""
        actions = [0] * 30  # action_levels=[0,1], index 0 = flat
        mismatches = _run_sequence(sample_ohlcv_df, actions, label="spot-hold")
        assert not mismatches, "\n".join(mismatches)

    def test_buy_and_hold(self, sample_ohlcv_df):
        """Buy then hold: PV should track identically."""
        actions = [1] + [1] * 25  # buy then hold (same-action opt)
        mismatches = _run_sequence(sample_ohlcv_df, actions, label="spot-buy-hold")
        assert not mismatches, "\n".join(mismatches)

    def test_buy_sell_cycles(self, sample_ohlcv_df):
        """Alternating buy/sell every step — exercises open/close cycle repeatedly."""
        actions = [1, 0] * 15
        mismatches = _run_sequence(sample_ohlcv_df, actions, label="spot-alternating")
        assert not mismatches, "\n".join(mismatches)

    def test_with_fees(self, sample_ohlcv_df):
        """Buy-hold-sell with transaction fees."""
        actions = [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0]
        mismatches = _run_sequence(sample_ohlcv_df, actions, fee=0.001, label="spot-fees")
        assert not mismatches, "\n".join(mismatches)

    def test_rapid_trading_with_fees(self, sample_ohlcv_df):
        """Rapid in/out with fees — maximum fee impact."""
        actions = [1, 0] * 10
        mismatches = _run_sequence(sample_ohlcv_df, actions, fee=0.005, label="spot-rapid-fees")
        assert not mismatches, "\n".join(mismatches)


# ============================================================================
# FUTURES MODE EQUIVALENCE
# ============================================================================


class TestScalarVecEquivalenceFutures:
    """Exhaustive futures mode equivalence (leverage>1)."""

    def test_hold_only(self, sample_ohlcv_df):
        """All-hold in futures mode."""
        actions = [1] * 30  # action_levels=[-1,0,1], index 1 = flat
        mismatches = _run_sequence(
            sample_ohlcv_df, actions, leverage=10, label="futures-hold"
        )
        assert not mismatches, "\n".join(mismatches)

    @pytest.mark.parametrize("open_idx,label", [(2, "long"), (0, "short")],
                             ids=["long", "short"])
    def test_open_and_hold(self, sample_ohlcv_df, open_idx, label):
        """Open position then hold — same-action optimization path."""
        actions = [open_idx] * 26
        mismatches = _run_sequence(
            sample_ohlcv_df, actions, leverage=10, label=f"futures-{label}-hold"
        )
        assert not mismatches, "\n".join(mismatches)

    @pytest.mark.parametrize("open_idx,label", [(2, "long"), (0, "short")],
                             ids=["long", "short"])
    def test_open_hold_close(self, sample_ohlcv_df, open_idx, label):
        """Open, hold, close to flat."""
        actions = [open_idx] * 5 + [1, 1, 1]
        mismatches = _run_sequence(
            sample_ohlcv_df, actions, leverage=10, label=f"futures-{label}-close"
        )
        assert not mismatches, "\n".join(mismatches)

    @pytest.mark.parametrize("actions,label", [
        ([2, 2, 2, 0, 0, 0], "l2s"),
        ([0, 0, 0, 2, 2, 2], "s2l"),
    ], ids=["l2s", "s2l"])
    def test_direction_switch(self, sample_ohlcv_df, actions, label):
        """Direction switch — close then reopen opposite."""
        mismatches = _run_sequence(
            sample_ohlcv_df, actions, leverage=10, label=f"futures-{label}"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_full_cycle(self, sample_ohlcv_df):
        """Long → flat → short → flat → long."""
        actions = [2, 2, 2, 1, 1, 0, 0, 0, 1, 1, 2, 2]
        mismatches = _run_sequence(
            sample_ohlcv_df, actions, leverage=10, label="futures-full-cycle"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_rapid_switches(self, sample_ohlcv_df):
        """Rapid direction changes every 2 steps."""
        actions = [2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 2, 2]
        mismatches = _run_sequence(
            sample_ohlcv_df, actions, leverage=10, label="futures-rapid-switch"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_with_fees(self, sample_ohlcv_df):
        """Full cycle with transaction fees."""
        actions = [2, 2, 2, 1, 0, 0, 0, 1, 2, 2, 1]
        mismatches = _run_sequence(
            sample_ohlcv_df, actions, leverage=10, fee=0.001, label="futures-fees"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_rapid_trading_with_fees(self, sample_ohlcv_df):
        """Rapid long/short with fees — maximum fee + margin impact."""
        actions = [2, 0, 2, 0, 1, 2, 1, 0, 1]
        mismatches = _run_sequence(
            sample_ohlcv_df, actions, leverage=10, fee=0.002, label="futures-rapid-fees"
        )
        assert not mismatches, "\n".join(mismatches)


# ============================================================================
# DIFFERENT LEVERAGE LEVELS
# ============================================================================


class TestScalarVecEquivalenceLeverages:
    """Verify equivalence across different leverage levels."""

    @pytest.mark.parametrize("open_idx,direction", [(2, "long"), (0, "short")],
                             ids=["long", "short"])
    @pytest.mark.parametrize("leverage", [2, 5, 10, 25, 50], ids=lambda l: f"lev{l}")
    def test_hold_close_at_leverage(self, sample_ohlcv_df, open_idx, direction, leverage):
        """Open→hold→close cycle at various leverage levels and directions."""
        actions = [open_idx] * 5 + [1, 1]
        mismatches = _run_sequence(
            sample_ohlcv_df, actions, leverage=leverage, fee=0.001,
            label=f"{direction}-lev{leverage}"
        )
        assert not mismatches, "\n".join(mismatches)


# ============================================================================
# TRENDING MARKET EQUIVALENCE
# ============================================================================


class TestScalarVecEquivalenceTrending:
    """Verify equivalence in trending markets (stronger price moves)."""

    def test_long_in_uptrend(self, trending_up_df):
        """Long in uptrend — positive PnL should match."""
        actions = [2] * 20
        mismatches = _run_sequence(
            trending_up_df, actions, leverage=10, label="long-uptrend"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_short_in_downtrend(self, trending_down_df):
        """Short in downtrend — positive PnL should match."""
        actions = [0] * 20
        mismatches = _run_sequence(
            trending_down_df, actions, leverage=10, label="short-downtrend"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_long_in_downtrend_with_fees(self, trending_down_df):
        """Long in downtrend with fees — negative PnL should match."""
        actions = [2, 2, 2, 2, 2, 1]
        mismatches = _run_sequence(
            trending_down_df, actions, leverage=10, fee=0.001,
            label="long-downtrend-fees"
        )
        assert not mismatches, "\n".join(mismatches)

    def test_short_in_uptrend_with_fees(self, trending_up_df):
        """Short in uptrend with fees — negative PnL should match."""
        actions = [0, 0, 0, 0, 0, 1]
        mismatches = _run_sequence(
            trending_up_df, actions, leverage=10, fee=0.001,
            label="short-uptrend-fees"
        )
        assert not mismatches, "\n".join(mismatches)


# ============================================================================
# LIQUIDATION EQUIVALENCE
# ============================================================================


class TestScalarVecEquivalenceLiquidation:
    """Verify liquidation behavior matches between scalar and vectorized envs."""

    @pytest.mark.parametrize("open_idx,direction", [(2, "long"), (0, "short")],
                             ids=["long", "short"])
    def test_liquidation_equivalence(self, open_idx, direction, trending_down_df, trending_up_df):
        """Both envs should liquidate at the same step with the same balance."""
        # Long gets liquidated in downtrend, short in uptrend
        df = trending_down_df if direction == "long" else trending_up_df
        actions = [open_idx] * 200
        mismatches = _run_sequence(
            df, actions, leverage=20, fee=0.001, max_traj=200,
            label=f"liquidation-{direction}"
        )
        assert not mismatches, "\n".join(mismatches)
