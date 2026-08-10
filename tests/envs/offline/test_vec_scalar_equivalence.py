"""
Exhaustive equivalence test: SequentialTradingEnv == VectorizedSequentialTradingEnv(num_envs=1).

Runs both environments with identical configs and action sequences,
comparing ALL observable state at every step. Any divergence is a bug.
"""

import pandas as pd
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


# Both engines compute money in float64 (MONEY_DTYPE); measured worst-case disagreement
# across both equivalence files is under 1e-12, so 1e-9 leaves ~1000x headroom. The previous 5e-4 was
# sized to absorb float32-vs-float64 ULP, which no longer exists.
EQUIV_ATOL = 1e-9
EQUIV_RTOL = 1e-9


def _compare_state(scalar, vec):
    """Compare all observable state between scalar and vectorized envs.

    Uses numpy-style allclose: |s - v| <= atol + rtol * max(|s|, |v|).

    Returns list of (field, scalar_val, vec_val) mismatches.
    """
    mismatches = []

    def check(field, s_val, v_val, atol=EQUIV_ATOL, rtol=EQUIV_RTOL):
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
    mismatches = _compare_state(scalar, vec)
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

        # Compare rewards
        r_s = td_s["next"]["reward"].item()
        r_v = td_v["next"]["reward"].squeeze().item()
        r_diff = abs(r_s - r_v)
        if r_diff > EQUIV_ATOL + EQUIV_RTOL * max(abs(r_s), abs(r_v)):
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
            if diff > EQUIV_ATOL + EQUIV_RTOL * max(abs(s_val), abs(v_val)):
                all_mismatches.append(
                    f"[{label}] Step {step+1} account_state[{name}]: scalar={s_val:.6f} vec={v_val:.6f} diff={diff:.6f}"
                )

        # Compare market data observations
        for key in td_s["next"].keys():
            if not str(key).startswith("market_data_"):
                continue
            md_s = td_s["next"][key]
            md_v = td_v["next"][key].squeeze(0)
            if not torch.allclose(md_s, md_v, atol=EQUIV_ATOL, rtol=EQUIV_RTOL):
                max_diff = (md_s - md_v).abs().max().item()
                all_mismatches.append(
                    f"[{label}] Step {step+1} {key}: max_diff={max_diff:.6f}"
                )

        # Compare internal state
        mismatches = _compare_state(scalar, vec)
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
# ACTION-LEVEL PRECISION
# ============================================================================


class TestScalarVecEquivalenceIntermediateLevel:
    """An intermediate action level must reach position sizing undamaged.

    Every action_levels value elsewhere in this file is in {-1, 0, 1}, exact in float32
    and float64 alike, which leaves MONEY_DTYPE on _action_levels_tensor unfalsifiable --
    reverting that one line to float32 keeps the whole suite green. 0.1 is not exact:
    float32 holds it as 0.10000000149, putting a 1.5e-8 relative error into the notional.

    Only the open from flat is compared here; resizing is covered by
    TestScalarVecEquivalenceResize below.
    """

    def test_open_at_intermediate_level_matches_scalar(self):
        # Awkward cash and price on purpose: at round numbers the level's float32 error
        # rounds away and the test silently stops discriminating.
        price, cash = 97.3, 1234.5678
        n = 60
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
            "open": price, "high": price, "low": price, "close": price, "volume": 1000.0,
        }, index=range(n))
        common = dict(
            action_levels=[0.0, 0.1, 1.0], leverage=3, initial_cash=cash,
            time_frames=[TF_1MIN], window_sizes=[10], execute_on=TF_1MIN,
            transaction_fee=0.0007, seed=42, max_traj_length=20, random_start=False,
        )
        scalar = SequentialTradingEnv(df, SequentialTradingEnvConfig(**common))
        vec = VectorizedSequentialTradingEnv(
            df, VectorizedSequentialTradingEnvConfig(num_envs=1, **common)
        )
        td_s, td_v = scalar.reset(), vec.reset()
        td_s["action"] = torch.tensor(1)
        td_v["action"] = torch.tensor([1])
        scalar.step(td_s)
        vec.step(td_v)

        s_size = scalar.position.position_size
        v_size = float(vec._position_sizes[0])
        assert s_size != 0, "the level did not open a position -- nothing is being compared"
        rel = abs(s_size - v_size) / abs(s_size)
        assert rel < 1e-12, (
            f"position_size scalar={s_size!r} vec={v_size!r} rel={rel:.3e} -- an "
            "intermediate action level is being rounded before it reaches the notional"
        )
        scalar.close()
        vec.close()


# ============================================================================
# PARTIAL RESIZE
# ============================================================================


class TestScalarVecEquivalenceResize:
    """Changing an open position's size must trade only the delta in both engines (#274).

    The vectorized env used to route every target change through close-then-reopen, so a
    same-direction resize paid a full round trip of fees and threw away the entry price --
    which also moves the liquidation price and unrealized_pnl_pct. Measured 3.46% of
    portfolio over 8 steps at leverage 5.

    This is the axis `action_levels` exists for and no cell ever used: with the default
    {-1, 0, 1} every action is flat-or-full, so the resize branch is unreachable and the
    two implementations coincide.
    """

    LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0]

    @pytest.mark.parametrize("leverage,actions,label", [
        (1, [4, 3, 4, 3, 4, 3], "oscillate-full-half"),
        (5, [4, 3, 4, 3, 4, 3, 4, 3], "oscillate-levered"),
        (5, [4, 4, 3, 3, 4, 4, 3, 3], "gradual"),
        (5, [4, 3, 1, 0, 1, 3, 4, 0], "through-flat-and-short"),
    ], ids=lambda v: v if isinstance(v, str) else None)
    def test_resize_matches_scalar(self, sample_ohlcv_df, leverage, actions, label):
        mismatches = _run_sequence(
            sample_ohlcv_df, actions, leverage=leverage, fee=0.001,
            action_levels=self.LEVELS, max_traj=60, label=f"resize-{label}",
        )
        assert not mismatches, "\n".join(mismatches)

    def test_resize_pays_only_the_delta_fee(self, sample_ohlcv_df):
        """Halving a position must cost the fee on the half, not on a full round trip.

        The equivalence cells above would also fail if BOTH engines started closing and
        reopening, so this pins the absolute cost rather than just agreement.
        """
        scalar, vec = _make_pair(
            sample_ohlcv_df, leverage=1, fee=0.01, action_levels=self.LEVELS, max_traj=20
        )
        td_s, td_v = scalar.reset(), vec.reset()
        td_s["action"] = torch.tensor(4)  # full
        td_v["action"] = torch.tensor([4])
        td_s, td_v = scalar.step(td_s)["next"], vec.step(td_v)["next"]
        before = scalar.balance + abs(scalar.position.position_size) * scalar.position.entry_price

        td_s["action"] = torch.tensor(3)  # half
        td_v["action"] = torch.tensor([3])
        scalar.step(td_s)
        vec.step(td_v)

        # A close-and-reopen at 1% would cost ~1.5% of notional here; a delta trade
        # touches only the half being closed.
        held = abs(scalar.position.position_size) * scalar.position.entry_price
        cost = before - (scalar.balance + held)
        assert cost < 0.011 * before, (
            f"resize cost {cost:.4f} on a base of {before:.4f} -- that is a round trip, "
            "not a delta trade"
        )
        assert abs(float(vec._position_sizes[0]) - scalar.position.position_size) < 1e-9
        scalar.close()
        vec.close()


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
