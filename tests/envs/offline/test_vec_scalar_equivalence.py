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

    # Signed levels only where shorts exist: at leverage 1 the env clips negatives and
    # warns, so a spot cell declaring them reads as an oversight.
    SPOT_LEVELS = [0.0, 0.5, 1.0]
    FUTURES_LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0]

    @pytest.mark.parametrize("leverage,levels,actions", [
        # same sequence spot and levered, so the cell pair isolates leverage
        (1, SPOT_LEVELS, [2, 1, 2, 1, 2, 1]),
        (5, FUTURES_LEVELS, [4, 3, 4, 3, 4, 3]),
        # shorts resize through the same branches with every sign flipped: closed_qty,
        # pnl and freed_margin all carry it. Nothing exercised that until this cell --
        # the sequence it replaced refused one short resize for affordability and held
        # the other inside the tolerance band, so both compared two idle engines.
        (5, FUTURES_LEVELS, [0, 1, 0, 1]),
        # open, decrease, close to flat, open short, flip, increase, flip
        (5, FUTURES_LEVELS, [4, 3, 2, 1, 3, 4, 0]),
        # 0.5 -> 0.505 is a 1% target change: inside POSITION_TOLERANCE_PCT while a
        # position is open. Both engines drop that step before classifying it, so the
        # canonical ageing call is the only thing that ages the position -- and nothing
        # else in the suite ever enters that branch holding anything. Re-adding either
        # engine's removed `hold_counter += 1` there passed all 691 offline tests.
        (5, [0.0, 0.5, 0.505, 1.0], [1, 2, 2, 1, 3]),
    ], ids=["oscillate-spot", "oscillate-levered", "oscillate-short", "through-flat",
            "tolerance-hold"])
    def test_resize_matches_scalar(self, sample_ohlcv_df, leverage, levels, actions):
        mismatches = _run_sequence(
            sample_ohlcv_df, actions, leverage=leverage, fee=0.001,
            action_levels=levels, max_traj=60, label=f"resize-lev{leverage}",
        )
        assert not mismatches, "\n".join(mismatches)

    def test_an_unaffordable_increase_changes_nothing_but_the_age(self):
        """An increase the balance cannot pay for must be refused whole, and still age.

        This is the case that forced ageing out of the trade branches into one call per
        step: the position is still held, so it must still age, but no trade branch runs
        to age it. An earlier version of this PR incremented inside the resize branch and
        got exactly this wrong in the vectorized env while the scalar env got it right.

        Reaching the affordability path takes care. Scaling up under fractional sizing is
        almost always affordable, because the target is derived from a portfolio value
        that already contains the position; and a small increase is swallowed by the 2%
        tolerance band before affordability is ever consulted. The drifting price here is
        the shape TestAffordabilitySlackOnScaleUp (fundamental/test_margin_accounting.py)
        uses: the delta is ~10 units against a ~0.4 tolerance, so the refusal is
        genuinely about cost.
        """
        drift, n = 1e-6, 40
        prices = [100.0 * (1 + drift) ** i for i in range(n)]
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
            "open": prices, "high": prices, "low": prices, "close": prices,
            "volume": [1000.0] * n,
        }, index=range(n))
        scalar, vec = _make_pair(
            df, leverage=2, fee=0.0, action_levels=[0.0, 0.5, 1.0], max_traj=20
        )
        td_s, td_v = scalar.reset(), vec.reset()
        td_s["action"] = torch.tensor(1)  # half
        td_v["action"] = torch.tensor([1])
        td_s, td_v = scalar.step(td_s)["next"], vec.step(td_v)["next"]
        size_before, balance_before = scalar.position.position_size, scalar.balance

        td_s["action"] = torch.tensor(2)  # full -- costs more than the balance
        td_v["action"] = torch.tensor([2])
        scalar.step(td_s)
        vec.step(td_v)

        for name, size, balance, held in (
            ("scalar", scalar.position.position_size, scalar.balance,
             scalar.position.hold_counter),
            ("vec", float(vec._position_sizes[0]), float(vec._balances[0]),
             int(vec._hold_counters[0])),
        ):
            assert size == size_before, f"{name} filled an unaffordable increase"
            assert balance == balance_before, f"{name} paid for a refused increase"
            assert held == 2, (
                f"{name} hold_counter={held} -- a refused increase leaves the position "
                "held, so it must still age"
            )
        scalar.close()
        vec.close()

    def test_resize_charges_the_delta_fee_exactly(self):
        """Halving a position must cost the fee on the delta it traded, to the cent.

        Precisely: fee == |change in size| * price * rate. It pins the charge against the
        size actually reached, not that the size reached was the right target -- any target
        error lights up dozens of tests elsewhere, so that division of labour is deliberate.

        The equivalence cells above compare the two engines, so they stay green if BOTH
        regress to close-and-reopen together. This pins the absolute number instead, and
        pins it tightly: an earlier version of this test asserted only "cost < 1.1% of
        base", which a doubled fee on the closed half slipped straight through.

        Flat prices so the expectation is exact arithmetic with no PnL term.
        """
        price, cash, fee = 100.0, 10000.0, 0.01
        n = 40
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
            "open": price, "high": price, "low": price, "close": price, "volume": 1000.0,
        }, index=range(n))
        scalar, vec = _make_pair(
            df, leverage=1, fee=fee, action_levels=self.SPOT_LEVELS, max_traj=20
        )
        td_s, td_v = scalar.reset(), vec.reset()
        td_s["action"] = torch.tensor(2)  # full
        td_v["action"] = torch.tensor([2])
        td_s, td_v = scalar.step(td_s)["next"], vec.step(td_v)["next"]

        opened_qty = scalar.position.position_size
        equity_before = scalar.balance + abs(opened_qty) * price
        vec_equity_before = float(vec._balances[0]) + abs(float(vec._position_sizes[0])) * price

        td_s["action"] = torch.tensor(1)  # half
        td_v["action"] = torch.tensor([1])
        scalar.step(td_s)
        vec.step(td_v)

        # At a flat price the only cost of halving is the fee on the half being closed.
        expected_fee = abs(opened_qty - scalar.position.position_size) * price * fee
        for name, equity_before_, balance, size in (
            ("scalar", equity_before, scalar.balance, scalar.position.position_size),
            ("vec", vec_equity_before, float(vec._balances[0]),
             float(vec._position_sizes[0])),
        ):
            charged = equity_before_ - (balance + abs(size) * price)
            assert abs(charged - expected_fee) < 1e-6, (
                f"{name} halving charged {charged:.6f}, expected {expected_fee:.6f} -- "
                "that is not a delta trade"
            )
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


class TestActionOnTheBreachingBarIsNotDiscarded:
    """An action submitted on a liquidating bar must still execute (#292, via #281).

    The agent's order fills at close(N), chronologically before bar N+1's wick exists, so
    a close or a switch on that bar is a real trade that happened -- it cannot be undone
    by what the next bar does. Both engines used to gate on liquidation and throw it away:
    the scalar skipped _execute_trade_if_needed entirely, the vectorized rewrote the
    action to 0.0.

    That difference decides what each cell can catch. 0.0 IS the close action, so zeroing
    it changes nothing: restoring the vectorized gate fails only the switch cell, while
    restoring the scalar gate fails both of its cells. The vectorized close cell is grid
    completeness, not a killer.

    Money: holding into the wick ends at 200, closing on that same bar at 10000 -- a 98%
    difference that nothing tested, because the liquidation equivalence cells submit the
    same action every step and so never act on a liquidating bar.

    The hold cell is a control, and it is load-bearing: the step count below is coupled to
    wick_liquidation_df's breach bar, and that fixture is now shared. Moving the breach
    by five bars makes every other cell pass while asserting nothing, because the exit
    never lands on the wick at all. The control fails loudly when that happens.

    Note this suite does NOT fail against pre-PR code -- reverting the ordering moves the
    liquidation off this step entirely. It pins the gate, not the ordering.
    """

    @pytest.mark.parametrize("is_vec", [False, True], ids=["scalar", "vectorized"])
    @pytest.mark.parametrize("exit_action,expected_direction,expected_pv", [
        (2, 0.0, 200.0),     # control: hold into the wick and it liquidates
        (1, 0.0, 10000.0),   # close
        (0, -1.0, 10000.0),  # switch to short
    ], ids=["hold-liquidates", "close", "switch-to-short"])
    def test_action_survives_the_liquidation_on_the_same_bar(
        self, wick_liquidation_df, is_vec, exit_action, expected_direction, expected_pv
    ):
        common = dict(
            action_levels=[-1.0, 0.0, 1.0], leverage=5, initial_cash=10000,
            time_frames=[TF_1MIN], execute_on=TF_1MIN, window_sizes=[10],
            transaction_fee=0.0, slippage=0.0, seed=42, max_traj_length=25,
            random_start=False,
        )
        if is_vec:
            env = VectorizedSequentialTradingEnv(
                wick_liquidation_df,
                VectorizedSequentialTradingEnvConfig(num_envs=1, **common),
                simple_feature_fn,
            )
        else:
            env = SequentialTradingEnv(
                wick_liquidation_df, SequentialTradingEnvConfig(**common), simple_feature_fn
            )

        td = env.reset()
        wrap = (lambda a: torch.tensor([a])) if is_vec else torch.tensor
        for _ in range(9):  # hold the long up to, but not onto, the wick bar
            td["action"] = wrap(2)
            td = env.step(td)["next"]
        td["action"] = wrap(exit_action)
        td = env.step(td)["next"]

        # Portfolio value, not balance: the switch leaves a position open, so its balance
        # is 0.0 with the margin deducted, and asserting on balance fails correct code.
        pv = float(env._portfolio_values[0]) if is_vec else env._get_portfolio_value()
        state = td["account_state"]
        direction = float(state[0][1] if is_vec else state[1])

        assert pv == pytest.approx(expected_pv), (
            f"portfolio {pv:.2f}, expected {expected_pv} -- the order filled at "
            "close(N)=100 before the wick, so the wick cannot discard it"
        )
        assert direction == expected_direction, (
            f"direction {direction}, expected {expected_direction} -- the action was "
            "swallowed by the liquidation check"
        )
        env.close()


class TestScalarVecEquivalenceLiquidation:
    """Verify liquidation behavior matches between scalar and vectorized envs."""

    @pytest.mark.parametrize("open_idx,direction", [(2, "long"), (0, "short"), (2, "wick")],
                             ids=["long", "short", "wick"])
    def test_liquidation_equivalence(self, open_idx, direction, trending_down_df,
                                     trending_up_df, wick_liquidation_df):
        """Both envs should liquidate at the same step with the same balance.

        The wick cell is the #281 shape: the trending frames breach on the close, so the
        observation reads unhealthy either way and the check-order cannot be isolated.
        """
        df = {"long": trending_down_df, "short": trending_up_df,
              "wick": wick_liquidation_df}[direction]
        actions = [open_idx] * 200
        mismatches = _run_sequence(
            df, actions, leverage=20, fee=0.001, max_traj=200,
            label=f"liquidation-{direction}"
        )
        assert not mismatches, "\n".join(mismatches)
