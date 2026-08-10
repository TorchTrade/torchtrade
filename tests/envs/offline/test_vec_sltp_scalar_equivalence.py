"""
Exhaustive equivalence test: SequentialTradingEnvSLTP == VectorizedSequentialTradingEnvSLTP(num_envs=1).

Runs both environments with identical configs and action sequences,
comparing ALL observable state at every step. Any divergence is a bug.
"""

import collections

import numpy as np
import pandas as pd
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
# One home for the equivalence tolerance: two copies in two files is the same drift hazard
# this PR removed from AFFORDABILITY_REL_TOL, one level up.
from tests.envs.offline.test_vec_scalar_equivalence import EQUIV_ATOL, EQUIV_RTOL

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
    trade_mode="fractional",
    lock=False,
    initial_cash=1000,
):
    """Create matched scalar and N=1 vectorized SLTP envs."""
    if sl_levels is None:
        sl_levels = [-0.025, -0.05]
    if tp_levels is None:
        tp_levels = [0.05, 0.1]

    common = dict(
        leverage=leverage,
        stoploss_levels=sl_levels,
        takeprofit_levels=tp_levels,
        include_hold_action=include_hold,
        include_close_action=include_close,
        initial_cash=initial_cash,
        time_frames=[TF_1MIN],
        window_sizes=[10],
        execute_on=TF_1MIN,
        transaction_fee=fee,
        slippage=0.0,
        seed=42,
        max_traj_length=max_traj,
        random_start=False,
        trade_mode=trade_mode,
        lock_position_until_sltp=lock,
    )
    # Ignored by the fractional path. 100.0 is $100 of notional; in quantity mode it is
    # 100 *units*, ~$10,000 at these fixtures' ~$100 prices, which no leverage-1 balance can
    # afford -- so that cell would open nothing and compare two idle envs.
    common["quantity_per_trade"] = 100.0 if trade_mode == "notional" else 1.0

    scalar = SequentialTradingEnvSLTP(
        df, SequentialTradingEnvSLTPConfig(**common), simple_feature_fn
    )
    vec = VectorizedSequentialTradingEnvSLTP(
        df, VectorizedSequentialTradingEnvSLTPConfig(num_envs=1, **common), simple_feature_fn
    )
    return scalar, vec


def _compare_sltp_state(scalar, vec):
    """Compare all observable state between scalar and vectorized SLTP envs.

    Returns list of (field, scalar_val, vec_val) mismatches.
    """
    mismatches = []

    def check(field, s_val, v_val, atol=EQUIV_ATOL, rtol=EQUIV_RTOL):
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
    action_indices=None,
    label="",
    expect_action_type=None,
    random_steps=None,
    **pair_kwargs,
):
    """Run a sequence of actions through both envs and compare at every step.

    Args:
        expect_action_type: if given, the run must actually record this action type
            (e.g. "sltp_sl"), so a scenario that stops reaching its bracket is reported
            rather than passing as a pair of idle envs.
        random_steps: if given, `action_indices` is ignored and this many random actions
            are drawn instead.
        pair_kwargs: forwarded to `_make_sltp_pair`.
    """
    scalar, vec = _make_sltp_pair(df, **pair_kwargs)
    if random_steps is not None:
        # Drawn after construction so the draw respects the real action-space size, which
        # changes with leverage (shorts) and include_close.
        action_indices = (
            np.random.default_rng(0)
            .integers(0, scalar.action_spec.n, size=random_steps)
            .tolist()
        )
    all_mismatches = []

    td_s = scalar.reset()
    td_v = vec.reset()

    # Compare initial state
    mismatches = _compare_sltp_state(scalar, vec)
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

        # Compare internal state
        mismatches = _compare_sltp_state(scalar, vec)
        for field, s_val, v_val, diff in mismatches:
            all_mismatches.append(
                f"[{label}] Step {step+1} {field}: scalar={s_val:.6f} vec={v_val:.6f} diff={diff:.6f}"
            )

        if td_s["next"]["done"].item() or td_v["next"]["done"].item():
            break

    # Without this a scenario whose action indices or wick stop reaching a bracket
    # degrades into "both envs held cash identically" and still passes.
    if expect_action_type is not None and expect_action_type not in scalar.history.action_types:
        all_mismatches.append(
            f"[{label}] path never exercised: no {expect_action_type} in "
            f"{scalar.history.action_types}"
        )

    # The random runs cannot name one expected action type, so this is their version of the
    # same guard, and it counts what the run exercised rather than how long a position sat
    # open. Both weaker forms shipped: first cells that opened nothing at all, then cells
    # that opened once and held for 99 steps with no bracket ever firing -- which a
    # "position was open" check reads as maximally healthy. Healthy cells measure 8-38
    # opens and 4-13 bracket exits, so these floors have room.
    if random_steps is not None:
        kinds = collections.Counter(scalar.history.action_types)
        opens = kinds["long"] + kinds["short"]
        brackets = kinds["sltp_sl"] + kinds["sltp_tp"]
        if opens < 2 or brackets < 1:
            all_mismatches.append(
                f"[{label}] exercised {opens} opens and {brackets} bracket exits "
                f"({dict(kinds)}) -- this cell is not reaching the SLTP machinery"
            )

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
# ENTRY-BAR EQUIVALENCE
# ============================================================================


def _flat_df(bar=20, low=None, high=None, n=50):
    """Flat 100.0 series, optionally with one bar carrying a single excursion."""
    prices = np.full(n, 100.0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": prices.copy(),
        "high": prices.copy(),
        "low": prices.copy(),
        "close": prices.copy(),
        "volume": np.ones(n) * 1000,
    })
    if low is not None:
        df.loc[bar, "low"] = low
    if high is not None:
        df.loc[bar, "high"] = high
    return df


class TestSLTPScalarVecEquivalenceEntryBar:
    """Both envs must apply the entry bar to a freshly-opened position (#268)."""

    @pytest.mark.parametrize("leverage,sl_levels,tp_levels,wick_low,wick_high,expected_exit", [
        (1, [-0.025], [0.50], 80.0, None, "sltp_sl"),
        (10, [-0.50], [0.50], 88.0, None, "liquidation"),
        (1, [-0.50], [0.025], None, 120.0, "sltp_tp"),  # the branch the fix never ran
    ], ids=["stop-loss", "liquidation", "take-profit"])
    def test_entry_bar_exit_matches_scalar(
        self, leverage, sl_levels, tp_levels, wick_low, wick_high, expected_exit
    ):
        """Both envs forked this ordering separately, so both could drift separately.

        The equivalence suite passed on main because the bug was symmetric — both envs
        skipped the entry bar identically. The value-pinning tests in
        test_sequential_sltp.py are what catch the shared bug; this catches divergence.
        Only the entry-bar timing is covered here: the held-position path is already
        exercised by the trend-based cases above.
        """
        crash_bar = 20
        df = _flat_df(bar=crash_bar, low=wick_low, high=wick_high)

        # reset caches bar 10 (window_sizes=[10]), so step k executes at bar 10+k, and
        # the position opens one bar before the wick.
        open_step = crash_bar - 1 - 10
        actions = [0] * open_step + [1] + [0] * 3

        mismatches = _run_sltp_sequence(
            df, actions, leverage=leverage,
            sl_levels=sl_levels, tp_levels=tp_levels,
            label=f"sltp-entry-bar-{leverage}x-{expected_exit}",
            expect_action_type=expected_exit,
        )
        assert not mismatches, "\n".join(mismatches)

    def test_switch_exposes_the_new_position_to_the_entry_bar(self):
        """A switch opens a position too, and each env recognises that differently.

        Both envs open a switch through their own code path -- _execute_sltp_action
        against switch_mask | open_from_flat -- so a refactor could reopen #268 for
        switches in one env only. The incoming long's own TP sits inside bar 20 and
        must NOT pre-empt the switch (#292): the agent closed that long at close(N),
        so bar 20 belongs to the short, which its own SL at 102.5 then stops out.
        """
        df = _flat_df(bar=20, high=120.0)
        long_tight, short_tight = 1, 5
        actions = [0] * 5 + [long_tight] + [0] * 3 + [short_tight] + [0] * 2

        mismatches = _run_sltp_sequence(
            df, actions, leverage=2,
            sl_levels=[-0.025, -0.50], tp_levels=[0.025, 0.50],
            label="sltp-entry-bar-switch",
            # The hardcoded indices and the delicately balanced levels make this the
            # scenario most likely to quietly stop testing anything.
            expect_action_type="sltp_sl",
        )
        assert not mismatches, "\n".join(mismatches)


class TestSLTPScalarVecEquivalenceCloseVsTrigger:
    """A close action and a firing bracket on the same bar (#292).

    The reorder applies to closes as well as switches, and each env recognises a close
    through its own path -- _execute_sltp_action against close_action_mask -- so a
    vec-only regression here is invisible to any test that never puts a close and a
    trigger on the same bar.
    """

    @pytest.mark.parametrize("leverage,sl_levels,tp_levels,wick_high,wick_low", [
        (2, [-0.025, -0.50], [0.025, 0.50], 120.0, None),
        # Liquidation flavour: a 10x long liquidates near 90.4 and bar 20 wicks to 88.
        # The held-and-hold version of this scenario passes under BOTH orderings, which
        # is what made the scalar liquidation test dead before it sent a close.
        (10, [-0.50], [0.50], None, 88.0),
    ], ids=["bracket", "liquidation"])
    def test_close_action_beats_an_exit_on_the_same_bar(
        self, leverage, sl_levels, tp_levels, wick_high, wick_low
    ):
        df = _flat_df(bar=20, high=wick_high, low=wick_low)
        # With the close action enabled every index shifts by one: 2 is the first long.
        long_action, close_action = 2, 1
        actions = [0] * 5 + [long_action] + [0] * 3 + [close_action] + [0] * 2

        mismatches = _run_sltp_sequence(
            df, actions, leverage=leverage,
            sl_levels=sl_levels, tp_levels=tp_levels,
            label=f"sltp-close-vs-exit-{leverage}x",
            include_close=True,
            expect_action_type="close",
        )
        assert not mismatches, "\n".join(mismatches)


class TestSLTPScalarVecEquivalencePriority:
    """A bar breaching both the stop and the liquidation price must liquidate in BOTH
    envs (#298).

    They encode the precedence differently -- the scalar as an if/else chain, the
    vectorized by zeroing the position inside the liquidation branch so the trigger
    masks cannot see it -- so one could be reordered without the other. At this
    harness's initial_cash of 1000 the liquidation leaves 40 where the stop would
    leave 500 -- the same scenario the scalar test pins at ten times the cash.

    The vec side has no action_types record, so its liquidation is pinned by numeric
    parity with the scalar rather than by a label of its own.
    """

    def test_liquidation_wins_over_the_bracket_in_both_envs(self):
        df = _flat_df(bar=20, low=88.0)
        # 10x long at 100: SL 95, liquidation 90.4, bar low 88 breaches both.
        long_action = 1
        actions = [0] * 5 + [long_action] + [0] * 4

        mismatches = _run_sltp_sequence(
            df, actions, leverage=10,
            sl_levels=[-0.05], tp_levels=[0.50],
            label="sltp-liquidation-over-bracket",
            expect_action_type="liquidation",
        )
        assert not mismatches, "\n".join(mismatches)


class TestSLTPScalarVecEquivalenceTradeModesAndLock:
    """trade_mode and lock crossed with leverage -- the axes this harness never varied.

    Randomised actions rather than another hand-picked sequence: hand-picked sequences are
    what let the float32 divergence in #293 live here for months. Leverage is crossed in
    because it is what scales an epsilon up onto a threshold and flips an episode.
    """

    @pytest.mark.parametrize("trade_mode", ["fractional", "notional", "quantity"])
    @pytest.mark.parametrize("lock", [False, True], ids=["unlocked", "locked"])
    @pytest.mark.parametrize("leverage", [1, 25], ids=["spot", "lev25"])
    def test_random_actions_match(self, sample_ohlcv_df, trade_mode, lock, leverage):
        if lock and leverage == 1:
            pytest.skip(
                "lock is inert in spot: with no shorts and no close action the agent can "
                "only ask for long or hold, and re-asking for long is already a no-op via "
                "the duplicate-action guard. Measured byte-identical to the unlocked twin "
                "(same action histogram, balance and position), so this cell is a copy."
            )
        mismatches = _run_sltp_sequence(
            sample_ohlcv_df,
            leverage=leverage,
            fee=0.001,
            max_traj=120,
            random_steps=100,
            trade_mode=trade_mode,
            lock=lock,
            # The fixture only spans +5.9%/-1.7%, so the default +/-5% brackets can never
            # trigger: as shipped this grid fired zero stops across all twelve cells.
            sl_levels=[-0.005],
            tp_levels=[0.005],
            label=f"sltp-{trade_mode}-{'locked' if lock else 'unlocked'}-lev{leverage}",
        )
        assert not mismatches, "\n".join(mismatches)


class TestAffordabilitySlackIsShared:
    """Both engines must make the same accept/reject call on a marginal open (#293).

    The vectorized envs carried 1e-5 slack against the scalar envs' 1e-9 -- a 10,000x
    window in which one engine opened a position the other refused, leaving one in cash
    and the other fully deployed at a zeroed balance. Reverting either side to 1e-5 left
    the whole offline suite green.
    """

    @pytest.mark.parametrize("shortfall,expect_open", [
        (1e-11, True),
        (3e-6, False),
    ], ids=["inside-slack", "inside-the-old-1e-5-window"])
    def test_both_engines_make_the_same_call(self, shortfall, expect_open):
        # Flat at 100, and _make_sltp_pair sizes notional mode at quantity_per_trade=100,
        # so the cost is exact: 100 margin at leverage 1 plus 0.1 fee. Cash is set just
        # short of that 100.1 by `shortfall`.
        df = _flat_df(n=40)
        scalar, vec = _make_sltp_pair(
            df, leverage=1, fee=0.001, sl_levels=[-0.05], tp_levels=[0.1],
            max_traj=20, trade_mode="notional", initial_cash=100.1 / (1 + shortfall),
        )
        td_s, td_v = scalar.reset(), vec.reset()
        td_s["action"] = torch.tensor(1)
        td_v["action"] = torch.tensor([1])
        scalar.step(td_s)
        vec.step(td_v)

        opened = (scalar.position.position_size != 0, float(vec._position_sizes[0]) != 0)
        assert opened == (expect_open, expect_open), (
            f"shortfall={shortfall}: (scalar, vec) opened={opened}, expected both "
            f"{expect_open} -- the engines disagree about what is affordable"
        )
        scalar.close()
        vec.close()


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
