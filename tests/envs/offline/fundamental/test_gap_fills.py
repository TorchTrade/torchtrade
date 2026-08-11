"""A stop that a bar gapped past must fill at the open, not at the stop (#280).

All four engines that trigger brackets share the rule, so all four are pinned here:
scalar SLTP, vectorized SLTP, OneStep, and the replay executor. The bug shipped in every
one of them, and a fix that reached only some would look done at the PR level while
staying live in the rest.
"""

import math

import pandas as pd
import pytest
import torch

from torchtrade.envs.offline import (
    OneStepTradingEnv,
    OneStepTradingEnvConfig,
    SequentialTradingEnvSLTP,
    SequentialTradingEnvSLTPConfig,
    VectorizedSequentialTradingEnvSLTP,
    VectorizedSequentialTradingEnvSLTPConfig,
)
from torchtrade.envs.replay.order_executor import ReplayOrderExecutor
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

TF_1MIN = TimeFrame(1, TimeFrameUnit.Minute)


def _gap_df(gap_to, n=40, bar=20):
    """Flat series at 100 where one bar opens at `gap_to`, having gapped there.

    The gap bar is a real bar, not a point: open, high, low and close all differ, so a
    fill booked at the low or the close is distinguishable from one booked at the open.
    """
    o, h, l, c = ([100.0] * n for _ in range(4))
    drift = 2.0 if gap_to < 100.0 else -2.0
    o[bar], c[bar] = gap_to, gap_to + drift
    h[bar], l[bar] = max(gap_to, c[bar]) + 1.0, min(gap_to, c[bar]) - 1.0
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": o, "high": h, "low": l, "close": c, "volume": [1000.0] * n,
    })


@pytest.mark.parametrize("gap_to,open_idx,expected_balance", [
    # 2x long, entry 100, stop 97.5. Gapping to 85 is a 15% adverse move -> 30% of equity.
    # Booking the stop instead would leave 9500, overstating by 2500.
    (85.0, 1, 7000.0),
    (115.0, 2, 7000.0),
    # Take-profit is a limit order, so it is NOT chased: 2.5% at 2x = 5%, whatever the
    # gap. This is the deliberate asymmetry -- the cells that fail if someone adds a
    # take-profit counterpart to stop_fill_price for symmetry's sake.
    (130.0, 1, 10500.0),
    (70.0, 2, 10500.0),
], ids=["long-stop-gap", "short-stop-gap", "long-tp-gap", "short-tp-gap"])
@pytest.mark.parametrize("is_vec", [False, True], ids=["scalar", "vectorized"])
def test_sltp_engines_fill_a_gapped_bracket_alike(gap_to, open_idx, expected_balance, is_vec):
    df = _gap_df(gap_to)
    common = dict(
        leverage=2, stoploss_levels=[-0.025], takeprofit_levels=[0.025],
        initial_cash=10000, time_frames=[TF_1MIN], window_sizes=[10], execute_on=TF_1MIN,
        transaction_fee=0.0, slippage=0.0, seed=42, max_traj_length=25, random_start=False,
    )
    if is_vec:
        env = VectorizedSequentialTradingEnvSLTP(
            df, VectorizedSequentialTradingEnvSLTPConfig(num_envs=1, **common)
        )
    else:
        env = SequentialTradingEnvSLTP(df, SequentialTradingEnvSLTPConfig(**common))

    td = env.reset()
    wrap = (lambda a: torch.tensor([a])) if is_vec else torch.tensor
    # Enter on step 8, two bars before the fixture's gap at bar 20; it fires on step 9.
    for step in range(10):
        td["action"] = wrap(open_idx if step == 8 else 0)
        td = env.step(td)["next"]

    balance = float(env._balances[0]) if is_vec else env.balance
    assert balance == pytest.approx(expected_balance), (
        f"balance {balance:.2f}, expected {expected_balance} -- a stop that the bar "
        "gapped past must fill at the open, and a take-profit must not chase a gap"
    )
    env.close()


@pytest.mark.parametrize("gap_to,action,expected_balance", [
    (85.0, 1, 7000.0),      # long, entry 100, stop 97.5, bar opens at 85
    (115.0, 2, 7000.0),     # short, entry 100, stop 102.5, bar opens at 115
    (130.0, 1, 10500.0),    # take-profit is not chased: 2.5% at 2x, whatever the gap
    (70.0, 2, 10500.0),
], ids=["long-stop-gap", "short-stop-gap", "long-tp-gap", "short-tp-gap"])
def test_onestep_prices_a_gapped_bracket(gap_to, action, expected_balance):
    """OneStep re-forks the trigger check (#316), so the scalar cells do not cover it.

    The reward assertion earns its place by pinning the reward FORMULA, which nothing
    else does: test_onestep.py asserts only reward == sum(rollout_returns), which is
    invariant under swapping the log return for a simple one. It is not what guards the
    deleted reward-side fill -- verified by mutation that no price can reach the reward.
    """
    config = OneStepTradingEnvConfig(
        leverage=2, stoploss_levels=[-0.025], takeprofit_levels=[0.025],
        initial_cash=10000, time_frames=[TF_1MIN], window_sizes=[10], execute_on=TF_1MIN,
        transaction_fee=0.0, slippage=0.0, seed=42, max_traj_length=45,
        include_hold_action=True,
    )
    env = OneStepTradingEnv(_gap_df(gap_to, n=60, bar=45), config)

    td = env.reset()
    td["action"] = torch.tensor(action)
    out = env.step(td)["next"]

    assert env.position.position_size == 0, "the bracket should have closed the position"
    assert env.balance == pytest.approx(expected_balance), (
        f"balance {env.balance:.2f}, expected {expected_balance} -- a stop the bar "
        f"gapped past must fill at the open ({gap_to}), and a take-profit must not chase"
    )
    # abs=1e-6: reward is stored float32. Pre-fix the two disagree by ~0.3.
    assert out["reward"].item() == pytest.approx(math.log(expected_balance / 10000), abs=1e-6)
    env.close()


@pytest.mark.parametrize("qty,sl,tp,bar_price,expected_balance", [
    # Filled at the open: 10 * (85 - 100) = -150 against the returned 1000 of margin.
    # Booking the stop at 97.5 would give 9975, understating the loss by five sixths.
    (10.0, 97.5, 110.0, 85.0, 9850.0),
    (-10.0, 102.5, 90.0, 115.0, 9850.0),
    # Take-profit stays at its bracket: 10 * 10 = +100, not the 300 the open would give.
    (10.0, 97.5, 110.0, 130.0, 10100.0),
    (-10.0, 102.5, 90.0, 70.0, 10100.0),
], ids=["long-stop-gap", "short-stop-gap", "long-tp-gap", "short-tp-gap"])
def test_replay_executor_prices_a_gapped_bracket(qty, sl, tp, bar_price, expected_balance):
    """The replay path had to start reading ohlc["open"], which it never did before."""
    ex = ReplayOrderExecutor(initial_balance=10000.0)
    # Set the position directly rather than through trade(): this pins the fill rule, not
    # the sizing path. Deduct the margin by hand, since _close_at_price returns it.
    entry = 100.0
    ex.position_qty, ex.entry_price = qty, entry
    ex.balance -= abs(qty) * entry / ex.leverage
    ex.sl_price, ex.tp_price = sl, tp

    ex.advance_bar({k: bar_price for k in ("open", "high", "low", "close")})

    assert ex.position_qty == 0, "the bracket should have triggered"
    assert ex.balance == pytest.approx(expected_balance), (
        f"balance {ex.balance:.2f}, expected {expected_balance}"
    )
