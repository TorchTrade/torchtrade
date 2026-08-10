"""A bracket that a bar gapped past must fill at the open, not at the bracket (#280).

All four engines that trigger brackets share the rule, so all four are pinned here:
scalar SLTP, vectorized SLTP, OneStep, and the replay executor. The bug shipped in every
one of them, and a fix that reached only some would look done at the PR level while
staying live in the rest.
"""

import pandas as pd
import pytest
import torch

from torchtrade.envs.offline import (
    SequentialTradingEnvSLTP,
    SequentialTradingEnvSLTPConfig,
    VectorizedSequentialTradingEnvSLTP,
    VectorizedSequentialTradingEnvSLTPConfig,
)
from torchtrade.envs.replay.order_executor import ReplayOrderExecutor
from torchtrade.envs.utils.sltp_helpers import gap_aware_fill
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

TF_1MIN = TimeFrame(1, TimeFrameUnit.Minute)


@pytest.mark.parametrize("bracket,open_price,is_long,is_stop,expected", [
    # A stop is a market order once touched: a gap past it fills at the open.
    (97.5, 85.0, True, True, 85.0),
    (102.5, 115.0, False, True, 115.0),
    # A bar that merely wicks through opens beyond the bracket, so min/max returns the
    # bracket unchanged -- which is why the rule needs no separate is-this-a-gap branch.
    (97.5, 100.0, True, True, 97.5),
    (102.5, 100.0, False, True, 102.5),
    # A take-profit is a limit order and is deliberately NOT gap-adjusted. Chasing a
    # favourable gap would make the backtest optimistic, against this module's stated
    # pessimism bias ("live trading can only outperform the backtest").
    (110.0, 130.0, True, False, 110.0),
    (90.0, 70.0, False, False, 90.0),
], ids=["long-stop-gap", "short-stop-gap", "long-stop-wick", "short-stop-wick",
        "long-tp-favourable-gap", "short-tp-favourable-gap"])
def test_gap_aware_fill_rule(bracket, open_price, is_long, is_stop, expected):
    assert gap_aware_fill(bracket, open_price, is_long, is_stop) == expected


def _gap_df(gap_to, n=40, bar=20, price=100.0):
    """Flat series where one bar gaps to `gap_to` and stays there for the whole bar."""
    o, h, l, c = ([price] * n for _ in range(4))
    o[bar] = h[bar] = l[bar] = c[bar] = gap_to
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": o, "high": h, "low": l, "close": c, "volume": [1000.0] * n,
    })


@pytest.mark.parametrize("gap_to,open_idx,expected_balance", [
    # 2x long, entry 100, stop 97.5. Gapping to 85 is a 15% adverse move -> 30% of equity.
    # Booking the stop instead would leave 9500, overstating by 2500.
    (85.0, 1, 7000.0),
    (115.0, 2, 7000.0),
    # Take-profit is NOT chased: 2.5% at 2x = 5%, whatever the gap.
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
    for step in range(12):
        td["action"] = wrap(open_idx if step == 8 else 0)
        td = env.step(td)["next"]
        if td["done"].all() if is_vec else td["done"].item():
            break

    balance = float(env._balances[0]) if is_vec else env.balance
    assert balance == pytest.approx(expected_balance), (
        f"balance {balance:.2f}, expected {expected_balance} -- a stop that the bar "
        "gapped past must fill at the open, and a take-profit must not chase a gap"
    )
    env.close()


def test_replay_executor_fills_a_gapped_stop_at_the_open():
    """The replay path had to start reading ohlc["open"], which it never did before."""
    ex = ReplayOrderExecutor(initial_balance=10000.0)
    # Set the position directly rather than through trade(): this pins the fill rule, not
    # the sizing path. Deduct the margin by hand, since _close_at_price returns it.
    qty, entry = 10.0, 100.0
    ex.position_qty, ex.entry_price = qty, entry
    ex.balance -= qty * entry / ex.leverage
    ex.sl_price, ex.tp_price = 97.5, 110.0

    ex.advance_bar({"open": 85.0, "high": 85.0, "low": 85.0, "close": 85.0})

    assert ex.position_qty == 0, "the stop should have triggered"
    # Filled at the open: 10 * (85 - 100) = -150 against the returned 1000 of margin.
    assert ex.balance == pytest.approx(9850.0), (
        f"balance {ex.balance:.2f} -- booking the stop at 97.5 would give 9975 and "
        "understate the loss by five sixths"
    )
