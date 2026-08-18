"""One copy of the shared executor helpers, and a guard against re-forking (#288)."""

import inspect

import pytest

from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass
from torchtrade.envs.live.bitget.order_executor import BitgetFuturesOrderClass
from torchtrade.envs.live.bybit.order_executor import BybitFuturesOrderClass
from torchtrade.envs.live.okx.order_executor import OKXFuturesOrderClass
from torchtrade.envs.live.shared.executor_helpers import (
    ExecutorHelpersMixin,
    TickSizeMixin,
)

EXECUTORS = [BinanceFuturesOrderClass, BitgetFuturesOrderClass,
             BybitFuturesOrderClass, OKXFuturesOrderClass]


@pytest.mark.parametrize("cls", EXECUTORS, ids=lambda c: c.__name__)
def test_no_executor_re_forks_the_shared_pnl(cls):
    """Three copies of this existed. It feeds `account_state[2]`, which the policy reads
    every step and a reward function sees, so a drifted copy is expensive twice over --
    and this repo has already shipped a fix landing on some exchanges but not others
    three times (lot size #271, full_done_spec #272, hedge-mode surface).

    A copy that has not drifted YET is still a copy.
    """
    assert "_calculate_unrealized_pnl_pct" not in cls.__dict__, (
        f"{cls.__name__} defines its own _calculate_unrealized_pnl_pct; use the shared one"
    )
    assert cls._calculate_unrealized_pnl_pct is ExecutorHelpersMixin._calculate_unrealized_pnl_pct


@pytest.mark.parametrize("cls", [BinanceFuturesOrderClass, BybitFuturesOrderClass,
                                 OKXFuturesOrderClass], ids=lambda c: c.__name__)
def test_tick_rounding_goes_through_the_shared_helper(cls):
    """These three had byte-identical tick arithmetic. bitget is deliberately absent --
    it OVERRIDES with CCXT's `price_to_precision`, a different mechanism with a different
    failure mode, and collapsing it in to shorten the file would be the opposite error.
    """
    assert cls._round_price is TickSizeMixin._round_price


@pytest.mark.parametrize("cls", EXECUTORS, ids=lambda c: c.__name__)
def test_no_executor_re_implements_the_pnl_rule_INLINE(cls):
    """By RULE, not by name -- the name check alone certified the one that was wrong.

    Binance never had a method called `_calculate_unrealized_pnl_pct`; it had the same
    arithmetic inlined in `get_status`, still branching on `qty > 0`. So
    `"_calculate_unrealized_pnl_pct" not in cls.__dict__` passed VACUOUSLY on the single
    exchange still carrying the dust bug, and the dust parametrization below calls the
    mixin directly and never touches an executor. The guard certified the miss.

    A re-inlined copy is spelled `(mark_price - entry_price) / entry_price` or its short
    counterpart; anything computing that outside the mixin is a fourth copy.
    """
    source = inspect.getsource(cls)
    for expression in ("(mark_price - entry_price) / entry_price",
                       "(entry_price - mark_price) / entry_price"):
        assert expression not in source, (
            f"{cls.__name__} recomputes the PnL rule inline instead of calling the "
            f"shared helper -- a copy that is not a method is still a copy"
        )


@pytest.mark.parametrize("cls", [BybitFuturesOrderClass, OKXFuturesOrderClass],
                         ids=lambda c: c.__name__)
def test_price_formatting_goes_through_the_shared_helper(cls):
    """Two verbatim copies. It emits a STRING because the venues parse the wire value and
    `repr` of a rounded float can carry more digits than the tick allows -- so a drifted
    copy is a rejected or silently re-rounded order, not a cosmetic difference."""
    assert cls._format_price is TickSizeMixin._format_price


def test_bitget_keeps_its_ccxt_rounding():
    """Pinned so a later dedup pass does not 'finish the job' by folding it in."""
    assert "price_to_precision" in inspect.getsource(BitgetFuturesOrderClass._round_price)


def test_bitget_does_not_inherit_the_tick_helpers_at_all():
    """It has no `_tick_size`, so those methods would raise on call.

    An override only covers the one you remember to write: `_round_price` had a bitget
    override and `_format_price` did not, so bitget silently inherited a method that
    always AttributeErrors. Not mixing them in cannot be half-done.
    """
    assert not issubclass(BitgetFuturesOrderClass, TickSizeMixin)
    assert not hasattr(BitgetFuturesOrderClass, "_format_price")


@pytest.mark.parametrize("cls", [BybitFuturesOrderClass, OKXFuturesOrderClass],
                         ids=lambda c: c.__name__)
def test_price_formatting_is_not_re_inlined_at_a_call_site(cls):
    """By RULE, not identity -- the identity check is blind to an inlined copy.

    That is the same hole the name-based PnL guard had, and this file already carries
    that lesson: replacing `self._format_price(x)` with an inline
    `f"{rounded:.{self._tick_decimals}f}"` leaves every identity assertion passing.
    Tick-decimal formatting belongs in the mixin and nowhere else.
    """
    source = inspect.getsource(cls)
    assert "_tick_decimals}f}" not in source, (
        f"{cls.__name__} formats a price against _tick_decimals inline instead of "
        f"calling the shared _format_price -- an inlined copy is still a copy"
    )


@pytest.mark.parametrize("qty,expected", [
    (1.0, 0.10),      # long
    (-1.0, -0.10),    # short
    (1e-12, 0.0),     # dust left by a full close is NOT a position
    (-1e-12, 0.0),
    (0.0, 0.0),
])
def test_unrealized_pnl_is_signed_by_direction_and_ignores_dust(qty, expected):
    """The three originals branched on `qty > 0`, so a 1e-12 residual took the SHORT
    branch and reported -10% on a position that does not exist -- into account_state[2],
    which is invariant 1 (the dust rule) in the observation.
    """
    assert ExecutorHelpersMixin()._calculate_unrealized_pnl_pct(qty, 100.0, 110.0) == pytest.approx(expected)


def test_a_zero_entry_price_reports_no_pnl_rather_than_dividing_by_it():
    assert ExecutorHelpersMixin()._calculate_unrealized_pnl_pct(1.0, 0.0, 110.0) == 0.0
