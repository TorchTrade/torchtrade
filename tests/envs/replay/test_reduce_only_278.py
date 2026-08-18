"""reduce_only reduces BY a quantity; it does not always flatten (#278)."""

import pytest

from torchtrade.envs.replay.order_executor import ReplayOrderExecutor


def _opened(qty=1.0, price=100.0, leverage=5, fee=0.0, side="BUY"):
    executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=leverage,
                                   transaction_fee=fee)
    executor.advance_bar({"open": price, "high": price + 1, "low": price - 1, "close": price})
    executor.trade(side, qty)
    return executor


@pytest.mark.parametrize("reduce_by,expected_remaining", [
    (0.25, 0.75),   # the case that was silently flattening
    (0.5, 0.5),
    (1.0, 0.0),     # exactly flat
    (5.0, 0.0),     # oversized: clamps, never flips the side
])
def test_reduce_only_closes_the_quantity_it_was_given(reduce_by, expected_remaining):
    """`if reduce_only: return self.close_position()` discarded the quantity, so every
    reduce was a full close. All four live executors forward the quantity with a
    reduceOnly flag and honour a partial, so replay silently disagreed with all of them.
    """
    executor = _opened()
    executor.trade("SELL", reduce_by, reduce_only=True)
    assert executor.position_qty == pytest.approx(expected_remaining)


def test_a_partial_reduce_keeps_the_entry_price_and_the_brackets():
    """The remainder was opened at that price; selling part of it does not change its
    cost basis. And a partially reduced position still HAS brackets -- clearing them
    would leave a live position with no stop, which is what a full close does.

    The bar MUST move first. The earlier version reduced at the opening price, so
    `fill == entry` and the entry assertion was a tautology: re-basing the remainder to
    the fill on every partial passed it. That matters -- `entry_price` feeds
    `liquidation_price`, the next reduce's margin release, and unrealized PnL.
    """
    executor = _opened()
    # Levels the move below does NOT cross: at tp=103 the bar advance fires the take
    # profit and flattens the position, so the test would assert against a closed one.
    executor.sl_price, executor.tp_price = 90.0, 130.0
    executor.advance_bar({"open": 110.0, "high": 111.0, "low": 109.0, "close": 110.0})

    executor.trade("SELL", 0.25, reduce_only=True)

    assert executor.entry_price == pytest.approx(100.0), (
        "the remainder was opened at 100 and must keep that cost basis, not re-base to "
        "the 110 it was partially sold at"
    )
    assert (executor.sl_price, executor.tp_price) == (90.0, 130.0)


def test_reducing_a_position_away_in_uneven_steps_leaves_no_phantom_dust():
    """Seven reduces of 1/7 leave -2.2e-16, which `get_status()` reads as an open short.

    That is CLAUDE.md invariant 1: an exchange residual read as a position freezes the
    duplicate-action guard AND puts a phantom position in the observation. The dust rule
    is why this file imports POSITION_DUST_EPS, and nothing exercised it -- replacing the
    guard with `== 0.0` passed every replay test.
    """
    executor = _opened(qty=1.0, side="SELL")
    for _ in range(7):
        assert executor.trade("BUY", 1.0 / 7, reduce_only=True) is True

    assert executor.position_qty == 0.0
    assert executor.get_status()["position_status"] is None


@pytest.mark.parametrize("open_side,reduce_side,direction", [
    ("BUY", "SELL", 1),    # long
    ("SELL", "BUY", -1),   # short: the classic place a PnL sign flips
])
def test_a_partial_reduce_books_pnl_fee_and_margin_against_the_right_bases(
    open_side, reduce_side, direction,
):
    """The FEE is charged on the closed notional at the FILL price, and the margin is
    released at the ENTRY price. Both must be asserted absolutely.

    The first version compared a 0.25 reduce against the remaining 0.75 and asserted only
    that the second was three times the first. That is pure linearity, which holds for
    ANY basis that scales with the closed quantity -- corrupting the fee basis to entry
    price AND the margin basis to fill price left all seven tests passing. It also never
    set `transaction_fee`, which defaults to 0.0, so the word "fee" in its name was
    untested.
    """
    entry, fill, qty, closed, leverage, fee = 100.0, 110.0, 1.0, 0.25, 5, 0.0004
    executor = _opened(qty=qty, price=entry, leverage=leverage, fee=fee, side=open_side)
    balance_before = executor.balance
    executor.advance_bar({"open": fill, "high": fill + 1, "low": fill - 1, "close": fill})

    executor.trade(reduce_side, closed, reduce_only=True)

    expected = (
        direction * closed * (fill - entry)     # PnL on the closed portion only
        - closed * fill * fee                   # fee on the FILL notional, not entry
        + closed * entry / leverage             # margin released at ENTRY, not fill
    )
    assert executor.balance - balance_before == pytest.approx(expected, rel=1e-9)
    assert executor.position_qty == pytest.approx(direction * (qty - closed))


@pytest.mark.parametrize("open_side,bad_reduce_side", [("BUY", "BUY"), ("SELL", "SELL")])
def test_a_same_direction_reduce_only_is_rejected(open_side, bad_reduce_side):
    """`side` must OPPOSE the position, as it does on every venue: bybit routes a BUY
    reduceOnly to the short leg, okx sets `posSide = "short" if buy else "long"`, and
    both binance and bybit build a close side as the inverse of the held one. In one-way
    mode the venue rejects a same-direction reduceOnly outright.

    Deriving the direction from the POSITION instead accepted the nonsense order and
    quietly applied it to the other side -- the same divergence this fix exists to
    remove, one argument over.
    """
    executor = _opened(side=open_side)
    before = executor.position_qty

    assert executor.trade(bad_reduce_side, 0.25, reduce_only=True) is False
    assert executor.position_qty == pytest.approx(before)


def test_reduce_only_on_a_flat_account_is_a_no_op():
    executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5)
    executor.advance_bar({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    assert executor.trade("SELL", 0.5, reduce_only=True) is False
    assert executor.balance == pytest.approx(10000.0)
