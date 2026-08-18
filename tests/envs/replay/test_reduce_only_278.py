"""reduce_only reduces BY a quantity; it does not always flatten (#278)."""

import pytest

from torchtrade.envs.replay.order_executor import ReplayOrderExecutor


def _opened(qty=1.0, price=100.0, leverage=5):
    executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=leverage)
    executor.advance_bar({"open": price, "high": price + 1, "low": price - 1, "close": price})
    executor.trade("BUY", qty)
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
    would leave a live position with no stop, which is what a full close does."""
    executor = _opened()
    executor.sl_price, executor.tp_price = 98.0, 103.0
    executor.trade("SELL", 0.25, reduce_only=True)

    assert executor.entry_price == pytest.approx(100.0)
    assert (executor.sl_price, executor.tp_price) == (98.0, 103.0)


def test_a_partial_reduce_books_pnl_fee_and_margin_pro_rata():
    """Booking the whole position's PnL on a partial close would credit profit on units
    still held -- and releasing all the margin would let the remainder be levered past
    the account."""
    executor = _opened(qty=1.0, price=100.0, leverage=5)
    balance_before = executor.balance
    executor.advance_bar({"open": 110.0, "high": 111.0, "low": 109.0, "close": 110.0})

    executor.trade("SELL", 0.25, reduce_only=True)
    quarter = executor.balance - balance_before

    executor.trade("SELL", 0.75, reduce_only=True)
    remaining_three_quarters = executor.balance - balance_before - quarter

    assert remaining_three_quarters == pytest.approx(quarter * 3, rel=1e-6)
    assert executor.position_qty == 0.0


def test_reduce_only_on_a_flat_account_is_a_no_op():
    executor = ReplayOrderExecutor(initial_balance=10000.0, leverage=5)
    executor.advance_bar({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    assert executor.trade("SELL", 0.5, reduce_only=True) is False
    assert executor.balance == pytest.approx(10000.0)
