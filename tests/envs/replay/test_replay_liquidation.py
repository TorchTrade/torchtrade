"""#269: the replay executor had no liquidation, and reported liquidation_price=0.0."""

import pytest

from torchtrade.envs.replay.order_executor import ReplayOrderExecutor


def _long_at(entry, leverage, balance=1000.0):
    ex = ReplayOrderExecutor(initial_balance=balance, leverage=leverage)
    ex.current_price = entry
    ex.trade(side="buy", quantity=(balance * leverage) / entry)
    return ex


@pytest.mark.parametrize("leverage,expect_liquidation", [
    (1, False), (5, True), (20, True),
], ids=["spot-never", "5x", "20x"])
def test_a_leveraged_replay_position_liquidates(leverage, expect_liquidation):
    """Equity went to -1000 at 5x and then FULLY RECOVERED when price came back.

    The offline env on identical input liquidates and terminates, so any leveraged replay
    evaluation was unboundedly optimistic -- the one number a backtest exists to produce.
    """
    ex = _long_at(40000.0, leverage)
    ex.advance_bar({"open": 30000.0, "high": 30000.0, "low": 30000.0, "close": 30000.0})

    if expect_liquidation:
        assert ex.position_qty == 0, "a 25% adverse move did not liquidate"
        assert ex.balance >= 0.0, "isolated margin cannot lose more than the margin posted"
    else:
        assert ex.position_qty > 0, "an unlevered position has nothing to be liquidated by"


@pytest.mark.parametrize("leverage,positive", [(1, False), (10, True)])
def test_the_reported_liquidation_price_is_real(leverage, positive):
    """Hardcoded 0.0 made futures_live_base fail open, so account_state[5] read 1.0 for
    EVERY position -- a levered position shown to the policy as maximally far from
    liquidation. CLAUDE.md invariant 3."""
    ex = _long_at(40000.0, leverage)
    reported = ex.get_status()["position_status"].liquidation_price

    assert (reported > 0) is positive
    if positive:
        assert 0 < reported < 40000.0, "a long is liquidated below its entry"


def test_liquidation_is_checked_without_any_bracket_set():
    """The old guard returned early when sl and tp were both 0, so a position with no
    bracket -- the default -- could never liquidate at all."""
    ex = _long_at(40000.0, 10)
    assert ex.sl_price == 0 and ex.tp_price == 0

    ex.advance_bar({"open": 20000.0, "high": 20000.0, "low": 20000.0, "close": 20000.0})
    assert ex.position_qty == 0
