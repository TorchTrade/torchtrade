"""Tests for ReplayOrderExecutor's trader-interface obligations."""

import pytest


def test_replay_rounds_quantities_to_the_grid_it_has_always_used():
    """#271 made `round_quantity` part of the trader interface, and returning the quantity
    untouched silently moved every existing backtest's numbers.

    Before this, binance's env reached `futures_exchange_info()` through a trader with no
    `client`; the AttributeError fell through to a default filter set with stepSize 0.001.
    An accident of the error path, but it is what recorded backtests were run against --
    so it is preserved deliberately rather than changed silently. Leaving replay ungridded
    would also have WIDENED the live/replay divergence in a PR meant to narrow it.
    """
    from torchtrade.envs.replay.order_executor import ReplayOrderExecutor, REPLAY_QTY_STEP

    assert REPLAY_QTY_STEP == 0.001
    assert ReplayOrderExecutor.round_quantity(None, 0.9796155) == pytest.approx(0.979)
    # And the same exact-multiple hazard the live executor guards against.
    assert ReplayOrderExecutor.round_quantity(None, 0.29) == pytest.approx(0.29)
