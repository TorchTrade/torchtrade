import math

import pytest

from torchtrade.envs.core.state import PortfolioHistoryTracker
from torchtrade.metrics import portfolio_metrics


def test_portfolio_metrics_totals_and_final_value():
    """Hand-built three-step history: the portfolio totals and final value are exact, the
    return and drawdown match the value path, and the discrete-action trade count is gone."""
    h = PortfolioHistoryTracker()
    h.record_step("t0", 1000.0, [1.0, 0.0])
    h.record_step("t1", 1100.0, [0.0, 1.0], commission=1.0, funding=0.5, turnover=1.0)
    h.record_step("t2", 990.0, [0.0, 1.0], commission=0.0, funding=0.25, turnover=0.0)
    h.record_step("t3", 1089.0, [0.5, 0.5], commission=2.0, funding=0.0, turnover=0.5)
    h.rewards[1:] = [math.log(1.1), math.log(0.9), math.log(1.1)]

    m = portfolio_metrics(h, periods_per_year=6 * 365)

    assert m["final_value"] == pytest.approx(1.089)
    assert m["total_return"] == pytest.approx(0.089)
    assert m["turnover"] == 1.5 and m["commission"] == 3.0 and m["funding"] == 0.75
    assert m["max_drawdown"] == pytest.approx(-0.1)
    assert m["win_rate (reward>0)"] == pytest.approx(0.5)  # two of four rewards are positive
    assert "num_trades" not in m
    assert portfolio_metrics(h, periods_per_year=4 * 6 * 365)["sharpe_ratio"] == pytest.approx(2 * m["sharpe_ratio"])
