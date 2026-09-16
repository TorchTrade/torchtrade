from typing import Dict

import torch

from torchtrade.metrics.trading_metrics import compute_all_metrics


def portfolio_metrics(history, periods_per_year: float) -> Dict[str, float]:
    """`compute_all_metrics` over a `PortfolioHistoryTracker`, plus the portfolio totals.

    Adds `final_value` (p_f / p_0), and `turnover`, `commission` and `funding` summed over
    the episode. `num_trades` is dropped: it counts non-zero discrete actions, which the
    portfolio env does not have.
    """
    pv = torch.tensor(history.portfolio_values, dtype=torch.float64)
    out = compute_all_metrics(pv, torch.tensor(history.rewards, dtype=torch.float64), [], periods_per_year)
    del out["num_trades"]
    out.update(
        final_value=(pv[-1] / pv[0]).item(),
        turnover=float(sum(history.turnovers)),
        commission=float(sum(history.commissions)),
        funding=float(sum(history.fundings)),
    )
    return out
