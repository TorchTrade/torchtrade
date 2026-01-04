"""Trading metrics module for TorchTrade.

This module provides functions for calculating trading performance metrics
including risk-adjusted returns, drawdowns, and win rates.
"""

from torchtrade.metrics.trading_metrics import (
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_calmar_ratio,
    compute_win_rate,
    compute_portfolio_returns,
    compute_all_metrics,
)

__all__ = [
    "compute_max_drawdown",
    "compute_sharpe_ratio",
    "compute_sortino_ratio",
    "compute_calmar_ratio",
    "compute_win_rate",
    "compute_portfolio_returns",
    "compute_all_metrics",
]
