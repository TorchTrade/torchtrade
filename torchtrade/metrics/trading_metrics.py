"""Trading metrics for evaluating agent performance.

This module provides functions for calculating common trading performance metrics
including MaxDrawDown, MaxDrawDownDuration, Sharpe Ratio, and other risk-adjusted
return measures.
"""

import torch
from typing import Dict, Tuple


def compute_max_drawdown(portfolio_values: torch.Tensor) -> Dict[str, float]:
    """
    Compute maximum drawdown and related metrics.

    Drawdown is the decline from a historical peak in portfolio value.
    Maximum drawdown is the largest peak-to-trough decline.

    Args:
        portfolio_values: 1D torch.Tensor of portfolio values over time

    Returns:
        Dictionary containing:
            - max_drawdown: Maximum percentage drawdown (negative value, e.g., -0.15 for 15% drawdown)
            - max_drawdown_duration: Duration of max drawdown in number of periods
            - current_drawdown: Current drawdown from peak
            - peak_value: Historical peak value
            - trough_value: Value at maximum drawdown trough
    """
    if len(portfolio_values) == 0:
        return {
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'current_drawdown': 0.0,
            'peak_value': 0.0,
            'trough_value': 0.0,
        }

    # Calculate running maximum (peak)
    running_max = torch.cummax(portfolio_values, dim=0)[0]

    # Calculate drawdown at each point
    drawdown = (portfolio_values - running_max) / running_max

    # Find maximum drawdown
    max_dd_value = torch.min(drawdown).item()
    max_dd_idx = torch.argmin(drawdown).item()

    # Find the peak that corresponds to this maximum drawdown
    peak_idx = torch.argmax(running_max[:max_dd_idx + 1]).item() if max_dd_idx > 0 else 0

    # Calculate duration
    max_dd_duration = max_dd_idx - peak_idx

    # Current drawdown
    current_dd = drawdown[-1].item()

    # Peak and trough values
    peak_value = running_max[max_dd_idx].item()
    trough_value = portfolio_values[max_dd_idx].item()

    return {
        'max_drawdown': max_dd_value,
        'max_drawdown_duration': max_dd_duration,
        'current_drawdown': current_dd,
        'peak_value': peak_value,
        'trough_value': trough_value,
    }


def compute_sharpe_ratio(
    returns: torch.Tensor,
    periods_per_year: float,
    rf_annual: float = 0.0
) -> float:
    """
    Compute annualized Sharpe ratio.

    The Sharpe ratio measures risk-adjusted return by comparing excess returns
    (returns above the risk-free rate) to volatility.

    Args:
        returns: 1D torch.Tensor of per-period returns
        periods_per_year: Number of periods in a year (e.g., 525600 for 1-minute crypto)
        rf_annual: Annual risk-free rate (default: 0.0)

    Returns:
        Annualized Sharpe ratio. Higher is better. Typical interpretation:
        - < 1.0: Poor
        - 1.0-2.0: Good
        - > 2.0: Very good
        - > 3.0: Excellent
    """
    # Import here to avoid circular dependency
    from torchtrade.envs.utils import compute_sharpe_torch

    return compute_sharpe_torch(returns, periods_per_year, rf_annual).item()


def compute_sortino_ratio(
    returns: torch.Tensor,
    periods_per_year: float,
    rf_annual: float = 0.0,
    target_return: float = 0.0,
) -> float:
    """
    Compute annualized Sortino ratio.

    Similar to Sharpe ratio, but only penalizes downside volatility (returns below target).
    This is often more relevant for trading strategies since upside volatility is desirable.

    Args:
        returns: 1D torch.Tensor of per-period returns
        periods_per_year: Number of periods in a year
        rf_annual: Annual risk-free rate (default: 0.0)
        target_return: Target return threshold (default: 0.0)

    Returns:
        Annualized Sortino ratio. Higher is better.
    """
    # Remove NaNs
    returns = returns[~torch.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    # Convert annual RF to per-period
    rf_period = (1 + rf_annual) ** (1 / periods_per_year) - 1

    # Excess returns
    excess_returns = returns - rf_period

    # Downside returns (below target)
    downside_returns = torch.where(
        returns < target_return,
        returns - target_return,
        torch.zeros_like(returns)
    )

    # Compute mean and downside deviation
    mean_excess = torch.mean(excess_returns)
    downside_std = torch.sqrt(torch.mean(downside_returns ** 2))

    if downside_std == 0:
        return 0.0

    # Annualized Sortino
    sortino = (mean_excess / downside_std) * torch.sqrt(torch.tensor(periods_per_year, dtype=returns.dtype))

    return sortino.item()


def compute_calmar_ratio(
    portfolio_values: torch.Tensor,
    periods_per_year: float,
) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).

    The Calmar ratio measures return relative to maximum drawdown risk.
    Higher is better.

    Args:
        portfolio_values: 1D torch.Tensor of portfolio values over time
        periods_per_year: Number of periods in a year

    Returns:
        Calmar ratio. Typical interpretation:
        - < 0.5: Poor
        - 0.5-1.0: Acceptable
        - 1.0-3.0: Good
        - > 3.0: Excellent
    """
    if len(portfolio_values) < 2:
        return 0.0

    # Calculate total return
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

    # Annualize the return
    num_periods = len(portfolio_values) - 1
    years = num_periods / periods_per_year
    if years <= 0:
        return 0.0

    annualized_return = (1 + total_return) ** (1 / years) - 1

    # Calculate max drawdown
    dd_metrics = compute_max_drawdown(portfolio_values)
    max_dd = abs(dd_metrics['max_drawdown'])

    if max_dd == 0:
        return 0.0

    calmar = annualized_return / max_dd

    # Convert to float if it's a tensor
    if isinstance(calmar, torch.Tensor):
        calmar = calmar.item()

    return calmar


def compute_win_rate(returns: torch.Tensor) -> Dict[str, float]:
    """
    Compute win rate and related statistics.

    Args:
        returns: 1D torch.Tensor of per-period returns

    Returns:
        Dictionary containing:
            - win_rate: Percentage of profitable periods
            - avg_win: Average return of winning periods
            - avg_loss: Average return of losing periods
            - profit_factor: Ratio of total profits to total losses
    """
    # Remove NaNs
    returns = returns[~torch.isnan(returns)]

    if len(returns) == 0:
        return {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
        }

    # Separate wins and losses
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
    avg_win = torch.mean(wins).item() if len(wins) > 0 else 0.0
    avg_loss = torch.mean(losses).item() if len(losses) > 0 else 0.0

    # Profit factor
    total_wins = torch.sum(wins).item() if len(wins) > 0 else 0.0
    total_losses = abs(torch.sum(losses).item()) if len(losses) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

    return {
        'win_rate (reward>0)': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
    }


def compute_portfolio_returns(portfolio_values: torch.Tensor) -> torch.Tensor:
    """
    Compute per-period returns from portfolio values.

    Args:
        portfolio_values: 1D torch.Tensor of portfolio values over time

    Returns:
        1D torch.Tensor of returns (length = len(portfolio_values) - 1)
    """
    if len(portfolio_values) < 2:
        return torch.tensor([])

    returns = torch.diff(portfolio_values) / portfolio_values[:-1]
    return returns


def compute_all_metrics(
    portfolio_values: torch.Tensor,
    rewards: torch.Tensor,
    action_history: list,
    periods_per_year: float,
) -> Dict[str, float]:
    """
    Compute all trading metrics from episode data.

    This is a convenience function that computes all standard metrics
    and can be used by both environments and training scripts.

    Args:
        portfolio_values: 1D torch.Tensor of portfolio values over time
        rewards: 1D torch.Tensor of per-period rewards
        action_history: List of actions taken (for counting trades)
        periods_per_year: Number of periods in a year for annualization

    Returns:
        Dictionary containing all metrics:
            - total_return: Total portfolio return
            - sharpe_ratio: Annualized Sharpe ratio
            - sortino_ratio: Annualized Sortino ratio
            - calmar_ratio: Calmar ratio (return / max drawdown)
            - max_drawdown: Maximum drawdown (negative value)
            - max_dd_duration: Maximum drawdown duration in periods
            - num_trades: Number of trades executed
            - win_rate (reward>0): Percentage of profitable periods
            - avg_win: Average win amount
            - avg_loss: Average loss amount
            - profit_factor: Ratio of total wins to total losses
    """
    # Compute returns
    returns = compute_portfolio_returns(portfolio_values)

    # Compute drawdown metrics
    dd_metrics = compute_max_drawdown(portfolio_values)

    # Compute Sharpe ratio
    sharpe = compute_sharpe_ratio(returns, periods_per_year) if len(returns) > 0 else 0.0

    # Compute Sortino ratio
    sortino = compute_sortino_ratio(returns, periods_per_year) if len(returns) > 0 else 0.0

    # Compute Calmar ratio
    calmar = compute_calmar_ratio(portfolio_values, periods_per_year) if len(portfolio_values) > 1 else 0.0

    # Compute win rate
    win_rate_metrics = compute_win_rate(rewards)

    # Count number of trades (non-zero actions)
    num_trades = sum(1 for a in action_history if a != 0)

    # Calculate total return
    if len(portfolio_values) > 0:
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        total_return = total_return.item() if isinstance(total_return, torch.Tensor) else total_return
    else:
        total_return = 0.0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': dd_metrics['max_drawdown'],
        'max_dd_duration': dd_metrics['max_drawdown_duration'],
        'num_trades': num_trades,
        **win_rate_metrics,
    }
