"""Default reward functions for TorchTrade environments.

This module provides simple, well-tested reward functions that work with
the history tracker interface. Users can use these as-is or create their own.
"""

import math

import numpy as np


def log_return_reward(history) -> float:
    """Default log return reward using history tracker.

    Computes: log(portfolio_value[-1] / portfolio_value[-2])

    This is a simple, interpretable reward that:
    - Is symmetric (equal magnitude for +10% and -10% returns)
    - Naturally handles compounding
    - Is scale-invariant (works for any portfolio size)
    - Handles bankruptcy gracefully (returns large negative reward)

    Args:
        history: HistoryTracker instance with portfolio_values

    Returns:
        Log return reward, or -10.0 if bankrupt, or 0.0 if insufficient history

    Examples:
        >>> # Use as default
        >>> config = SequentialTradingEnvConfig(
        ...     trading_mode="spot",
        ...     reward_function=log_return_reward  # This is the default
        ... )

        >>> # Or customize it
        >>> def scaled_log_return(history):
        ...     return log_return_reward(history) * 10.0
    """
    if len(history.portfolio_values) < 2:
        return 0.0

    old_value = history.portfolio_values[-2]
    new_value = history.portfolio_values[-1]

    if old_value <= 0:
        raise ValueError(
            f"Invalid old portfolio value: {old_value}. "
            f"Portfolio value must be positive. This indicates a calculation error."
        )

    # Handle bankruptcy gracefully: return large negative reward
    if new_value <= 0:
        return -10.0

    # PERF: math.log is ~3-5x faster than np.log for single floats
    return math.log(new_value / old_value)


def sharpe_ratio_reward(history) -> float:
    """Reward based on running Sharpe ratio.

    Encourages risk-adjusted returns instead of raw returns.
    Uses all historical portfolio values to compute Sharpe ratio.

    Args:
        history: HistoryTracker with portfolio_values

    Returns:
        Sharpe ratio clipped to [-10.0, 10.0], or 0.0 if insufficient history

    Examples:
        >>> config = SequentialTradingEnvConfig(
        ...     trading_mode="spot",
        ...     reward_function=sharpe_ratio_reward
        ... )
    """
    if len(history.portfolio_values) < 3:
        return 0.0

    values = np.array(history.portfolio_values)

    # Guard against non-positive values (e.g. after liquidation)
    if np.any(values <= 0):
        return -10.0

    returns = np.diff(np.log(values))

    mean_return = returns.mean()
    std_return = returns.std() + 1e-9  # Avoid division by zero

    sharpe = mean_return / std_return
    return float(np.clip(sharpe, -10.0, 10.0))


def drawdown_penalty_reward(history) -> float:
    """Log return with penalty for drawdown from peak.

    Combines step-wise log return with a penalty proportional to current
    drawdown from historical peak. Encourages consistent gains while
    discouraging large drawdowns.

    Args:
        history: HistoryTracker with portfolio_values

    Returns:
        Log return with drawdown penalty

    Examples:
        >>> config = SequentialTradingEnvConfig(
        ...     trading_mode="spot",
        ...     reward_function=drawdown_penalty_reward
        ... )
    """
    if len(history.portfolio_values) < 2:
        return 0.0

    # Base reward: log return
    old_value = history.portfolio_values[-2]
    new_value = history.portfolio_values[-1]

    if old_value <= 0 or new_value <= 0:
        return -10.0

    log_ret = float(np.log(new_value / old_value))

    # Compute drawdown from peak
    peak = float(np.max(history.portfolio_values))
    if peak <= 0:
        return log_ret

    current_drawdown = (peak - new_value) / peak

    # Apply penalty for significant drawdown (> 10%)
    dd_penalty = -5.0 * current_drawdown if current_drawdown > 0.1 else 0.0

    return log_ret + dd_penalty
