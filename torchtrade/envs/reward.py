"""
Custom reward function infrastructure for TorchTrade environments.

This module provides a flexible interface for defining custom reward functions
that can be used across all TorchTrade environments. Users can either use the
default log return reward or provide their own custom implementations.

Examples:
    Using default reward (log return):
        >>> config = SeqLongOnlyEnvConfig(symbol="BTC/USD")
        >>> env = SeqLongOnlyEnv(config)  # Uses default reward

    Using custom reward function:
        >>> from torchtrade.envs.reward import sharpe_ratio_reward
        >>> config = SeqLongOnlyEnvConfig(
        ...     symbol="BTC/USD",
        ...     reward_function=sharpe_ratio_reward
        ... )
        >>> env = SeqLongOnlyEnv(config)

    Creating custom reward:
        >>> def my_custom_reward(ctx: RewardContext) -> float:
        ...     log_return = np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)
        ...     # Add your custom logic here
        ...     return log_return
        >>> config = SeqLongOnlyEnvConfig(reward_function=my_custom_reward)
"""

from dataclasses import dataclass
from typing import Protocol, List, Optional
import numpy as np


@dataclass
class RewardContext:
    """
    Context passed to custom reward functions containing all necessary state and history.

    This dataclass provides comprehensive information about the current environment state,
    trade execution details, and historical data needed to compute sophisticated reward
    metrics like Sharpe ratio, drawdown, win rate, etc.

    Attributes:
        old_portfolio_value: Portfolio value before the action
        new_portfolio_value: Portfolio value after the action
        action: Action taken (integer action index)
        current_step: Current step in the episode (0-indexed)
        max_steps: Maximum steps in the episode

        trade_executed: Whether a trade was executed this step
        trade_side: Side of the trade ('buy', 'sell', 'hold', 'long', 'short')
        fee_paid: Transaction fees paid this step
        slippage_amount: Slippage incurred this step

        cash: Current cash balance
        position_size: Current position size (shares/contracts)
        position_value: Current position value in base currency
        entry_price: Entry price of current position (0 if no position)
        current_price: Current market price
        unrealized_pnl_pct: Unrealized PnL as percentage of entry value
        holding_time: Steps holding current position

        portfolio_value_history: Historical portfolio values
        action_history: Historical actions taken
        reward_history: Historical rewards received
        base_price_history: Historical close prices

        position_history: Historical position sizes (futures only)
        liquidated: Whether position was liquidated this step
        leverage: Current leverage (futures only)
        margin_ratio: Current margin ratio (futures only)
        liquidation_price: Liquidation price (futures only)

        initial_portfolio_value: Portfolio value at episode start
        buy_and_hold_value: Buy & hold benchmark value (computed at terminal step)
    """

    # Current step state
    old_portfolio_value: float
    new_portfolio_value: float
    action: int
    current_step: int
    max_steps: int

    # Trade execution info
    trade_executed: bool
    trade_side: str  # 'buy', 'sell', 'hold', 'long', 'short'
    fee_paid: float
    slippage_amount: float

    # Account state (7-10 elements depending on environment)
    cash: float
    position_size: float
    position_value: float
    entry_price: float
    current_price: float
    unrealized_pnl_pct: float
    holding_time: int

    # History for complex metrics (drawdown, Sharpe, etc.)
    portfolio_value_history: List[float]
    action_history: List[int]
    reward_history: List[float]
    base_price_history: List[float]

    # Benchmark (required fields must come before optional fields)
    initial_portfolio_value: float

    # Optional environment-specific fields
    position_history: Optional[List[float]] = None
    liquidated: bool = False
    leverage: Optional[float] = None
    margin_ratio: Optional[float] = None
    liquidation_price: Optional[float] = None
    buy_and_hold_value: Optional[float] = None


class RewardFunction(Protocol):
    """
    Protocol for custom reward functions.

    Custom reward functions must be callables that accept a RewardContext
    and return a float reward value.

    Example:
        >>> def my_reward(ctx: RewardContext) -> float:
        ...     return np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)
    """

    def __call__(self, ctx: RewardContext) -> float:
        """
        Compute reward given current step context.

        Args:
            ctx: RewardContext containing all necessary state and history

        Returns:
            Reward value (float)
        """
        ...


# ============================================================================
# Default Reward Function
# ============================================================================


def default_reward_function(ctx: RewardContext) -> float:
    """
    Default reward function: log return of portfolio value.

    Computes: log(portfolio_value_t / portfolio_value_t-1)

    This is a simple, interpretable reward that:
    - Is symmetric (equal magnitude for +10% and -10% returns)
    - Naturally handles compounding
    - Is scale-invariant (works for any portfolio size)

    Args:
        ctx: RewardContext containing portfolio values

    Returns:
        Log return reward, or 0.0 if old portfolio value is invalid
    """
    if ctx.old_portfolio_value <= 0:
        return 0.0
    return float(np.log(ctx.new_portfolio_value / ctx.old_portfolio_value))


# ============================================================================
# Example Custom Reward Functions
# ============================================================================


def sharpe_ratio_reward(ctx: RewardContext) -> float:
    """
    Reward based on Sharpe ratio of historical returns.

    Computes the Sharpe ratio from all historical portfolio value returns.
    Useful for encouraging risk-adjusted returns rather than raw returns.

    Args:
        ctx: RewardContext containing portfolio value history

    Returns:
        Sharpe ratio clipped to [-10.0, 10.0], or 0.0 if insufficient history
    """
    if len(ctx.portfolio_value_history) < 2:
        return 0.0

    # Compute log returns
    returns = []
    for i in range(1, len(ctx.portfolio_value_history)):
        if ctx.portfolio_value_history[i - 1] <= 0:
            continue
        ret = np.log(ctx.portfolio_value_history[i] / ctx.portfolio_value_history[i - 1])
        returns.append(ret)

    if len(returns) == 0:
        return 0.0

    # Sharpe ratio = mean / std
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns)) + 1e-9
    sharpe = mean_ret / std_ret

    return float(np.clip(sharpe, -10.0, 10.0))


def drawdown_penalty_reward(ctx: RewardContext) -> float:
    """
    Log return with penalty for drawdown.

    Combines step-wise log return with a penalty proportional to current
    drawdown from historical peak. Encourages consistent gains while
    discouraging large drawdowns.

    Args:
        ctx: RewardContext containing portfolio values and history

    Returns:
        Log return with drawdown penalty
    """
    # Base reward: log return
    if ctx.old_portfolio_value <= 0:
        return 0.0
    log_return = float(np.log(ctx.new_portfolio_value / ctx.old_portfolio_value))

    # Compute drawdown from peak
    if len(ctx.portfolio_value_history) == 0:
        return log_return

    peak = float(np.max(ctx.portfolio_value_history))
    if peak <= 0:
        return log_return

    current_dd = (peak - ctx.new_portfolio_value) / peak

    # Apply penalty for significant drawdown (> 10%)
    dd_penalty = -5.0 * current_dd if current_dd > 0.1 else 0.0

    return log_return + dd_penalty


def terminal_comparison_reward(ctx: RewardContext) -> float:
    """
    Sparse terminal reward comparing final portfolio to buy & hold benchmark.

    Returns 0.0 for all non-terminal steps, and on the terminal step returns
    the percentage outperformance vs. max(initial_value, buy_and_hold_value).

    Useful for training agents to maximize final portfolio value without
    providing intermediate feedback.

    Args:
        ctx: RewardContext with terminal state information

    Returns:
        0.0 for non-terminal steps, portfolio outperformance for terminal step
    """
    is_terminal = ctx.current_step >= ctx.max_steps - 1

    if not is_terminal:
        return 0.0

    if ctx.buy_and_hold_value is None:
        return 0.0

    compare_value = max(ctx.initial_portfolio_value, ctx.buy_and_hold_value)
    if compare_value <= 0:
        return 0.0

    terminal_return = (ctx.new_portfolio_value - compare_value) / compare_value

    return float(np.clip(terminal_return, -5.0, 5.0))


def hybrid_dense_sparse_reward(ctx: RewardContext) -> float:
    """
    Hybrid reward combining dense step-wise and sparse terminal components.

    Dense component (all steps):
    - Log return of portfolio value
    - Transaction fee penalty

    Terminal component (last step only):
    - Comparison to buy & hold benchmark

    This balances immediate feedback (dense) with long-term objective (sparse).

    Args:
        ctx: RewardContext with full state and history

    Returns:
        Dense reward for non-terminal steps, dense + terminal for terminal step
    """
    is_terminal = ctx.current_step >= ctx.max_steps - 1

    # Dense component: log return with fee penalty
    if ctx.old_portfolio_value <= 0:
        dense_reward = 0.0
    else:
        log_return = np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)
        fee_penalty = -ctx.fee_paid / ctx.old_portfolio_value
        dense_reward = float(log_return + fee_penalty)

    # Terminal component: compare to benchmark
    if is_terminal and ctx.buy_and_hold_value is not None:
        compare_value = max(ctx.initial_portfolio_value, ctx.buy_and_hold_value)
        if compare_value > 0:
            terminal_return = (ctx.new_portfolio_value - compare_value) / compare_value
            terminal_reward = float(np.clip(terminal_return, -5.0, 5.0))
            return float(np.clip(dense_reward, -0.1, 0.1)) + terminal_reward

    return float(np.clip(dense_reward, -0.1, 0.1))


def realized_pnl_reward(ctx: RewardContext) -> float:
    """
    Reward only on trade execution (realized PnL).

    Returns portfolio return only when trades are executed, 0.0 otherwise.
    Useful for environments where holding positions shouldn't generate rewards
    until they're closed.

    Args:
        ctx: RewardContext with trade execution info

    Returns:
        Portfolio return if trade executed, 0.0 otherwise
    """
    if not ctx.trade_executed:
        return 0.0

    if ctx.old_portfolio_value <= 0:
        return 0.0

    portfolio_return = (ctx.new_portfolio_value - ctx.old_portfolio_value) / ctx.old_portfolio_value
    return float(portfolio_return)


# ============================================================================
# Helper Functions
# ============================================================================


def compute_drawdown(portfolio_values: List[float]) -> float:
    """
    Compute current drawdown from historical peak.

    Args:
        portfolio_values: List of historical portfolio values

    Returns:
        Drawdown as fraction (0.0 = no drawdown, 1.0 = 100% drawdown)
    """
    if len(portfolio_values) == 0:
        return 0.0

    peak = float(np.max(portfolio_values))
    current = float(portfolio_values[-1])

    if peak <= 0:
        return 0.0

    return (peak - current) / peak


def compute_max_drawdown(portfolio_values: List[float]) -> float:
    """
    Compute maximum drawdown from historical peak.

    Args:
        portfolio_values: List of historical portfolio values

    Returns:
        Maximum drawdown as fraction
    """
    if len(portfolio_values) == 0:
        return 0.0

    max_dd = 0.0
    peak = portfolio_values[0]

    for value in portfolio_values:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

    return float(max_dd)


def compute_sharpe_ratio(portfolio_values: List[float], periods_per_year: float = 252.0) -> float:
    """
    Compute annualized Sharpe ratio from portfolio values.

    Args:
        portfolio_values: List of historical portfolio values
        periods_per_year: Number of periods per year for annualization

    Returns:
        Annualized Sharpe ratio
    """
    if len(portfolio_values) < 2:
        return 0.0

    # Compute log returns
    returns = []
    for i in range(1, len(portfolio_values)):
        if portfolio_values[i - 1] <= 0:
            continue
        ret = np.log(portfolio_values[i] / portfolio_values[i - 1])
        returns.append(ret)

    if len(returns) == 0:
        return 0.0

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns)) + 1e-9

    # Annualize
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year)

    return float(sharpe)


def compute_win_rate(action_history: List[int], reward_history: List[float]) -> float:
    """
    Compute win rate: fraction of actions with positive rewards.

    Args:
        action_history: List of actions taken
        reward_history: List of rewards received

    Returns:
        Win rate as fraction (0.0 to 1.0)
    """
    if len(reward_history) == 0:
        return 0.0

    wins = sum(1 for r in reward_history if r > 0)
    total = len(reward_history)

    return float(wins) / float(total)
