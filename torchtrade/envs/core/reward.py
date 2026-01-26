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
        >>> from torchtrade.envs.core.reward import sharpe_ratio_reward
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
from typing import Protocol, List, Optional, Callable
import numpy as np
import inspect


@dataclass
class RewardContext:
    """
    Minimal context for custom reward functions.

    Provides core state needed for most reward computations. For advanced metrics
    (Sharpe ratio, drawdown, win rate), use the helper functions or access the
    environment directly via the metadata dict.

    Attributes:
        old_portfolio_value: Portfolio value before the action
        new_portfolio_value: Portfolio value after the action
        action: Action taken (integer action index)
        current_step: Current step in the episode (0-indexed)
        max_steps: Maximum steps in the episode
        trade_executed: Whether a trade was executed this step
        fee_paid: Transaction fees paid this step (default: 0.0)
        slippage_amount: Slippage incurred this step (default: 0.0)
        metadata: Optional dict for environment-specific data.
                  Available fields vary by environment type:

                  **Offline Sequential Envs** (SeqLongOnly, SeqLongOnlySLTP):
                  - portfolio_value_history, action_history, reward_history
                  - base_price_history, initial_portfolio_value
                  - buy_and_hold_value (terminal step only)

                  **Offline Futures Envs** (SeqFutures, SeqFuturesSLTP, FuturesOneStep):
                  - All sequential env fields above, plus:
                  - position_history, leverage, margin_ratio
                  - liquidation_price, liquidated (bool)

                  **Offline One-Step Envs** (LongOnlyOneStep):
                  - All sequential env fields, plus:
                  - rollout_returns (list of returns during rollout)

                  **Live Envs** (Alpaca, Binance):
                  - No history tracking (for performance)
                  - Futures live envs may include: leverage, margin_ratio, liquidation_price

                  Access via: ctx.metadata.get('key', default_value)

    Example:
        >>> def my_reward(ctx: RewardContext) -> float:
        ...     # Simple log return
        ...     return np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)
        ...
        >>> def advanced_reward(ctx: RewardContext) -> float:
        ...     # Access environment-specific data via metadata
        ...     leverage = ctx.metadata.get('leverage', 1.0)
        ...     history = ctx.metadata.get('portfolio_value_history', [])
        ...     # Compute metrics using helpers
        ...     sharpe = compute_sharpe_ratio(history)
        ...     return sharpe / leverage
    """

    # Core state (always present)
    old_portfolio_value: float
    new_portfolio_value: float
    action: int
    current_step: int
    max_steps: int
    trade_executed: bool

    # Transaction costs (optional, default to 0)
    fee_paid: float = 0.0
    slippage_amount: float = 0.0

    # Optional extensions for advanced use cases
    # Access environment-specific data like:
    # - ctx.metadata['portfolio_value_history']
    # - ctx.metadata['leverage']
    # - ctx.metadata['liquidated']
    # - ctx.metadata['buy_and_hold_value']
    metadata: Optional[dict] = None


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
# Utility Functions
# ============================================================================


def validate_reward_function(reward_function: Callable) -> None:
    """
    Validate that a custom reward function has the correct signature.

    Checks that the function accepts exactly one parameter (RewardContext)
    and provides a helpful error message if not.

    Args:
        reward_function: The custom reward function to validate

    Raises:
        TypeError: If the reward function doesn't have the correct signature

    Example:
        >>> def my_reward(ctx: RewardContext) -> float:
        ...     return 1.0
        >>> validate_reward_function(my_reward)  # OK
        >>>
        >>> def bad_reward(x, y):
        ...     return 1.0
        >>> validate_reward_function(bad_reward)  # Raises TypeError
    """
    if reward_function is None:
        return

    sig = inspect.signature(reward_function)
    params = list(sig.parameters.values())

    if len(params) != 1:
        raise TypeError(
            f"Reward function must accept exactly 1 parameter (RewardContext), "
            f"but got {len(params)} parameters: {list(sig.parameters.keys())}. "
            f"Expected signature: def reward_function(ctx: RewardContext) -> float"
        )


def build_reward_context(
    env,
    old_portfolio_value: float,
    new_portfolio_value: float,
    action,
    trade_info: dict,
    **metadata_fields
) -> RewardContext:
    """
    Build reward context from environment state.

    This is a utility function to eliminate code duplication across all environments.
    Extracts common fields from any TorchTrade environment and allows passing
    environment-specific data via metadata_fields.

    Args:
        env: The environment instance (any TorchTrade environment)
        old_portfolio_value: Portfolio value before action
        new_portfolio_value: Portfolio value after action
        action: Action taken (will be converted to int)
        trade_info: Dict with trade execution details
        **metadata_fields: Additional environment-specific data to include in metadata

    Returns:
        RewardContext with core fields populated and metadata dict

    Example:
        >>> # In environment's _calculate_reward method:
        >>> ctx = build_reward_context(
        ...     self, old_pv, new_pv, action, trade_info,
        ...     portfolio_value_history=self.portfolio_value_history,
        ...     leverage=float(self.leverage)
        ... )
        >>> return self.config.reward_function(ctx)
    """
    return RewardContext(
        old_portfolio_value=old_portfolio_value,
        new_portfolio_value=new_portfolio_value,
        action=int(action) if isinstance(action, (int, float)) else 0,
        current_step=getattr(env, 'step_counter', 0),
        max_steps=getattr(env, 'max_traj_length', 1),
        trade_executed=trade_info.get('executed', False),
        fee_paid=trade_info.get('fee_paid', 0.0),
        slippage_amount=trade_info.get('price_noise', 0.0),
        metadata=metadata_fields if metadata_fields else None,
    )


def default_log_return(old_portfolio_value: float, new_portfolio_value: float) -> float:
    """
    Simple default reward: log return of portfolio value.

    Computes: log(portfolio_value_t / portfolio_value_t-1)

    This is a simple, interpretable reward that:
    - Is symmetric (equal magnitude for +10% and -10% returns)
    - Naturally handles compounding
    - Is scale-invariant (works for any portfolio size)
    - Handles bankruptcy gracefully (returns large negative reward)

    Args:
        old_portfolio_value: Portfolio value before action
        new_portfolio_value: Portfolio value after action

    Returns:
        Log return reward, or -10.0 if bankrupt (new_portfolio_value <= 0)

    Raises:
        ValueError: If old_portfolio_value <= 0, which indicates a calculation
                   error (the previous state was already invalid)

    Note:
        This function doesn't require a RewardContext, making it more efficient
        for the default case. Use this when you don't need custom reward logic.
    """
    if old_portfolio_value <= 0:
        raise ValueError(
            f"Invalid old_portfolio_value: {old_portfolio_value}. "
            f"Portfolio value must be positive. This indicates a calculation error in the previous step."
        )

    # Handle bankruptcy gracefully: return large negative reward instead of crashing
    # The environment will terminate in the next step via _check_termination
    if new_portfolio_value <= 0:
        return -10.0  # Severe penalty for bankruptcy (log(0.01) ≈ -4.6, so -10 is ~99% loss)

    return float(np.log(new_portfolio_value / old_portfolio_value))


# Backward compatibility: keep default_reward_function for users who may reference it
def default_reward_function(ctx: RewardContext) -> float:
    """
    Default reward function: log return (context-based version).

    This is kept for backward compatibility. For better performance, environments
    should use default_log_return() directly instead of building a context.

    Args:
        ctx: RewardContext containing portfolio values

    Returns:
        Log return reward
    """
    return default_log_return(ctx.old_portfolio_value, ctx.new_portfolio_value)


# ============================================================================
# Example Custom Reward Functions
# ============================================================================


def sharpe_ratio_reward(ctx: RewardContext) -> float:
    """
    Reward based on Sharpe ratio of historical returns.

    Computes the Sharpe ratio from all historical portfolio value returns.
    Useful for encouraging risk-adjusted returns rather than raw returns.

    Args:
        ctx: RewardContext with portfolio_value_history in metadata

    Returns:
        Sharpe ratio clipped to [-10.0, 10.0], or 0.0 if insufficient history

    Example:
        >>> # When building context, pass history:
        >>> ctx = build_reward_context(
        ...     env, old_pv, new_pv, action, trade_info,
        ...     portfolio_value_history=env.portfolio_value_history
        ... )
    """
    if ctx.metadata is None:
        return 0.0

    portfolio_history = ctx.metadata.get('portfolio_value_history', [])
    sharpe = compute_sharpe_ratio(portfolio_history)
    return float(np.clip(sharpe, -10.0, 10.0))


def drawdown_penalty_reward(ctx: RewardContext) -> float:
    """
    Log return with penalty for drawdown.

    Combines step-wise log return with a penalty proportional to current
    drawdown from historical peak. Encourages consistent gains while
    discouraging large drawdowns.

    Args:
        ctx: RewardContext with portfolio_value_history in metadata

    Returns:
        Log return with drawdown penalty
    """
    # Base reward: log return
    log_return = default_log_return(ctx.old_portfolio_value, ctx.new_portfolio_value)

    # Compute drawdown from peak if history available
    if ctx.metadata is None:
        return log_return

    portfolio_history = ctx.metadata.get('portfolio_value_history', [])
    if len(portfolio_history) == 0:
        return log_return

    current_dd = compute_drawdown(portfolio_history + [ctx.new_portfolio_value])

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
        ctx: RewardContext with buy_and_hold_value and initial_portfolio_value in metadata

    Returns:
        0.0 for non-terminal steps, portfolio outperformance for terminal step
    """
    is_terminal = ctx.current_step >= ctx.max_steps - 1

    if not is_terminal or ctx.metadata is None:
        return 0.0

    buy_and_hold_value = ctx.metadata.get('buy_and_hold_value')
    initial_value = ctx.metadata.get('initial_portfolio_value', ctx.old_portfolio_value)

    if buy_and_hold_value is None:
        return 0.0

    compare_value = max(initial_value, buy_and_hold_value)
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
        ctx: RewardContext with buy_and_hold_value and initial_portfolio_value in metadata

    Returns:
        Dense reward for non-terminal steps, dense + terminal for terminal step
    """
    # Dense component: log return with fee penalty
    log_return = default_log_return(ctx.old_portfolio_value, ctx.new_portfolio_value)
    fee_penalty = -ctx.fee_paid / ctx.old_portfolio_value if ctx.old_portfolio_value > 0 else 0.0
    dense_reward = log_return + fee_penalty
    clipped_dense = float(np.clip(dense_reward, -0.1, 0.1))

    # Early return if not terminal or no metadata
    is_terminal = ctx.current_step >= ctx.max_steps - 1
    if not is_terminal or ctx.metadata is None:
        return clipped_dense

    # Terminal component: compare to benchmark
    buy_and_hold_value = ctx.metadata.get('buy_and_hold_value')
    initial_value = ctx.metadata.get('initial_portfolio_value', ctx.old_portfolio_value)

    if buy_and_hold_value is None:
        return clipped_dense

    compare_value = max(initial_value, buy_and_hold_value)
    if compare_value <= 0:
        return clipped_dense

    terminal_return = (ctx.new_portfolio_value - compare_value) / compare_value
    terminal_reward = float(np.clip(terminal_return, -5.0, 5.0))
    return clipped_dense + terminal_reward


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
