"""State management dataclasses for TorchTrade environments."""

from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Union


def binarize_action_type(action_type: str) -> int:
    """Convert action type string to binarized action value.

    Args:
        action_type: Action type string ("buy"/"long" -> 1, "sell"/"short" -> -1, others -> 0)

    Returns:
        Binarized action value (-1, 0, or 1)

    Examples:
        >>> binarize_action_type("buy")
        1
        >>> binarize_action_type("long")
        1
        >>> binarize_action_type("sell")
        -1
        >>> binarize_action_type("short")
        -1
        >>> binarize_action_type("hold")
        0
        >>> binarize_action_type("liquidation")
        0
    """
    if action_type in ("buy", "long"):
        return 1
    elif action_type in ("sell", "short"):
        return -1
    return 0


@dataclass
class PositionState:
    """Encapsulates all position-related state.

    Attributes:
        current_position: Current position indicator (0=no position, 1=long, -1=short)
        position_size: Number of units held (can be negative for shorts)
        position_value: Current market value of the position
        entry_price: Price at which the position was entered
        unrealized_pnlpc: Unrealized profit/loss as percentage of entry price
        hold_counter: Number of steps the position has been held
    """
    current_position: float = 0.0
    position_size: float = 0.0
    position_value: float = 0.0
    entry_price: float = 0.0
    unrealized_pnlpc: float = 0.0
    hold_counter: int = 0

    def reset(self):
        """Reset all position state to initial values."""
        for field in fields(self):
            setattr(self, field.name, field.default)


@dataclass
class HistoryTracker:
    """Encapsulates episode history tracking for all environments.

    Tracks price, action, reward, portfolio value, and position history across an episode.
    Provides methods to record steps, reset history, and export data for analysis.

    Attributes:
        base_prices: List of asset base prices at each step
        actions: List of actions taken at each step
        rewards: List of rewards received at each step
        portfolio_values: List of total portfolio values at each step
        positions: List of position sizes at each step (positive=long, negative=short, 0=flat)
        action_types: List of action types at each step ("hold", "buy", "sell", "long", "short", etc.)

    Example:
        >>> history = HistoryTracker()
        >>> history.record_step(price=50000.0, action=1.0, reward=0.05, portfolio_value=5000.0, position=0.5, action_type="long")
        >>> history.to_dict()
        {'base_prices': [50000.0], 'actions': [1.0], 'rewards': [0.05], 'portfolio_values': [5000.0], 'positions': [0.5], 'action_types': ['long']}
    """

    base_prices: List[float] = field(default_factory=list)
    actions: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    portfolio_values: List[float] = field(default_factory=list)
    positions: List[float] = field(default_factory=list)
    action_types: List[str] = field(default_factory=list)

    def reset(self) -> None:
        """Clear all history arrays.

        Resets the tracker to empty state, typically called at the start of a new episode.
        """
        self.base_prices.clear()
        self.actions.clear()
        self.rewards.clear()
        self.portfolio_values.clear()
        self.positions.clear()
        self.action_types.clear()

    def record_step(
        self,
        price: Union[int, float],
        action: Union[int, float],
        reward: Union[int, float],
        portfolio_value: Union[int, float],
        position: Union[int, float] = 0.0,
        action_type: str = "hold"
    ) -> None:
        """Record a single step's history.

        Args:
            price: Base asset price at this step
            action: Action taken at this step
            reward: Reward received at this step
            portfolio_value: Total portfolio value at this step
            position: Position size at this step (positive=long, negative=short, 0=flat). Defaults to 0.0.
            action_type: Type of action taken ("hold", "buy", "sell", "long", "short", etc.)
        """
        self.base_prices.append(price)
        self.actions.append(action)
        self.rewards.append(reward)
        self.portfolio_values.append(portfolio_value)
        self.positions.append(position)
        self.action_types.append(action_type)

    def to_dict(self) -> Dict[str, List[float]]:
        """Export history as dictionary for plotting/analysis.

        Returns:
            Dictionary mapping history field names to their value lists
        """
        return asdict(self)

    def __len__(self) -> int:
        """Return the number of steps recorded.

        Returns:
            Number of steps in the history
        """
        return len(self.base_prices)
