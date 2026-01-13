"""State management dataclasses for TorchTrade environments."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HistoryTracker:
    """Encapsulates episode history tracking for offline environments.

    Tracks price, action, reward, and portfolio value history across an episode.
    Provides methods to record steps, reset history, and export data for analysis.

    Attributes:
        base_prices: List of asset base prices at each step
        actions: List of actions taken at each step
        rewards: List of rewards received at each step
        portfolio_values: List of total portfolio values at each step

    Example:
        >>> history = HistoryTracker()
        >>> history.record_step(price=50000.0, action=1.0, reward=0.05, portfolio_value=5000.0)
        >>> history.to_dict()
        {'base_prices': [50000.0], 'actions': [1.0], 'rewards': [0.05], 'portfolio_values': [5000.0]}
    """

    base_prices: List[float] = field(default_factory=list)
    actions: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    portfolio_values: List[float] = field(default_factory=list)

    def reset(self) -> None:
        """Clear all history arrays.

        Resets the tracker to empty state, typically called at the start of a new episode.
        """
        self.base_prices.clear()
        self.actions.clear()
        self.rewards.clear()
        self.portfolio_values.clear()

    def record_step(
        self,
        price: float,
        action: float,
        reward: float,
        portfolio_value: float
    ) -> None:
        """Record a single step's history.

        Args:
            price: Base asset price at this step
            action: Action taken at this step
            reward: Reward received at this step
            portfolio_value: Total portfolio value at this step
        """
        self.base_prices.append(price)
        self.actions.append(action)
        self.rewards.append(reward)
        self.portfolio_values.append(portfolio_value)

    def to_dict(self) -> Dict[str, List[float]]:
        """Export history as dictionary for plotting/analysis.

        Returns:
            Dictionary mapping history field names to their value lists
        """
        return {
            'base_prices': self.base_prices,
            'actions': self.actions,
            'rewards': self.rewards,
            'portfolio_values': self.portfolio_values
        }

    def __len__(self) -> int:
        """Return the number of steps recorded.

        Returns:
            Number of steps in the history
        """
        return len(self.base_prices)


@dataclass
class FuturesHistoryTracker(HistoryTracker):
    """Extended history tracker for futures environments.

    Adds position size tracking to the base history tracker for environments
    that support long/short positions (futures trading).

    Attributes:
        positions: List of position sizes at each step (positive for long, negative for short)

    Example:
        >>> history = FuturesHistoryTracker()
        >>> history.record_step(price=50000.0, action=1.0, reward=0.05, portfolio_value=5000.0, position=0.5)
        >>> history.to_dict()
        {'base_prices': [50000.0], 'actions': [1.0], 'rewards': [0.05],
         'portfolio_values': [5000.0], 'positions': [0.5]}
    """

    positions: List[float] = field(default_factory=list)

    def reset(self) -> None:
        """Clear all history arrays including position history."""
        super().reset()
        self.positions.clear()

    def record_step(
        self,
        price: float,
        action: float,
        reward: float,
        portfolio_value: float,
        position: Optional[float] = None
    ) -> None:
        """Record a single step's history including position.

        Args:
            price: Base asset price at this step
            action: Action taken at this step
            reward: Reward received at this step
            portfolio_value: Total portfolio value at this step
            position: Position size at this step (positive=long, negative=short)
        """
        super().record_step(price, action, reward, portfolio_value)
        if position is not None:
            self.positions.append(position)

    def to_dict(self) -> Dict[str, List[float]]:
        """Export history as dictionary including position history.

        Returns:
            Dictionary mapping history field names to their value lists
        """
        result = super().to_dict()
        result['positions'] = self.positions
        return result
