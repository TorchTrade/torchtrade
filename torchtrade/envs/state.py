"""State classes for trading environments."""

from dataclasses import dataclass


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
        self.current_position = 0.0
        self.position_size = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.unrealized_pnlpc = 0.0
        self.hold_counter = 0
