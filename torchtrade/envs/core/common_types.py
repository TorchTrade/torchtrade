"""Common type definitions used across TorchTrade environments."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MarginMode(Enum):
    """Margin type for futures trading.

    - ISOLATED: Margin is isolated to individual positions
    - CROSSED: Margin is shared across all positions
    """
    ISOLATED = "ISOLATED"
    CROSSED = "CROSSED"


@dataclass
class OrderStatus:
    """Standard order status structure across exchanges."""
    is_open: bool
    filled_qty: Optional[float]
    filled_avg_price: Optional[float]
    status: str
    side: str
