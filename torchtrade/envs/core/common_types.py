"""Common type definitions used across TorchTrade environments."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MarginMode(Enum):
    """Margin mode for futures trading.

    - ISOLATED: Margin is isolated to individual positions
    - CROSSED: Margin is shared across all positions
    """
    ISOLATED = "ISOLATED"
    CROSSED = "CROSSED"


@dataclass
class PositionStatus:
    """What a futures venue says it is holding, in one shape for all of them.

    Was five dataclasses -- four venues plus replay -- with identical field NAMES and two
    diverging annotations (#288). Three of the five were byte-identical. A copy that has
    not drifted yet is still a copy, and this one had already drifted once: replay carried
    both `margin_type` and `margin_mode` as separate fields to satisfy two spellings of
    the same concept, which only surfaced when #289 collapsed the names.

    Alpaca is deliberately NOT here. It is spot, and its status object is a different
    shape (`market_value`, `avg_entry_price`, `unrealized_pl`) rather than the same shape
    under other names.
    """

    qty: float  # Positive for long, negative for short
    notional_value: float
    entry_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    mark_price: float
    leverage: float  # float, not int: int() truncated 1.5x to 1x, which then took
    # the no-liquidation branch and reported a levered position as safe (#277).
    margin_mode: Optional[str]  # Optional: bybit reports none for a cross-margin account
    liquidation_price: float


@dataclass
class OrderStatus:
    """Standard order status structure across exchanges."""
    is_open: bool
    filled_qty: Optional[float]
    filled_avg_price: Optional[float]
    status: str
    side: str
