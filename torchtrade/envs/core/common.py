"""Common utilities shared across all environments."""

import math
from typing import Literal


# Type alias for trade mode with autocomplete and validation
# Use string literals "notional" or "quantity" throughout the codebase
TradeMode = Literal["notional", "quantity"]
"""
Position sizing mode for trading environments.

- "quantity": Fixed quantity per trade (e.g., 0.001 BTC)
  - Used when you want consistent position size in base asset units
  - Example: Always trade 0.001 BTC regardless of price or balance

- "notional": Fixed notional value per trade (e.g., $100 USD)
  - Used when you want consistent position size in quote currency
  - Example: Always trade $100 worth of BTC, quantity varies with price
"""


def validate_trade_mode(trade_mode: str) -> str:
    """
    Validate trade_mode configuration parameter.

    Args:
        trade_mode: The trade mode string to validate

    Returns:
        Validated trade mode string in lowercase

    Raises:
        ValueError: If trade_mode is not "notional" or "quantity" (case-insensitive)
    """
    trade_mode_lower = trade_mode.lower()
    if trade_mode_lower not in ("notional", "quantity"):
        raise ValueError(
            f"trade_mode must be 'notional' or 'quantity' (case-insensitive), got '{trade_mode}'"
        )
    return trade_mode_lower


def validate_quantity_per_trade(quantity_per_trade: float) -> None:
    """
    Validate quantity_per_trade configuration parameter.

    Args:
        quantity_per_trade: The quantity per trade value to validate

    Raises:
        TypeError: If quantity_per_trade is not a number
        ValueError: If quantity_per_trade is not positive, finite, or is NaN
    """
    if not isinstance(quantity_per_trade, (int, float)):
        raise TypeError(
            f"quantity_per_trade must be a number, got {type(quantity_per_trade).__name__}"
        )
    if math.isnan(quantity_per_trade) or math.isinf(quantity_per_trade):
        raise ValueError(f"quantity_per_trade must be finite, got {quantity_per_trade}")
    if quantity_per_trade <= 0:
        raise ValueError(f"quantity_per_trade must be positive, got {quantity_per_trade}")
