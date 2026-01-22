"""Common enums and utilities shared across all environments."""

import math
from enum import Enum


class TradeMode(Enum):
    """
    Position sizing mode for trading environments.

    - QUANTITY: Fixed quantity per trade (e.g., 0.001 BTC)
      - Used when you want consistent position size in base asset units
      - Example: Always trade 0.001 BTC regardless of price or balance

    - NOTIONAL: Fixed notional value per trade (e.g., $100 USD)
      - Used when you want consistent position size in quote currency
      - Example: Always trade $100 worth of BTC, quantity varies with price
    """
    QUANTITY = "quantity"
    NOTIONAL = "notional"


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
