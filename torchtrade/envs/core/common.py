"""Common utilities shared across all environments."""

from typing import Literal


# Type alias for trade mode with autocomplete and validation
# Use string literals "fractional", "notional", or "quantity" throughout the codebase
TradeMode = Literal["fractional", "notional", "quantity"]
"""
Position sizing mode for trading environments.

- "fractional": Fraction of portfolio per trade (e.g., 0.1 = 10%)
  - Position size: portfolio_value * position_fraction * leverage / price
  - Best for: training and adaptive live sizing

- "quantity": Fixed quantity per trade (e.g., 0.001 BTC)
  - Used when you want consistent position size in base asset units

- "notional": Fixed notional value per trade (e.g., $100 USD)
  - Used when you want consistent position size in quote currency
"""


def validate_trade_mode(trade_mode: str) -> str:
    """
    Validate trade_mode configuration parameter.

    Args:
        trade_mode: The trade mode string to validate

    Returns:
        Validated trade mode string in lowercase

    Raises:
        ValueError: If trade_mode is not "fractional", "notional", or "quantity" (case-insensitive)
    """
    trade_mode_lower = trade_mode.lower()
    if trade_mode_lower not in ("fractional", "notional", "quantity"):
        raise ValueError(
            f"trade_mode must be 'fractional', 'notional', or 'quantity' (case-insensitive), got '{trade_mode}'"
        )
    return trade_mode_lower
