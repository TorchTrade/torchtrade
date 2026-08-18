"""Common utilities shared across all environments."""

from typing import Literal


# Type alias for trade mode with autocomplete and validation
# Use string literals "fractional", "notional", or "quantity" throughout the codebase
TradeMode = Literal["fractional", "notional", "quantity"]
"""
Position sizing mode for trading environments.

- "fractional": Fraction of portfolio per trade (e.g., 0.1 = 10%)
  - Position size: portfolio_value * position_fraction * leverage / price,
    net of the entry fee (see fractional_sizing.fee_multiplier, #278)
  - Best for: training and adaptive live sizing

- "quantity": Fixed quantity per trade (e.g., 0.001 BTC)
  - Used when you want consistent position size in base asset units

- "notional": Fixed notional value per trade (e.g., $100 USD)
  - Used when you want consistent position size in quote currency
"""


def validate_position_sizing(
    trade_mode: str, position_fraction: float, quantity_per_trade: float
) -> None:
    """Reject a sizing config that would produce a zero, negative or >100% position.

    Was seven copies. Six agreed; alpaca's `elif` read `== "notional"` and so never
    checked `quantity_per_trade` in "quantity" mode -- which `alpaca/env_sltp.py` sizes
    every trade from. A zero there is a silent no-op order, a negative one is a reversed
    order, and both passed construction.
    """
    if trade_mode == "fractional":
        if not (0 < position_fraction <= 1.0):
            raise ValueError(f"position_fraction must be in (0, 1.0], got {position_fraction}")
    elif trade_mode in ("notional", "quantity"):
        if quantity_per_trade <= 0:
            raise ValueError(f"quantity_per_trade must be positive, got {quantity_per_trade}")


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
