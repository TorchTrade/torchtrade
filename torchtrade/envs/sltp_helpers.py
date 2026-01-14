"""Helper functions for Stop-Loss/Take-Profit (SLTP) price calculations."""

from typing import Tuple


def calculate_long_bracket_prices(entry_price: float, sl_pct: float, tp_pct: float) -> Tuple[float, float]:
    """
    Calculate SL/TP prices for long positions.

    For long positions:
    - Stop loss must be BELOW entry (loss when price falls)
    - Take profit must be ABOVE entry (profit when price rises)

    Args:
        entry_price: Entry price for the position
        sl_pct: Stop loss percentage (negative, e.g., -0.02 for 2% below entry)
        tp_pct: Take profit percentage (positive, e.g., 0.05 for 5% above entry)

    Returns:
        Tuple of (stop_loss_price, take_profit_price)

    Example:
        >>> calculate_long_bracket_prices(50000, -0.02, 0.05)
        (49000.0, 52500.0)  # SL 2% below, TP 5% above
    """
    stop_loss = entry_price * (1 + sl_pct)
    take_profit = entry_price * (1 + tp_pct)
    return stop_loss, take_profit


def calculate_short_bracket_prices(entry_price: float, sl_pct: float, tp_pct: float) -> Tuple[float, float]:
    """
    Calculate SL/TP prices for short positions.

    For short positions:
    - Stop loss must be ABOVE entry (loss when price rises)
    - Take profit must be BELOW entry (profit when price falls)

    IMPORTANT: The action_map already swaps SL/TP for shorts, so:
    - sl_pct will be POSITIVE (from the original tp_pct)
    - tp_pct will be NEGATIVE (from the original sl_pct)

    Args:
        entry_price: Entry price for the position
        sl_pct: Stop loss percentage from action_map (positive, e.g., 0.03 for 3% above entry)
        tp_pct: Take profit percentage from action_map (negative, e.g., -0.02 for 2% below entry)

    Returns:
        Tuple of (stop_loss_price, take_profit_price)

    Example:
        >>> # Action map provides: sl_pct=0.03, tp_pct=-0.02 for shorts
        >>> calculate_short_bracket_prices(50000, 0.03, -0.02)
        (51500.0, 49000.0)  # SL 3% above, TP 2% below
    """
    # For shorts, action_map has already swapped the percentages
    # sl_pct is positive (above entry), tp_pct is negative (below entry)
    stop_loss = entry_price * (1 + sl_pct)
    take_profit = entry_price * (1 + tp_pct)
    return stop_loss, take_profit


def calculate_bracket_prices(side: str, entry_price: float, sl_pct: float, tp_pct: float) -> Tuple[float, float]:
    """
    Calculate SL/TP prices for either long or short positions.

    Convenience function that dispatches to the appropriate calculator based on side.

    Args:
        side: Position side ("long" or "short")
        entry_price: Entry price for the position
        sl_pct: Stop loss percentage from action_map
        tp_pct: Take profit percentage from action_map

    Returns:
        Tuple of (stop_loss_price, take_profit_price)

    Raises:
        ValueError: If side is not "long" or "short"

    Example:
        >>> calculate_bracket_prices("long", 50000, -0.02, 0.05)
        (49000.0, 52500.0)
        >>> calculate_bracket_prices("short", 50000, 0.03, -0.02)
        (51500.0, 49000.0)
    """
    if side == "long":
        return calculate_long_bracket_prices(entry_price, sl_pct, tp_pct)
    elif side == "short":
        return calculate_short_bracket_prices(entry_price, sl_pct, tp_pct)
    else:
        raise ValueError(f"Invalid side: {side}. Must be 'long' or 'short'.")
