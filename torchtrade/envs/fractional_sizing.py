"""Shared utilities for fractional position sizing across environments."""

from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PositionCalculationParams:
    """Parameters for fractional position calculation."""
    balance: float
    action_value: float
    current_price: float
    leverage: int = 1
    transaction_fee: float = 0.0
    allow_short: bool = True


# Position tolerance constants
# Used to determine if a position is already close enough to target
POSITION_TOLERANCE_PCT = 0.001  # 0.1% - relative tolerance as fraction of target position
POSITION_TOLERANCE_ABS = 0.0001  # Absolute minimum tolerance for very small positions


def calculate_fractional_position(params: PositionCalculationParams) -> Tuple[float, float, str]:
    """Calculate position size from fractional action value.

    This is the core position sizing formula used across all fractional environments.

    Args:
        params: Position calculation parameters

    Returns:
        (position_size, notional_value, side) where:
        - position_size: Quantity in base currency (positive=long, negative=short, 0=flat)
        - notional_value: Absolute value of position in quote currency
        - side: "long", "short", or "flat"

    Formula:
        For long/short with leverage:
            position_size = (balance × |action| × leverage) / price
            (accounting for fees in margin calculation)

        For long-only (leverage=1):
            position_size = (balance × action) / price

    Examples:
        >>> params = PositionCalculationParams(
        ...     balance=10000, action_value=0.5, current_price=50000, leverage=5
        ... )
        >>> pos_size, notional, side = calculate_fractional_position(params)
        >>> # (10000 × 0.5 × 5) / 50000 = 0.5 BTC long
        >>> assert abs(pos_size - 0.5) < 0.01
        >>> assert side == "long"
    """
    # Handle neutral case
    if params.action_value == 0.0:
        return 0.0, 0.0, "flat"

    # Validate short positions if not allowed
    if params.action_value < 0 and not params.allow_short:
        # For long-only environments, negative actions mean "reduce/sell"
        # But target calculation should be handled by caller
        # Here we just validate the formula works for allowed range
        pass

    # Calculate fraction and direction
    fraction = abs(params.action_value)
    direction = 1 if params.action_value > 0 else -1

    # Allocate fraction of balance
    capital_allocated = params.balance * fraction

    # Account for fees in margin calculation
    # We need to ensure: margin + fee <= capital_allocated
    # Where: margin = notional / leverage, fee = notional × transaction_fee
    #
    # Solving for notional:
    #   notional × (1/leverage + transaction_fee) <= capital_allocated
    #   notional <= capital_allocated / (1/leverage + transaction_fee)
    #
    # Simplified form (multiply numerator and denominator by leverage):
    #   fee_multiplier = 1 + (leverage × transaction_fee)
    #   notional = (capital_allocated / fee_multiplier) × leverage
    fee_multiplier = 1 + (params.leverage * params.transaction_fee)
    margin_required = capital_allocated / fee_multiplier
    notional_value = margin_required * params.leverage

    # Convert to position size
    position_qty = notional_value / params.current_price

    # Apply direction
    position_size = position_qty * direction
    side = "long" if direction > 0 else "short"

    return position_size, notional_value, side


def build_default_action_levels(
    position_sizing_mode: str,
    include_hold_action: bool = True,
    include_close_action: bool = False,
    allow_short: bool = True
) -> list[float]:
    """Build default action levels based on environment configuration.

    Args:
        position_sizing_mode: "fractional" or "fixed"
        include_hold_action: Include HOLD action (only used in fixed mode)
        include_close_action: Include CLOSE action (only used in fixed mode for futures)
        allow_short: Allow short positions (futures vs long-only)

    Returns:
        List of action level values

    Examples:
        >>> # Fractional futures (default)
        >>> build_default_action_levels("fractional", allow_short=True)
        [-1.0, -0.5, 0.0, 0.5, 1.0]

        >>> # Fractional long-only (default)
        >>> build_default_action_levels("fractional", allow_short=False)
        [0.0, 0.5, 1.0]

        >>> # Legacy fixed mode with hold and close
        >>> build_default_action_levels("fixed", True, True, True)
        [-1.0, 0.0, 0.5, 1.0]

        >>> # Legacy fixed mode with only hold
        >>> build_default_action_levels("fixed", True, False, True)
        [-1.0, 0.0, 1.0]
    """
    if position_sizing_mode == "fractional":
        # Fractional sizing with neutral at 0
        if allow_short:
            # Futures: allow both long and short positions
            return [-1.0, -0.5, 0.0, 0.5, 1.0]
        else:
            # Long-only: only non-negative actions (0 = close, positive = buy)
            return [0.0, 0.5, 1.0]

    else:  # "fixed" - legacy mode
        # Build legacy action levels based on flags
        if allow_short:
            # Futures legacy mode
            if include_hold_action and include_close_action:
                return [-1.0, 0.0, 0.5, 1.0]  # Short, Hold, Close, Long
            elif include_hold_action:
                return [-1.0, 0.0, 1.0]  # Short, Hold, Long
            elif include_close_action:
                return [-1.0, 0.5, 1.0]  # Short, Close, Long
            else:
                return [-1.0, 1.0]  # Short, Long
        else:
            # Long-only legacy mode
            if include_hold_action:
                return [-1.0, 0.0, 1.0]  # Sell-all, Hold, Buy-all
            else:
                return [-1.0, 1.0]  # Sell-all, Buy-all


def validate_position_sizing_mode(mode: str) -> None:
    """Validate position sizing mode parameter.

    Args:
        mode: Position sizing mode string

    Raises:
        ValueError: If mode is not "fractional" or "fixed"
    """
    if mode not in ["fractional", "fixed"]:
        raise ValueError(
            f"position_sizing_mode must be 'fractional' or 'fixed', got '{mode}'"
        )


def validate_action_levels(action_levels: list[float]) -> None:
    """Validate custom action levels.

    Args:
        action_levels: List of action level values

    Raises:
        ValueError: If action levels are invalid
    """
    if not all(-1.0 <= a <= 1.0 for a in action_levels):
        raise ValueError(
            f"All action_levels must be in range [-1.0, 1.0], got {action_levels}"
        )

    if len(action_levels) != len(set(action_levels)):
        raise ValueError(
            f"action_levels must not contain duplicates, got {action_levels}"
        )

    if len(action_levels) < 2:
        raise ValueError(
            f"action_levels must contain at least 2 actions, got {len(action_levels)}"
        )


def round_to_step_size(quantity: float, step_size: float) -> float:
    """Round quantity to exchange step size.

    Args:
        quantity: Quantity to round
        step_size: Exchange step size (e.g., 0.001 for BTC)

    Returns:
        Rounded quantity

    Examples:
        >>> round_to_step_size(0.1234, 0.001)
        0.123
        >>> round_to_step_size(1.9999, 0.01)
        2.0
    """
    if step_size == 0:
        return quantity
    return round(quantity / step_size) * step_size
