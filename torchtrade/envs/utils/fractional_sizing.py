"""Shared utilities for fractional position sizing across environments."""

from typing import Tuple
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
# Tolerance needs to be large enough to avoid churn from normal price fluctuations
# When using portfolio_value for position sizing, small price changes cause target to drift
POSITION_TOLERANCE_PCT = 0.02  # 2% - relative tolerance as fraction of target position
POSITION_TOLERANCE_ABS = 0.001  # Absolute minimum tolerance for very small positions


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
    allow_short: bool = True
) -> list[float]:
    """Build default action levels for fractional position sizing.

    Args:
        allow_short: Allow short positions (futures vs long-only)

    Returns:
        List of action level values in range [-1.0, 1.0]

    Examples:
        >>> # Futures (default)
        >>> build_default_action_levels(allow_short=True)
        [-1, 0, 1]

        >>> # Long-only
        >>> build_default_action_levels(allow_short=False)
        [0, 1]
    """
    if allow_short:
        # Futures: short, flat, long
        return [-1, 0, 1]
    else:
        # Long-only: flat, long
        return [0, 1]


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
