"""Common utilities shared across all environments."""

from typing import Literal, Sequence

from torchtrade.envs.core.common_types import MarginMode


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

    Was seven copies. Six agreed; alpaca's `elif` read `== "notional"`, so in "quantity"
    mode it validated nothing. Not a money bug -- alpaca SLTP raises NotImplementedError
    for that mode at the first trade -- but it failed one trade late instead of at
    construction, which is the boundary this function exists to hold.
    """
    if trade_mode == "fractional":
        if not (0 < position_fraction <= 1.0):
            raise ValueError(f"position_fraction must be in (0, 1.0], got {position_fraction}")
    elif trade_mode in ("notional", "quantity"):
        if quantity_per_trade <= 0:
            raise ValueError(f"quantity_per_trade must be positive, got {quantity_per_trade}")


def validate_offline_margin_mode(margin_mode) -> None:
    """CROSSED is not implemented offline, so refuse it instead of silently isolating.

    Offline liquidation always uses isolated_liquidation_price, so a CROSSED config
    produced isolated math while the user believed otherwise (#289).
    """
    if margin_mode is not MarginMode.ISOLATED:
        raise ValueError(
            f"margin_mode={margin_mode} is not implemented for the offline "
            f"environments: liquidation always uses isolated_liquidation_price, so "
            f"CROSSED silently produced isolated liquidation math (#289). Nothing "
            f"reads this field. Use ISOLATED, or the live envs, which honour it."
        )


def validate_sltp_levels(
    stoploss_levels: Sequence[float], takeprofit_levels: Sequence[float]
) -> None:
    """Reject SL/TP levels whose sign would invert the bracket.

    The action map NEGATES these for shorts (#279), so the sign convention is what puts
    each leg on the correct side of entry. A positive stop level silently places a LONG's
    stop above entry -- an inverted bracket that exits on a favourable move.

    Both offline SLTP envs already raised here; the four live futures configs and alpaca's
    did not, and they are the ones passing include_short_positions=True.
    """
    for sl in stoploss_levels:
        if sl >= 0:
            raise ValueError(
                f"Stop-loss levels must be negative (e.g., -0.05 for 5% loss), got {sl}"
            )
    for tp in takeprofit_levels:
        if tp <= 0:
            raise ValueError(
                f"Take-profit levels must be positive (e.g., 0.1 for 10% profit), got {tp}"
            )


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


def validate_unknown_status_budget(steps) -> None:
    """Checked at the boundary: a negative budget reads as "disabled" (#295)."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError(
            f"max_unknown_status_steps must be an int >= 0 (0 disables the grace "
            f"period), got {steps!r}"
        )
