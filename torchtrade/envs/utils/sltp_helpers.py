"""Helper functions for Stop-Loss/Take-Profit (SLTP) price calculations."""

from typing import Tuple


def calculate_bracket_prices(side: str, entry_price: float, sl_pct: float, tp_pct: float) -> Tuple[float, float]:
    """Bracket prices for a long or short, from percentages the action map already signed.

    One formula for both sides: the action map hands over percentages already oriented
    against the position (long -0.02/+0.05, short +0.02/-0.05), so `entry * (1 + pct)`
    puts each leg on the correct side without a per-side branch. There were two
    byte-identical functions here whose only job was to document the SWAP the map used
    to apply to shorts; #279 negates instead, so the pair had nothing left to say.

    `side` is still validated -- a typo would otherwise price a bracket off percentages
    meant for the other direction.

    Args:
        side: Position side ("long" or "short")
        entry_price: Entry price for the position
        sl_pct: Stop loss percentage from action_map (negative long, positive short)
        tp_pct: Take profit percentage from action_map (positive long, negative short)

    Returns:
        Tuple of (stop_loss_price, take_profit_price)

    Raises:
        ValueError: If side is not "long" or "short"

    Example:
        >>> calculate_bracket_prices("long", 50000, -0.02, 0.05)
        (49000.0, 52500.0)
        >>> calculate_bracket_prices("short", 50000, 0.02, -0.05)
        (51000.0, 47500.0)
    """
    if side not in ("long", "short"):
        raise ValueError(f"Invalid side: {side}. Must be 'long' or 'short'.")
    return entry_price * (1 + sl_pct), entry_price * (1 + tp_pct)


def stop_fill_price(stop_price: float, open_price: float, is_long: bool) -> float:
    """Price a triggered stop, accounting for a bar that gapped past it (#280).

    A stop is a market order once touched, so a bar that opens beyond it fills at the
    open, not at the stop: long entry 100 with a stop at 97.5, on a bar opening at 85,
    fills near 85.

    There is deliberately no take-profit counterpart. A take-profit is a limit order,
    which fills at its limit even when price runs past, and chasing a favourable gap up
    to the open would make the backtest more optimistic -- the opposite of the pessimism
    this module is built on ("we assume the worst case ... so live trading can only
    outperform the backtest").

    The min/max self-selects, so there is no separate is-this-a-gap branch: a bar that
    merely wicks through opens on the untriggered side of the stop -- above it for a
    long -- so min returns the stop unchanged.
    """
    return min(open_price, stop_price) if is_long else max(open_price, stop_price)
