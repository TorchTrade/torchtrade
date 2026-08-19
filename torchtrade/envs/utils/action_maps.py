"""Shared action map utilities for SLTP (Stop-Loss/Take-Profit) environments."""

from itertools import product
from typing import Dict, List, Optional, Tuple


def create_sltp_action_map(
    stoploss_levels: List[float],
    takeprofit_levels: List[float],
    include_short_positions: bool = False,
    include_hold_action: bool = True,
    include_close_action: bool = False,  # Changed default to False for SLTP envs
) -> Dict[int, Tuple[Optional[str], Optional[float], Optional[float]]]:
    """Create a mapping from action indices to (side, stop_loss_pct, take_profit_pct) tuples.

    This unified function supports both long-only environments (Alpaca) and
    long/short environments (Binance futures).

    Action Space:
        - Action 0 (optional): HOLD (no trade) - if include_hold_action=True
        - Action 1 (optional): CLOSE (close position) - if include_close_action=True
        - Actions 2..N+1: LONG positions with (stop_loss_pct, take_profit_pct) combinations
        - Actions N+2..M+1: SHORT positions with (stop_loss_pct, take_profit_pct) combinations
                         (only if include_short_positions=True)

    Args:
        stoploss_levels: List of stop-loss percentages (typically negative, e.g., -0.02 = -2%)
        takeprofit_levels: List of take-profit percentages (typically positive, e.g., 0.05 = 5%)
        include_short_positions: If True, include short position actions with swapped SL/TP
        include_hold_action: If True, action 0 = HOLD (default: True)
        include_close_action: If True, add CLOSE action to exit positions (default: False for SLTP)

    Returns:
        Dict mapping action index to (side, stop_loss_pct, take_profit_pct) tuple where:
        - side is "long", "short", "close", or None for HOLD
        - stop_loss_pct and take_profit_pct are the percentage levels (None for CLOSE)

    Examples:
        >>> # Long-only environment (Alpaca) WITHOUT CLOSE (default)
        >>> action_map = create_sltp_action_map([-0.02, -0.05], [0.03, 0.06])
        >>> action_map[0]  # HOLD
        (None, None, None)
        >>> action_map[1]  # Long with SL=-2%, TP=3%
        ('long', -0.02, 0.03)

        >>> # Long/short environment (Binance) with CLOSE enabled
        >>> action_map = create_sltp_action_map([-0.02], [0.03], include_short_positions=True, include_close_action=True)
        >>> action_map[1]  # CLOSE
        ('close', None, None)
        >>> action_map[2]  # Long
        ('long', -0.02, 0.03)
        >>> action_map[3]  # Short (SL/TP swapped)
        ('short', 0.03, -0.02)

    Note:
        For short positions SL must be above entry (positive %) and TP below entry
        (negative %), so the levels are NEGATED: ``-sl`` becomes the stop and ``-tp`` the
        target. Short action k is then the exact mirror of long action k.

        It used to SWAP the two lists instead. That puts the sides right and the
        magnitudes wrong, because both halves iterate the same
        ``product(stoploss_levels, takeprofit_levels)`` -- so the nth short carried the
        nth long's risk and reward exchanged, unconditionally, and visibly only when the
        magnitudes differed. With ``stoploss_levels=(-0.025, -0.05, -0.1)`` and
        ``takeprofit_levels=(0.05, 0.1, 0.2)`` long stops were {2.5, 5, 10}% while short
        stops were {5, 10, 20}%: no short action had a 2.5% stop anywhere in the space,
        so a policy that learned "tight stop, wide target" got the opposite geometry the
        moment it went short (#279).

        The action-space SIZE is unchanged (19 actions either way) and only short indices
        change MEANING, so an SLTP checkpoint trained before this loads without complaint
        and trades a different strategy. Retrain rather than reload.
    """
    action_map = {}
    idx = 0

    # Optional HOLD action
    if include_hold_action:
        action_map[0] = (None, None, None)
        idx = 1

    # Optional CLOSE action
    if include_close_action:
        action_map[idx] = ("close", None, None)
        idx += 1

    # Long positions with SL/TP combinations
    for sl, tp in product(stoploss_levels, takeprofit_levels):
        action_map[idx] = ("long", sl, tp)
        idx += 1

    # Short positions with SL/TP combinations (if enabled)
    if include_short_positions:
        for sl, tp in product(stoploss_levels, takeprofit_levels):
            # Negate, do not swap. Shorts need SL above entry (positive) and TP below
            # (negative), which a swap also produces -- but it takes the magnitudes from
            # the OTHER list, so the nth short carried the nth long's risk and reward
            # exchanged (#279). Negating keeps the sides and mirrors the geometry.
            action_map[idx] = ("short", -sl, -tp)
            idx += 1

    return action_map


def create_alpaca_sltp_action_map(
    stoploss_levels: List[float],
    takeprofit_levels: List[float],
    include_hold_action: bool = True,
    include_close_action: bool = False,
) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    """Create action map for Alpaca SLTP environment (long-only, returns (sl, tp) tuples).

    Wrapper around create_sltp_action_map() that returns the Alpaca-specific format.

    Args:
        stoploss_levels: List of stop-loss percentages
        takeprofit_levels: List of take-profit percentages
        include_hold_action: If True, action 0 = HOLD (default: True)
        include_close_action: If True, add CLOSE action to exit positions (default: False)

    Returns:
        Dict mapping action index to (stop_loss_pct, take_profit_pct) tuple
        Action 0 returns (None, None) for HOLD if include_hold_action=True
        Action 1 returns (None, None) for CLOSE if include_close_action=True
    """
    full_map = create_sltp_action_map(
        stoploss_levels,
        takeprofit_levels,
        include_short_positions=False,
        include_hold_action=include_hold_action,
        include_close_action=include_close_action,
    )

    # Convert (side, sl, tp) -> (sl, tp) for Alpaca compatibility
    alpaca_map = {}
    for idx, (side, sl, tp) in full_map.items():
        alpaca_map[idx] = (sl, tp)

    return alpaca_map
