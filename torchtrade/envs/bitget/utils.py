"""Utility functions for Bitget environments."""
from functools import partial
from typing import List, Union, Tuple

from torchtrade.envs.timeframe import (
    TimeFrame,
    TimeFrameUnit,
    normalize_timeframe_config,
    create_provider_parser,
)


# Bitget interval mapping
BITGET_INTERVAL_MAP = {
    TimeFrameUnit.Minute: {
        1: "1m",
        3: "3m",
        5: "5m",
        15: "15m",
        30: "30m",
    },
    TimeFrameUnit.Hour: {
        1: "1H",
        2: "2H",
        4: "4H",
        6: "6H",
        12: "12H",
    },
    TimeFrameUnit.Day: {
        1: "1D",
        3: "3D",
    },
}

# Pre-computed reverse mapping for efficient lookups
_BITGET_REVERSE_MAP = {
    interval_str: (value, unit)
    for unit, value_map in BITGET_INTERVAL_MAP.items()
    for value, interval_str in value_map.items()
}


def timeframe_to_bitget(tf: TimeFrame) -> str:
    """Convert TimeFrame to Bitget interval string.

    Args:
        tf: Custom TimeFrame object

    Returns:
        Bitget interval string (e.g., "1m", "5m", "1H", "1D")

    Raises:
        ValueError: If timeframe is not supported by Bitget

    Examples:
        >>> tf = TimeFrame(5, TimeFrameUnit.Minute)
        >>> timeframe_to_bitget(tf)
        '5m'
        >>> tf = TimeFrame(1, TimeFrameUnit.Hour)
        >>> timeframe_to_bitget(tf)
        '1H'
    """
    if tf.unit not in BITGET_INTERVAL_MAP:
        raise ValueError(f"Unsupported TimeFrameUnit for Bitget: {tf.unit}")

    unit_map = BITGET_INTERVAL_MAP[tf.unit]
    if tf.value not in unit_map:
        raise ValueError(
            f"Unsupported timeframe value for Bitget: {tf.value}{tf.unit.value}. "
            f"Valid values for {tf.unit.value}: {list(unit_map.keys())}"
        )

    return unit_map[tf.value]


def bitget_to_timeframe(interval: str) -> TimeFrame:
    """Convert Bitget interval string to custom TimeFrame.

    Args:
        interval: Bitget interval string (e.g., "1m", "5m", "1H", "1D")

    Returns:
        Custom TimeFrame object

    Raises:
        ValueError: If interval is not valid for Bitget

    Examples:
        >>> bitget_to_timeframe("5m")
        TimeFrame(5, TimeFrameUnit.Minute)
        >>> bitget_to_timeframe("1H")
        TimeFrame(1, TimeFrameUnit.Hour)
    """
    if interval not in _BITGET_REVERSE_MAP:
        raise ValueError(
            f"Invalid Bitget interval: {interval}. "
            f"Valid intervals: {list(_BITGET_REVERSE_MAP.keys())}"
        )

    value, unit = _BITGET_REVERSE_MAP[interval]
    return TimeFrame(value, unit)


# Create Bitget-specific parser using factory (tries Bitget format first, then standard format)
parse_bitget_timeframe_string = create_provider_parser(bitget_to_timeframe)

# Convenience wrapper: normalize_timeframe_config with Bitget-specific parsing
# Accepts Bitget interval strings ("1m", "1H"), standard strings ("1Min"), or TimeFrame objects
normalize_bitget_timeframe_config = partial(
    normalize_timeframe_config,
    parse_fn=parse_bitget_timeframe_string
)


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to CCXT perpetual swap format.

    Args:
        symbol: Trading symbol in various formats:
                - "BTCUSDT" -> "BTC/USDT:USDT"
                - "BTC/USDT" -> "BTC/USDT:USDT"
                - "BTC/USDT:USDT" -> "BTC/USDT:USDT" (no change)

    Returns:
        Normalized symbol in CCXT perpetual swap format (e.g., "BTC/USDT:USDT")
    """
    import warnings

    # Already in correct format
    if "/" in symbol and ":" in symbol:
        return symbol

    # Has slash but no settlement currency
    if "/" in symbol and ":" not in symbol:
        return f"{symbol}:USDT"

    # No slash - parse and add both
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}/USDT:USDT"

    # Fallback for other formats
    normalized = f"{symbol}/USDT:USDT"
    warnings.warn(
        f"Symbol '{symbol}' doesn't match expected format. "
        f"Auto-converted to '{normalized}'. "
        f"Please use CCXT format 'BTC/USDT:USDT' directly to avoid ambiguity."
    )
    return normalized
