"""Utility functions for Bybit environments."""
from functools import partial

from torchtrade.envs.utils.timeframe import (
    TimeFrame,
    TimeFrameUnit,
    normalize_timeframe_config,
    create_provider_parser,
)


# Bybit interval mapping (pybit uses string numbers for minutes, letters for D/W/M)
BYBIT_INTERVAL_MAP = {
    TimeFrameUnit.Minute: {
        1: "1",
        3: "3",
        5: "5",
        15: "15",
        30: "30",
    },
    TimeFrameUnit.Hour: {
        1: "60",
        2: "120",
        4: "240",
        6: "360",
        12: "720",
    },
    TimeFrameUnit.Day: {
        1: "D",
    },
}

# Pre-computed reverse mapping for efficient lookups
_BYBIT_REVERSE_MAP = {
    interval_str: (value, unit)
    for unit, value_map in BYBIT_INTERVAL_MAP.items()
    for value, interval_str in value_map.items()
}


def timeframe_to_bybit(tf: TimeFrame) -> str:
    """Convert TimeFrame to Bybit interval string.

    Args:
        tf: Custom TimeFrame object

    Returns:
        Bybit interval string (e.g., "1", "15", "60", "D")

    Raises:
        ValueError: If timeframe is not supported by Bybit

    Examples:
        >>> tf = TimeFrame(5, TimeFrameUnit.Minute)
        >>> timeframe_to_bybit(tf)
        '5'
        >>> tf = TimeFrame(1, TimeFrameUnit.Hour)
        >>> timeframe_to_bybit(tf)
        '60'
    """
    if tf.unit not in BYBIT_INTERVAL_MAP:
        raise ValueError(f"Unsupported TimeFrameUnit for Bybit: {tf.unit}")

    unit_map = BYBIT_INTERVAL_MAP[tf.unit]
    if tf.value not in unit_map:
        raise ValueError(
            f"Unsupported timeframe value for Bybit: {tf.value}{tf.unit.value}. "
            f"Valid values for {tf.unit.value}: {list(unit_map.keys())}"
        )

    return unit_map[tf.value]


def bybit_to_timeframe(interval: str) -> TimeFrame:
    """Convert Bybit interval string to custom TimeFrame.

    Args:
        interval: Bybit interval string (e.g., "1", "15", "60", "D")

    Returns:
        Custom TimeFrame object

    Raises:
        ValueError: If interval is not valid for Bybit

    Examples:
        >>> bybit_to_timeframe("5")
        TimeFrame(5, TimeFrameUnit.Minute)
        >>> bybit_to_timeframe("60")
        TimeFrame(1, TimeFrameUnit.Hour)
    """
    if interval not in _BYBIT_REVERSE_MAP:
        raise ValueError(
            f"Invalid Bybit interval: {interval}. "
            f"Valid intervals: {list(_BYBIT_REVERSE_MAP.keys())}"
        )

    value, unit = _BYBIT_REVERSE_MAP[interval]
    return TimeFrame(value, unit)


# Create Bybit-specific parser using factory (tries Bybit format first, then standard format)
parse_bybit_timeframe_string = create_provider_parser(bybit_to_timeframe)

# Convenience wrapper: normalize_timeframe_config with Bybit-specific parsing
# Accepts Bybit interval strings ("1", "60"), standard strings ("1Min"), or TimeFrame objects
normalize_bybit_timeframe_config = partial(
    normalize_timeframe_config,
    parse_fn=parse_bybit_timeframe_string
)


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to Bybit linear perpetual format.

    Bybit uses simple concatenated symbols like "BTCUSDT" for linear perpetuals.

    Args:
        symbol: Trading symbol in various formats:
                - "BTCUSDT" -> "BTCUSDT" (no change)
                - "BTC/USDT" -> "BTCUSDT"
                - "BTC/USDT:USDT" -> "BTCUSDT"

    Returns:
        Normalized symbol in Bybit format (e.g., "BTCUSDT")
    """
    symbol = symbol.strip().upper()

    # Strip CCXT settlement suffix
    if ":" in symbol:
        symbol = symbol.split(":")[0]

    # Remove slash
    symbol = symbol.replace("/", "")

    if not symbol:
        raise ValueError("Symbol cannot be empty after normalization")

    return symbol
