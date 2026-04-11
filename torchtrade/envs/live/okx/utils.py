"""Utility functions for OKX environments."""
from functools import partial

from torchtrade.envs.utils.timeframe import (
    TimeFrame,
    TimeFrameUnit,
    normalize_timeframe_config,
    create_provider_parser,
)


# OKX interval mapping
OKX_INTERVAL_MAP = {
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
_OKX_REVERSE_MAP = {
    interval_str: (value, unit)
    for unit, value_map in OKX_INTERVAL_MAP.items()
    for value, interval_str in value_map.items()
}


def timeframe_to_okx(tf: TimeFrame) -> str:
    """Convert TimeFrame to OKX interval string.

    Args:
        tf: Custom TimeFrame object

    Returns:
        OKX interval string (e.g., "1m", "15m", "1H", "1D")

    Raises:
        ValueError: If timeframe is not supported by OKX

    Examples:
        >>> tf = TimeFrame(5, TimeFrameUnit.Minute)
        >>> timeframe_to_okx(tf)
        '5m'
        >>> tf = TimeFrame(1, TimeFrameUnit.Hour)
        >>> timeframe_to_okx(tf)
        '1H'
    """
    if tf.unit not in OKX_INTERVAL_MAP:
        raise ValueError(f"Unsupported TimeFrameUnit for OKX: {tf.unit}")

    unit_map = OKX_INTERVAL_MAP[tf.unit]
    if tf.value not in unit_map:
        raise ValueError(
            f"Unsupported timeframe value for OKX: {tf.value}{tf.unit.value}. "
            f"Valid values for {tf.unit.value}: {list(unit_map.keys())}"
        )

    return unit_map[tf.value]


def okx_to_timeframe(interval: str) -> TimeFrame:
    """Convert OKX interval string to custom TimeFrame.

    Args:
        interval: OKX interval string (e.g., "1m", "15m", "1H", "1D")

    Returns:
        Custom TimeFrame object

    Raises:
        ValueError: If interval is not valid for OKX

    Examples:
        >>> okx_to_timeframe("5m")
        TimeFrame(5, TimeFrameUnit.Minute)
        >>> okx_to_timeframe("1H")
        TimeFrame(1, TimeFrameUnit.Hour)
    """
    if interval not in _OKX_REVERSE_MAP:
        raise ValueError(
            f"Invalid OKX interval: {interval}. "
            f"Valid intervals: {list(_OKX_REVERSE_MAP.keys())}"
        )

    value, unit = _OKX_REVERSE_MAP[interval]
    return TimeFrame(value, unit)


# Create OKX-specific parser using factory (tries OKX format first, then standard format)
parse_okx_timeframe_string = create_provider_parser(okx_to_timeframe)

# Convenience wrapper: normalize_timeframe_config with OKX-specific parsing
# Accepts OKX interval strings ("1m", "1H"), standard strings ("1Min"), or TimeFrame objects
normalize_okx_timeframe_config = partial(
    normalize_timeframe_config,
    parse_fn=parse_okx_timeframe_string
)


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to OKX perpetual swap format.

    OKX uses dash-separated symbols with product suffix for perpetual futures:
    "BTC-USDT-SWAP"

    Args:
        symbol: Trading symbol in various formats:
                - "BTC-USDT-SWAP" -> "BTC-USDT-SWAP" (no change)
                - "BTC-USDT" -> "BTC-USDT-SWAP"
                - "BTCUSDT" -> "BTC-USDT-SWAP"
                - "BTC/USDT" -> "BTC-USDT-SWAP"
                - "BTC/USDT:USDT" -> "BTC-USDT-SWAP"

    Returns:
        Normalized symbol in OKX swap format (e.g., "BTC-USDT-SWAP")
    """
    symbol = symbol.strip().upper()

    # Already in OKX swap format
    if symbol.endswith("-SWAP"):
        return symbol

    # Strip CCXT settlement suffix (e.g., ":USDT")
    if ":" in symbol:
        symbol = symbol.split(":")[0]

    # Handle slash-separated (e.g., "BTC/USDT")
    if "/" in symbol:
        symbol = symbol.replace("/", "-")
    # Handle concatenated (e.g., "BTCUSDT") — insert dash before quote currency
    elif "-" not in symbol:
        # Common quote currencies in order of specificity
        for quote in ("USDT", "USDC", "USD"):
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                symbol = f"{base}-{quote}"
                break

    if not symbol or "-" not in symbol:
        raise ValueError(f"Symbol cannot be parsed into OKX format: '{symbol}'")

    return f"{symbol}-SWAP"
