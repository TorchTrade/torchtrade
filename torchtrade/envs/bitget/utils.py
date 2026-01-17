"""Utility functions for Bitget environments."""
from typing import List, Union, Tuple
import warnings
from torchtrade.envs.timeframe import (
    TimeFrame,
    TimeFrameUnit,
    parse_timeframe_string,
    normalize_timeframe_config,
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
    import re

    # Reverse mapping
    reverse_map = {}
    for unit, value_map in BITGET_INTERVAL_MAP.items():
        for value, interval_str in value_map.items():
            reverse_map[interval_str] = (value, unit)

    if interval not in reverse_map:
        raise ValueError(
            f"Invalid Bitget interval: {interval}. "
            f"Valid intervals: {list(reverse_map.keys())}"
        )

    value, unit = reverse_map[interval]
    return TimeFrame(value, unit)


def parse_bitget_timeframe_string(s: str) -> TimeFrame:
    """Parse timeframe string to custom TimeFrame object.

    Accepts both Bitget interval format ("1m", "5m", "1H") and standard format
    ("1Min", "5Min", "1Hour") for flexibility.

    Args:
        s: Timeframe string (e.g., "5Min", "1Hour", or "5m", "1H")

    Returns:
        Custom TimeFrame object

    Raises:
        ValueError: If string format is invalid or unit is unknown

    Examples:
        >>> parse_bitget_timeframe_string("5Min")
        TimeFrame(5, TimeFrameUnit.Minute)
        >>> parse_bitget_timeframe_string("5m")  # Bitget format
        TimeFrame(5, TimeFrameUnit.Minute)
        >>> parse_bitget_timeframe_string("1Hour")
        TimeFrame(1, TimeFrameUnit.Hour)
        >>> parse_bitget_timeframe_string("1H")  # Bitget format
        TimeFrame(1, TimeFrameUnit.Hour)
    """
    # Try Bitget interval format first (e.g., "1m", "5m", "1H")
    try:
        return bitget_to_timeframe(s)
    except ValueError:
        # Fall back to standard format (e.g., "1Min", "5Min", "1Hour")
        return parse_timeframe_string(s)


def normalize_bitget_timeframe_config(
    execute_on: Union[str, TimeFrame],
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]],
    window_sizes: Union[List[int], int],
) -> Tuple[TimeFrame, List[TimeFrame], List[int]]:
    """Normalize timeframe configuration for Bitget environment configs.

    Accepts both custom TimeFrame objects, standard timeframe strings ("1Min"),
    and Bitget interval strings ("1m") for backwards compatibility.

    Args:
        execute_on: Execution timeframe (string or custom TimeFrame)
        time_frames: List of timeframes or single timeframe
        window_sizes: List of window sizes or single window size

    Returns:
        Tuple of (execute_on_tf, time_frames_list, window_sizes_list) using custom TimeFrame

    Examples:
        >>> # Using standard strings
        >>> execute_on, tfs, ws = normalize_bitget_timeframe_config("5Min", ["1Min", "5Min"], 10)

        >>> # Using Bitget interval strings (backwards compatible)
        >>> execute_on, tfs, ws = normalize_bitget_timeframe_config("5m", ["1m", "5m"], 10)

        >>> # Using custom TimeFrame
        >>> from torchtrade.envs.timeframe import TimeFrame, TimeFrameUnit
        >>> tf = TimeFrame(5, TimeFrameUnit.Minute)
        >>> execute_on, tfs, ws = normalize_bitget_timeframe_config(tf, [tf], 10)
    """
    # Use shared normalization with Bitget-aware parse function
    return normalize_timeframe_config(
        execute_on, time_frames, window_sizes, parse_fn=parse_bitget_timeframe_string
    )
