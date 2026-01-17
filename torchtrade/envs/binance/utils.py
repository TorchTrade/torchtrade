"""Utility functions for Binance environments."""
from typing import List, Union, Tuple
import warnings
from torchtrade.envs.timeframe import (
    TimeFrame,
    TimeFrameUnit,
    parse_timeframe_string,
    normalize_timeframe_config,
    binance_to_timeframe,
)


def parse_binance_timeframe_string(s: str) -> TimeFrame:
    """Parse timeframe string to custom TimeFrame object.

    Accepts both Binance interval format ("1m", "5m", "1h") and standard format
    ("1Min", "5Min", "1Hour") for flexibility.

    Args:
        s: Timeframe string (e.g., "5Min", "1Hour", or "5m", "1h")

    Returns:
        Custom TimeFrame object

    Raises:
        ValueError: If string format is invalid or unit is unknown

    Examples:
        >>> parse_binance_timeframe_string("5Min")
        TimeFrame(5, TimeFrameUnit.Minute)
        >>> parse_binance_timeframe_string("5m")  # Binance format
        TimeFrame(5, TimeFrameUnit.Minute)
        >>> parse_binance_timeframe_string("1Hour")
        TimeFrame(1, TimeFrameUnit.Hour)
        >>> parse_binance_timeframe_string("1h")  # Binance format
        TimeFrame(1, TimeFrameUnit.Hour)
    """
    # Try Binance interval format first (e.g., "1m", "5m", "1h")
    try:
        return binance_to_timeframe(s)
    except ValueError:
        # Fall back to standard format (e.g., "1Min", "5Min", "1Hour")
        return parse_timeframe_string(s)


def normalize_binance_timeframe_config(
    execute_on: Union[str, TimeFrame],
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]],
    window_sizes: Union[List[int], int],
) -> Tuple[TimeFrame, List[TimeFrame], List[int]]:
    """Normalize timeframe configuration for Binance environment configs.

    Accepts both custom TimeFrame objects, standard timeframe strings ("1Min"),
    and Binance interval strings ("1m") for backwards compatibility.

    Args:
        execute_on: Execution timeframe (string or custom TimeFrame)
        time_frames: List of timeframes or single timeframe
        window_sizes: List of window sizes or single window size

    Returns:
        Tuple of (execute_on_tf, time_frames_list, window_sizes_list) using custom TimeFrame

    Examples:
        >>> # Using standard strings
        >>> execute_on, tfs, ws = normalize_binance_timeframe_config("5Min", ["1Min", "5Min"], 10)

        >>> # Using Binance interval strings (backwards compatible)
        >>> execute_on, tfs, ws = normalize_binance_timeframe_config("5m", ["1m", "5m"], 10)

        >>> # Using custom TimeFrame
        >>> from torchtrade.envs.timeframe import TimeFrame, TimeFrameUnit
        >>> tf = TimeFrame(5, TimeFrameUnit.Minute)
        >>> execute_on, tfs, ws = normalize_binance_timeframe_config(tf, [tf], 10)
    """
    # Use shared normalization with Binance-aware parse function
    return normalize_timeframe_config(
        execute_on, time_frames, window_sizes, parse_fn=parse_binance_timeframe_string
    )
