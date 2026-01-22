"""Utility functions for Alpaca environments."""
from typing import Any, List, Union, Tuple
import warnings
from torchtrade.envs.utils.timeframe import (
    TimeFrame,
    TimeFrameUnit,
    parse_timeframe_string,
    normalize_timeframe_config,
    alpaca_to_timeframe,
)


def parse_alpaca_timeframe_string(s: str) -> TimeFrame:
    """Parse timeframe string to custom TimeFrame object.

    Args:
        s: Timeframe string (e.g., "5Min", "1Hour", "15Minute")

    Returns:
        Custom TimeFrame object

    Raises:
        ValueError: If string format is invalid or unit is unknown

    Examples:
        >>> parse_alpaca_timeframe_string("5Min")
        TimeFrame(5, TimeFrameUnit.Minute)
        >>> parse_alpaca_timeframe_string("1Hour")
        TimeFrame(1, TimeFrameUnit.Hour)
    """
    return parse_timeframe_string(s)


def normalize_alpaca_timeframe_config(
    execute_on: Union[str, TimeFrame, "AlpacaTimeFrame"],
    time_frames: Union[List[Union[str, TimeFrame, "AlpacaTimeFrame"]], Union[str, TimeFrame, "AlpacaTimeFrame"]],
    window_sizes: Union[List[int], int],
) -> Tuple[TimeFrame, List[TimeFrame], List[int]]:
    """Normalize timeframe configuration for Alpaca environment configs.

    Accepts both custom TimeFrame objects and Alpaca's TimeFrame objects for
    backwards compatibility. Alpaca TimeFrame objects are automatically converted
    to custom TimeFrame objects.

    Args:
        execute_on: Execution timeframe (string, custom TimeFrame, or Alpaca TimeFrame)
        time_frames: List of timeframes or single timeframe
        window_sizes: List of window sizes or single window size

    Returns:
        Tuple of (execute_on_tf, time_frames_list, window_sizes_list) using custom TimeFrame

    Examples:
        >>> # Using strings
        >>> execute_on, tfs, ws = normalize_alpaca_timeframe_config("5Min", ["1Min", "5Min"], 10)

        >>> # Using custom TimeFrame
        >>> from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
        >>> tf = TimeFrame(5, TimeFrameUnit.Minute)
        >>> execute_on, tfs, ws = normalize_alpaca_timeframe_config(tf, [tf], 10)

        >>> # Backwards compatibility: Using Alpaca TimeFrame (deprecated)
        >>> from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame, TimeFrameUnit as AlpacaUnit
        >>> alpaca_tf = AlpacaTimeFrame(5, AlpacaUnit.Minute)
        >>> execute_on, tfs, ws = normalize_alpaca_timeframe_config(alpaca_tf, [alpaca_tf], 10)
    """
    # Convert Alpaca TimeFrame to custom TimeFrame for backwards compatibility
    def convert_if_alpaca(tf: Union[TimeFrame, Any]) -> TimeFrame:
        """Convert Alpaca TimeFrame to custom TimeFrame if needed."""
        # Check if it's an Alpaca TimeFrame by checking for 'amount' attribute
        if hasattr(tf, 'amount') and hasattr(tf, 'unit'):
            warnings.warn(
                "Using Alpaca's TimeFrame class is deprecated. "
                "Please use torchtrade.envs.timeframe.TimeFrame instead.",
                DeprecationWarning,
                stacklevel=3
            )
            return alpaca_to_timeframe(tf)
        return tf

    # Convert execute_on if it's an Alpaca TimeFrame
    if not isinstance(execute_on, str):
        execute_on = convert_if_alpaca(execute_on)

    # Convert time_frames if any are Alpaca TimeFrame objects
    if isinstance(time_frames, list):
        time_frames = [convert_if_alpaca(tf) if not isinstance(tf, str) else tf for tf in time_frames]
    elif not isinstance(time_frames, str):
        time_frames = convert_if_alpaca(time_frames)

    # Use shared normalization with custom parse function
    return normalize_timeframe_config(
        execute_on, time_frames, window_sizes, parse_fn=parse_alpaca_timeframe_string
    )
