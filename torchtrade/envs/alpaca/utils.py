"""Utility functions for Alpaca environments."""
from typing import List, Union, Tuple
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from torchtrade.envs.offline.utils import parse_timeframe_string as _parse_offline_tf


def parse_alpaca_timeframe_string(s: str) -> TimeFrame:
    """Parse timeframe string to Alpaca TimeFrame object.

    Uses shared parsing logic from offline utils and converts to Alpaca's TimeFrame.

    Args:
        s: Timeframe string (e.g., "5Min", "1Hour", "15Minute")

    Returns:
        Alpaca TimeFrame object

    Raises:
        ValueError: If string format is invalid or unit is unknown

    Examples:
        >>> parse_alpaca_timeframe_string("5Min")
        TimeFrame(5, TimeFrameUnit.Minute)
        >>> parse_alpaca_timeframe_string("1Hour")
        TimeFrame(1, TimeFrameUnit.Hour)
    """
    # Parse using shared offline logic
    offline_tf = _parse_offline_tf(s)

    # Map offline TimeFrameUnit enum values to Alpaca TimeFrameUnit
    alpaca_unit_map = {
        "Min": TimeFrameUnit.Minute,
        "H": TimeFrameUnit.Hour,
        "D": TimeFrameUnit.Day,
    }

    # Convert to Alpaca TimeFrame
    alpaca_unit = alpaca_unit_map[offline_tf.unit.value]
    return TimeFrame(offline_tf.value, alpaca_unit)


def normalize_alpaca_timeframe_config(
    execute_on: Union[str, TimeFrame],
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]],
    window_sizes: Union[List[int], int],
) -> Tuple[TimeFrame, List[TimeFrame], List[int]]:
    """Normalize timeframe configuration for Alpaca environment configs.

    Wrapper around normalize_timeframe_config that uses Alpaca's TimeFrame class.

    Args:
        execute_on: Execution timeframe (string or Alpaca TimeFrame)
        time_frames: List of timeframes or single timeframe
        window_sizes: List of window sizes or single window size

    Returns:
        Tuple of (execute_on_tf, time_frames_list, window_sizes_list)
    """
    from torchtrade.envs.offline.utils import normalize_timeframe_config

    return normalize_timeframe_config(
        execute_on, time_frames, window_sizes, parse_fn=parse_alpaca_timeframe_string
    )
