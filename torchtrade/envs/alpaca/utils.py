"""Utility functions for Alpaca environments."""
import re
from typing import Union
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def parse_alpaca_timeframe_string(s: str) -> TimeFrame:
    """Parse timeframe string to Alpaca TimeFrame object.

    Supports formats: "5Min", "5min", "5Minute", "5 minutes", "5M", "5H", "5D"

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
        >>> parse_alpaca_timeframe_string("15 minutes")
        TimeFrame(15, TimeFrameUnit.Minute)
    """
    s = s.strip()

    # Pattern: <number><optional space><unit>
    pattern = r'^(\d+)\s*([a-zA-Z]+)$'
    match = re.match(pattern, s)

    if not match:
        raise ValueError(
            f"Invalid timeframe format: '{s}'. "
            f"Expected format: '<number><unit>' (e.g., '5Min', '1Hour', '15Minute')"
        )

    value = int(match.group(1))
    unit_str = match.group(2).lower()

    # Map unit variations to Alpaca TimeFrameUnit
    if unit_str in ('min', 'minute', 'minutes', 'm'):
        unit = TimeFrameUnit.Minute
    elif unit_str in ('hour', 'hours', 'h', 'hr'):
        unit = TimeFrameUnit.Hour
    elif unit_str in ('day', 'days', 'd'):
        unit = TimeFrameUnit.Day
    else:
        raise ValueError(
            f"Unknown time unit: '{unit_str}'. "
            f"Supported units: Min/Minute/Minutes/M, Hour/Hours/H, Day/Days/D"
        )

    return TimeFrame(value, unit)
