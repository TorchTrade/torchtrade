"""Shared TimeFrame infrastructure for all TorchTrade environments.

This module provides a unified TimeFrame abstraction used across offline, Alpaca,
and Binance environments. It includes:
- TimeFrame and TimeFrameUnit classes for representing time intervals
- Parsing and normalization utilities
- Provider-specific conversion functions (Alpaca ↔ custom, Binance ↔ custom)
"""

from enum import Enum
from functools import total_ordering
from typing import List, Union, Optional, Tuple
import pandas as pd
import re
import warnings


class TimeFrameUnit(Enum):
    """Time frame units supported across all environments."""
    Minute = 'Min'  # Pandas freq for minutes
    Hour = 'H'      # Pandas freq for hours
    Day = 'D'       # Pandas freq for days


@total_ordering
class TimeFrame:
    """Represents a time interval with value and unit.

    Provides consistent timeframe representation across all TorchTrade environments
    (offline, Alpaca, Binance). Supports comparison, hashing, and conversion to
    provider-specific formats.

    Args:
        value: Numeric value of the timeframe (e.g., 5 for "5 minutes")
        unit: TimeFrameUnit indicating the time unit

    Examples:
        >>> tf = TimeFrame(5, TimeFrameUnit.Minute)
        >>> tf.to_pandas_freq()
        '5Min'
        >>> tf.obs_key_freq()
        '5Minute'
        >>> tf.to_minutes()
        5.0
    """

    def __init__(self, value: int, unit: TimeFrameUnit):
        self.value = value
        self.unit = unit

    def to_pandas_freq(self) -> str:
        """Convert to pandas frequency string (e.g., '5Min', '1H', '1D')."""
        return f"{self.value}{self.unit.value}"

    def obs_key_freq(self) -> str:
        """Convert to observation key format (e.g., '5Minute', '1Hour', '1Day')."""
        if self.unit == TimeFrameUnit.Minute:
            return f"{self.value}Minute"
        elif self.unit == TimeFrameUnit.Hour:
            return f"{self.value}Hour"
        elif self.unit == TimeFrameUnit.Day:
            return f"{self.value}Day"
        else:
            raise ValueError(f"Unknown TimeFrameUnit {self.unit}")

    def __eq__(self, other):
        """Compare TimeFrames by value and unit."""
        if not isinstance(other, TimeFrame):
            return NotImplemented
        return self.value == other.value and self.unit == other.unit

    def __lt__(self, other):
        """Less than comparison based on total minutes."""
        if not isinstance(other, TimeFrame):
            return NotImplemented
        return self.to_minutes() < other.to_minutes()

    def __hash__(self):
        """Make TimeFrame hashable for use in sets and as dict keys."""
        return hash((self.value, self.unit))

    def __repr__(self):
        """String representation for debugging."""
        return f"TimeFrame({self.value}, {self.unit})"

    def to_minutes(self) -> float:
        """Convert timeframe to total minutes for comparison."""
        if self.unit == TimeFrameUnit.Minute:
            return float(self.value)
        elif self.unit == TimeFrameUnit.Hour:
            return float(self.value * 60)
        elif self.unit == TimeFrameUnit.Day:
            return float(self.value * 60 * 24)
        else:
            raise ValueError(f"Unknown TimeFrameUnit {self.unit}")


def timeframe_to_seconds(tf: TimeFrame) -> int:
    """Convert TimeFrame to seconds for timing purposes.

    Args:
        tf: TimeFrame object

    Returns:
        Number of seconds in the timeframe

    Examples:
        >>> tf = TimeFrame(1, TimeFrameUnit.Minute)
        >>> timeframe_to_seconds(tf)
        60
        >>> tf = TimeFrame(1, TimeFrameUnit.Hour)
        >>> timeframe_to_seconds(tf)
        3600
        >>> tf = TimeFrame(1, TimeFrameUnit.Day)
        >>> timeframe_to_seconds(tf)
        86400
    """
    if tf.unit == TimeFrameUnit.Minute:
        return tf.value * 60
    elif tf.unit == TimeFrameUnit.Hour:
        return tf.value * 3600
    elif tf.unit == TimeFrameUnit.Day:
        return tf.value * 86400
    else:
        raise ValueError(f"Unsupported TimeFrameUnit: {tf.unit}")


def parse_timeframe_string(s: str) -> TimeFrame:
    """Parse timeframe string to TimeFrame object.

    Supports formats: "5Min", "5min", "5Minute", "5 minutes", "5M", "5H", "5D"

    **Important**: Non-canonical forms like "60min" create different observation keys
    than canonical forms like "1hour". A warning will be issued if a timeframe can be
    normalized to a cleaner form (e.g., 60min → 1hour, 120min → 2hour, 24hour → 1day).

    Args:
        s: Timeframe string (e.g., "5Min", "1Hour", "15Minute")

    Returns:
        TimeFrame object

    Raises:
        ValueError: If string format is invalid or unit is unknown

    Warns:
        UserWarning: If timeframe can be normalized to a canonical form

    Examples:
        >>> parse_timeframe_string("5Min")
        TimeFrame(5, TimeFrameUnit.Minute)
        >>> parse_timeframe_string("1Hour")
        TimeFrame(1, TimeFrameUnit.Hour)
        >>> parse_timeframe_string("15 minutes")
        TimeFrame(15, TimeFrameUnit.Minute)
        >>> parse_timeframe_string("60min")  # Issues warning, suggests "1hour"
        TimeFrame(60, TimeFrameUnit.Minute)
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

    # Map unit variations to TimeFrameUnit using dictionary for clarity
    unit_map = {
        'min': TimeFrameUnit.Minute,
        'minute': TimeFrameUnit.Minute,
        'minutes': TimeFrameUnit.Minute,
        'm': TimeFrameUnit.Minute,
        'hour': TimeFrameUnit.Hour,
        'hours': TimeFrameUnit.Hour,
        'h': TimeFrameUnit.Hour,
        'hr': TimeFrameUnit.Hour,
        'day': TimeFrameUnit.Day,
        'days': TimeFrameUnit.Day,
        'd': TimeFrameUnit.Day,
    }

    unit = unit_map.get(unit_str)
    if unit is None:
        raise ValueError(
            f"Unknown time unit: '{unit_str}'. "
            f"Supported units: Min/Minute/Minutes/M, Hour/Hours/H, Day/Days/D"
        )

    # Warn about non-canonical timeframe formats
    # Check if minutes could be normalized to hours or days
    if unit == TimeFrameUnit.Minute:
        if value >= 1440 and value % 1440 == 0:
            # Can be normalized to days
            days = value // 1440
            canonical = f"{days}day" if days == 1 else f"{days}days"
            warnings.warn(
                f"Timeframe '{s}' creates observation key '{value}Minute'. "
                f"Consider using '{canonical}' for a cleaner key '{days}Day'. "
                f"Note: These are treated as DIFFERENT timeframes - models trained with one won't work with the other.",
                UserWarning,
                stacklevel=2
            )
        elif value >= 60 and value % 60 == 0:
            # Can be normalized to hours
            hours = value // 60
            canonical = f"{hours}hour" if hours == 1 else f"{hours}hours"
            warnings.warn(
                f"Timeframe '{s}' creates observation key '{value}Minute'. "
                f"Consider using '{canonical}' for a cleaner key '{hours}Hour'. "
                f"Note: These are treated as DIFFERENT timeframes - models trained with one won't work with the other.",
                UserWarning,
                stacklevel=2
            )
    # Check if hours could be normalized to days
    elif unit == TimeFrameUnit.Hour:
        if value >= 24 and value % 24 == 0:
            days = value // 24
            canonical = f"{days}day" if days == 1 else f"{days}days"
            warnings.warn(
                f"Timeframe '{s}' creates observation key '{value}Hour'. "
                f"Consider using '{canonical}' for a cleaner key '{days}Day'. "
                f"Note: These are treated as DIFFERENT timeframes - models trained with one won't work with the other.",
                UserWarning,
                stacklevel=2
            )

    return TimeFrame(value, unit)


def create_provider_parser(provider_converter_fn):
    """Factory function to create provider-specific timeframe parsers.

    Creates a parser that tries provider-specific format first, then falls back
    to standard format parsing. This eliminates duplicate parsing logic across
    providers.

    Args:
        provider_converter_fn: Provider-specific converter function that converts
                              provider interval strings to TimeFrame objects.
                              Should raise ValueError if format is invalid.

    Returns:
        Parser function that accepts a string and returns a TimeFrame object.

    Examples:
        >>> # Create a Binance-specific parser
        >>> parse_binance = create_provider_parser(binance_to_timeframe)
        >>> parse_binance("5m")  # Binance format
        TimeFrame(5, TimeFrameUnit.Minute)
        >>> parse_binance("5Min")  # Standard format
        TimeFrame(5, TimeFrameUnit.Minute)
    """
    def parse_provider_timeframe_string(s: str) -> TimeFrame:
        """Parse timeframe string (provider-specific or standard format)."""
        try:
            # Try provider-specific format first
            return provider_converter_fn(s)
        except ValueError:
            # Fall back to standard format
            return parse_timeframe_string(s)

    return parse_provider_timeframe_string


def normalize_timeframe_config(
    execute_on: Union[str, TimeFrame],
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]],
    window_sizes: Union[List[int], int],
    parse_fn=None,
) -> Tuple[TimeFrame, List[TimeFrame], List[int]]:
    """Normalize timeframe configuration for environment configs.

    Converts string timeframes to TimeFrame objects, normalizes to lists,
    and validates that window_sizes matches time_frames length.

    Args:
        execute_on: Execution timeframe (string or TimeFrame)
        time_frames: List of timeframes or single timeframe (strings or TimeFrame objects)
        window_sizes: List of window sizes or single window size
        parse_fn: Function to parse timeframe strings (defaults to parse_timeframe_string)

    Returns:
        Tuple of (execute_on_tf, time_frames_list, window_sizes_list)

    Raises:
        ValueError: If window_sizes length doesn't match time_frames length

    Examples:
        >>> execute_on, tfs, ws = normalize_timeframe_config("5Min", ["1Min", "5Min"], 10)
        >>> execute_on.to_pandas_freq()
        '5Min'
        >>> len(tfs)
        2
        >>> ws
        [10, 10]
    """
    if parse_fn is None:
        parse_fn = parse_timeframe_string

    # Convert execute_on string to TimeFrame
    if isinstance(execute_on, str):
        execute_on = parse_fn(execute_on)

    # Normalize time_frames to list
    # Handle both regular lists, tuples, and Hydra ListConfig
    # Duck typing is needed because Hydra's OmegaConf.ListConfig is not a standard list type,
    # but implements __iter__ and __len__. This approach works with any sequence-like object.
    try:
        # Try to iterate and check if it's list-like
        if hasattr(time_frames, '__iter__') and hasattr(time_frames, '__len__') and not isinstance(time_frames, (str, TimeFrame)):
            # It's list-like (list, tuple, ListConfig, etc.) - convert to regular list
            time_frames = list(time_frames)
        else:
            # It's a single element
            time_frames = [time_frames]
    except TypeError:
        # Not iterable, wrap in list
        time_frames = [time_frames]

    # Convert all string timeframes to TimeFrame objects
    time_frames = [
        parse_fn(tf) if isinstance(tf, str) else tf
        for tf in time_frames
    ]

    # Normalize window_sizes to list
    if isinstance(window_sizes, int):
        window_sizes = [window_sizes] * len(time_frames)
    else:
        # Convert to regular list (handles ListConfig, tuples, etc.)
        window_sizes = list(window_sizes)

    # Validate lengths match
    if len(window_sizes) != len(time_frames):
        raise ValueError(
            f"window_sizes length ({len(window_sizes)}) must match "
            f"time_frames length ({len(time_frames)})"
        )

    return execute_on, time_frames, window_sizes


def tf_to_timedelta(tf: TimeFrame) -> pd.Timedelta:
    """Convert TimeFrame to pandas Timedelta.

    Args:
        tf: TimeFrame object to convert

    Returns:
        pd.Timedelta object representing the same duration

    Examples:
        >>> tf = TimeFrame(5, TimeFrameUnit.Minute)
        >>> tf_to_timedelta(tf)
        Timedelta('0 days 00:05:00')
    """
    if tf.unit == TimeFrameUnit.Minute:
        return pd.Timedelta(minutes=tf.value)
    elif tf.unit == TimeFrameUnit.Hour:
        return pd.Timedelta(hours=tf.value)
    elif tf.unit == TimeFrameUnit.Day:
        return pd.Timedelta(days=tf.value)
    else:
        raise ValueError(f"Unknown TimeFrameUnit {tf.unit}")


# ============================================================================
# Provider-Specific Conversion Functions
# ============================================================================

def timeframe_to_alpaca(tf: TimeFrame):
    """Convert custom TimeFrame to Alpaca TimeFrame.

    Args:
        tf: Custom TimeFrame object

    Returns:
        Alpaca TimeFrame object

    Examples:
        >>> from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
        >>> tf = TimeFrame(5, TimeFrameUnit.Minute)
        >>> alpaca_tf = timeframe_to_alpaca(tf)
        >>> alpaca_tf.amount
        5
    """
    from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame, TimeFrameUnit as AlpacaUnit

    unit_map = {
        TimeFrameUnit.Minute: AlpacaUnit.Minute,
        TimeFrameUnit.Hour: AlpacaUnit.Hour,
        TimeFrameUnit.Day: AlpacaUnit.Day,
    }

    alpaca_unit = unit_map.get(tf.unit)
    if alpaca_unit is None:
        raise ValueError(f"Unsupported TimeFrameUnit for Alpaca: {tf.unit}")

    return AlpacaTimeFrame(tf.value, alpaca_unit)


def alpaca_to_timeframe(alpaca_tf) -> TimeFrame:
    """Convert Alpaca TimeFrame to custom TimeFrame.

    Provides backwards compatibility for code using Alpaca's TimeFrame objects.

    Args:
        alpaca_tf: Alpaca TimeFrame object

    Returns:
        Custom TimeFrame object

    Examples:
        >>> from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame, TimeFrameUnit as AlpacaUnit
        >>> alpaca_tf = AlpacaTimeFrame(5, AlpacaUnit.Minute)
        >>> tf = alpaca_to_timeframe(alpaca_tf)
        >>> tf.value
        5
    """
    from alpaca.data.timeframe import TimeFrameUnit as AlpacaUnit

    unit_map = {
        AlpacaUnit.Minute: TimeFrameUnit.Minute,
        AlpacaUnit.Hour: TimeFrameUnit.Hour,
        AlpacaUnit.Day: TimeFrameUnit.Day,
    }

    custom_unit = unit_map.get(alpaca_tf.unit)
    if custom_unit is None:
        raise ValueError(f"Unsupported Alpaca TimeFrameUnit: {alpaca_tf.unit}")

    return TimeFrame(alpaca_tf.amount, custom_unit)


def timeframe_to_binance(tf: TimeFrame) -> str:
    """Convert custom TimeFrame to Binance interval string.

    Args:
        tf: Custom TimeFrame object

    Returns:
        Binance interval string (e.g., "1m", "5m", "1h", "1d")

    Examples:
        >>> tf = TimeFrame(5, TimeFrameUnit.Minute)
        >>> timeframe_to_binance(tf)
        '5m'
        >>> tf = TimeFrame(1, TimeFrameUnit.Hour)
        >>> timeframe_to_binance(tf)
        '1h'
    """
    unit_map = {
        TimeFrameUnit.Minute: "m",
        TimeFrameUnit.Hour: "h",
        TimeFrameUnit.Day: "d",
    }

    binance_unit = unit_map.get(tf.unit)
    if binance_unit is None:
        raise ValueError(f"Unsupported TimeFrameUnit for Binance: {tf.unit}")

    return f"{tf.value}{binance_unit}"


def binance_to_timeframe(interval: str) -> TimeFrame:
    """Convert Binance interval string to custom TimeFrame.

    Provides backwards compatibility for code using Binance interval strings.

    Args:
        interval: Binance interval string (e.g., "1m", "5m", "1h", "1d")

    Returns:
        Custom TimeFrame object

    Raises:
        ValueError: If interval format is invalid

    Examples:
        >>> tf = binance_to_timeframe("5m")
        >>> tf.value
        5
        >>> tf.unit
        <TimeFrameUnit.Minute: 'Min'>
        >>> binance_to_timeframe("1h")
        TimeFrame(1, TimeFrameUnit.Hour)
    """
    match = re.match(r'^(\d+)([mhd])$', interval)
    if not match:
        raise ValueError(
            f"Invalid Binance interval: '{interval}'. "
            f"Expected format: '<number><unit>' where unit is m/h/d (e.g., '1m', '5m', '1h')"
        )

    value = int(match.group(1))
    unit_char = match.group(2)

    unit_map = {
        'm': TimeFrameUnit.Minute,
        'h': TimeFrameUnit.Hour,
        'd': TimeFrameUnit.Day,
    }

    return TimeFrame(value, unit_map[unit_char])
