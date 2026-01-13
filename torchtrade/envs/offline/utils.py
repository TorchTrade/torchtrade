from enum import Enum
from typing import List, Union, Optional, Dict, Tuple
from itertools import product
import pandas as pd
import numpy as np
import re
import warnings

def get_timeframe_unit(tf_str: "Min"):
    """DEPRECATED: Use parse_timeframe_string() instead.

    This function will be removed in version 2.0.
    """
    warnings.warn(
        "get_timeframe_unit() is deprecated. Use parse_timeframe_string() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if tf_str == "Min" or tf_str == "min" or tf_str == "Minute":
        return TimeFrameUnit.Minute
    elif tf_str == "Hour" or tf_str == "h" or tf_str == "H" or tf_str == "hour":
        return TimeFrameUnit.Hour
    elif tf_str == "Day" or tf_str == "D" or tf_str == "day" or tf_str == "day":
        return TimeFrameUnit.Day
    else:
        raise ValueError(f"Unknown TimeFrameUnit {tf_str}")


def parse_timeframe_string(s: str) -> "TimeFrame":
    """Parse timeframe string to TimeFrame object.

    Supports formats: "5Min", "5min", "5Minute", "5 minutes", "5M", "5H", "5D"

    Args:
        s: Timeframe string (e.g., "5Min", "1Hour", "15Minute")

    Returns:
        TimeFrame object

    Raises:
        ValueError: If string format is invalid or unit is unknown

    Examples:
        >>> parse_timeframe_string("5Min")
        TimeFrame(5, TimeFrameUnit.Minute)
        >>> parse_timeframe_string("1Hour")
        TimeFrame(1, TimeFrameUnit.Hour)
        >>> parse_timeframe_string("15 minutes")
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

    return TimeFrame(value, unit)


def normalize_timeframe_config(
    execute_on: Union[str, "TimeFrame"],
    time_frames: Union[List[Union[str, "TimeFrame"]], Union[str, "TimeFrame"]],
    window_sizes: Union[List[int], int],
    parse_fn=None,
) -> Tuple["TimeFrame", List["TimeFrame"], List[int]]:
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
    # Use duck typing: if it has __iter__ and __len__, treat it as a sequence
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


def compute_periods_per_year_crypto(execute_on_unit: str, execute_on_value: float):
    """
    Compute periods per year for crypto trading (24/7).
    
    execute_on_unit: 'S', 'Min', 'H', 'D'
    execute_on_value: number of units per trade
    """
    minutes_per_year = 365 * 24 * 60  # total minutes in a year

    if execute_on_unit == 'S':
        periods_per_year = minutes_per_year * 60 / execute_on_value
    elif execute_on_unit == 'Min':
        periods_per_year = minutes_per_year / execute_on_value
    elif execute_on_unit == 'H':
        periods_per_year = 365 * 24 / execute_on_value
    elif execute_on_unit == 'D':
        periods_per_year = 365 / execute_on_value
    else:
        raise ValueError(f"Unknown execute_on_unit: {execute_on_unit}")

    return periods_per_year

class TimeFrameUnit(Enum):
    Minute = 'Min'  # Pandas freq for minutes
    Hour = 'H'    # Pandas freq for hours
    Day = 'D'    # Pandas freq for days
    # Add more if needed, e.g., week = 'W'

class TimeFrame:
    def __init__(self, value: int, unit: TimeFrameUnit):
        self.value = value
        self.unit = unit

    def to_pandas_freq(self) -> str:
        return f"{self.value}{self.unit.value}"

    def obs_key_freq(self) -> str:
        if self.unit == TimeFrameUnit.Minute:
            return f"{self.value}Minute"
        elif self.unit == TimeFrameUnit.Hour:
            return f"{self.value}Hour"
        elif self.unit == TimeFrameUnit.Day:
            return f"{self.value}Day"
        else:
            raise ValueError(f"Unknown TimeFrameUnit {self.unit}")


# Correct tf_to_timedelta
def tf_to_timedelta(tf: TimeFrame) -> pd.Timedelta:
    if tf.unit == TimeFrameUnit.Minute:
        return pd.Timedelta(minutes=tf.value)
    elif tf.unit == TimeFrameUnit.Hour:
        return pd.Timedelta(hours=tf.value)
    elif tf.unit == TimeFrameUnit.Day:
        return pd.Timedelta(days=tf.value)
    else:
        raise ValueError(f"Unknown TimeFrameUnit {tf.unit}")


class InitialBalanceSampler:
    """Sampler for initial balance with optional domain randomization.

    Args:
        initial_cash: Fixed amount (int) or range [min, max] (list) for randomization
        seed: Optional random seed for reproducibility
    """
    def __init__(self, initial_cash: Union[List[int], int], seed: Optional[int] = None):
        self.initial_cash = initial_cash
        if seed is not None:
            np.random.seed(seed)

    def sample(self) -> float:
        """Sample an initial balance value.

        Returns:
            Initial balance as float
        """
        if isinstance(self.initial_cash, int):
            return float(self.initial_cash)
        else:
            return float(np.random.randint(self.initial_cash[0], self.initial_cash[1]))


def build_sltp_action_map(
    stoploss_levels: List[float],
    takeprofit_levels: List[float],
    include_hold_action: bool = True,
    include_short_positions: bool = False,
) -> Dict[int, Union[Tuple[Optional[float], Optional[float]], Tuple[Optional[str], Optional[float], Optional[float]]]]:
    """
    Build action map for environments with stop-loss/take-profit.

    Creates a combinatorial action space from SL and TP levels. Supports both long-only
    and futures (long/short) trading environments.

    Args:
        stoploss_levels: List of stop-loss percentages (e.g., [-0.025, -0.05, -0.1])
        takeprofit_levels: List of take-profit percentages (e.g., [0.05, 0.1, 0.2])
        include_hold_action: If True, action 0 is HOLD. If False, starts at index 0 with first position
        include_short_positions: If True, creates actions for both long and short positions (futures).
                                 If False, only creates long positions (long-only)

    Returns:
        Dictionary mapping action index to tuples:

        **Long-only mode (include_short_positions=False):**
        - Returns: (sl, tp) tuples
        - If include_hold_action=True: {0: (None, None), 1: (sl1, tp1), 2: (sl1, tp2), ...}
        - If include_hold_action=False: {0: (sl1, tp1), 1: (sl1, tp2), ...}

        **Futures mode (include_short_positions=True):**
        - Returns: (side, sl, tp) tuples where side is None, "long", or "short"
        - If include_hold_action=True:
            - 0: (None, None, None) - HOLD/Close
            - 1 to N: ("long", sl, tp) - Long positions with all SL/TP combinations
            - N+1 to 2N: ("short", sl, tp) - Short positions with all SL/TP combinations
        - If include_hold_action=False:
            - 0 to N-1: ("long", sl, tp) - Long positions
            - N to 2N-1: ("short", sl, tp) - Short positions

    Example:
        >>> # Long-only with 2 SL levels, 2 TP levels
        >>> build_sltp_action_map([-0.05, -0.1], [0.05, 0.1], include_short_positions=False)
        {0: (None, None), 1: (-0.05, 0.05), 2: (-0.05, 0.1), 3: (-0.1, 0.05), 4: (-0.1, 0.1)}

        >>> # Futures with same levels
        >>> build_sltp_action_map([-0.05, -0.1], [0.05, 0.1], include_short_positions=True)
        {0: (None, None, None), 1: ("long", -0.05, 0.05), ..., 5: ("short", -0.05, 0.05), ...}
    """
    action_map = {}
    idx = 0

    # Add HOLD action if requested
    if include_hold_action:
        if include_short_positions:
            action_map[0] = (None, None, None)
        else:
            action_map[0] = (None, None)
        idx = 1

    # Long positions
    for sl, tp in product(stoploss_levels, takeprofit_levels):
        if include_short_positions:
            action_map[idx] = ("long", sl, tp)
        else:
            action_map[idx] = (sl, tp)
        idx += 1

    # Short positions (only for futures environments)
    if include_short_positions:
        for sl, tp in product(stoploss_levels, takeprofit_levels):
            action_map[idx] = ("short", sl, tp)
            idx += 1

    return action_map