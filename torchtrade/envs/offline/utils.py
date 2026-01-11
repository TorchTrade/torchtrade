from enum import Enum
from typing import List, Union, Optional
import pandas as pd
import numpy as np

def get_timeframe_unit(tf_str: "Min"):
    if tf_str == "Min" or tf_str == "min" or tf_str == "Minute":
        return TimeFrameUnit.Minute
    elif tf_str == "Hour" or tf_str == "h" or tf_str == "H" or tf_str == "hour":
        return TimeFrameUnit.Hour
    elif tf_str == "Day" or tf_str == "D" or tf_str == "day" or tf_str == "day":
        return TimeFrameUnit.Day
    else:
        raise ValueError(f"Unknown TimeFrameUnit {tf_str}")


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