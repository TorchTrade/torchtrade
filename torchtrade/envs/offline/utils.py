from enum import Enum
from typing import List
import pandas as pd

def get_timeframe_unit(tf_str: "Min"):
    if tf_str == "Min" or tf_str == "min" or tf_str == "Minute":
        return TimeFrameUnit.Minute
    elif tf_str == "Hour" or tf_str == "h" or tf_str == "H" or tf_str == "hour":
        return TimeFrameUnit.Hour
    elif tf_str == "Day" or tf_str == "D" or tf_str == "day" or tf_str == "day":
        return TimeFrameUnit.Day
    else:
        raise ValueError(f"Unknown TimeFrameUnit {tf_str}")

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