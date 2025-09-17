from enum import Enum
from typing import List

class TimeFrameUnit(Enum):
    Minute = 'min'  # Pandas freq for minutes
    Hour = 'H'    # Pandas freq for hours
    # Add more if needed, e.g., Day = 'D'

class TimeFrame:
    def __init__(self, value: int, unit: TimeFrameUnit):
        self.value = value
        self.unit = unit

    def to_pandas_freq(self) -> str:
        return f"{self.value}{self.unit.value}"