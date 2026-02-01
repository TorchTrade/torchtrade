from typing import List, Union, Optional
import pandas as pd
import numpy as np

# Import TimeFrame infrastructure from shared module
from torchtrade.envs.utils.timeframe import (  # noqa: F401
    TimeFrame,
    TimeFrameUnit,
    parse_timeframe_string,
    normalize_timeframe_config,
    tf_to_timedelta,
)


def load_torch_trade_dataset(
    dataset_name: str = "Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025",
    split: str = "train"
) -> pd.DataFrame:
    """Load TorchTrade dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load (default: "train")

    Returns:
        DataFrame with OHLCV data (columns: timestamp, open, high, low, close, volume)

    Example:
        >>> df = load_torch_trade_dataset()
        >>> df = load_torch_trade_dataset("Torch-Trade/ethusdt_spot_1m", split="test")
    """
    import datasets
    dataset = datasets.load_dataset(dataset_name)
    return dataset[split].to_pandas()


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


class InitialBalanceSampler:
    """Sampler for initial balance with optional domain randomization.

    Args:
        initial_cash: Fixed amount (int) or range [min, max] (list) for randomization
        seed: Optional random seed for reproducibility
    """
    def __init__(self, initial_cash: Union[List[int], int], seed: Optional[int] = None):
        self.initial_cash = initial_cash
        self.np_rng = np.random.default_rng(seed)

    def sample(self) -> float:
        """Sample an initial balance value.

        Returns:
            Initial balance as float
        """
        if isinstance(self.initial_cash, int):
            return float(self.initial_cash)

        return float(self.np_rng.integers(self.initial_cash[0], self.initial_cash[1]))
