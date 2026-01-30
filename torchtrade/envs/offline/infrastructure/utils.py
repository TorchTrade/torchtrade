from typing import List, Union, Optional, Dict, Tuple
from itertools import product
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


def build_sltp_action_map(
    stoploss_levels: List[float],
    takeprofit_levels: List[float],
    include_hold_action: bool = True,
    include_close_action: bool = True,
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
        include_close_action: If True, add CLOSE action to exit positions (default: True)
        include_short_positions: If True, creates actions for both long and short positions (futures).
                                 If False, only creates long positions (long-only)

    Returns:
        Dictionary mapping action index to tuples:

        **Long-only mode (include_short_positions=False):**
        - Returns: (sl, tp) tuples
        - If include_hold_action=True and include_close_action=True:
            {0: (None, None) - HOLD, 1: (None, None) - CLOSE, 2: (sl1, tp1), 3: (sl1, tp2), ...}
        - If include_hold_action=False: {0: (sl1, tp1), 1: (sl1, tp2), ...}

        **Futures mode (include_short_positions=True):**
        - Returns: (side, sl, tp) tuples where side is None, "close", "long", or "short"
        - If include_hold_action=True and include_close_action=True:
            - 0: (None, None, None) - HOLD
            - 1: ("close", None, None) - CLOSE
            - 2 to N+1: ("long", sl, tp) - Long positions with all SL/TP combinations
            - N+2 to 2N+1: ("short", sl, tp) - Short positions with all SL/TP combinations
        - If include_hold_action=False:
            - 0 to N-1: ("long", sl, tp) - Long positions
            - N to 2N-1: ("short", sl, tp) - Short positions

    Example:
        >>> # Long-only with 2 SL levels, 2 TP levels
        >>> build_sltp_action_map([-0.05, -0.1], [0.05, 0.1], include_short_positions=False)
        {0: (None, None), 1: (None, None), 2: (-0.05, 0.05), 3: (-0.05, 0.1), 4: (-0.1, 0.05), 5: (-0.1, 0.1)}

        >>> # Futures with same levels
        >>> build_sltp_action_map([-0.05, -0.1], [0.05, 0.1], include_short_positions=True)
        {0: (None, None, None), 1: ("close", None, None), 2: ("long", -0.05, 0.05), ..., 6: ("short", -0.05, 0.05), ...}
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

    # Add CLOSE action if requested
    if include_close_action:
        if include_short_positions:
            action_map[idx] = ("close", None, None)
        else:
            action_map[idx] = ("close", None)  # Long-only CLOSE marker
        idx += 1

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