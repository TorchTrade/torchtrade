"""The SLTP config fields every futures exchange declares identically (#288)."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from torchtrade.envs.core.common import (
    TradeMode,
    validate_position_sizing,
    validate_trade_mode,
)
from torchtrade.envs.core.live import ObservationFailurePolicy
from torchtrade.envs.utils.timeframe import TimeFrame


@dataclass
class BaseFuturesSLTPConfig:
    """Fields the four venue SLTP configs declared identically, name, type and default.

    Each venue keeps only its margin surface, and overrides `symbol`'s default. `symbol`
    itself lives here so it stays parameter #1, as it was before the extraction.
    """

    symbol: str = "BTCUSDT"
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, TimeFrame] = "1Hour"  # timeframe that gates trade execution
    leverage: int = 1  # 1-125
    quantity_per_trade: float = 0.001  # used when trade_mode is "quantity"/"notional"
    trade_mode: TradeMode = "quantity"
    position_fraction: float = 1.0  # used when trade_mode="fractional"
    lock_position_until_sltp: bool = False  # if True, actions are ignored while in position
    stoploss_levels: Tuple[float, ...] = (-0.025, -0.05, -0.1)
    takeprofit_levels: Tuple[float, ...] = (0.05, 0.1, 0.2)
    include_short_positions: bool = True
    include_hold_action: bool = True
    include_close_action: bool = False
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1  # fraction of INITIAL balance, not current
    demo: bool = True
    seed: Optional[int] = 42
    include_base_features: bool = False
    close_position_on_init: bool = True
    close_position_on_reset: bool = False
    observation_failure_policy: Union[ObservationFailurePolicy, str] = (
        ObservationFailurePolicy.HALT
    )

    # Each subclass sets this to its venue normalizer -- all four are a partial of the
    # same normalize_timeframe_config, differing only in parse_fn.
    _normalize_timeframes = None

    def __post_init__(self):
        """Validate at the boundary, so nothing downstream has to guard."""
        self.observation_failure_policy = ObservationFailurePolicy(
            self.observation_failure_policy
        )
        self.trade_mode = validate_trade_mode(self.trade_mode)
        validate_position_sizing(
            self.trade_mode, self.position_fraction, self.quantity_per_trade
        )
        self.execute_on, self.time_frames, self.window_sizes = self._normalize_timeframes(
            self.execute_on, self.time_frames, self.window_sizes
        )
