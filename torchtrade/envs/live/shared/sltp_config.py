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
    """Fields the four SLTP configs declared identically, name, type AND default.

    Each exchange still owns what genuinely differs: `symbol`, and the margin surface,
    which differs in NAME -- binance calls it `margin_type: MarginType`, the other three
    `margin_mode: MarginMode`, and bitget adds `product_type`. That naming split is
    #289's call, since whichever name wins changes one venue's public API.
    """

    symbol: str = "BTCUSDT"
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, TimeFrame] = "1Hour"  # timeframe that gates trade execution
    leverage: int = 1  # max is venue-specific (binance 125, others differ)
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

    # Each subclass sets this to its venue normalizer. All four are a partial of the SAME
    # normalize_timeframe_config differing only in parse_fn, so the venue difference is
    # one callable. staticmethod so pointing it at a plain function would not bind it.
    _normalize_timeframes = None

    def __post_init__(self):
        """Validate at the boundary, so nothing downstream has to guard.

        The failure-policy coercion is here to REJECT an invalid string at construction.
        It is not needed for comparisons -- ObservationFailurePolicy subclasses str, so
        `"halt" == ObservationFailurePolicy.HALT` is already True.
        """
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
