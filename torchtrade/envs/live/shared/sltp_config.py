"""The SLTP config fields every futures exchange declares identically (#288)."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from torchtrade.envs.core.common import TradeMode
from torchtrade.envs.core.live import ObservationFailurePolicy


@dataclass
class BaseFuturesSLTPConfig:
    """19 fields the four SLTP configs declared identically, name, type AND default.

    Each exchange still owns what genuinely differs: `symbol`, `time_frames` and
    `execute_on` differ only in DEFAULT (venue timeframe spellings), and the margin
    surface differs in NAME -- binance calls it `margin_type: MarginType`, the other
    three `margin_mode: MarginMode`, and bitget adds `product_type`.

    That naming split is #289's, not this extraction's: whichever name wins changes one
    venue's public API.
    """

    window_sizes: Union[List[int], int] = 10
    leverage: int = 1
    quantity_per_trade: float = 0.001
    trade_mode: TradeMode = "quantity"
    position_fraction: float = 1.0
    lock_position_until_sltp: bool = False
    stoploss_levels: Tuple[float, ...] = (-0.025, -0.05, -0.1)
    takeprofit_levels: Tuple[float, ...] = (0.05, 0.1, 0.2)
    include_short_positions: bool = True
    include_hold_action: bool = True
    include_close_action: bool = False
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1
    demo: bool = True
    seed: Optional[int] = 42
    include_base_features: bool = False
    close_position_on_init: bool = True
    close_position_on_reset: bool = False
    observation_failure_policy: Union[ObservationFailurePolicy, str] = (
        ObservationFailurePolicy.HALT
    )

    def __post_init__(self):
        """Coerce the failure policy. Identical in all four configs (#288).

        A string from a yaml must become the enum before anything compares against it --
        `'halt' == ObservationFailurePolicy.HALT` is False, so a config loaded from hydra
        would silently take the wrong branch. Subclasses that override __post_init__ for
        their own timeframe normalisation MUST call super().
        """
        self.observation_failure_policy = ObservationFailurePolicy(
            self.observation_failure_policy
        )
