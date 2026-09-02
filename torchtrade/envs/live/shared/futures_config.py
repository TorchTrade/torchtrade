"""The plain-futures config fields every exchange declares identically (#288).

The SLTP twin of this file is `sltp_config.py`, and it landed first. This one is the same
extraction for the non-SLTP configs, which the four venues had still been repeating.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

from torchtrade.envs.core.common import validate_unknown_status_budget
from torchtrade.envs.core.live import ObservationFailurePolicy
from torchtrade.envs.utils.fractional_sizing import (
    build_default_action_levels,
    validate_action_levels,
)
from torchtrade.envs.utils.timeframe import TimeFrame


@dataclass
class BaseFuturesTradingConfig:
    """Fields the four venue configs declared identically, name, type and default.

    Each venue keeps only its margin surface -- `margin_mode` is a different Enum
    class per venue -- binance uses the shared `core.common_types.MarginMode`, the other
    three declare their own carrying the venue's wire value -- so there is no single type
    to hoist. `symbol` does
    live here so it stays parameter #1; okx overrides its default.
    """

    symbol: str = "BTCUSDT"
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, TimeFrame] = "1Hour"  # timeframe that gates trade execution
    leverage: int = 1  # 1-125

    # `None` means the default below. The list's LENGTH is the Categorical's n, and a
    # checkpoint is bound to the length it trained on, so changing this needs a retrain.
    # `validate_action_levels` checks range [-1, 1], distinctness and len >= 2 -- NOT
    # ordering. e.g. [-1.0, -0.5, 0.0, 0.5, 1.0] for half-size steps.
    action_levels: Optional[List[float]] = None

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
    # Bars to ride out an unreadable venue before truncating; 0 disables (#295).
    max_unknown_status_steps: int = 0

    # Each subclass sets this to its venue normalizer -- all four are a partial of the
    # same normalize_timeframe_config, differing only in parse_fn.
    _normalize_timeframes = None

    def __post_init__(self):
        """Validate at the boundary, so nothing downstream has to guard."""
        self.observation_failure_policy = ObservationFailurePolicy(
            self.observation_failure_policy
        )
        validate_unknown_status_budget(self.max_unknown_status_steps)
        self.execute_on, self.time_frames, self.window_sizes = self._normalize_timeframes(
            self.execute_on, self.time_frames, self.window_sizes
        )

        # One default for all four. binance derived it from this helper while the other
        # three hard-coded a five-level list, so the SAME unset config produced a
        # 3-action space on one venue and a 5-action space on the other three -- and
        # `action_spec.n` is what a checkpoint is bound to, so a policy could not move
        # between venues it was meant to be portable across (#288).
        if self.action_levels is None:
            self.action_levels = build_default_action_levels(allow_short=True)

        validate_action_levels(self.action_levels)
