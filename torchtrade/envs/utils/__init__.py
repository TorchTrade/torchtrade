"""Utility modules for TorchTrade environments."""

from torchtrade.envs.utils.timeframe import (
    TimeFrame,
    TimeFrameUnit,
    parse_timeframe_string,
    create_provider_parser,
    normalize_timeframe_config,
    tf_to_timedelta,
    timeframe_to_seconds,
    timeframe_to_alpaca,
    alpaca_to_timeframe,
    timeframe_to_binance,
    binance_to_timeframe,
)
from torchtrade.envs.utils.action_maps import (
    create_sltp_action_map,
    create_alpaca_sltp_action_map,
)
from torchtrade.envs.utils.sltp_helpers import (
    calculate_bracket_prices,
    calculate_long_bracket_prices,
    calculate_short_bracket_prices,
)
from torchtrade.envs.utils.sltp_mixin import SLTPMixin
from torchtrade.envs.utils.fractional_sizing import (
    PositionCalculationParams,
    calculate_fractional_position,
    build_default_action_levels,
)
from torchtrade.envs.utils.metrics import compute_sharpe_torch

__all__ = [
    # Timeframe utilities
    "TimeFrame",
    "TimeFrameUnit",
    "parse_timeframe_string",
    "create_provider_parser",
    "normalize_timeframe_config",
    "tf_to_timedelta",
    "timeframe_to_seconds",
    "timeframe_to_alpaca",
    "alpaca_to_timeframe",
    "timeframe_to_binance",
    "binance_to_timeframe",
    # Action maps
    "create_sltp_action_map",
    "create_alpaca_sltp_action_map",
    # SL/TP helpers
    "calculate_bracket_prices",
    "calculate_long_bracket_prices",
    "calculate_short_bracket_prices",
    "SLTPMixin",
    # Position sizing
    "PositionCalculationParams",
    "calculate_fractional_position",
    "build_default_action_levels",
    # Metrics
    "compute_sharpe_torch",
]
