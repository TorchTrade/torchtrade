"""Utility functions for Binance environments."""
from functools import partial
from typing import List, Union, Tuple

from torchtrade.envs.timeframe import (
    TimeFrame,
    TimeFrameUnit,
    normalize_timeframe_config,
    binance_to_timeframe,
    create_provider_parser,
)


# Create Binance-specific parser using factory (tries Binance format first, then standard format)
parse_binance_timeframe_string = create_provider_parser(binance_to_timeframe)

# Convenience wrapper: normalize_timeframe_config with Binance-specific parsing
# Accepts Binance interval strings ("1m", "5m"), standard strings ("1Min"), or TimeFrame objects
normalize_binance_timeframe_config = partial(
    normalize_timeframe_config,
    parse_fn=parse_binance_timeframe_string
)
