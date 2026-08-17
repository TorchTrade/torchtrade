# Utility Modules

Shared utility functions and helpers used across all TorchTrade environments.

## Modules

### `timeframe.py`

Time period management and provider-specific conversions.

**Key Classes:**
- `TimeFrame`: Represents a time period (e.g., "1 day", "5 minutes")
- `TimeFrameUnit`: Enum of time units (SECOND, MINUTE, HOUR, DAY, WEEK, MONTH)

**Functions:**
- `parse_timeframe_string()`: Parse strings like "1d", "5m" into TimeFrame objects
- `normalize_timeframe_config()`: Validate and normalize timeframe configurations
- `tf_to_timedelta()`: Convert TimeFrame to Python timedelta
- `timeframe_to_seconds()`: Convert TimeFrame to seconds
- `timeframe_to_alpaca()` / `alpaca_to_timeframe()`: Alpaca API conversions
- `timeframe_to_binance()` / `binance_to_timeframe()`: Binance API conversions

**Example:**
```python
from torchtrade.envs.utils import TimeFrame, TimeFrameUnit
from torchtrade.envs.utils.timeframe import (
    parse_timeframe_string,
    timeframe_to_alpaca,
    timeframe_to_binance,
)

# Create timeframe
tf = TimeFrame(5, TimeFrameUnit.Minute)

# Parse from string
tf = parse_timeframe_string("1d")

# Convert to provider format
alpaca_tf = timeframe_to_alpaca(tf)  # "1Day"
binance_tf = timeframe_to_binance(tf)  # "1d"
```

### `action_maps.py`

Discrete action space mappings for different trading strategies.

**Functions:**
- `create_alpaca_sltp_action_map(stoploss_levels, takeprofit_levels)`: long-only bracket
  map of `(sl_pct, tp_pct)`
- `create_sltp_action_map(stoploss_levels, takeprofit_levels)`: bracket map of
  `(side, sl_pct, tp_pct)` for envs that can short

Neither is a BUY/SELL/HOLD map. Both are `1 + len(sl) * len(tp)` entries, index 0 being
the flat/no-bracket action.

**Example:**
```python
from torchtrade.envs.utils.action_maps import create_alpaca_sltp_action_map

action_map = create_alpaca_sltp_action_map([0.02], [0.05])
# {0: (None, None), 1: (0.02, 0.05)}
#   index 0 -> stay flat; index 1 -> enter with a 2% stop and a 5% target

sl_pct, tp_pct = action_map[1]
```

### `sltp_helpers.py`

Stop-loss and take-profit calculation utilities.

**Functions:**
- `calculate_bracket_prices()`: SL/TP price levels for a bracket order

**Example:**
```python
from torchtrade.envs.utils.sltp_helpers import calculate_bracket_prices, stop_fill_price

# sl_pct is SIGNED and its sign is not the same for both sides. For a long the stop
# sits BELOW entry, so sl_pct is negative; for a short it sits above, so it is
# positive. The helper does not normalise this -- passing +0.02 for a long puts the
# stop 2% ABOVE the entry, where it is not a stop at all.
sl_price, tp_price = calculate_bracket_prices(
    side="long",
    entry_price=100.0,
    sl_pct=-0.02,   # 2% BELOW entry
    tp_pct=0.05,    # 5% above entry
)
# sl_price = 98.0, tp_price = 105.0

# Whether a bracket triggered is decided by the environment against the bar's high and
# low; there is no check-if-hit helper here. What this module does provide is the price
# a triggered stop actually FILLS at, which is not the stop price when the bar gaps
# through it.
stop_fill_price(stop_price=98.0, open_price=97.0, is_long=True)   # -> 97.0, the gap open
stop_fill_price(stop_price=98.0, open_price=99.0, is_long=True)   # -> 98.0, the stop
```

### `sltp_mixin.py`

Mixin class for adding SL/TP functionality to environments.

**Key Classes:**
- `SLTPMixin`: Mixin providing SL/TP tracking and execution

**Usage:**
```python
from torchtrade.envs.utils import SLTPMixin

# SLTPMixin is small and deliberately so: SIDE_DIRECTION, _record_sltp_position,
# _reset_sltp_state and _sync_position_from_exchange. It does NOT own bracket pricing,
# trigger detection or the exit -- those live in the environment, because whether a
# bracket fired is a question about the bar's high and low, which the mixin cannot see.
class MyEnvWithSLTP(SLTPMixin):
    def _open(self, side):
        # Records the position the ACTION targets, never the order side (#276): a long
        # bracket is placed with SELL stop orders, and recording those would invert the
        # position direction the policy observes.
        self._record_sltp_position(side)

    def _close(self):
        self._reset_sltp_state()
```

### `fractional_sizing.py`

Position sizing utilities for fractional share/contract trading.

**Key Classes:**
- `calculate_fractional_position()`: position size from a fractional action value

**Example:**
```python
from torchtrade.envs.utils.fractional_sizing import (
    PositionCalculationParams,
    calculate_fractional_position,
)

qty, notional, side = calculate_fractional_position(
    PositionCalculationParams(
        balance=10000.0, action_value=0.5, current_price=50000.0, leverage=5
    )
)
```

### `metrics.py`

Performance metrics.

**Functions:**
- `compute_sharpe_torch()`: annualised Sharpe from a returns tensor


### Setting Up an Environment with Utilities

```python
from torchtrade.envs.offline import SequentialTradingEnvSLTP, SequentialTradingEnvSLTPConfig
from torchtrade.envs.utils.action_maps import create_alpaca_sltp_action_map

action_map = create_alpaca_sltp_action_map([-0.02, -0.05], [0.05, 0.10])
print(f"{len(action_map)} bracket actions, e.g. {action_map[1]}")

config = SequentialTradingEnvSLTPConfig(
    symbol="BTCUSD",
    time_frames=["1Minute"],
    window_sizes=[50],
    execute_on="1Minute",
    stoploss_levels=[-0.02, -0.05],
    takeprofit_levels=[0.05, 0.10],
)
env = SequentialTradingEnvSLTP(df, config)
```

### Converting Timeframes for Different Providers

```python
from torchtrade.envs.utils import (
    TimeFrame,
    TimeFrameUnit,
    timeframe_to_alpaca,
    timeframe_to_binance,
)

# Universal timeframe
tf = TimeFrame(5, TimeFrameUnit.Minute)

# Convert for different providers
alpaca_format = timeframe_to_alpaca(tf)  # "5Min"
binance_format = timeframe_to_binance(tf)  # "5m"

# Use in API calls
alpaca_client.get_bars(symbol, timeframe=alpaca_format)
binance_client.get_klines(symbol, interval=binance_format)
```

### Adding SL/TP to Custom Environment

See the SLTPMixin section above for the real surface.

## Design Principles

1. **Provider Agnostic**: Core utilities work across all providers
2. **Type Safety**: Strong typing and dataclasses for configurations
3. **Extensibility**: Easy to add new conversions, metrics, etc.
4. **Performance**: Optimized for both backtesting and live trading
5. **Testability**: All utilities have comprehensive test coverage

## See Also

- [Core Base Classes](../core/README.md)
- [Offline Environments](../offline/README.md)
- [Live Environments](../live/README.md)
- [Main README](../README.md)
