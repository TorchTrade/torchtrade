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

# Create timeframe
tf = TimeFrame(5, TimeFrameUnit.MINUTE)

# Parse from string
tf = parse_timeframe_string("1d")

# Convert to provider format
alpaca_tf = timeframe_to_alpaca(tf)  # "1Day"
binance_tf = timeframe_to_binance(tf)  # "1d"
```

### `action_maps.py`

Discrete action space mappings for different trading strategies.

**Functions:**
- `create_alpaca_sltp_action_map()`: 3-action map (BUY, SELL, HOLD)
- `discrete_action_map_long_short()`: 5-action map (LONG, SHORT, CLOSE, HOLD, etc.)
- `create_sltp_action_map()`: Simplified 3-action map
- `discrete_action_map_futures_positions()`: Futures-specific position actions

**Example:**
```python
from torchtrade.envs.utils.action_maps import create_alpaca_sltp_action_map

action_map = create_alpaca_sltp_action_map()
# Returns: {0: "BUY", 1: "SELL", 2: "HOLD"}

# Use in environment
action = 0  # BUY
action_name = action_map[action]
```

### `sltp_helpers.py`

Stop-loss and take-profit calculation utilities.

**Functions:**
- `calculate_bracket_prices()`: SL/TP price levels for a bracket order
- `update_sltp_prices()`: Update SL/TP levels (e.g., trailing stop)

**Example:**
```python
from torchtrade.envs.utils.sltp_helpers import calculate_bracket_prices

# Calculate SL/TP levels
entry_price = 100.0
sl_price, tp_price = calculate_sltp_prices(
    entry_price=entry_price,
    direction="long",
    sl_percent=0.02,  # 2% stop loss
    tp_percent=0.05,  # 5% take profit
)
# sl_price = 98.0, tp_price = 105.0

# Check if hit
current_price = 97.5
sl_hit, tp_hit = check_sltp_hit(
    current_price=current_price,
    sl_price=sl_price,
    tp_price=tp_price,
    direction="long"
)
# sl_hit = True, tp_hit = False
```

### `sltp_mixin.py`

Mixin class for adding SL/TP functionality to environments.

**Key Classes:**
- `SLTPMixin`: Mixin providing SL/TP tracking and execution

**Usage:**
```python
from torchtrade.envs.utils import SLTPMixin
from torchtrade.envs.core import TorchTradeOfflineEnv

class MyEnvWithSLTP(SLTPMixin, TorchTradeOfflineEnv):
    def __init__(self, config):
        super().__init__(config)
        self._init_sltp(
            sl_percent=config.sl_percent,
            tp_percent=config.tp_percent
        )

    def _step(self, action):
        # Check SL/TP before processing action
        if self._check_sltp_triggered(current_price):
            return self._execute_sltp_exit()

        # Normal step logic
        return super()._step(action)
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

Performance metric calculations for trading strategies.

**Functions:**

**Example:**


## Common Use Cases

### Setting Up an Environment with Utilities

```python
from torchtrade.envs.utils import (
    TimeFrame,
    TimeFrameUnit,
    discrete_action_map_long_only,
    calculate_metrics
)

# Configure environment
    timeframe=TimeFrame(1, TimeFrameUnit.DAY),
    window_size=50,
)

# Create environment

# Get action mapping
action_map = create_alpaca_sltp_action_map()
print(f"Available actions: {action_map}")

# After training, calculate metrics
metrics = calculate_metrics(
    returns=env.get_returns(),
    trades=env.get_trades(),
    portfolio_values=env.get_portfolio_values()
)
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
tf = TimeFrame(5, TimeFrameUnit.MINUTE)

# Convert for different providers
alpaca_format = timeframe_to_alpaca(tf)  # "5Min"
binance_format = timeframe_to_binance(tf)  # "5m"

# Use in API calls
alpaca_client.get_bars(symbol, timeframe=alpaca_format)
binance_client.get_klines(symbol, interval=binance_format)
```

### Adding SL/TP to Custom Environment

```python
from torchtrade.envs.core import TorchTradeOfflineEnv
from torchtrade.envs.utils import SLTPMixin
from dataclasses import dataclass

@dataclass
class MyEnvConfig:
    sl_percent: float = 0.02
    tp_percent: float = 0.05

class MyEnv(SLTPMixin, TorchTradeOfflineEnv):
    def __init__(self, config):
        super().__init__(config)
        self._init_sltp(config.sl_percent, config.tp_percent)

    def _step(self, action):
        # SL/TP check happens automatically
        if self.has_position and self._check_sltp():
            return self._execute_sltp_exit()

        # Normal logic
        return super()._step(action)
```

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
