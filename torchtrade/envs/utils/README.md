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
- `discrete_action_map_long_only()`: 3-action map (BUY, SELL, HOLD)
- `discrete_action_map_long_short()`: 5-action map (LONG, SHORT, CLOSE, HOLD, etc.)
- `discrete_action_map_long_only_ternary()`: Simplified 3-action map
- `discrete_action_map_futures_positions()`: Futures-specific position actions

**Example:**
```python
from torchtrade.envs.utils import discrete_action_map_long_only

action_map = discrete_action_map_long_only()
# Returns: {0: "BUY", 1: "SELL", 2: "HOLD"}

# Use in environment
action = 0  # BUY
action_name = action_map[action]
```

### `sltp_helpers.py`

Stop-loss and take-profit calculation utilities.

**Functions:**
- `calculate_sltp_prices()`: Calculate SL/TP price levels
- `check_sltp_hit()`: Check if SL or TP was triggered
- `update_sltp_prices()`: Update SL/TP levels (e.g., trailing stop)

**Example:**
```python
from torchtrade.envs.utils import calculate_sltp_prices, check_sltp_hit

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
- `FractionalPositionSizing`: Calculate position sizes based on portfolio percentage

**Example:**
```python
from torchtrade.envs.utils import FractionalPositionSizing

sizer = FractionalPositionSizing(
    portfolio_value=10000.0,
    max_position_fraction=0.1  # Max 10% per position
)

# Calculate position size
position_size = sizer.calculate(
    action_size=0.5,  # Use 50% of available
    current_price=100.0
)
# position_size = 5.0 shares (50% of 10% = 5% of portfolio)
```

### `metrics.py`

Performance metric calculations for trading strategies.

**Functions:**
- `calculate_metrics()`: Comprehensive metric calculation
  - Total return
  - Sharpe ratio
  - Maximum drawdown
  - Win rate
  - Profit factor
  - And more...

**Example:**
```python
from torchtrade.envs.utils import calculate_metrics

metrics = calculate_metrics(
    returns=daily_returns,
    trades=trade_history,
    portfolio_values=portfolio_values
)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
```

## Common Use Cases

### Setting Up an Environment with Utilities

```python
from torchtrade.envs import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.utils import (
    TimeFrame,
    TimeFrameUnit,
    discrete_action_map_long_only,
    calculate_metrics
)

# Configure environment
config = SeqLongOnlyEnvConfig(
    timeframe=TimeFrame(1, TimeFrameUnit.DAY),
    window_size=50,
)

# Create environment
env = SeqLongOnlyEnv(df=data, config=config)

# Get action mapping
action_map = discrete_action_map_long_only()
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
