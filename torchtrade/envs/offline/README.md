# Offline Backtesting Environments

Backtesting environments for strategy development, evaluation, and research.

## Directory Structure

```
offline/
├── infrastructure/  # Internal components (samplers, utilities)
├── longonly/        # Long-only trading environments
└── futures/         # Futures trading environments
```

## Environment Types

### Long-Only Environments (`longonly/`)

Traditional buy-and-hold style trading:

- **`sequential.py`** (`SeqLongOnlyEnv`): Multi-step sequential trading
  - Buy, sell, or hold at each timestep
  - Portfolio value tracks performance
  - Full episode history

- **`sequential_sltp.py`** (`SeqLongOnlySLTPEnv`): Sequential with stop-loss/take-profit
  - Automatic SL/TP execution
  - Risk management built-in
  - Trailing stop options

- **`onestep.py`** (`LongOnlyOneStepEnv`): Single-step decision making
  - Simplified for fast iteration
  - Ideal for supervised learning
  - Quick backtesting

### Futures Environments (`futures/`)

Leveraged long/short trading with margin:

- **`sequential.py`** (`SeqFuturesEnv`): Multi-step futures trading
  - Long and short positions
  - Leverage and margin management
  - Liquidation handling
  - Funding fee simulation

- **`sequential_sltp.py`** (`SeqFuturesSLTPEnv`): Futures with SL/TP
  - Risk management for leveraged trades
  - Automatic position exit
  - Margin call prevention

- **`onestep.py`** (`FuturesOneStepEnv`): Single-step futures
  - Fast futures backtesting
  - Simplified leverage model
  - Quick strategy testing

### Infrastructure (`infrastructure/`)

Internal components (not directly used by end users):

- **`sampler.py`**: Market data observation sampling and windowing
- **`utils.py`**: Helper functions for offline environments

## Quick Start

### Long-Only Sequential Trading

```python
from torchtrade.envs import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.utils import TimeFrame, TimeFrameUnit
import pandas as pd

# Load your data
df = pd.read_csv("market_data.csv")

# Configure environment
config = SeqLongOnlyEnvConfig(
    timeframe=TimeFrame(1, TimeFrameUnit.DAY),
    window_size=50,
    initial_cash=10000.0,
    commission=0.001,  # 0.1% commission
)

# Create environment
env = SeqLongOnlyEnv(df=df, config=config)

# Run episode
obs = env.reset()
done = False
while not done:
    action = agent.get_action(obs)
    obs, reward, done, info = env.step(action)

print(f"Final portfolio value: ${env.portfolio_value:.2f}")
```

### Futures Trading with Leverage

```python
from torchtrade.envs import SeqFuturesEnv, SeqFuturesEnvConfig, MarginType
from torchtrade.envs.utils import TimeFrame, TimeFrameUnit

config = SeqFuturesEnvConfig(
    timeframe=TimeFrame(1, TimeFrameUnit.HOUR),
    window_size=100,
    initial_margin=10000.0,
    max_leverage=10.0,
    margin_type=MarginType.ISOLATED,
    commission=0.0004,  # 0.04% per trade
)

env = SeqFuturesEnv(df=df, config=config)

# Trade with leverage
obs = env.reset()
action = {"position": 0.5, "leverage": 5.0}  # 50% long with 5x leverage
obs, reward, done, info = env.step(action)
```

### Using Stop-Loss and Take-Profit

```python
from torchtrade.envs import SeqLongOnlySLTPEnv, SeqLongOnlySLTPEnvConfig

config = SeqLongOnlySLTPEnvConfig(
    timeframe=TimeFrame(1, TimeFrameUnit.DAY),
    window_size=50,
    initial_cash=10000.0,
    sl_percent=0.02,  # 2% stop loss
    tp_percent=0.05,  # 5% take profit
    use_trailing_stop=True,
)

env = SeqLongOnlySLTPEnv(df=df, config=config)

# SL/TP triggers automatically during stepping
obs = env.reset()
action = 0  # BUY
obs, reward, done, info = env.step(action)

if info.get("sl_triggered"):
    print("Stop loss was hit!")
if info.get("tp_triggered"):
    print("Take profit was hit!")
```

## Configuration Options

### Common Config Parameters

All environments share these base parameters:

```python
@dataclass
class BaseOfflineEnvConfig:
    timeframe: TimeFrame           # Data timeframe
    window_size: int              # Observation window size
    commission: float = 0.001     # Trading commission (0.1%)
    slippage: float = 0.0         # Price slippage simulation
    reward_scaling: float = 1.0   # Reward scale factor
```

### Long-Only Specific

```python
@dataclass
class SeqLongOnlyEnvConfig(BaseOfflineEnvConfig):
    initial_cash: float = 10000.0     # Starting capital
    max_position_size: float = 1.0    # Max % of portfolio per trade
```

### Futures Specific

```python
@dataclass
class SeqFuturesEnvConfig(BaseOfflineEnvConfig):
    initial_margin: float = 10000.0      # Starting margin
    max_leverage: float = 10.0           # Maximum leverage
    margin_type: MarginType = ISOLATED   # Cross or isolated margin
    maintenance_margin_rate: float = 0.5 # Liquidation threshold
    funding_rate: float = 0.0001         # Funding fee per hour
```

### SL/TP Specific

```python
@dataclass
class SLTPEnvConfig(BaseOfflineEnvConfig):
    sl_percent: float = 0.02           # Stop loss percentage
    tp_percent: float = 0.05           # Take profit percentage
    use_trailing_stop: bool = False    # Enable trailing stop
    trailing_stop_distance: float = 0.01  # Trailing distance
```

## Data Format

Environments expect DataFrames with OHLCV columns:

```python
df = pd.DataFrame({
    'timestamp': pd.date_range('2020-01-01', periods=1000, freq='1H'),
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})
```

Optional columns for enhanced simulation:
- `bid` / `ask`: More realistic execution prices
- `spread`: Bid-ask spread for slippage modeling

## Action Spaces

### Long-Only Actions

**Discrete (default):**
- 0: BUY (enter long position)
- 1: SELL (close position)
- 2: HOLD (do nothing)

**Continuous (optional):**
- Value in [-1, 1]: Position size (-1=fully short, 0=flat, 1=fully long)

### Futures Actions

**Discrete:**
- 0: LONG (enter/increase long)
- 1: SHORT (enter/increase short)
- 2: CLOSE (exit position)
- 3: HOLD (maintain position)

**Continuous:**
- Position: [-1, 1] (long/short direction and size)
- Leverage: [1, max_leverage] (optional)

## Observation Space

Observations are TensorDict objects containing:

```python
{
    "observation": Tensor[window_size, n_features],  # OHLCV window
    "position": Tensor[1],                            # Current position
    "cash": Tensor[1],                                # Available cash/margin
    "portfolio_value": Tensor[1],                     # Total value
    "timestamp": Tensor[1],                           # Current time
}
```

## Reward Functions

Default reward is percentage return per step:

```python
reward = (portfolio_value_t1 - portfolio_value_t0) / portfolio_value_t0
```

Custom rewards can be provided:

```python
from torchtrade.envs.core import SharpeRatioReward

env = SeqLongOnlyEnv(
    df=df,
    config=config,
    reward_fn=SharpeRatioReward(window=252)  # Annualized Sharpe
)
```

## Performance Metrics

After backtesting, retrieve metrics:

```python
metrics = env.get_metrics()

print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
print(f"Number of Trades: {metrics['num_trades']}")
```

## Visualization

Generate plots of backtest results:

```python
import matplotlib.pyplot as plt

# Plot portfolio value over time
env.render(mode='human')

# Or manually
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Portfolio value
ax1.plot(env.portfolio_history)
ax1.set_title('Portfolio Value')

# Positions
ax2.plot(env.position_history)
ax2.set_title('Position Size Over Time')

plt.tight_layout()
plt.show()
```

## Best Practices

1. **Data Quality**: Ensure clean, complete OHLCV data
2. **Timeframe Alignment**: Match timeframe to your strategy
3. **Window Size**: Balance context vs. computation (typically 20-100)
4. **Commission**: Use realistic values (0.05-0.1% for crypto, 0.01-0.05% for stocks)
5. **Slippage**: Add slippage for realistic execution modeling
6. **Risk Management**: Use SL/TP environments for safer strategies
7. **Validation**: Always validate on out-of-sample data

## Common Pitfalls

1. **Lookahead Bias**: Ensure no future data in observations
2. **Survivorship Bias**: Include delisted/failed assets in data
3. **Overfitting**: Don't over-optimize on training data
4. **Ignoring Costs**: Always include commissions and slippage
5. **Unrealistic Leverage**: Use conservative leverage ratios

## Testing Strategies

```python
import pytest
from torchtrade.envs import SeqLongOnlyEnv, SeqLongOnlyEnvConfig

def test_buy_and_hold():
    """Test simple buy-and-hold strategy"""
    env = SeqLongOnlyEnv(df=test_data, config=test_config)

    obs = env.reset()
    obs, reward, done, info = env.step(0)  # BUY

    while not done:
        obs, reward, done, info = env.step(2)  # HOLD

    assert env.portfolio_value > 0
    assert len(env.trades) == 1  # Only one buy trade
```

## See Also

- [Core Base Classes](../core/README.md)
- [Utilities](../utils/README.md)
- [Live Environments](../live/README.md)
- [Main README](../README.md)
