# Live Trading Environments

Production-ready environments for live trading with real market data and order execution.

## Directory Structure

```
live/
├── shared/      # Shared components (futures base observation)
├── alpaca/      # Alpaca (US equities & crypto spot)
├── binance/     # Binance Futures (crypto)
└── bitget/      # Bitget Futures (crypto)
```

## Supported Providers

### Alpaca (`alpaca/`)
- **Markets**: US equities, crypto spot
- **Environments**: `AlpacaTorchTradingEnv`, `AlpacaSLTPTorchTradingEnv`
- **Features**: Paper trading, fractional shares, extended hours

### Binance (`binance/`)
- **Markets**: Crypto futures (USDT-margined)
- **Environments**: `BinanceFuturesTorchTradingEnv`, `BinanceFuturesSLTPTorchTradingEnv`
- **Features**: Leverage trading, isolated/cross margin, funding fees

### Bitget (`bitget/`)
- **Markets**: Crypto futures (USDT-margined)
- **Environments**: `BitgetFuturesTorchTradingEnv`, `BitgetFuturesSLTPTorchTradingEnv`
- **Features**: Leverage trading, low fees, copy trading integration

## Quick Start

### Alpaca Live Trading

```python
from torchtrade.envs import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from torchtrade.envs.utils import TimeFrame, TimeFrameUnit

config = AlpacaTradingEnvConfig(
    api_key="YOUR_API_KEY",
    api_secret="YOUR_SECRET_KEY",
    paper=True,  # Use paper trading for testing
    symbol="AAPL",
    timeframe=TimeFrame(1, TimeFrameUnit.MINUTE),
    initial_cash=10000.0,
)

env = AlpacaTorchTradingEnv(config=config)

# Run live trading loop
obs = env.reset()
while True:
    action = agent.get_action(obs)
    obs, reward, done, info = env.step(action)

    if done:
        break
```

### Binance Futures Trading

```python
from torchtrade.envs import BinanceFuturesTorchTradingEnv, BinanceFuturesTradingEnvConfig
from torchtrade.envs.utils import TimeFrame, TimeFrameUnit

config = BinanceFuturesTradingEnvConfig(
    api_key="YOUR_API_KEY",
    api_secret="YOUR_SECRET_KEY",
    symbol="BTCUSDT",
    timeframe=TimeFrame(1, TimeFrameUnit.MINUTE),
    testnet=True,  # Use testnet for testing
    max_leverage=10.0,
    margin_type="ISOLATED",
)

env = BinanceFuturesTorchTradingEnv(config=config)

# Run trading loop
obs = env.reset()
while not done:
    action = agent.get_action(obs)
    obs, reward, done, info = env.step(action)
```

## Configuration

### Common Parameters

All live environments share these base parameters:

```python
@dataclass
class LiveEnvConfig:
    api_key: str                   # API key
    api_secret: str                # API secret
    symbol: str                    # Trading symbol
    timeframe: TimeFrame           # Bar timeframe
    window_size: int = 50         # Observation window
    initial_cash: float = 10000.0  # Starting capital
```

### Provider-Specific

**Alpaca:**
```python
@dataclass
class AlpacaTradingEnvConfig(LiveEnvConfig):
    paper: bool = True             # Paper or live trading
    base_url: str = None           # Custom API endpoint
    extended_hours: bool = False   # Trade extended hours
```

**Binance:**
```python
@dataclass
class BinanceFuturesTradingEnvConfig(LiveEnvConfig):
    testnet: bool = True           # Testnet or mainnet
    max_leverage: float = 10.0     # Maximum leverage
    margin_type: str = "ISOLATED"  # ISOLATED or CROSS
```

**Bitget:**
```python
@dataclass
class BitgetFuturesTradingEnvConfig(LiveEnvConfig):
    testnet: bool = True           # Testnet or mainnet
    max_leverage: float = 10.0     # Maximum leverage
    margin_mode: str = "isolated"  # isolated or crossed
```

## Safety Features

### Paper Trading / Testnets

Always test strategies in safe environments:

```python
# Alpaca paper trading
config = AlpacaTradingEnvConfig(
    api_key=key,
    api_secret=secret,
    paper=True,  # No real money
)

# Binance testnet
config = BinanceFuturesTradingEnvConfig(
    api_key=key,
    api_secret=secret,
    testnet=True,  # Fake funds
)
```

### Position Limits

Set maximum position sizes:

```python
config = AlpacaTradingEnvConfig(
    # ...
    max_position_size=0.1,  # Max 10% of portfolio
)
```

### Stop-Loss / Take-Profit

Use SL/TP environments for automatic risk management:

```python
from torchtrade.envs import AlpacaSLTPTorchTradingEnv

config = AlpacaSLTPTradingEnvConfig(
    # ...
    sl_percent=0.02,  # 2% stop loss
    tp_percent=0.05,  # 5% take profit
)

env = AlpacaSLTPTorchTradingEnv(config=config)
```

### Error Handling

Environments handle common errors:
- Network failures → Auto-retry with exponential backoff
- API rate limits → Automatic throttling
- Invalid orders → Error logging, no crash
- Position desync → Automatic reconciliation

## Real-Time Data

### Data Streaming

Live environments stream real-time market data:

```python
env = AlpacaTorchTradingEnv(config=config)

# Data updates automatically
obs = env.reset()  # Gets latest market data

# Step waits for new bar
obs, reward, done, info = env.step(action)  # Waits for next bar
```

### Latency Considerations

- **Observation latency**: 100-500ms (API call + processing)
- **Order execution**: 200-1000ms (varies by provider)
- **Total step time**: Depends on timeframe (1min bars = wait 1min)

### Time Synchronization

Environments sync with market time:

```python
# 1-minute bars: step() returns when new bar arrives
env = AlpacaTorchTradingEnv(
    config=AlpacaTradingEnvConfig(
        timeframe=TimeFrame(1, TimeFrameUnit.MINUTE)
    )
)

# Step waits until next minute
start = time.time()
obs, reward, done, info = env.step(action)
elapsed = time.time() - start
# elapsed ≈ 60 seconds (wait for new bar)
```

## Order Execution

### Order Types

**Market Orders** (default):
```python
action = 0  # BUY at market price
```

**Limit Orders** (future feature):
```python
action = {"type": "limit", "price": 100.0, "side": "buy"}
```

### Execution Flow

1. Agent produces action
2. Environment validates action
3. Order submitted to exchange
4. Environment waits for fill confirmation
5. Position updated
6. Next observation returned

### Partial Fills

Environments handle partial fills:
- Retry until fully filled
- Or adjust position size accordingly
- Logged in `info` dict

## Position Management

### Position Tracking

Environments track positions automatically:

```python
obs, reward, done, info = env.step(action)

current_position = info["position"]
current_pnl = info["unrealized_pnl"]
```

### Position Synchronization

Environments sync with exchange:
- On reset: Fetch current positions
- On step: Verify positions match exchange
- On error: Reconcile discrepancies

### Closing Positions

```python
# Close all positions
action = 1  # SELL action
obs, reward, done, info = env.step(action)

# Or use close_all_positions() method
env.close_all_positions()
```

## Monitoring

### Logging

Environments log important events:

```python
import logging

logging.basicConfig(level=logging.INFO)

env = AlpacaTorchTradingEnv(config=config)
# Logs:
# - Order submissions
# - Fills
# - Position updates
# - Errors
```

### Metrics Tracking

Track live performance:

```python
# Get real-time metrics
metrics = env.get_metrics()

print(f"Portfolio Value: ${metrics['portfolio_value']:.2f}")
print(f"Total PnL: ${metrics['realized_pnl']:.2f}")
print(f"Open Positions: {metrics['num_positions']}")
```

### Integration with Monitoring Tools

Environments can integrate with monitoring:

```python
from torchtrade.integrations import WandbLogger

logger = WandbLogger(project="live-trading")

env = AlpacaTorchTradingEnv(config=config, logger=logger)
# Automatically logs metrics to Weights & Biases
```

## Best Practices

### Development Workflow

1. **Backtest** on historical data (`offline/` environments)
2. **Paper trade** with live data (Alpaca paper, Binance testnet)
3. **Small live trade** with minimal capital
4. **Scale up** gradually if successful

### Risk Management

1. **Start small**: Begin with minimal capital
2. **Use stop-losses**: Always use SL/TP environments
3. **Limit leverage**: Start with low leverage (2-3x max)
4. **Monitor constantly**: Check positions regularly
5. **Have kill switch**: Be able to close all positions quickly

### Error Recovery

```python
import signal

def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    env.close_all_positions()
    env.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
```

### API Key Security

**Never hardcode keys:**
```python
# Bad
config = AlpacaTradingEnvConfig(
    api_key="AKXXXXXXXXXXXX",  # Don't do this!
)

# Good
import os
config = AlpacaTradingEnvConfig(
    api_key=os.environ["ALPACA_API_KEY"],
    api_secret=os.environ["ALPACA_SECRET_KEY"],
)
```

## Testing Live Environments

```python
import pytest
from torchtrade.envs import AlpacaTorchTradingEnv

@pytest.mark.live
def test_alpaca_connection():
    """Test connection to Alpaca paper trading"""
    config = AlpacaTradingEnvConfig(
        api_key=os.environ["ALPACA_KEY"],
        api_secret=os.environ["ALPACA_SECRET"],
        paper=True,
        symbol="SPY",
    )

    env = AlpacaTorchTradingEnv(config=config)
    obs = env.reset()

    assert obs is not None
    assert env.is_connected()
```

## Troubleshooting

### Common Issues

**Connection errors:**
- Check API keys
- Verify network connection
- Ensure API endpoint is correct

**Order rejections:**
- Insufficient funds
- Invalid symbol
- Market closed
- Leverage too high

**Position desync:**
- Environment resets automatically
- Check exchange for manual trades
- Use `sync_positions()` method

## See Also

- [Alpaca README](alpaca/README.md)
- [Binance README](binance/README.md)
- [Bitget README](bitget/README.md)
- [Core Base Classes](../core/README.md)
- [Main README](../README.md)
