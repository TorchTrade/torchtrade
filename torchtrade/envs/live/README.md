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
import os

import torch
from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from torchtrade.envs.utils import TimeFrame, TimeFrameUnit

config = AlpacaTradingEnvConfig(
    paper=True,  # Use paper trading for testing
    symbol="AAPL",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
)

env = AlpacaTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["ALPACA_API_KEY"],
    api_secret=os.environ["ALPACA_SECRET_KEY"],
)

# Run live trading loop
td = env.reset()
while True:
    action = agent.get_action(obs)
    td["action"] = action
    td = env.step(td)

    if done:
        break
```

### Binance Futures Trading

```python
import os
from torchtrade.envs.live.binance.env import BinanceFuturesTorchTradingEnv, BinanceFuturesTradingEnvConfig
from torchtrade.envs.utils import TimeFrame, TimeFrameUnit

config = BinanceFuturesTradingEnvConfig(
    symbol="BTCUSDT",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
    demo=True,  # Use testnet for testing
    leverage=10.0,
    margin_type="ISOLATED",
)

env = BinanceFuturesTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["BINANCE_API_KEY"],
    api_secret=os.environ["BINANCE_SECRET_KEY"],
)

# Run trading loop
td = env.reset()
while not done:
    action = agent.get_action(obs)
    td["action"] = action
    td = env.step(td)
```

## Configuration

### Common Parameters

All live environments share these base parameters:

```python
# There is no shared LiveEnvConfig -- each exchange ships its own dataclass. What they
# have in common:
#
#   symbol, time_frames (a LIST), window_sizes (one per timeframe), execute_on,
#   action_levels, done_on_bankruptcy, bankrupt_threshold, seed,
#   include_base_features
#
# Futures configs add: leverage, margin_type/margin_mode, demo,
#   close_position_on_init, close_position_on_reset, observation_failure_policy
# Alpaca adds: paper, trade_mode
#
# Credentials are CONSTRUCTOR arguments on every exchange, never config fields.
# See the per-exchange READMEs for the exact field list.
```

### Provider-Specific

These are standalone dataclasses -- there is no shared base to inherit from. The fields
that differ between exchanges:

**Alpaca:**
```python
paper: bool = True                          # Paper or live trading
trade_mode: str = "notional"                # "fractional" | "notional" | "quantity"
```

**Binance:**
```python
demo: bool = True                           # Testnet or mainnet
leverage: int = 1                           # the leverage APPLIED, not a cap
margin_type: MarginType = MarginType.ISOLATED   # ISOLATED or CROSSED (the enum)
```

**Bitget:**
```python
demo: bool = True                           # Testnet or mainnet
leverage: int = 1                           # the leverage APPLIED, not a cap
margin_mode: MarginMode = MarginMode.ISOLATED   # ISOLATED or CROSSED (the enum)
position_mode: PositionMode = PositionMode.ONE_WAY
product_type: str = "USDT-FUTURES"
```

## Safety Features

### Paper Trading / Testnets

Always test strategies in safe environments:

```python
# Alpaca paper trading
config = AlpacaTradingEnvConfig(
    paper=True,  # No real money
)

# Binance testnet
config = BinanceFuturesTradingEnvConfig(
    demo=True,  # Fake funds
)
```

### Position Limits

Position size is capped by `action_levels` -- each entry is the fraction of the
portfolio an action allocates, so the largest level is the maximum exposure:

```python
config = AlpacaTradingEnvConfig(
    symbol="BTC/USD",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
    action_levels=[0.0, 0.25, 0.5],  # never more than 50% of the portfolio
)
```

### Stop-Loss / Take-Profit

Use SL/TP environments for automatic risk management:

```python
import os
from torchtrade.envs.live.alpaca.env_sltp import AlpacaSLTPTorchTradingEnv

config = AlpacaSLTPTradingEnvConfig(
    # ...
    stoploss_levels=[-0.02],  # 2% stop loss
    takeprofit_levels=[0.05],  # 5% take profit
)

env = AlpacaSLTPTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["ALPACA_API_KEY"],
    api_secret=os.environ["ALPACA_SECRET_KEY"],
)
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
import os
env = AlpacaTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["ALPACA_API_KEY"],
    api_secret=os.environ["ALPACA_SECRET_KEY"],
)

# Data updates automatically
td = env.reset()  # Gets latest market data

# Step waits for new bar
td["action"] = action
td = env.step(td)  # Waits for next bar
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
        time_frames=["1Min"],
        window_sizes=[10],
        execute_on="1Min",
    )
)

# Step waits until next minute
start = time.time()
td["action"] = action
td = env.step(td)
elapsed = time.time() - start
# elapsed ≈ 60 seconds (wait for new bar)
```

## Order Execution

### Order Types

**Market Orders** (default):
```python
action = torch.tensor(2)  # action_levels [0.0, 0.5, 1.0] -> 2 = fully long
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
td["action"] = action
td = env.step(td)

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
action = torch.tensor(0)  # 0 allocates 0% -- i.e. go flat
td["action"] = action
td = env.step(td)

# Or close directly through the trader
env.trader.close_position()
```

## Monitoring

### Logging

Environments log important events:

```python
import os
import logging

logging.basicConfig(level=logging.INFO)

env = AlpacaTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["ALPACA_API_KEY"],
    api_secret=os.environ["ALPACA_SECRET_KEY"],
)
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

print(f"Portfolio Value: ${metrics['portfolio_value']:.2f}")
print(f"Total PnL: ${metrics['realized_pnl']:.2f}")
print(f"Open Positions: {metrics['num_positions']}")
```

### Integration with Monitoring Tools

Environments can integrate with monitoring:


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
    env.trader.close_position()
    env.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
```

### API Key Security

**Never hardcode keys.** They are constructor arguments, so this is where the contrast
lives -- the config itself never holds a credential:

```python
import os

# Bad -- a key in the source is a key in your git history
env = AlpacaTorchTradingEnv(config, api_key="PKXXXXXXXXXXXXXXXXXX")

# Good
env = AlpacaTorchTradingEnv(
    config,
    api_key=os.environ["ALPACA_API_KEY"],
    api_secret=os.environ["ALPACA_SECRET_KEY"],
)
```

## Testing Live Environments

```python
import os
import pytest
from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv

@pytest.mark.live
def test_alpaca_connection():
    """Test connection to Alpaca paper trading"""
    config = AlpacaTradingEnvConfig(
        paper=True,
        symbol="SPY",
    )

    env = AlpacaTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["ALPACA_API_KEY"],
    api_secret=os.environ["ALPACA_SECRET_KEY"],
)
    td = env.reset()

    assert obs is not None
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
- The env re-reads the exchange position every step, so a manual trade is picked
  up on the next bar (see `_sync_position_from_exchange`)

## See Also

- [Alpaca README](alpaca/README.md)
- [Binance README](binance/README.md)
- [Bitget README](bitget/README.md)
- [Core Base Classes](../core/README.md)
- [Main README](../README.md)
