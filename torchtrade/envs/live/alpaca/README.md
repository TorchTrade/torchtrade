# Alpaca Trading Environment

Live trading integration with Alpaca for US equities and crypto spot markets.

## Files

- **`base.py`**: Base Alpaca environment with common functionality
- **`observation.py`**: Alpaca-specific observation handling
- **`order_executor.py`**: Order execution logic for Alpaca API
- **`env.py`**: Main Alpaca trading environment
- **`env_sltp.py`**: Alpaca environment with stop-loss/take-profit
- **`utils.py`**: Helper functions and utilities

## Quick Start

```python
from torchtrade.envs import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from torchtrade.envs.utils import TimeFrame, TimeFrameUnit

config = AlpacaTradingEnvConfig(
    api_key="YOUR_KEY",
    api_secret="YOUR_SECRET",
    paper=True,  # Paper trading
    symbol="AAPL",
    timeframe=TimeFrame(1, TimeFrameUnit.MINUTE),
    initial_cash=10000.0,
)

env = AlpacaTorchTradingEnv(config=config)
obs = env.reset()
```

## Features

- **Paper Trading**: Risk-free testing with simulated funds
- **Fractional Shares**: Buy partial shares (e.g., 0.5 shares of AAPL)
- **Extended Hours**: Trade during pre-market and after-hours
- **Real-time Data**: WebSocket streaming for live prices
- **Multiple Assets**: Stocks, ETFs, and crypto (BTC, ETH, etc.)

## Configuration

```python
@dataclass
class AlpacaTradingEnvConfig:
    api_key: str
    api_secret: str
    symbol: str
    timeframe: TimeFrame
    paper: bool = True              # Paper or live trading
    base_url: str = None            # Custom API endpoint
    extended_hours: bool = False    # Trade extended hours
    initial_cash: float = 10000.0
    window_size: int = 50
```

## Supported Symbols

### US Equities
- Stocks: AAPL, GOOGL, MSFT, TSLA, etc.
- ETFs: SPY, QQQ, IWM, etc.

### Crypto
- BTC/USD, ETH/USD, etc. (spot trading only)

Check Alpaca docs for full list: https://alpaca.markets/docs/trading/

## Market Hours

**Regular Hours:**
- 9:30 AM - 4:00 PM ET (Monday-Friday)

**Extended Hours** (with `extended_hours=True`):
- Pre-market: 4:00 AM - 9:30 AM ET
- After-hours: 4:00 PM - 8:00 PM ET

## Order Types

**Market Orders** (default):
```python
action = 0  # BUY at current market price
```

**Time-in-Force**: Day orders (default), good-til-canceled (GTC) optional

## Example: Paper Trading

```python
import os
from torchtrade.envs import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig

# Load keys from environment
config = AlpacaTradingEnvConfig(
    api_key=os.environ["ALPACA_KEY"],
    api_secret=os.environ["ALPACA_SECRET"],
    paper=True,
    symbol="SPY",
    timeframe=TimeFrame(1, TimeFrameUnit.MINUTE),
)

env = AlpacaTorchTradingEnv(config=config)

# Trading loop
obs = env.reset()
for _ in range(100):
    action = agent.get_action(obs)
    obs, reward, done, info = env.step(action)

    if done:
        break

print(f"Final value: ${env.portfolio_value:.2f}")
```

## Example: With Stop-Loss/Take-Profit

```python
from torchtrade.envs import AlpacaSLTPTorchTradingEnv, AlpacaSLTPTradingEnvConfig

config = AlpacaSLTPTradingEnvConfig(
    api_key=os.environ["ALPACA_KEY"],
    api_secret=os.environ["ALPACA_SECRET"],
    paper=True,
    symbol="AAPL",
    timeframe=TimeFrame(1, TimeFrameUnit.MINUTE),
    sl_percent=0.02,  # 2% stop loss
    tp_percent=0.05,  # 5% take profit
)

env = AlpacaSLTPTorchTradingEnv(config=config)
obs = env.reset()

# SL/TP triggers automatically
action = 0  # BUY
obs, reward, done, info = env.step(action)

if info.get("sl_triggered"):
    print("Stop loss hit!")
```

## Best Practices

1. **Start with paper trading**: Test thoroughly before live trading
2. **Verify market hours**: Don't trade when market is closed
3. **Handle holidays**: Market closed on US holidays
4. **Monitor positions**: Check Alpaca dashboard regularly
5. **Use stop-losses**: Protect against large losses

## API Rate Limits

- **Market Data**: 200 requests/minute
- **Orders**: 200 requests/minute
- **Account**: 200 requests/minute

Environments handle rate limiting automatically.

## Common Issues

**"Market is closed"**: Check market hours and holidays

**"Insufficient funds"**: Not enough cash for order

**"Symbol not found"**: Verify symbol is supported by Alpaca

**Connection timeout**: Check internet connection, API keys

## Resources

- [Alpaca Documentation](https://alpaca.markets/docs/)
- [Paper Trading Dashboard](https://paper-api.alpaca.markets/)
- [Market Calendar](https://alpaca.markets/docs/market-hours/)

## See Also

- [Live Environments README](../README.md)
- [Core Base Classes](../../core/README.md)
