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
from torchrl.envs.utils import step_mdp

from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig

# Credentials are CONSTRUCTOR arguments, not config fields. The config carries the
# market and observation shape: `time_frames` (a list) with matching `window_sizes`,
# and `execute_on`. There is no `timeframe` or `initial_cash` -- a live account's cash
# comes from the broker.
config = AlpacaTradingEnvConfig(
    paper=True,
    close_position_on_init=True,
    close_position_on_reset=False,  # Paper trading
    symbol="AAPL",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
)

env = AlpacaTorchTradingEnv(config, api_key="YOUR_KEY", api_secret="YOUR_SECRET")
td = env.reset()  # a TensorDict, not a bare array
```

## Features

- **Paper Trading**: Risk-free testing with simulated funds
- **Fractional Shares**: Buy partial shares (e.g., 0.5 shares of AAPL)
- **Extended Hours**: Trade during pre-market and after-hours
- **Real-time Data**: WebSocket streaming for live prices
- **Multiple Assets**: Stocks, ETFs, and crypto (BTC, ETH, etc.)

## Configuration

```python
from torchtrade.envs.live.alpaca import AlpacaTradingEnvConfig

# Every field of AlpacaTradingEnvConfig, with its real default.
# Credentials are NOT fields -- they are constructor arguments on the env.
config = AlpacaTradingEnvConfig(
    symbol='BTC/USD',
    action_levels=None,
    time_frames='1Hour',
    window_sizes=10,
    execute_on='1Hour',
    done_on_bankruptcy=True,
    bankrupt_threshold=0.1,
    paper=True,
    close_position_on_init=True,
    close_position_on_reset=False,
    trade_mode='notional',
    seed=42,
    include_base_features=False,
)
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
action = 1  # action 0 is flat/HOLD; 1..N open a position
```

**Time-in-Force**: Day orders (default), good-til-canceled (GTC) optional

## Example: Paper Trading

```python
import os
from torchrl.envs.utils import step_mdp

from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig

# Load keys from environment
config = AlpacaTradingEnvConfig(
    paper=True,
    close_position_on_init=True,
    close_position_on_reset=False,
    symbol="SPY",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
)

env = AlpacaTorchTradingEnv(
    config,
    api_key=os.environ["ALPACA_KEY"],
    api_secret=os.environ["ALPACA_SECRET"],
)

# Trading loop
td = env.reset()
for _ in range(100):
    td["action"] = agent.get_action(td)
    td = env.step(td)
    if td["next", "done"]:
        break
    # step_mdp, NOT step_and_maybe_reset: the latter returns a 2-tuple, and its
    # auto-reset calls cancel_open_orders() on the real broker -- it would cancel the
    # SL/TP brackets placed below and clear env.history.
    td = step_mdp(td)

print(f"Final value: ${env.history.portfolio_values[-1]:.2f}")
```

## Example: With Stop-Loss/Take-Profit

```python
import os

import torch

from torchtrade.envs.live.alpaca.env_sltp import AlpacaSLTPTorchTradingEnv, AlpacaSLTPTradingEnvConfig

# SL/TP are LEVELS lists, not single percentages -- the action space is one entry per
# (stoploss, takeprofit) pair, so a single number could not describe it.
config = AlpacaSLTPTradingEnvConfig(
    paper=True,
    close_position_on_init=True,
    close_position_on_reset=False,
    symbol="AAPL",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
    stoploss_levels=[-0.02],   # 2% stop loss
    takeprofit_levels=[0.05],  # 5% take profit
)

env = AlpacaSLTPTorchTradingEnv(
    config,
    api_key=os.environ["ALPACA_KEY"],
    api_secret=os.environ["ALPACA_SECRET"],
)
td = env.reset()

# The bracket is placed with the entry; SL/TP then trigger on the venue.
# Action 0 is HOLD -- 1..N are the (stoploss, takeprofit) pairs.
td["action"] = torch.tensor(1)
td = env.step(td)
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
