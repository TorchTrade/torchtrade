# Alpaca Trading Environment

Live trading integration with Alpaca for **crypto spot** markets.

> **Equities are not supported.** `observation.py` fetches crypto bars only. An equity
> symbol still constructs, because the order client accepts one, and then returns no data.

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
    close_position_on_reset=False,
    symbol="BTC/USD",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
)

env = AlpacaTorchTradingEnv(config, api_key="YOUR_KEY", api_secret="YOUR_SECRET")
td = env.reset()  # a TensorDict, not a bare array
```

## Features

- **Paper Trading**: Risk-free testing with simulated funds
- **Fractional Quantities**: crypto orders are not rounded to whole units
- **Bar Polling**: REST. `get_observations()` makes one date-range fetch per configured
  timeframe, each call.
- **Crypto Spot**: BTC/USD, ETH/USD and Alpaca's other crypto pairs

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

### Crypto spot
- BTC/USD, ETH/USD, etc.

### US equities
Not supported. See the note at the top.

Check Alpaca docs for the full list: <https://docs.alpaca.markets/>

## Market Hours

Crypto trades 24/7.

## Order Types

**Market Orders** (default):
```python
action = 1  # action 0 is flat/HOLD; 1..N open a position
```

**Time-in-Force**: the executor defaults to `ioc`. The SLTP bracket order passes `gtc`.

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
    symbol="BTC/USD",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
)

env = AlpacaTorchTradingEnv(
    config,
    api_key=os.environ["ALPACA_API_KEY"],
    api_secret=os.environ["ALPACA_SECRET_KEY"],
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
    symbol="BTC/USD",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
    stoploss_levels=[-0.02],   # 2% stop loss
    takeprofit_levels=[0.05],  # 5% take profit
)

env = AlpacaSLTPTorchTradingEnv(
    config,
    api_key=os.environ["ALPACA_API_KEY"],
    api_secret=os.environ["ALPACA_SECRET_KEY"],
)
td = env.reset()

# The bracket is placed with the entry; SL/TP then trigger on the venue.
# Action 0 is HOLD; then CLOSE if `include_close_action`; then the
# ('long', stoploss, takeprofit) grid.
td["action"] = torch.tensor(1)
td = env.step(td)
```

## Best Practices

1. **Start with paper trading**: Test thoroughly before live trading
2. **Monitor positions**: Check Alpaca dashboard regularly
3. **Use stop-losses**: Protect against large losses

## API Rate Limits

Alpaca's limits depend on the API and the plan; check
<https://docs.alpaca.markets/> for your account.

**This package does not throttle.** One env on a 1-minute bar is unlikely to reach any of
them. Many envs, or a sub-minute `execute_on`, need your own request budget.

## Common Issues

**No bars returned**: check the symbol is a crypto pair. An equity symbol reaches a
crypto data client and comes back empty.

**"Insufficient funds"**: Not enough cash for order

**"Symbol not found"**: Verify symbol is supported by Alpaca

**Connection timeout**: Check internet connection, API keys

## Resources

- [Alpaca Documentation](https://docs.alpaca.markets/)
- [Paper Trading Dashboard](https://app.alpaca.markets/paper/dashboard/overview)

## See Also

- [Live Environments README](../README.md)
- [Core Base Classes](../../core/README.md)
