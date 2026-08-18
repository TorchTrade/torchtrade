# Live Trading Environments

Production-ready environments for live trading with real market data and order execution.

## Directory Structure

```
live/
├── shared/      # Shared components (futures base observation)
├── alpaca/      # Alpaca (US equities & crypto spot)
├── binance/     # Binance Futures (crypto)
├── bitget/      # Bitget Futures (crypto)
├── bybit/       # Bybit Futures (crypto, cross/isolated margin)
├── okx/         # OKX Futures (crypto, bybit-derived)
└── polymarket/  # Polymarket prediction markets (has a dry_run paper path)
```

## Supported Providers

### Alpaca (`alpaca/`)
- **Markets**: US equities, crypto spot
- **Environments**: `AlpacaTorchTradingEnv`, `AlpacaSLTPTorchTradingEnv`
- **Features**: Paper trading, fractional shares, extended hours

### Binance (`binance/`)
- **Markets**: Crypto futures (USDT-margined)
- **Environments**: `BinanceFuturesTorchTradingEnv`, `BinanceFuturesSLTPTorchTradingEnv`
- **Features**: Leverage trading, isolated/cross margin

### Bitget (`bitget/`)
- **Markets**: Crypto futures (USDT-margined)
- **Environments**: `BitgetFuturesTorchTradingEnv`, `BitgetFuturesSLTPTorchTradingEnv`
- **Features**: Leverage trading, cross/isolated margin, one-way or hedge position mode

### Bybit (`bybit/`)
- **Markets**: Crypto futures (USDT-margined)
- **Environments**: `BybitFuturesTorchTradingEnv`, `BybitFuturesSLTPTorchTradingEnv`
- **Features**: Leverage trading, cross/isolated margin, testnet

### OKX (`okx/`)
- **Markets**: Crypto futures (USDT-margined)
- **Environments**: `OKXFuturesTorchTradingEnv`, `OKXFuturesSLTPTorchTradingEnv`
- **Features**: Bybit-derived integration, leverage trading, testnet

### Polymarket (`polymarket/`)
- **Markets**: Prediction markets
- **Components**: `PolymarketBetEnv`, `MarketScanner`, `PolymarketOrderExecutor`
- **Features**: `dry_run` paper path

No environment on any venue models funding fees -- see the note in `binance/README.md`.

## Quick Start

### Alpaca Live Trading

```python
import os

import torch
from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig

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
    td["action"] = agent.get_action(td)
    # Returns (transition_td, next_root_td). The transition carries ("next", ...) --
    # the outcome and the done flag -- while ITS root is the observation you just acted
    # on. The second element is the step_mdp'd observation to act on next. Keeping only
    # the first stalls the policy on the reset observation forever.
    transition, td = env.step_and_maybe_reset(td)
    if transition["next", "done"]:
        break
```

### Binance Futures Trading

```python
from torchtrade.envs.core.common_types import MarginType

import os
from torchtrade.envs.live.binance.env import BinanceFuturesTorchTradingEnv, BinanceFuturesTradingEnvConfig

config = BinanceFuturesTradingEnvConfig(
    symbol="BTCUSDT",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
    demo=True,  # Use testnet for testing
    leverage=10.0,
    margin_type=MarginType.ISOLATED,
)

env = BinanceFuturesTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["BINANCE_API_KEY"],
    api_secret=os.environ["BINANCE_SECRET_KEY"],
)

# Run trading loop
td = env.reset()
while True:
    td["action"] = agent.get_action(td)
    # step_and_maybe_reset returns (transition_td, next_root_td). BOTH are needed:
    # `transition` carries ("next", ...) -- the outcome and the done flag -- while its
    # ROOT is still the observation you just acted on. `root` is the step_mdp'd
    # observation to act on next. Keeping only [0] leaves the policy reading the reset
    # observation forever, which is the same silent stall as `td = env.step(td)`.
    transition, root = env.step_and_maybe_reset(td)
    if transition["next", "done"]:
        break
    td = root
```

## Configuration

### Common Parameters

The non-SLTP live environments share these base parameters:

```python
# There is no shared LiveEnvConfig -- each exchange ships its own dataclass. What they
# have in common:
#
#   symbol, time_frames (a LIST), window_sizes (one per timeframe), execute_on,
#   action_levels, done_on_bankruptcy, bankrupt_threshold, seed,
#   include_base_features
#
# Futures configs add: leverage, margin_type/margin_mode, demo,
#   observation_failure_policy (close_position_on_init/_on_reset are on every
#   exchange including alpaca)
# Alpaca adds: paper, trade_mode (and shares close_position_on_init/_on_reset)
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
trade_mode: str = "notional"                # "notional" | "quantity" here.
# "fractional" is validated and implemented only on AlpacaSLTPTradingEnvConfig; this
# config accepts the string without validating and then executes it as quantity mode.
```

**Binance:**
```python
from torchtrade.envs.core.common_types import MarginType

demo: bool = True                           # Testnet or mainnet
leverage: int = 1                           # the leverage APPLIED, not a cap
margin_type: MarginType = MarginType.ISOLATED   # ISOLATED or CROSSED (the enum)
```

**Bitget:**
```python
from torchtrade.envs.live.bitget.order_executor import MarginMode
from torchtrade.envs.live.bitget.order_executor import PositionMode

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
from torchtrade.envs.live.alpaca import AlpacaTradingEnvConfig
from torchtrade.envs.live.binance import BinanceFuturesTradingEnvConfig

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
from torchtrade.envs.live.alpaca import AlpacaTradingEnvConfig

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
from torchtrade.envs.live.alpaca import AlpacaSLTPTradingEnvConfig

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
- API rate limits → Automatic throttling
- Invalid orders → Error logging, no crash
- Position desync → Automatic reconciliation

## Real-Time Data

### Data Streaming

Live environments stream real-time market data:

```python
from torchtrade.envs.live.alpaca import AlpacaTorchTradingEnv
import os
import torch

env = AlpacaTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["ALPACA_API_KEY"],
    api_secret=os.environ["ALPACA_SECRET_KEY"],
)

# Data updates automatically
td = env.reset()  # Gets latest market data

# Step blocks until the new bar arrives
td["action"] = torch.tensor(1)
transition, td = env.step_and_maybe_reset(td)  # Waits for next bar

```

### Latency Considerations

- **Observation latency**: 100-500ms (API call + processing)
- **Order execution**: 200-1000ms (varies by provider)
- **Total step time**: Depends on timeframe (1min bars = wait 1min)

### Time Synchronization

Environments sync with market time:

```python
from torchtrade.envs.live.alpaca import AlpacaTorchTradingEnv
from torchtrade.envs.live.alpaca import AlpacaTradingEnvConfig
import os
import time
import torch

# 1-minute bars: step() returns when the new bar arrives
env = AlpacaTorchTradingEnv(
    AlpacaTradingEnvConfig(
        time_frames=["1Min"],
        window_sizes=[10],
        execute_on="1Min",
    ),
    api_key=os.environ["ALPACA_API_KEY"],
    api_secret=os.environ["ALPACA_SECRET_KEY"],
)

# Step blocks until the next minute closes
td = env.reset()
start = time.time()
td["action"] = torch.tensor(1)
transition, td = env.step_and_maybe_reset(td)

elapsed = time.time() - start
# elapsed ≈ 60 seconds (wait for new bar)
```

## Order Execution

### Order Types

**Market Orders** (default):
```python
import torch

action = torch.tensor(2)  # action_levels [0.0, 0.5, 1.0] -> 2 = fully long
```

**Limit Orders**: not exposed through the action space. Actions are categorical indices
into `action_levels`; order type is decided by the executor, which places market orders
for entries and native bracket orders for SL/TP.

### Execution Flow

1. Agent produces action
2. Environment validates action
3. Order submitted to exchange
4. Environment waits for fill confirmation
5. Position updated
6. Next observation returned

### Partial Fills

There is **no retry-to-fill**: the env does not resubmit the unfilled remainder. What it
does do is stop the agent from being locked out of correcting it. A partial fill leaves
the position *direction* intact, so the duplicate-action guard would otherwise suppress
every corrective order silently and permanently; `core/live.py` compares held size
against the requested size and releases the guard when they differ by more than the
venue's minimum tradeable size, so the next action can resize. `account_state` reports
what the exchange actually holds, so a partial fill shows up as a smaller
`exposure_pct` than the action asked for.

## Position Management

### Position Tracking

Environments track positions automatically:

```python
td["action"] = action
transition, td = env.step_and_maybe_reset(td)


# account_state is [exposure, direction, unrealized_pnl_pct, holding_time,
# leverage, distance_to_liquidation] -- there is no `info` dict.
direction = transition["next", "account_state"][1].item()
unrealized_pnl_pct = transition["next", "account_state"][2].item()
```

### Position Synchronization

Environments sync with exchange:
- On reset: Fetch current positions
- On step: Verify positions match exchange
- On error: Reconcile discrepancies

### Closing Positions

```python
import torch

# Close all positions
action = torch.tensor(env.action_levels.index(0.0))  # the flat level
# Non-SLTP envs only -- SLTP envs have no action_levels -- and only when 0.0 is among
# them: validate_action_levels checks range and duplicates, never that a flat level exists.
td["action"] = action
transition, td = env.step_and_maybe_reset(td)


# Or close directly through the trader
env.trader.close_position()
```

## Monitoring

### Logging

Environments log important events:

```python
from torchtrade.envs.live.alpaca import AlpacaTorchTradingEnv

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

print(f"Portfolio Value: ${env.history.portfolio_values[-1]:.2f}")
print(f"Steps recorded: {len(env.history.portfolio_values)}")
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
import sys

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
from torchtrade.envs.live.alpaca import AlpacaTorchTradingEnv

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
from torchtrade.envs.live.alpaca import AlpacaTradingEnvConfig

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

    assert td is not None
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
- Check exchange for manual trades
- The env re-reads the exchange position every step, so a manual trade is picked
  up on the next bar (see `_sync_position_from_exchange`)

## See Also

- [Alpaca README](alpaca/README.md)
- [Binance README](binance/README.md)
- [Bitget README](bitget/README.md)
- [Core Base Classes](../core/README.md)
- [Main README](../README.md)
