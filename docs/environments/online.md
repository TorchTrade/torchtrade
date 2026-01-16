# Online Environments

Online environments connect to real trading APIs for paper trading or live execution. They provide the same TorchRL interface as offline environments but fetch real-time market data from exchanges.

## Overview

TorchTrade supports 3 exchanges with 6 live trading environments:

| Environment | Exchange | Asset Type | Futures | Leverage | Bracket Orders |
|-------------|----------|------------|---------|----------|----------------|
| **AlpacaTorchTradingEnv** | Alpaca | Crypto/Stocks | ❌ | ❌ | ❌ |
| **AlpacaSLTPTorchTradingEnv** | Alpaca | Crypto/Stocks | ❌ | ❌ | ✅ |
| **BinanceFuturesTorchTradingEnv** | Binance | Crypto | ✅ | ✅ | ❌ |
| **BinanceFuturesSLTPTorchTradingEnv** | Binance | Crypto | ✅ | ✅ | ✅ |
| **BitgetFuturesTorchTradingEnv** | Bitget | Crypto | ✅ | ✅ | ❌ |
| **BitgetFuturesSLTPTorchTradingEnv** | Bitget | Crypto | ✅ | ✅ | ✅ |

---

## Alpaca Environments

[Alpaca](https://alpaca.markets/) provides commission-free trading for US stocks and cryptocurrencies with paper trading support.

### AlpacaTorchTradingEnv

Simple long-only spot trading with Alpaca API.

#### Setup

```bash
# Get API keys from: https://alpaca.markets/signup
# Create .env file
cat > .env << EOF
API_KEY=your_alpaca_api_key
SECRET_KEY=your_alpaca_secret_key
EOF
```

#### Configuration

```python
from torchtrade.envs.alpaca import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

config = AlpacaTradingEnvConfig(
    symbol="BTC/USD",                       # Trading symbol
    time_frames=[
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
        TimeFrame(15, TimeFrameUnit.Minute),
    ],
    window_sizes=[12, 8, 8],                # Lookback per timeframe
    execute_on=TimeFrame(5, TimeFrameUnit.Minute),  # Execute every 5 minutes
    paper=True,                             # Paper trading (recommended!)
    feature_preprocessing_fn=None,          # Optional custom features
    reward_function=None,                   # Optional custom reward
)

env = AlpacaTorchTradingEnv(config)
```

#### Features
- **Paper trading**: Safe testing without real money
- **Commission-free**: No transaction fees
- **Real-time data**: Live market data from Alpaca
- **US markets**: Stocks and crypto available

#### Example Usage

```python
from torchtrade.envs.alpaca import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import torch

# Configure environment
config = AlpacaTradingEnvConfig(
    symbol="BTC/USD",
    time_frames=[
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
    ],
    window_sizes=[12, 8],
    execute_on=TimeFrame(5, TimeFrameUnit.Minute),
    paper=True  # Always start with paper trading!
)

# Create environment
env = AlpacaTorchTradingEnv(config)

# Run episode
tensordict = env.reset()

for step in range(100):
    # Your policy selects action
    action = policy(tensordict)
    tensordict["action"] = action

    # Execute trade
    tensordict = env.step(tensordict)

    print(f"Step {step}: Reward = {tensordict['reward'].item():.4f}")
```

### AlpacaSLTPTorchTradingEnv

Alpaca environment with stop-loss/take-profit bracket orders.

#### Configuration

```python
from torchtrade.envs.alpaca import AlpacaSLTPTorchTradingEnv, AlpacaSLTPTradingEnvConfig
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

config = AlpacaSLTPTradingEnvConfig(
    symbol="BTC/USD",
    time_frames=[
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
    ],
    window_sizes=[12, 8],
    execute_on=TimeFrame(5, TimeFrameUnit.Minute),

    # Bracket order configuration
    stoploss_levels=[-0.02, -0.05],         # -2%, -5%
    takeprofit_levels=[0.05, 0.10],         # +5%, +10%

    paper=True,
)

env = AlpacaSLTPTorchTradingEnv(config)
```

#### Action Space

With `stoploss_levels=[-0.02, -0.05]` and `takeprofit_levels=[0.05, 0.10]`:

- **Action 0**: HOLD / Close position
- **Action 1**: BUY with SL=-2%, TP=+5%
- **Action 2**: BUY with SL=-2%, TP=+10%
- **Action 3**: BUY with SL=-5%, TP=+5%
- **Action 4**: BUY with SL=-5%, TP=+10%

Total: **5 actions**

---

## Binance Environments

[Binance](https://www.binance.com/) is a leading cryptocurrency exchange with futures trading and testnet support.

### BinanceFuturesTorchTradingEnv

Futures trading environment with leverage support.

#### Setup

```bash
# Get API keys from: https://www.binance.com/en/my/settings/api-management
# Create .env file
cat > .env << EOF
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
EOF
```

#### Configuration

```python
from torchtrade.envs.binance import (
    BinanceFuturesTorchTradingEnv,
    BinanceFuturesTradingEnvConfig
)

config = BinanceFuturesTradingEnvConfig(
    symbol="BTCUSDT",                       # Trading pair
    intervals=["1m", "5m", "15m"],          # Timeframes
    window_sizes=[12, 8, 8],                # Lookback per timeframe
    execute_on="1m",                        # Execute every 1 minute

    # Futures parameters
    leverage=5,                             # 5x leverage
    quantity_per_trade=0.01,                # Position size (BTC)

    # Trading mode
    demo=True,                              # Use testnet (recommended!)

    # Optional
    feature_preprocessing_fn=None,
    reward_function=None,
)

env = BinanceFuturesTorchTradingEnv(config)
```

#### Features
- **Leverage**: Up to 125x leverage
- **Testnet**: Safe testing with fake funds
- **Real-time data**: Live market data from Binance
- **Low fees**: Competitive maker/taker fees
- **Long and short**: Both directions supported

#### Example Usage

```python
from torchtrade.envs.binance import (
    BinanceFuturesTorchTradingEnv,
    BinanceFuturesTradingEnvConfig
)
import os

config = BinanceFuturesTradingEnvConfig(
    symbol="BTCUSDT",
    intervals=["1m", "5m"],
    window_sizes=[12, 8],
    execute_on="1m",
    leverage=5,
    quantity_per_trade=0.01,
    demo=True,  # Testnet
)

env = BinanceFuturesTorchTradingEnv(config)

# Run trading loop
tensordict = env.reset()

for step in range(100):
    action = policy(tensordict)
    tensordict["action"] = action
    tensordict = env.step(tensordict)

    # Check margin ratio
    account = tensordict["account_state"]
    margin_ratio = account[7].item()

    if margin_ratio < 0.3:
        print(f"⚠️ Low margin ratio: {margin_ratio:.2f}")
```

### BinanceFuturesSLTPTorchTradingEnv

Binance futures with stop-loss/take-profit bracket orders.

#### Configuration

```python
from torchtrade.envs.binance import (
    BinanceFuturesSLTPTorchTradingEnv,
    BinanceFuturesSLTPTradingEnvConfig
)

config = BinanceFuturesSLTPTradingEnvConfig(
    symbol="BTCUSDT",
    intervals=["1m", "5m"],
    window_sizes=[12, 8],
    execute_on="1m",

    # Bracket orders
    stoploss_levels=[-0.02, -0.05],         # -2%, -5%
    takeprofit_levels=[0.03, 0.06, 0.10],   # +3%, +6%, +10%
    include_short_positions=True,           # Enable short bracket orders

    # Futures parameters
    leverage=5,
    quantity_per_trade=0.01,

    demo=True,
)

env = BinanceFuturesSLTPTorchTradingEnv(config)
```

#### Action Space

With `stoploss_levels=[-0.02, -0.05]`, `takeprofit_levels=[0.03, 0.06, 0.10]`, and `include_short_positions=True`:

- **Action 0**: HOLD / Close position
- **Actions 1-6**: LONG with SL/TP combinations (2 × 3)
- **Actions 7-12**: SHORT with SL/TP combinations (2 × 3)

Total: 1 + 2 × (2 × 3) = **13 actions**

---

## Bitget Environments

[Bitget](https://www.bitget.com/) is a fast-growing cryptocurrency exchange with competitive fees and testnet support.

### BitgetFuturesTorchTradingEnv

Futures trading environment for Bitget exchange.

#### Setup

```bash
# Get API keys from: https://www.bitget.com/en/support/articles/360038859731
# Create .env file
cat > .env << EOF
BITGET_API_KEY=your_bitget_api_key
BITGET_SECRET=your_bitget_secret
BITGET_PASSPHRASE=your_bitget_passphrase
EOF
```

#### Configuration

```python
from torchtrade.envs.bitget import (
    BitgetFuturesTorchTradingEnv,
    BitgetFuturesTradingEnvConfig
)
import os

config = BitgetFuturesTradingEnvConfig(
    symbol="BTCUSDT",
    intervals=["1m", "5m", "15m"],
    window_sizes=[12, 8, 8],
    execute_on="1m",

    # Futures parameters
    leverage=5,
    quantity_per_trade=0.002,               # Position size

    # Trading mode
    demo=True,                              # Testnet (recommended!)

    # Optional
    feature_preprocessing_fn=None,
    reward_function=None,
)

env = BitgetFuturesTorchTradingEnv(
    config,
    api_key=os.getenv("BITGET_API_KEY"),
    api_secret=os.getenv("BITGET_SECRET"),
    api_passphrase=os.getenv("BITGET_PASSPHRASE"),
)
```

#### Features
- **Leverage**: Up to 125x leverage
- **Testnet**: Safe testing environment
- **Competitive fees**: Lower fees than some competitors
- **Alternative to Binance**: Growing liquidity

#### Example Usage

```python
from torchtrade.envs.bitget import (
    BitgetFuturesTorchTradingEnv,
    BitgetFuturesTradingEnvConfig
)
import os

config = BitgetFuturesTradingEnvConfig(
    symbol="BTCUSDT",
    intervals=["1m", "5m"],
    window_sizes=[12, 8],
    execute_on="1m",
    leverage=5,
    quantity_per_trade=0.002,
    demo=True,
)

env = BitgetFuturesTorchTradingEnv(
    config,
    api_key=os.getenv("BITGET_API_KEY"),
    api_secret=os.getenv("BITGET_SECRET"),
    api_passphrase=os.getenv("BITGET_PASSPHRASE"),
)

# Trading loop
tensordict = env.reset()

for step in range(100):
    action = policy(tensordict)
    tensordict["action"] = action
    tensordict = env.step(tensordict)
```

### BitgetFuturesSLTPTorchTradingEnv

Bitget futures with stop-loss/take-profit bracket orders.

#### Configuration

```python
from torchtrade.envs.bitget import (
    BitgetFuturesSLTPTorchTradingEnv,
    BitgetFuturesSLTPTradingEnvConfig
)
import os

config = BitgetFuturesSLTPTradingEnvConfig(
    symbol="BTCUSDT",
    intervals=["1m", "5m"],
    window_sizes=[12, 8],
    execute_on="1m",

    # Bracket orders
    stoploss_levels=[-0.02, -0.05],
    takeprofit_levels=[0.03, 0.06, 0.10],
    include_short_positions=True,

    # Futures parameters
    leverage=5,
    quantity_per_trade=0.002,

    demo=True,
)

env = BitgetFuturesSLTPTorchTradingEnv(
    config,
    api_key=os.getenv("BITGET_API_KEY"),
    api_secret=os.getenv("BITGET_SECRET"),
    api_passphrase=os.getenv("BITGET_PASSPHRASE"),
)
```

---

## Safety and Best Practices

### Always Start with Paper/Testnet Trading

```python
# Alpaca
config = AlpacaTradingEnvConfig(paper=True)  # ✅ Paper trading

# Binance
config = BinanceFuturesTradingEnvConfig(demo=True)  # ✅ Testnet

# Bitget
config = BitgetFuturesTradingEnvConfig(demo=True)  # ✅ Testnet
```

### Use Low Leverage Initially

```python
config = BinanceFuturesTradingEnvConfig(
    leverage=2,  # Start with 2x, not 125x!
    ...
)
```

### Monitor Margin Ratio (Futures)

```python
while not done:
    tensordict = env.step(tensordict)
    account = tensordict["account_state"]
    margin_ratio = account[7].item()

    if margin_ratio < 0.3:
        print(f"⚠️ WARNING: Low margin ratio {margin_ratio:.2f}")
        # Consider reducing position or closing
```

### Handle API Rate Limits

```python
import time

for step in range(1000):
    tensordict = env.step(tensordict)

    # Add small delay to avoid rate limits
    if step % 10 == 0:
        time.sleep(0.5)
```

### Use Environment Variables for API Keys

Never hardcode API keys! Always use environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")
```

---

## Deployment Workflow

### 1. Train Offline

First, train and validate your policy on historical data:

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig

# Train on historical data
offline_config = SeqLongOnlyEnvConfig(...)
offline_env = SeqLongOnlyEnv(df, offline_config)

# Train policy with PPO/IQL/GRPO
train_policy(offline_env, policy)
```

### 2. Test on Paper Trading

Deploy to paper trading to validate in live conditions:

```python
from torchtrade.envs.alpaca import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig

# Deploy to paper trading
online_config = AlpacaTradingEnvConfig(paper=True, ...)
online_env = AlpacaTorchTradingEnv(online_config)

# Run policy in paper trading mode
evaluate_policy(online_env, policy)
```

### 3. Monitor Performance

Track key metrics during paper trading:
- **Sharpe ratio**: Risk-adjusted returns
- **Max drawdown**: Worst peak-to-trough decline
- **Win rate**: Percentage of profitable trades
- **Average trade duration**: How long positions are held

### 4. Go Live (Carefully!)

Only after extensive paper trading validation:

```python
# Switch to live trading
online_config = AlpacaTradingEnvConfig(
    paper=False,  # Live trading!
    ...
)

# Start with small position sizes
online_config.quantity_per_trade = 0.001  # Very small
```

---

## Exchange Comparison

| Feature | Alpaca | Binance | Bitget |
|---------|--------|---------|--------|
| **Asset Types** | Stocks, Crypto | Crypto | Crypto |
| **Futures** | ❌ | ✅ | ✅ |
| **Max Leverage** | 1x | 125x | 125x |
| **Paper Trading** | ✅ | ✅ (Testnet) | ✅ (Testnet) |
| **Commission** | Free | 0.02%/0.04% | 0.02%/0.06% |
| **Best For** | US markets, stocks | High leverage, low fees | Binance alternative |

---

## Requesting New Exchanges

Need support for another exchange? We're open to adding more integrations!

**Supported exchanges we're considering:**
- OKX
- Bybit
- Kraken
- Interactive Brokers
- OANDA (Forex/CFDs)

**To request a new exchange:**
1. [Create an issue](https://github.com/TorchTrade/torchtrade_envs/issues/new) with the exchange name
2. Email us at: torchtradecontact@gmail.com

---

## Next Steps

- **[Offline Environments](offline.md)** - Train on historical data first
- **[Actors](../components/actors.md)** - Alternative policies (RuleBasedActor, LLMActor for live trading)
- **[Transforms](../components/transforms.md)** - Data preprocessing and monitoring
- **[Custom Feature Engineering](../guides/custom-features.md)** - Add technical indicators
- **[Custom Reward Functions](../guides/reward-functions.md)** - Design better rewards
- **[Building Custom Environments](../guides/custom-environment.md)** - Extend TorchTrade

## Support

- 💬 **Questions**: [GitHub Discussions](https://github.com/TorchTrade/torchtrade_envs/discussions)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/TorchTrade/torchtrade_envs/issues)
- 📧 **Email**: torchtradecontact@gmail.com
