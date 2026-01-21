# Online Environments

Online environments connect to real trading APIs for paper trading or live execution. They provide the same TorchTrade interface as offline environments but fetch real-time market data from exchanges.

**Supported Exchanges:**

- **[Alpaca](https://alpaca.markets/)** - Commission-free US stocks and crypto with paper trading

- **[Binance](https://accounts.binance.com/register?ref=25015935)** - Cryptocurrency futures with high leverage and testnet

- **[Bitget](https://www.bitget.com/)** - Cryptocurrency futures with competitive fees and testnet

## Overview

TorchTrade provides 6 live trading environments across these exchanges:

| Environment | Exchange | Asset Type | Futures | Leverage | Bracket Orders |
|-------------|----------|------------|---------|----------|----------------|
| **AlpacaTorchTradingEnv** | Alpaca | Crypto/Stocks | ❌ | ❌ | ❌ |
| **AlpacaSLTPTorchTradingEnv** | Alpaca | Crypto/Stocks | ❌ | ❌ | ✅ |
| **BinanceFuturesTorchTradingEnv** | Binance | Crypto | ✅ | ✅ | ❌ |
| **BinanceFuturesSLTPTorchTradingEnv** | Binance | Crypto | ✅ | ✅ | ✅ |
| **BitgetFuturesTorchTradingEnv** | Bitget | Crypto | ✅ | ✅ | ❌ |
| **BitgetFuturesSLTPTorchTradingEnv** | Bitget | Crypto | ✅ | ✅ | ✅ |

## Fractional Position Sizing

**Non-SLTP live environments** (`AlpacaTorchTradingEnv`, `BinanceFuturesTorchTradingEnv`, `BitgetFuturesTorchTradingEnv`) support **fractional position sizing** where action values represent the fraction of balance to allocate.

### How It Works

**Action Interpretation:**
- Action values range from **-1.0 to 1.0** (futures) or **0.0 to 1.0** (long-only)
- **Magnitude** = fraction of balance to allocate (0.5 = 50%, 1.0 = 100%)
- **Sign** = direction (positive = long, negative = short)
- **Zero** = market neutral (close all positions, stay in cash)

**Key Feature**: When adjusting positions (e.g., 100% → 50%), the environment **only trades the delta** (50%), reducing transaction fees.

**Examples:**

```python
# Binance Futures - 5x leverage, $10k balance
from torchtrade.envs.binance import BinanceFuturesTorchTradingEnv, BinanceFuturesTradingEnvConfig

config = BinanceFuturesTradingEnvConfig(
    symbol="BTCUSDT",
    leverage=5,  # Fixed global leverage
    action_levels=[-1.0, -0.5, 0.0, 0.5, 1.0],  # Custom levels
    demo=True  # Use testnet
)
env = BinanceFuturesTorchTradingEnv(config, api_key="...", api_secret="...")

# Action interpretation (similar to offline environments):
# action = -1.0  → 100% short: (balance × 1.0 × 5) / price
# action = -0.5  → 50% short
# action =  0.0  → Market neutral (close all)
# action =  0.5  → 50% long
# action =  1.0  → 100% long
```

```python
# Alpaca - Long-only, no leverage
from torchtrade.envs.alpaca import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig

config = AlpacaTradingEnvConfig(
    symbol="BTC/USD",
    action_levels=[0.0, 0.5, 1.0],  # Only long positions
    paper=True
)
env = AlpacaTorchTradingEnv(config, api_key="...", api_secret="...")

# Action interpretation:
# action = 0.0  → 0% invested (all cash)
# action = 0.5  → 50% of portfolio invested
# action = 1.0  → 100% of portfolio invested
```

### Customizing Action Levels

Same flexibility as offline environments - any list of values in [-1.0, 1.0]:

```python
# Fine-grained (9 actions)
action_levels = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

# More precision near neutral (asymmetric)
action_levels = [-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0]

# Conservative
action_levels = [-0.5, -0.25, 0.0, 0.25, 0.5]

# Long-only with fine control
action_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
```

**Default Values:**
- Binance/Bitget Futures: `[-1.0, -0.5, 0.0, 0.5, 1.0]`
- Alpaca (long-only): `[0.0, 0.5, 1.0]`

### Query-First Pattern

Live environments use a **query-first pattern** for reliability:

1. **Query actual position** from exchange (source of truth)
2. **Calculate target position** based on action and actual balance
3. **Round to exchange constraints** (lot size, step size, min notional)
4. **Trade only the delta** between current and target
5. **Use exchange close APIs** for flat (action=0.0)

This ensures positions stay synchronized with exchange state even if network issues occur.

### Leverage Design (Futures Environments)

For futures environments (`BinanceFuturesTorchTradingEnv`, `BitgetFuturesTorchTradingEnv`), **leverage is a fixed global parameter**, not part of the action space.

**Design Philosophy:**
- **Leverage** = "How much risk?" (configuration/risk management)
- **Action** = "How much to deploy?" (learned policy)

**Benefits:**
1. **Simpler learning**: Smaller action space
2. **Better risk control**: Global leverage constraint
3. **Matches trader workflows**: Set leverage once, size positions with actions

**Dynamic Leverage (Not Currently Implemented):**

Could be added as multi-dimensional actions:

```python
# Future extension (not currently available):
action_space = {
    "position_fraction": Categorical([-1, -0.5, 0, 0.5, 1]),
    "leverage_multiplier": Categorical([1, 3, 5])
}
```

However, **fixed leverage is recommended** for most use cases.

!!! warning "Timeframe Format - Critical for Model Compatibility"
    When specifying `time_frames`, **always use canonical forms**:

    - ✅ **Correct**: `["1Min", "5Min", "15Min", "1Hour", "1Day"]` (Alpaca format)
    - ✅ **Correct**: `["1m", "5m", "15m", "1h", "1d"]` (Binance/Bitget format)
    - ❌ **Wrong**: `["60min"]`, `["60m"]`, `["24hour"]`, `["24h"]`

    **Why this matters:**

    - `time_frames=["60min"]` creates observation key `"market_data_60Minute"`
    - `time_frames=["1hour"]` creates observation key `"market_data_1Hour"`
    - These are **DIFFERENT keys** - your model trained with `"60min"` won't work with config using `"1hour"`

    The framework will issue a warning if you use non-canonical forms. Use the suggested canonical forms to ensure model compatibility and cleaner observation keys.

    **Common conversions:**

    - `60min` / `60m` → use `1Hour` / `1h`
    - `120min` / `120m` → use `2Hour` / `2h`
    - `1440min` → use `1Day` / `1d`
    - `24hour` / `24h` → use `1Day` / `1d`

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

config = AlpacaTradingEnvConfig(
    symbol="BTC/USD",                       # Trading symbol
    time_frames=["1Min", "5Min", "15Min", "1Hour"],  # 1min, 5min, 15min, 1hour
    window_sizes=[12, 8, 8, 24],            # Lookback per timeframe
    execute_on="5Min",                      # Execute every 5 minutes
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
import torch

# Configure environment
config = AlpacaTradingEnvConfig(
    symbol="BTC/USD",
    time_frames=["1Min", "5Min", "15Min"],
    window_sizes=[12, 8, 8],
    execute_on="5Min",
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

config = AlpacaSLTPTradingEnvConfig(
    symbol="BTC/USD",
    time_frames=["1Min", "5Min", "15Min"],
    window_sizes=[12, 8, 8],
    execute_on="5Min",

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

[Binance](https://accounts.binance.com/register?ref=25015935) is a leading cryptocurrency exchange with futures trading and testnet support.

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

[Bitget](https://www.bitget.com/) is a fast-growing cryptocurrency exchange with competitive fees and testnet support. TorchTrade uses [CCXT](https://github.com/ccxt/ccxt) library to interface with Bitget's V2 API.

!!! note "CCXT Symbol Format"
    Bitget environments use CCXT's perpetual swap format: `"BTC/USDT:USDT"` (not `"BTCUSDT"`).
    Format: `{BASE}/{QUOTE}:{SETTLE}` where BASE=BTC, QUOTE=USDT, SETTLE=USDT.

### BitgetFuturesTorchTradingEnv

Futures trading environment for Bitget exchange with configurable margin and position modes.

#### Setup

```bash
# Get API keys from: https://www.bitget.com/en/support/articles/360038859731
# Or use demo trading: https://www.bitget.com/demo-trading
# Create .env file
cat > .env << EOF
BITGETACCESSAPIKEY=your_bitget_api_key
BITGETSECRETKEY=your_bitget_secret_key
BITGETPASSPHRASE=your_bitget_passphrase
EOF
```

#### Configuration

```python
from torchtrade.envs.bitget import (
    BitgetFuturesTorchTradingEnv,
    BitgetFuturesTradingEnvConfig
)
from torchtrade.envs.bitget.futures_order_executor import (
    MarginMode,
    PositionMode,
)
import os
from dotenv import load_dotenv

load_dotenv()

config = BitgetFuturesTradingEnvConfig(
    symbol="BTC/USDT:USDT",                 # CCXT perpetual swap format
    time_frames=["5min", "15min"],          # Timeframes for observations
    window_sizes=[6, 32],                   # Lookback per timeframe
    execute_on="1min",                      # Execute every 1 minute

    # Futures parameters
    product_type="USDT-FUTURES",            # V2 API: USDT-FUTURES, COIN-FUTURES
    leverage=5,                             # Leverage (1-125)
    quantity_per_trade=0.002,               # Position size
    margin_mode=MarginMode.ISOLATED,        # ISOLATED or CROSSED
    position_mode=PositionMode.ONE_WAY,     # ONE_WAY (recommended) or HEDGE

    # Trading mode
    demo=True,                              # Testnet (recommended!)

    # Optional
    feature_preprocessing_fn=None,
    reward_function=None,
)

env = BitgetFuturesTorchTradingEnv(
    config,
    api_key=os.getenv("BITGETACCESSAPIKEY"),
    api_secret=os.getenv("BITGETSECRETKEY"),
    api_passphrase=os.getenv("BITGETPASSPHRASE"),
)
```

#### Margin Modes

**ISOLATED (Recommended)**:
- Each position has separate margin allocation
- If liquidated, only that position's margin is lost
- Lower risk, better for beginners

**CROSSED**:
- All positions share the entire account balance
- Higher risk but more capital efficient
- Advanced users only

```python
from torchtrade.envs.bitget.futures_order_executor import MarginMode

config = BitgetFuturesTradingEnvConfig(
    margin_mode=MarginMode.ISOLATED,  # Safer default
    # margin_mode=MarginMode.CROSSED,  # Advanced
)
```

!!! note "Margin Mode Implementation"
    Bitget's `set_margin_mode()` API doesn't work reliably. TorchTrade sets margin mode per-order by including `marginMode` in each order's parameters. This is the recommended approach per [CCXT issue #21435](https://github.com/ccxt/ccxt/issues/21435). Each order will use your configured margin mode.

#### Position Modes

**ONE_WAY (Recommended)**:
- Single net position per symbol
- Going LONG when SHORT automatically closes the short first
- Simpler position management

**HEDGE**:
- Can hold separate long and short positions simultaneously
- More complex, for advanced hedging strategies

```python
from torchtrade.envs.bitget.futures_order_executor import PositionMode

config = BitgetFuturesTradingEnvConfig(
    position_mode=PositionMode.ONE_WAY,  # Recommended
    # position_mode=PositionMode.HEDGE,   # Advanced
)
```

#### Features
- **CCXT Integration**: Uses CCXT library with Bitget V2 API
- **Leverage**: Up to 125x leverage
- **Testnet**: Safe testing environment ([Demo Trading](https://www.bitget.com/demo-trading))
- **Competitive fees**: Maker 0.02% / Taker 0.06%
- **Flexible Risk Management**: Configurable margin and position modes
- **Alternative to Binance**: Growing liquidity

#### Example Usage

```python
from torchtrade.envs.bitget import (
    BitgetFuturesTorchTradingEnv,
    BitgetFuturesTradingEnvConfig
)
from torchtrade.envs.bitget.futures_order_executor import (
    MarginMode,
    PositionMode,
)
import os
from dotenv import load_dotenv

load_dotenv()

config = BitgetFuturesTradingEnvConfig(
    symbol="BTC/USDT:USDT",                 # CCXT format!
    time_frames=["5min", "15min"],
    window_sizes=[6, 32],
    execute_on="1min",
    leverage=5,
    quantity_per_trade=0.002,
    margin_mode=MarginMode.ISOLATED,        # Safer for testing
    position_mode=PositionMode.ONE_WAY,     # Simpler management
    demo=True,
)

env = BitgetFuturesTorchTradingEnv(
    config,
    api_key=os.getenv("BITGETACCESSAPIKEY"),
    api_secret=os.getenv("BITGETSECRETKEY"),
    api_passphrase=os.getenv("BITGETPASSPHRASE"),
)

# Trading loop
tensordict = env.reset()

for step in range(100):
    action = policy(tensordict)
    tensordict["action"] = action
    tensordict = env.step(tensordict)

    # Monitor account state (10 elements for Bitget futures)
    account = tensordict["account_state"]
    # [cash, position_size, position_value, entry_price, current_price,
    #  unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]
```

### BitgetFuturesSLTPTorchTradingEnv

Bitget futures with stop-loss/take-profit bracket orders.

#### Configuration

```python
from torchtrade.envs.bitget import (
    BitgetFuturesSLTPTorchTradingEnv,
    BitgetFuturesSLTPTradingEnvConfig
)
from torchtrade.envs.bitget.futures_order_executor import (
    MarginMode,
    PositionMode,
)
import os
from dotenv import load_dotenv

load_dotenv()

config = BitgetFuturesSLTPTradingEnvConfig(
    symbol="BTC/USDT:USDT",                 # CCXT perpetual swap format
    time_frames=["5min", "15min"],
    window_sizes=[6, 32],
    execute_on="1min",

    # Bracket orders
    stoploss_levels=(-0.025, -0.05, -0.1),  # -2.5%, -5%, -10%
    takeprofit_levels=(0.05, 0.1, 0.2),     # 5%, 10%, 20%
    include_short_positions=True,           # Enable short bracket orders
    include_hold_action=True,               # Include HOLD action

    # Futures parameters
    product_type="USDT-FUTURES",
    leverage=5,
    quantity_per_trade=0.002,
    margin_mode=MarginMode.ISOLATED,
    position_mode=PositionMode.ONE_WAY,

    demo=True,
)

env = BitgetFuturesSLTPTorchTradingEnv(
    config,
    api_key=os.getenv("BITGETACCESSAPIKEY"),
    api_secret=os.getenv("BITGETSECRETKEY"),
    api_passphrase=os.getenv("BITGETPASSPHRASE"),
)
```

#### Action Space

With the configuration above (3 SL × 3 TP, both long/short):

- **Action 0**: HOLD / Close position
- **Actions 1-9**: LONG with SL/TP combinations (3 × 3)
- **Actions 10-18**: SHORT with SL/TP combinations (3 × 3)

Total: 1 + 2 × (3 × 3) = **19 actions**

For detailed documentation and examples, see:
- [Bitget Examples README](../../examples/live/bitget/README.md)

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
1. [Create an issue](https://github.com/TorchTrade/torchtrade/issues/new) with the exchange name
2. Email us at: torchtradecontact@gmail.com

---

## Next Steps

- **[Offline Environments](offline.md)** - Train on historical data first
- **[Actors](../components/actors.md)** - Alternative policies (RuleBasedActor, LLMActor for live trading)
- **[Transforms](../components/transforms.md)** - Data preprocessing and monitoring
- **[Feature Engineering](../guides/custom-features.md)** - Add technical indicators
- **[Reward Functions](../guides/reward-functions.md)** - Design better rewards
- **[Building Custom Environments](../guides/custom-environment.md)** - Extend TorchTrade

## Support

- 💬 **Questions**: [GitHub Discussions](https://github.com/TorchTrade/torchtrade/discussions)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/TorchTrade/torchtrade/issues)
- 📧 **Email**: torchtradecontact@gmail.com
