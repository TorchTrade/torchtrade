# Online Environments

Online environments connect to real trading APIs for paper trading or live execution. They provide the same TorchTrade interface as [offline environments](offline.md) but fetch real-time market data from exchanges. Each offline environment has a corresponding live counterpart:

| Offline (Training) | Live (Deployment) |
|---------------------|-------------------|
| SequentialTradingEnv (spot, `leverage=1`) | AlpacaTorchTradingEnv |
| SequentialTradingEnv (futures, `leverage>1`) | BinanceFutures / BitgetFutures / BybitFuturesTorchTradingEnv |
| SequentialTradingEnvSLTP (spot) | AlpacaSLTPTorchTradingEnv |
| SequentialTradingEnvSLTP (futures) | BinanceFuturesSLTP / BitgetFuturesSLTP / BybitFuturesSLTPTorchTradingEnv |
| OneStepTradingEnv (spot) | AlpacaSLTPTorchTradingEnv |
| OneStepTradingEnv (futures) | BinanceFuturesSLTP / BitgetFuturesSLTP / BybitFuturesSLTPTorchTradingEnv |

**Supported Exchanges:**

- **[Alpaca](https://alpaca.markets/)** - Commission-free US stocks and crypto with paper trading
- **[Binance](https://accounts.binance.com/register?ref=25015935)** - Cryptocurrency futures with high leverage and testnet
- **[Bitget](https://www.bitget.com/)** - Cryptocurrency futures with competitive fees and testnet
- **[Bybit](https://www.bybit.com/)** - Cryptocurrency derivatives with native bracket orders and testnet

## Overview

| Environment | Exchange | Asset Type | Futures | Leverage | Bracket Orders |
|-------------|----------|------------|---------|----------|----------------|
| **AlpacaTorchTradingEnv** | Alpaca | Crypto/Stocks | - | - | - |
| **AlpacaSLTPTorchTradingEnv** | Alpaca | Crypto/Stocks | - | - | Yes |
| **BinanceFuturesTorchTradingEnv** | Binance | Crypto | Yes | Yes | - |
| **BinanceFuturesSLTPTorchTradingEnv** | Binance | Crypto | Yes | Yes | Yes |
| **BitgetFuturesTorchTradingEnv** | Bitget | Crypto | Yes | Yes | - |
| **BitgetFuturesSLTPTorchTradingEnv** | Bitget | Crypto | Yes | Yes | Yes |
| **BybitFuturesTorchTradingEnv** | Bybit | Crypto | Yes | Yes | - |
| **BybitFuturesSLTPTorchTradingEnv** | Bybit | Crypto | Yes | Yes | Yes |

## Fractional Position Sizing

Non-SLTP environments use `action_levels` for fractional position sizing — same as [offline environments](offline.md#fractional-position-sizing). Action values in [-1.0, 1.0] represent the fraction of balance to allocate. When adjusting positions (e.g., 100% to 50%), the environment only trades the delta, reducing transaction fees.

Live environments use a **query-first pattern**: they query the actual exchange position (source of truth), calculate the target based on the action, round to exchange constraints (lot size, min notional), and trade only the delta.

!!! warning "Timeframe Format - Critical for Model Compatibility"
    Always use canonical timeframe forms:

    - Alpaca: `["1Min", "5Min", "15Min", "1Hour", "1Day"]`
    - Binance/Bitget/Bybit: `["1m", "5m", "15m", "1h", "1d"]`
    - **Wrong**: `["60min"]`, `["60m"]`, `["24hour"]` — these create different observation keys and break model compatibility.

---

## Alpaca Environments

[Alpaca](https://alpaca.markets/) provides commission-free trading for US stocks and cryptocurrencies with paper trading support.

### AlpacaTorchTradingEnv

```python
from torchtrade.envs.alpaca import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig

config = AlpacaTradingEnvConfig(
    symbol="BTC/USD",
    time_frames=["1Min", "5Min", "15Min", "1Hour"],
    window_sizes=[12, 8, 8, 24],
    execute_on="5Min",
    paper=True,  # Paper trading (recommended!)
)

env = AlpacaTorchTradingEnv(config)
```

### AlpacaSLTPTorchTradingEnv

```python
from torchtrade.envs.alpaca import AlpacaSLTPTorchTradingEnv, AlpacaSLTPTradingEnvConfig

config = AlpacaSLTPTradingEnvConfig(
    symbol="BTC/USD",
    time_frames=["1Min", "5Min", "15Min"],
    window_sizes=[12, 8, 8],
    execute_on="5Min",
    stoploss_levels=[-0.02, -0.05],
    takeprofit_levels=[0.05, 0.10],
    paper=True,
)

env = AlpacaSLTPTorchTradingEnv(config)
# Action space: HOLD + 4 SL/TP combinations = 5 actions
```

---

## Binance Environments

[Binance](https://accounts.binance.com/register?ref=25015935) provides cryptocurrency futures trading with up to 125x leverage and testnet support.

### BinanceFuturesTorchTradingEnv

```python
from torchtrade.envs.binance import BinanceFuturesTorchTradingEnv, BinanceFuturesTradingEnvConfig

config = BinanceFuturesTradingEnvConfig(
    symbol="BTCUSDT",
    intervals=["1m", "5m", "15m"],
    window_sizes=[12, 8, 8],
    execute_on="1m",
    leverage=5,
    quantity_per_trade=0.01,
    demo=True,  # Testnet (recommended!)
)

env = BinanceFuturesTorchTradingEnv(config)
```

### BinanceFuturesSLTPTorchTradingEnv

```python
from torchtrade.envs.binance import BinanceFuturesSLTPTorchTradingEnv, BinanceFuturesSLTPTradingEnvConfig

config = BinanceFuturesSLTPTradingEnvConfig(
    symbol="BTCUSDT",
    intervals=["1m", "5m"],
    window_sizes=[12, 8],
    execute_on="1m",
    stoploss_levels=[-0.02, -0.05],
    takeprofit_levels=[0.03, 0.06, 0.10],
    leverage=5,
    quantity_per_trade=0.01,
    demo=True,
)

env = BinanceFuturesSLTPTorchTradingEnv(config)
# Action space: HOLD + 2×(2 SL × 3 TP) = 13 actions (long + short)
```

---

## Bitget Environments

[Bitget](https://www.bitget.com/) provides cryptocurrency futures trading with competitive fees and testnet support. TorchTrade uses [CCXT](https://github.com/ccxt/ccxt) to interface with Bitget's V2 API.

!!! note "CCXT Symbol Format"
    Bitget uses CCXT's perpetual swap format: `"BTC/USDT:USDT"` (not `"BTCUSDT"`).

### BitgetFuturesTorchTradingEnv

```python
from torchtrade.envs.bitget import BitgetFuturesTorchTradingEnv, BitgetFuturesTradingEnvConfig
from torchtrade.envs.bitget.futures_order_executor import MarginMode, PositionMode
import os
from dotenv import load_dotenv

load_dotenv()

config = BitgetFuturesTradingEnvConfig(
    symbol="BTC/USDT:USDT",
    time_frames=["5min", "15min"],
    window_sizes=[6, 32],
    execute_on="1min",
    product_type="USDT-FUTURES",
    leverage=5,
    quantity_per_trade=0.002,
    margin_mode=MarginMode.ISOLATED,     # ISOLATED (safer) or CROSSED
    position_mode=PositionMode.ONE_WAY,  # ONE_WAY (simpler) or HEDGE
    demo=True,
)

env = BitgetFuturesTorchTradingEnv(
    config,
    api_key=os.getenv("BITGETACCESSAPIKEY"),
    api_secret=os.getenv("BITGETSECRETKEY"),
    api_passphrase=os.getenv("BITGETPASSPHRASE"),
)
```

### BitgetFuturesSLTPTorchTradingEnv

```python
from torchtrade.envs.bitget import BitgetFuturesSLTPTorchTradingEnv, BitgetFuturesSLTPTradingEnvConfig
from torchtrade.envs.bitget.futures_order_executor import MarginMode, PositionMode
import os
from dotenv import load_dotenv

load_dotenv()

config = BitgetFuturesSLTPTradingEnvConfig(
    symbol="BTC/USDT:USDT",
    time_frames=["5min", "15min"],
    window_sizes=[6, 32],
    execute_on="1min",
    stoploss_levels=[-0.025, -0.05, -0.1],
    takeprofit_levels=[0.05, 0.1, 0.2],
    include_hold_action=True,
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
# Action space: HOLD + 2×(3 SL × 3 TP) = 19 actions (long + short)
```

---

## Bybit Environments

[Bybit](https://www.bybit.com/) provides cryptocurrency derivatives trading with native bracket order support (stop-loss/take-profit set directly on orders). TorchTrade uses [pybit](https://github.com/bybit-exchange/pybit), Bybit's official Python SDK, for direct v5 API access.

!!! note "Symbol Format"
    Bybit uses simple concatenated symbols: `"BTCUSDT"` (not `"BTC/USDT:USDT"`).

### BybitFuturesTorchTradingEnv

```python
from torchtrade.envs.bybit import BybitFuturesTorchTradingEnv, BybitFuturesTradingEnvConfig
from torchtrade.envs.bybit import MarginMode, PositionMode
import os
from dotenv import load_dotenv

load_dotenv()

config = BybitFuturesTradingEnvConfig(
    symbol="BTCUSDT",
    time_frames=["5m", "15m"],
    window_sizes=[6, 32],
    execute_on="1m",
    leverage=5,
    margin_mode=MarginMode.ISOLATED,     # ISOLATED (safer) or CROSSED
    position_mode=PositionMode.ONE_WAY,  # ONE_WAY (simpler) or HEDGE
    demo=True,  # Testnet (recommended!)
)

env = BybitFuturesTorchTradingEnv(
    config,
    api_key=os.getenv("BYBIT_API_KEY"),
    api_secret=os.getenv("BYBIT_API_SECRET"),
)
```

### BybitFuturesSLTPTorchTradingEnv

```python
from torchtrade.envs.bybit import BybitFuturesSLTPTorchTradingEnv, BybitFuturesSLTPTradingEnvConfig
from torchtrade.envs.bybit import MarginMode, PositionMode
import os
from dotenv import load_dotenv

load_dotenv()

config = BybitFuturesSLTPTradingEnvConfig(
    symbol="BTCUSDT",
    time_frames=["5m", "15m"],
    window_sizes=[6, 32],
    execute_on="1m",
    stoploss_levels=(-0.025, -0.05, -0.1),
    takeprofit_levels=(0.05, 0.1, 0.2),
    include_hold_action=True,
    leverage=5,
    quantity_per_trade=0.002,
    margin_mode=MarginMode.ISOLATED,
    position_mode=PositionMode.ONE_WAY,
    demo=True,
)

env = BybitFuturesSLTPTorchTradingEnv(
    config,
    api_key=os.getenv("BYBIT_API_KEY"),
    api_secret=os.getenv("BYBIT_API_SECRET"),
)
# Action space: HOLD + 2×(3 SL × 3 TP) = 19 actions (long + short)
```

---

## API Key Setup

Each exchange requires API keys stored in a `.env` file:

```bash
# Alpaca (https://alpaca.markets/signup)
API_KEY=your_alpaca_api_key
SECRET_KEY=your_alpaca_secret_key

# Binance (https://www.binance.com/en/my/settings/api-management)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# Bitget (https://www.bitget.com/en/support/articles/360038859731)
BITGETACCESSAPIKEY=your_bitget_api_key
BITGETSECRETKEY=your_bitget_secret_key
BITGETPASSPHRASE=your_bitget_passphrase

# Bybit (https://www.bybit.com/app/user/api-management)
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret
```

!!! warning "Always Start with Paper/Testnet Trading"
    Set `paper=True` (Alpaca) or `demo=True` (Binance/Bitget/Bybit) before using real funds. Start with low leverage (2-5x) for futures.

---

## Exchange Comparison

| Feature | Alpaca | Binance | Bitget | Bybit |
|---------|--------|---------|--------|-------|
| **Asset Types** | Stocks, Crypto | Crypto | Crypto | Crypto |
| **Futures** | - | Yes | Yes | Yes |
| **Max Leverage** | 1x | 125x | 125x | 100x |
| **Paper Trading** | Yes | Yes (Testnet) | Yes (Testnet) | Yes (Testnet) |
| **Commission** | Free | 0.02%/0.04% | 0.02%/0.06% | 0.02%/0.055% |
| **SDK** | Alpaca SDK | CCXT | CCXT | pybit (native) |

---

## Requesting New Exchanges

Need support for another exchange (OKX, Interactive Brokers, etc.)? [Create an issue](https://github.com/TorchTrade/torchtrade/issues/new) or email us at torchtradecontact@gmail.com.
