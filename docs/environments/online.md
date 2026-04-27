# Online Environments

Online environments connect to real trading APIs for paper trading or live execution. They provide the same TorchTrade interface as [offline environments](offline.md) but fetch real-time market data from exchanges. Each offline environment has a corresponding live counterpart:

| Offline (Training) | Live (Deployment) |
|---------------------|-------------------|
| SequentialTradingEnv (spot, `leverage=1`) | AlpacaTorchTradingEnv |
| SequentialTradingEnv (futures, `leverage>1`) | BinanceFutures / BitgetFutures / BybitFutures / OKXFuturesTorchTradingEnv |
| SequentialTradingEnvSLTP (spot) | AlpacaSLTPTorchTradingEnv |
| SequentialTradingEnvSLTP (futures) | BinanceFuturesSLTP / BitgetFuturesSLTP / BybitFuturesSLTP / OKXFuturesSLTPTorchTradingEnv |
| OneStepTradingEnv (spot) | AlpacaSLTPTorchTradingEnv |
| OneStepTradingEnv (futures) | BinanceFuturesSLTP / BitgetFuturesSLTP / BybitFuturesSLTP / OKXFuturesSLTPTorchTradingEnv |

**Supported Exchanges:**

- **[Alpaca](https://alpaca.markets/)** - Commission-free US stocks and crypto with paper trading
- **[Binance](https://accounts.binance.com/register?ref=25015935)** - Cryptocurrency futures with high leverage and testnet
- **[Bitget](https://www.bitget.com/)** - Cryptocurrency futures with competitive fees and testnet
- **[Bybit](https://www.bybit.com/)** - Cryptocurrency derivatives with native bracket orders and testnet
- **[OKX](https://www.okx.com/)** - Global cryptocurrency exchange with bracket orders via attachAlgoOrds and demo trading
- **[Polymarket](https://polymarket.com/)** - Decentralized prediction markets on Polygon (CLOB), with dry-run paper trading

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
| **OKXFuturesTorchTradingEnv** | OKX | Crypto | Yes | Yes | - |
| **OKXFuturesSLTPTorchTradingEnv** | OKX | Crypto | Yes | Yes | Yes |
| **PolymarketBetEnv** | Polymarket | Prediction markets | - | - | - |

## Fractional Position Sizing

Non-SLTP environments use `action_levels` for fractional position sizing — same as [offline environments](offline.md#fractional-position-sizing). Action values in [-1.0, 1.0] represent the fraction of balance to allocate. When adjusting positions (e.g., 100% to 50%), the environment only trades the delta, reducing transaction fees.

Live environments use a **query-first pattern**: they query the actual exchange position (source of truth), calculate the target based on the action, round to exchange constraints (lot size, min notional), and trade only the delta.

!!! warning "Timeframe Format - Critical for Model Compatibility"
    Always use canonical timeframe forms:

    - Alpaca: `["1Min", "5Min", "15Min", "1Hour", "1Day"]`
    - Binance/Bitget/Bybit/OKX: `["1m", "5m", "15m", "1h", "1d"]`
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
    quantity_per_trade=0.01,  # 0.01 BTC per trade (quantity mode)
    trade_mode="quantity",    # "quantity", "notional", or "fractional"
    demo=True,
)

# For consistent USD-sized positions regardless of BTC price:
# config = BinanceFuturesSLTPTradingEnvConfig(
#     ...
#     quantity_per_trade=500.0,  # $500 per trade
#     trade_mode="notional",
# )

# For fractional portfolio-based sizing (10% of account per trade):
# config = BinanceFuturesSLTPTradingEnvConfig(
#     ...
#     trade_mode="fractional",
#     position_fraction=0.1,
# )

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
    quantity_per_trade=0.002,  # 0.002 BTC per trade (quantity mode)
    trade_mode="quantity",     # "quantity", "notional", or "fractional"
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
    quantity_per_trade=0.002,  # 0.002 BTC per trade (quantity mode)
    trade_mode="quantity",     # "quantity", "notional", or "fractional"
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

## OKX Environments

[OKX](https://www.okx.com/) provides global cryptocurrency trading with up to 125x leverage and demo trading support. TorchTrade uses [python-okx](https://pypi.org/project/python-okx/), OKX's official Python SDK. Bracket orders (SL/TP) are placed atomically via OKX's `attachAlgoOrds` parameter.

!!! note "Symbol Format"
    OKX uses dash-separated symbols with swap suffix: `"BTC-USDT-SWAP"`. The environment auto-normalizes common formats (`"BTCUSDT"`, `"BTC/USDT"`, `"BTC-USDT"`) to the correct OKX format.

!!! note "Passphrase Required"
    OKX API requires a passphrase in addition to API key and secret.

### OKXFuturesTorchTradingEnv

```python
from torchtrade.envs.live.okx import OKXFuturesTorchTradingEnv, OKXFuturesTradingEnvConfig
from torchtrade.envs.live.okx import MarginMode, PositionMode
import os
from dotenv import load_dotenv

load_dotenv()

config = OKXFuturesTradingEnvConfig(
    symbol="BTC-USDT-SWAP",
    time_frames=["5m", "15m"],
    window_sizes=[6, 32],
    execute_on="1m",
    leverage=5,
    margin_mode=MarginMode.ISOLATED,  # ISOLATED (safer) or CROSS
    position_mode=PositionMode.NET,   # NET (simpler) or LONG_SHORT
    demo=True,  # Demo trading (recommended!)
)

env = OKXFuturesTorchTradingEnv(
    config,
    api_key=os.getenv("OKX_API_KEY"),
    api_secret=os.getenv("OKX_API_SECRET"),
    passphrase=os.getenv("OKX_PASSPHRASE"),
)
```

### OKXFuturesSLTPTorchTradingEnv

```python
from torchtrade.envs.live.okx import OKXFuturesSLTPTorchTradingEnv, OKXFuturesSLTPTradingEnvConfig
from torchtrade.envs.live.okx import MarginMode, PositionMode
import os
from dotenv import load_dotenv

load_dotenv()

config = OKXFuturesSLTPTradingEnvConfig(
    symbol="BTC-USDT-SWAP",
    time_frames=["5m", "15m"],
    window_sizes=[6, 32],
    execute_on="1m",
    stoploss_levels=(-0.025, -0.05, -0.1),
    takeprofit_levels=(0.05, 0.1, 0.2),
    include_hold_action=True,
    leverage=5,
    quantity_per_trade=0.002,  # 0.002 BTC per trade (quantity mode)
    trade_mode="quantity",     # "quantity", "notional", or "fractional"
    margin_mode=MarginMode.ISOLATED,
    position_mode=PositionMode.NET,
    demo=True,
)

env = OKXFuturesSLTPTorchTradingEnv(
    config,
    api_key=os.getenv("OKX_API_KEY"),
    api_secret=os.getenv("OKX_API_SECRET"),
    passphrase=os.getenv("OKX_PASSPHRASE"),
)
# Action space: HOLD + 2×(3 SL × 3 TP) = 19 actions (long + short)
```

---

## Polymarket Environment

[Polymarket](https://polymarket.com/) is a decentralized prediction market on Polygon. TorchTrade exposes a single env, `PolymarketBetEnv`, tailored for short-cadence binary markets — Polymarket runs continuous **5-minute, 15-minute, 1-hour, and 4-hour** crypto "up/down" markets (BTC, ETH, SOL) plus daily markets. Each step is an independent bet: place direction, wait for resolution, collect realized payoff, advance to the next market in the series. There is no carried position, so the observation deliberately omits any `account_state`.

!!! note "Authentication"
    Polymarket uses a **Polygon private key**, not an API key/secret pair. The key is used to derive CLOB API credentials at runtime.

!!! warning "USDC Required"
    The wallet must hold USDC.e on Polygon to place real orders. Use `dry_run=True` to validate the pipeline without spending capital — `dry_run` works without `py-clob-client` installed.

### PolymarketBetEnv

```python
from torchtrade.envs.live.polymarket import PolymarketBetEnv, PolymarketBetEnvConfig
import os
from dotenv import load_dotenv

load_dotenv()

config = PolymarketBetEnvConfig(
    market_slug_prefix="btc-updown-5m-",  # discover via scan_markets.py
    bet_fraction=0.01,                    # stake 1 % of cash per bet
    max_steps=10,                         # 10 bets per episode
    initial_cash=1_000.0,                 # for dry-run accounting
    dry_run=True,                         # paper trading (recommended!)
)

env = PolymarketBetEnv(
    config=config,
    private_key=os.getenv("POLYGON_PRIVATE_KEY", ""),
)
# observation_spec: {"market_state": (4,)}  → [yes_price, spread, vol_24h, liquidity]
# action_spec:      Categorical(2)            → 0 = Down, 1 = Up
```

Each `step()`:
1. Submits the bet on the current market (skipped in `dry_run`).
2. Sleeps until the market's `endDate` plus a small grace period.
3. Fetches the resolved outcome from Gamma (`outcomePrices` snaps to `[1, 0]` or `[0, 1]`).
4. Computes realized payoff — a win pays `stake × (1 − fill) / fill`; a loss returns `−stake`.
5. Picks the next active market matching `market_slug_prefix` and returns its `market_state`.

### Discovering markets and slug prefixes

`MarketScanner` is the discovery tool. Use it once to identify the slug prefix that points at the series you want, then plug it into the env. Live scanning a fast-cadence series at the API level uses Gamma's `endDate`-ascending sort and the `end_date_min=now` filter — short-cadence markets typically have $0 24-hour volume, so the volume-sorted browsing path would miss them.

```python
from torchtrade.envs.live.polymarket import MarketScanner, MarketScannerConfig

# Generic discovery
scanner = MarketScanner(MarketScannerConfig(
    min_volume_24h=10_000.0,
    max_markets=10,
))
for m in scanner.scan():
    print(m.slug, m.yes_price, m.volume_24h)

# Lock to a series (5-minute Bitcoin)
scanner = MarketScanner(MarketScannerConfig(slug_prefix="btc-updown-5m-"))
for m in scanner.scan():
    print(m.slug, m.end_date, m.liquidity)

# Find any short-cadence market resolving in <30 minutes
scanner = MarketScanner(MarketScannerConfig(max_time_to_resolution_minutes=30))
```

The companion CLI is [`examples/broker/polymarket/scan_markets.py`](../../examples/broker/polymarket/scan_markets.py); it accepts `--slug-prefix`, `--keyword`, `--max-resolution-minutes`, and friends.

### Currently active short-cadence series

| Series | Cadence | Slug prefix |
|--------|---------|-------------|
| BTC up/down (5 min)   | 5 min  | `btc-updown-5m-` |
| BTC up/down (15 min)  | 15 min | `btc-updown-15m-` |
| BTC up/down (1 hour)  | 1 h    | `bitcoin-up-or-down-` |
| BTC up/down (4 hour)  | 4 h    | `btc-updown-4h-` |
| ETH up/down (5 min)   | 5 min  | `eth-updown-5m-` |
| ETH up/down (15 min)  | 15 min | `eth-updown-15m-` |
| SOL up/down (5 min)   | 5 min  | `sol-updown-5m-` |
| BTC daily up/down     | 1 day  | `bitcoin-up-or-down-on-` |

Polymarket adds and renames series occasionally; re-running `scan_markets.py` to verify the prefix before configuring the env is the safe move.

---

## Position Sizing (SLTP Environments)

All live SLTP environments support three position sizing modes, matching the offline SLTP environments for train-deploy consistency.

| Mode | Config field | Formula | Use case |
|------|-------------|---------|----------|
| `"fractional"` | `position_fraction` | `account_balance × fraction × leverage / price` | Adaptive sizing (scales with account) |
| `"notional"` | `quantity_per_trade` | `quantity_per_trade / price` | Fixed USD per trade |
| `"quantity"` | `quantity_per_trade` | `quantity_per_trade` directly | Fixed base-asset units per trade |

```python
# Fractional: risk 10% of account per bracket order
config = BinanceFuturesSLTPTradingEnvConfig(
    trade_mode="fractional",
    position_fraction=0.1,       # 10% of account balance
    leverage=5,
    ...
)

# Notional: always trade $500 USD worth
config = BinanceFuturesSLTPTradingEnvConfig(
    trade_mode="notional",
    quantity_per_trade=500.0,    # $500 per trade
    ...
)

# Quantity: always trade exactly 0.001 BTC (default)
config = BinanceFuturesSLTPTradingEnvConfig(
    trade_mode="quantity",       # default
    quantity_per_trade=0.001,
    ...
)
```

!!! note "Alpaca"
    Alpaca SLTP supports `"fractional"` and `"notional"` modes only. `"quantity"` mode is not supported because Alpaca's bracket order API requires dollar amounts.

### Position Locking

All live SLTP environments support `lock_position_until_sltp`. When enabled, agent actions are ignored while a position is open — positions can only exit via the exchange's bracket orders (SL/TP triggers).

```python
config = BinanceFuturesSLTPTradingEnvConfig(
    lock_position_until_sltp=True,  # Positions exit only via SL/TP
    ...
)
```

This is useful for deploying policies trained on `OneStepTradingEnv` where positions are inherently locked to a single decision. See the [offline Position Locking docs](offline.md#position-locking) for details.

### Replay Mode (Historical Data Simulation)

All live environments (both SLTP and non-SLTP) support replaying historical data through the live pipeline using `ReplayObserver` and `ReplayOrderExecutor`. This is useful for:

- Verifying that the live pipeline produces identical results to backtesting
- Data-driven integration tests with real price data
- Catching feature ordering, normalization, or action mapping bugs before deployment

```python
from torchtrade.envs.replay import ReplayObserver, ReplayOrderExecutor

# Create replay components from historical data
executor = ReplayOrderExecutor(initial_balance=10000, leverage=5, transaction_fee=0.0004)
observer = ReplayObserver(
    df=historical_df,  # DataFrame with timestamp, open, high, low, close, volume
    time_frames=config.time_frames,
    window_sizes=config.window_sizes,
    execute_on=config.execute_on,
    feature_preprocessing_fn=feature_fn,
    executor=executor,
)

# Inject into any live env — no code changes needed
env = BinanceFuturesSLTPTorchTradingEnv(config, observer=observer, trader=executor)
# Also works with non-SLTP envs:
# env = BinanceFuturesTorchTradingEnv(config, observer=observer, trader=executor)
td = env.reset()
# Run episode with historical data through the exact live pipeline
```

The `ReplayOrderExecutor` simulates bracket order execution with intrabar SL/TP detection (checks high/low, SL before TP).

**End-of-data handling:** The observer raises `StopIteration` when historical data is exhausted. Wrap your evaluation loop accordingly:

```python
try:
    while True:
        action = policy(td)
        td = env.step(action)["next"]
except StopIteration:
    pass  # End of historical data
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

# OKX (https://www.okx.com/account/my-api)
OKX_API_KEY=your_okx_api_key
OKX_API_SECRET=your_okx_api_secret
OKX_PASSPHRASE=your_okx_passphrase

# Polymarket (Polygon wallet — fund with USDC.e)
POLYGON_PRIVATE_KEY=your_polygon_wallet_private_key
```

!!! warning "Always Start with Paper/Testnet Trading"
    Set `paper=True` (Alpaca), `demo=True` (Binance/Bitget/Bybit/OKX), or `dry_run=True` (Polymarket) before using real funds. Start with low leverage (2-5x) for futures.

---

## Exchange Comparison

| Feature | Alpaca | Binance | Bitget | Bybit | OKX | Polymarket |
|---------|--------|---------|--------|-------|-----|------------|
| **Asset Types** | Stocks, Crypto | Crypto | Crypto | Crypto | Crypto | Prediction markets |
| **Futures** | - | Yes | Yes | Yes | Yes | - |
| **Max Leverage** | 1x | 125x | 125x | 100x | 125x | 1x |
| **Paper Trading** | Yes | Yes (Testnet) | Yes (Testnet) | Yes (Testnet) | Yes (Demo) | Yes (dry_run) |
| **Commission** | Free | 0.02%/0.04% | 0.02%/0.06% | 0.02%/0.055% | 0.02%/0.05% | Variable (CLOB) |
| **SDK** | Alpaca SDK | CCXT | CCXT | pybit (native) | python-okx (native) | py-clob-client |

---

## Requesting New Exchanges

Need support for another exchange (Interactive Brokers, Kraken, etc.)? [Create an issue](https://github.com/TorchTrade/torchtrade/issues/new) or email us at torchtradecontact@gmail.com.
