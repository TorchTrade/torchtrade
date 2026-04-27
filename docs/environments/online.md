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

[Polymarket](https://polymarket.com/) is a decentralized prediction market on Polygon. TorchTrade currently exposes a single env, `PolymarketBetEnv`, tailored for short-cadence binary markets, Polymarket runs continuous **5-minute, 15-minute, 1-hour, and 4-hour** crypto "up/down" markets (BTC, ETH, SOL) plus daily markets. Each step is an independent bet: place direction, wait for resolution, collect realized payoff, advance to the next market in the series. There is no carried position, so the observation deliberately omits any `account_state`.

!!! info "Starter environment, more to come"
    `PolymarketBetEnv` is intentionally a **starter env** matching the most common Polymarket use case for TorchTrade (rolling binary up/down bets). Polymarket also has multi-strike daily price markets, sports, politics, and longer-horizon markets that benefit from a different env shape. Additional Polymarket envs will be added based on user requests and as new market types appear, open an issue describing the pattern you need.

!!! note "Authentication"
    Polymarket uses a **Polygon private key**, not an API key/secret pair. The key is used to derive CLOB API credentials at runtime.

!!! warning "USDC Required"
    The wallet must hold USDC.e on Polygon to place real orders. Use `dry_run=True` to validate the pipeline without spending capital, `dry_run` works without `py-clob-client` installed.

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
3. Polls Polymarket's **CLOB** at `clob.polymarket.com/midpoint?token_id=…` for each outcome token; the market is resolved once the YES midpoint is `≥ 0.99` and the NO midpoint is `≤ 0.01` (Up won), or vice versa (Down won). The CLOB is used here rather than Gamma's `outcomePrices` because Gamma evicts short-cadence markets within minutes of `endDate` and its prices field is a stale snapshot, not a live mid.
4. Computes realized payoff, a win pays `stake × (1 − fill) / fill`; a loss returns `−stake`.
5. Picks the next active market matching `market_slug_prefix` and returns its `market_state`.

### Discovering markets and slug prefixes

`MarketScanner` is the discovery tool. Use it once to identify the slug prefix that points at the series you want, then plug it into the env. Two query strategies depending on your filters:

- **Browsing mode** (no `slug_prefix` and no `max_time_to_resolution_minutes`), sorts by 24-hour volume descending, surfaces high-volume markets first. Best for "what's hot right now?".
- **Targeting upcoming mode** (either of those is set), sorts by `endDate` ascending and uses Gamma's `end_date_min=now` filter. Required for short-cadence series like `btc-updown-5m-`, which typically have $0 24-hour volume and never make the volume-sorted top page.

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

# Fuzzy keyword filter (case-insensitive, OR-match across question + slug)
scanner = MarketScanner(MarketScannerConfig(keyword=["btc", "bitcoin", "crypto"]))
```

#### CLI flags

The companion CLI is [`examples/broker/polymarket/scan_markets.py`](../../examples/broker/polymarket/scan_markets.py). All flags map 1:1 to fields on `MarketScannerConfig`:

| Flag | Purpose |
|------|---------|
| `--slug-prefix` | **Exact, structural** prefix match on the market slug (case-sensitive). This is the env-side primitive, once you've found a prefix you're happy with, it's the identifier you paste into `PolymarketBetEnvConfig.market_slug_prefix`. Examples: `btc-updown-5m-`, `bitcoin-up-or-down-`, `eth-updown-15m-`. |
| `--keyword` | **Fuzzy, exploratory** substring match (case-insensitive, applied to both `question` and `slug`). Pass one or more terms, a market passes the filter if **any** term appears: `--keyword btc bitcoin crypto` matches anything containing `btc` OR `bitcoin` OR `crypto`. Quote multi-word terms (`--keyword "world cup"`). Use this to find a series you don't already know the prefix of, then read the matching `slug` column to identify the prefix for the env. |
| `--min-volume` | Minimum 24-hour USDC volume. Default `0`. Raise to skip thin markets when browsing. |
| `--min-liquidity` | Minimum order-book liquidity in USDC. Default `0`. |
| `--min-resolution-hours` | Lower bound on time-to-resolution. Default `0`. |
| `--max-resolution-minutes` | Upper bound on time-to-resolution. Default unset. Set to e.g. `30` to surface short-cadence markets that volume-sorted browsing would otherwise miss. |
| `--max` | Cap on rows printed. Default `20`. |

Typical workflow: discover with `--keyword`, identify the slug stem, then verify with `--slug-prefix`:

```bash
# 1. Fuzzy: "what crypto markets exist resolving soon?"
python examples/broker/polymarket/scan_markets.py --keyword btc bitcoin --max-resolution-minutes 30

# 2. Identify a stem in the output (e.g. btc-updown-5m-) and verify
python examples/broker/polymarket/scan_markets.py --slug-prefix btc-updown-5m- --max 5

# 3. Plug that stem into PolymarketBetEnvConfig(market_slug_prefix="btc-updown-5m-")
```

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

### Adding external context (OHLCV, microfeatures, multi-timeframe)

`PolymarketBetEnv`'s default observation is intentionally minimal: a 4-element `market_state = [yes_price, spread, volume_24h, liquidity]`. For "BTC up/down in 5 min?" markets, the obvious thing to add is **live BTC OHLCV** from a spot exchange, the policy needs price action to take an informed view, not just the implied probability already baked into `yes_price`.

The idiomatic TorchRL way to do this is `TransformedEnv` plus a small `Transform` that fetches your data each step and adds it to the observation `TensorDict`. TorchTrade ships [`BinanceOHLCVTransform`](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/envs/transforms/binance_ohlcv.py) for exactly this, wraps `BinanceObservationClass` (no API key required for public OHLCV), pulls multi-timeframe windows each step, and extends the observation spec to match.

#### Example: Binance OHLCV alongside the Polymarket observation

```python
from torchrl.envs import TransformedEnv

from torchtrade.envs.live.polymarket import PolymarketBetEnv, PolymarketBetEnvConfig
from torchtrade.envs.transforms import BinanceOHLCVTransform

config = PolymarketBetEnvConfig(market_slug_prefix="btc-updown-5m-", dry_run=True)
env = TransformedEnv(
    PolymarketBetEnv(config),
    BinanceOHLCVTransform(),  # defaults: BTCUSDT, 1m/5m/15m × 60/30/20
)

td = env.reset()
# td now has:
#   market_state         (4,)
#   ohlcv_1Minute_60    (60, n_features)
#   ohlcv_5Minute_30    (30, n_features)
#   ohlcv_15Minute_20   (20, n_features)
#   terminated, truncated
```

Override the symbol, timeframes, or window sizes to suit your market:

```python
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

env = TransformedEnv(
    PolymarketBetEnv(PolymarketBetEnvConfig(market_slug_prefix="eth-updown-15m-")),
    BinanceOHLCVTransform(
        symbol="ETHUSDT",
        time_frames=[TimeFrame(15, TimeFrameUnit.Minute), TimeFrame(1, TimeFrameUnit.Hour)],
        window_sizes=[40, 24],
        key_prefix="eth_ohlcv",  # → keys ``eth_ohlcv_15Minute_40`` etc.
    ),
)
```

#### Customizing the features (microfeatures / technicals / cross-asset)

`BinanceObservationClass` accepts a `feature_preprocessing_fn(df) -> df` that returns a DataFrame with feature columns prefixed `feature_`. The default is normalized OHLC pct-change. Use it to inject:

- **Technicals**, RSI, MACD, Bollinger bands, etc.
- **Microfeatures**, order-book imbalance, recent trade-flow imbalance, realized variance.
- **Cross-asset signals**, ETH/BTC correlation features, dollar index, anything pulled from another `requests` call.

```python
import numpy as np
from torchtrade.envs.transforms import BinanceOHLCVTransform

def my_features(df):
    df = df.copy()
    df["feature_log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    df["feature_rolling_vol"] = df["feature_log_return"].rolling(20).std().fillna(0)
    df["feature_volume_pct"] = df["volume"] / df["volume"].rolling(20).mean()
    return df.dropna()

augment = BinanceOHLCVTransform(feature_preprocessing_fn=my_features)
```

#### Other data sources

For purely on-chain signals, custom microfeatures pipelines, or non-Binance exchanges, write your own `Transform` following the same pattern, `_reset`, `_step`, and `transform_observation_spec`. The `BinanceOHLCVTransform` source ([`torchtrade/envs/transforms/binance_ohlcv.py`](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/envs/transforms/binance_ohlcv.py)) is short enough to read and adapt.

!!! note "Roadmap: text and news context"
    A natural next step for `PolymarketBetEnv` is augmenting the observation with **text-based signals** that the LLM actor can reason over directly: real-time news headlines (e.g. via the X / Twitter API or a news aggregator), on-chain whale-flow alerts, macro-event feeds, and similar. The same `Transform` skeleton applies, just with non-numeric output written into the TensorDict (e.g. as `NonTensorData` strings). Open an issue or PR with the data source you want and we can ship a `NewsContextTransform` (or similar) the same way `BinanceOHLCVTransform` was added.

### Driving the env with an LLM actor

TorchTrade ships [`FrontierLLMActor`](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/actor/frontier_llm_actor.py) (OpenAI-compatible) and [`LocalLLMActor`](https://github.com/TorchTrade/torchtrade/blob/main/torchtrade/actor/local_llm_actor.py) (local HF model via vLLM). Both work directly with `PolymarketBetEnv`, pass an empty `account_state_labels` to skip that prompt section, point `market_data_keys` at `"market_state"`, and override the action descriptions to use binary up/down language:

```python
import os
import torch
from dotenv import load_dotenv

from torchtrade.actor import FrontierLLMActor
from torchtrade.envs.live.polymarket import (
    PolymarketBetEnv,
    PolymarketBetEnvConfig,
)

load_dotenv()

env = PolymarketBetEnv(
    PolymarketBetEnvConfig(
        market_slug_prefix="btc-updown-5m-",
        max_steps=4,
        dry_run=True,
    ),
)
actor = FrontierLLMActor(
    model="gpt-5-nano",                              # any OpenAI-compatible model
    market_data_keys=["market_state"],               # the env's only obs key
    account_state_labels=[],                         # PolymarketBetEnv has no account_state
    action_levels=[0, 1],                            # binary action space
    action_descriptions=[                            # override "target exposure" wording
        "Action 0 → bet DOWN (price will be lower at resolution)",
        "Action 1 → bet UP (price will be higher at resolution)",
    ],
    feature_keys=["yes_price", "spread", "volume_24h", "liquidity"],
    symbol="BTC up/down (5 min)",
    execute_on="5Minute",
)

td = env.reset()
while not bool(td.get("done", torch.zeros(1, dtype=torch.bool)).item()):
    td = actor(td)                                   # writes "action" into td
    td = env.step(td)["next"]
env.close()
```

The actor expects responses in `<answer>N</answer>` format (handled by `BaseLLMActor._extract_action`); models that don't follow it default to action 0. Set `debug=True` on the actor to print the system prompt, user prompt, and raw model response each step, useful while tuning.

#### Combining with `BinanceOHLCVTransform`

The transform exposes its OHLCV windows under keys like `ohlcv_1Minute_60`. Add those to `market_data_keys` and the actor will render them alongside `market_state` using the standard 2D-window layout (one row per bar, one column per feature):

```python
from torchrl.envs import TransformedEnv
from torchtrade.envs.transforms import BinanceOHLCVTransform

env = TransformedEnv(
    PolymarketBetEnv(PolymarketBetEnvConfig(market_slug_prefix="btc-updown-5m-")),
    BinanceOHLCVTransform(),  # defaults: 1m/5m/15m × 60/30/20
)
actor = FrontierLLMActor(
    model="gpt-5-nano",
    market_data_keys=[
        "market_state",
        "ohlcv_1Minute_60",
        "ohlcv_5Minute_30",
        "ohlcv_15Minute_20",
    ],
    account_state_labels=[],
    action_levels=[0, 1],
    action_descriptions=[
        "Action 0 → bet DOWN",
        "Action 1 → bet UP",
    ],
)
```

For local inference (no API key required), swap `FrontierLLMActor` for `LocalLLMActor`, the constructor kwargs are identical.

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

# Polymarket (Polygon wallet, fund with USDC.e)
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
