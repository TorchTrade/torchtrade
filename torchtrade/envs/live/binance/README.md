# Binance Futures Trading Environment

Live trading integration with Binance for crypto futures markets (USDT-margined).

## Files

- **`base.py`**: Base Binance Futures environment
- **`observation.py`**: Binance-specific observation handling
- **`order_executor.py`**: Order execution for Binance Futures API
- **`env.py`**: Main Binance Futures environment
- **`env_sltp.py`**: Binance environment with SL/TP
- **`utils.py`**: Helper functions

## Quick Start

```python
import os

from torchtrade.envs.core.common_types import MarginType
from torchtrade.envs.live.binance.env import BinanceFuturesTorchTradingEnv, BinanceFuturesTradingEnvConfig
from torchtrade.envs.utils import TimeFrame, TimeFrameUnit

config = BinanceFuturesTradingEnvConfig(
    symbol="BTCUSDT",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
    demo=True,  # Use testnet first!
    leverage=10.0,
    margin_type=MarginType.ISOLATED,
)

env = BinanceFuturesTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["BINANCE_API_KEY"],
    api_secret=os.environ["BINANCE_SECRET_KEY"],
)
obs = env.reset()
```

## Features

- **Leverage Trading**: Up to 125x leverage (use responsibly!)
- **Isolated/Cross Margin**: Choose margin mode
- **Funding Fees**: Realistic funding fee simulation
- **Liquidation**: Automatic liquidation handling
- **Testnet**: Safe testing environment with fake funds

## Configuration

```python
from torchtrade.envs.core.common_types import MarginType
from torchtrade.envs.core.live import ObservationFailurePolicy
from dataclasses import dataclass

@dataclass
class BinanceFuturesTradingEnvConfig:
    symbol = 'BTCUSDT'
    time_frames = '1Hour'
    window_sizes = 10
    execute_on = '1Hour'
    leverage = 1
    margin_type = MarginType.ISOLATED
    action_levels = None
    done_on_bankruptcy = True
    bankrupt_threshold = 0.1
    demo = True
    seed = 42
    include_base_features = False
    close_position_on_init = True
    close_position_on_reset = False
    observation_failure_policy = ObservationFailurePolicy.HALT
    # Credentials: Env(config, api_key=..., api_secret=...)
```

## Testnet vs Mainnet

**Testnet** (recommended for development):
```python
from torchtrade.envs.live.binance import BinanceFuturesTradingEnvConfig

config = BinanceFuturesTradingEnvConfig(
    demo=True,  # Fake funds
)
```

Get testnet API keys: https://testnet.binancefuture.com/

**Mainnet** (real money):
```python
from torchtrade.envs.live.binance import BinanceFuturesTradingEnvConfig

config = BinanceFuturesTradingEnvConfig(
    demo=False,  # Real trading!
)
```

## Margin Types

### Isolated Margin
- Each position has separate margin
- Liquidation affects only that position
- Lower risk, position-specific leverage

```python
from torchtrade.envs.live.binance import BinanceFuturesTradingEnvConfig
from torchtrade.envs.core.common_types import MarginType

config = BinanceFuturesTradingEnvConfig(
    margin_type=MarginType.ISOLATED,
    leverage=10.0,
)
```

### Cross Margin
- All positions share account margin
- Liquidation affects entire account
- Higher leverage, shared risk

```python
from torchtrade.envs.live.binance import BinanceFuturesTradingEnvConfig
from torchtrade.envs.core.common_types import MarginType

config = BinanceFuturesTradingEnvConfig(
    margin_type=MarginType.CROSSED,
    leverage=20.0,
)
```

## Leverage

Set leverage per symbol:

```python
from torchtrade.envs.live.binance import BinanceFuturesTradingEnvConfig

# Conservative leverage
config = BinanceFuturesTradingEnvConfig(
    leverage=3.0,  # 3x leverage
)

# Higher leverage (risky!)
config = BinanceFuturesTradingEnvConfig(
    leverage=50.0,  # 50x leverage
)
```

**Warning**: Higher leverage = higher liquidation risk!

## Custom Feature Preprocessing

The Binance observation class exposes all fields from Binance klines to your custom `feature_preprocessing_fn`. Beyond standard OHLCV, you have access to:

| Column | Type | Description |
|--------|------|-------------|
| `open`, `high`, `low`, `close` | float | Standard price data |
| `volume` | float | Base asset volume |
| `quote_volume` | float | Quote asset volume (e.g., USDT volume) |
| `trades` | int | Number of trades in the candle |
| `taker_buy_base` | float | Taker buy volume (base asset) |
| `taker_buy_quote` | float | Taker buy volume (quote asset) |

These extra fields allow you to derive sentiment features without additional API calls:

```python
from torchtrade.envs.live.binance import BinanceFuturesTorchTradingEnv

def my_preprocessing(df):
    df = df.copy()
    # Taker buy ratio: proportion of volume from aggressive buyers
    df["features_taker_buy_ratio"] = df["taker_buy_base"] / (df["volume"] + 1e-9)
    # Quote volume change
    df["features_quote_volume_pct"] = df["quote_volume"].pct_change().fillna(0)
    # Average trade size
    df["features_avg_trade_size"] = df["volume"] / (df["trades"] + 1e-9)
    # Standard price features
    df["features_close"] = df["close"].pct_change().fillna(0)
    df.dropna(inplace=True)
    return df

env = BinanceFuturesTorchTradingEnv(
    config=config,
    feature_preprocessing_fn=my_preprocessing,
)
```

**Note**: These extra kline fields are Binance-specific. Bitget and Bybit observation classes only expose standard OHLCV and volume through their respective APIs (CCXT and pybit). Built-in support for auxiliary data fetching (funding rate, taker buy/sell ratio, open interest) across all exchanges is planned for a future release.

## Funding Fees

Futures have periodic funding fees:
- **Rate**: ±0.01% typically
- **Frequency**: Every 8 hours (00:00, 08:00, 16:00 UTC)
- **Direction**: Longs pay shorts (or vice versa)

Environments simulate funding fees automatically.

## Example: Basic Futures Trading

```python
from torchtrade.envs.core.common_types import MarginType

import os
import torch

from torchtrade.envs.live.binance.env import BinanceFuturesTorchTradingEnv, BinanceFuturesTradingEnvConfig

config = BinanceFuturesTradingEnvConfig(
    symbol="BTCUSDT",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
    demo=True,
    leverage=5.0,
    margin_type=MarginType.ISOLATED,
)

env = BinanceFuturesTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["BINANCE_API_KEY"],
    api_secret=os.environ["BINANCE_SECRET_KEY"],
)
td = env.reset()

# Actions are CATEGORICAL indices into action_levels, which defaults to [-1, 0, 1]:
# 0 = full short, 1 = flat, 2 = full long. Leverage and size come from the config,
# not from the action.
td["action"] = torch.tensor(2)  # go long
td = env.step(td)

# account_state: [exposure, direction, unrealized_pnl_pct, holding_time,
# leverage, distance_to_liquidation] -- step() returns no `info` dict.
acct = td["next", "account_state"]
print(f"Direction: {acct[1].item()}  Unrealized PnL %: {acct[2].item():.4f}")
```

## Example: With Risk Management

```python
from torchtrade.envs.live.binance import BinanceFuturesSLTPTradingEnvConfig

import os

from torchtrade.envs.live.binance.env_sltp import BinanceFuturesSLTPTorchTradingEnv

config = BinanceFuturesSLTPTradingEnvConfig(
    symbol="ETHUSDT",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
    demo=True,
    leverage=3.0,
    stoploss_levels=[-0.02],  # 2% stop loss (important with leverage!)
    takeprofit_levels=[0.04],  # 4% take profit
)

env = BinanceFuturesSLTPTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["BINANCE_API_KEY"],
    api_secret=os.environ["BINANCE_SECRET_KEY"],
)
obs = env.reset()
```

## Liquidation

Liquidation occurs when margin ratio drops below maintenance level:

```
Liquidation Price = Entry Price × (1 ± 1/Leverage ± Maintenance Margin Rate)
```

**Example**: Long BTC at $50,000 with 10x leverage:
- Liquidation price ≈ $45,454
- 9% move against you = liquidation

Environments handle liquidation automatically.

## Best Practices

1. **Start with testnet**: Never trade real money untested
2. **Use conservative leverage**: 2-5x max for beginners
3. **Always use stop-losses**: Especially with leverage
4. **Monitor liquidation price**: Stay far from liquidation
5. **Understand funding fees**: Can eat profits over time
6. **Test thoroughly**: Futures are high-risk

## API Rate Limits

- **Market Data**: 2400 requests/minute
- **Orders**: 1200 requests/minute (varies by endpoint)
- **WebSocket**: 300 connections max

## Common Issues

**"Insufficient margin"**: Not enough funds for leveraged position

**"Leverage too high"**: Exceeds symbol's max leverage

**"Position liquidated"**: Price moved against you

**"Invalid symbol"**: Use USDT-margined symbols (e.g., BTCUSDT, not BTCUSD)

## Supported Symbols

All USDT-margined perpetual futures:
- **Major**: BTCUSDT, ETHUSDT, BNBUSDT
- **Alts**: ADAUSDT, DOGEUSDT, SOLUSDT, etc.

Check current symbols: https://fapi.binance.com/fapi/v1/exchangeInfo

## Resources

- [Binance Futures Docs](https://binance-docs.github.io/apidocs/futures/en/)
- [Testnet](https://testnet.binancefuture.com/)
- [Risk Management Guide](https://www.binance.com/en/support/faq/360033524991)

## See Also

- [Live Environments README](../README.md)
- [Core Base Classes](../../core/README.md)
