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
from torchtrade.envs import BinanceFuturesTorchTradingEnv, BinanceFuturesTradingEnvConfig
from torchtrade.envs.utils import TimeFrame, TimeFrameUnit

config = BinanceFuturesTradingEnvConfig(
    api_key="YOUR_KEY",
    api_secret="YOUR_SECRET",
    symbol="BTCUSDT",
    timeframe=TimeFrame(1, TimeFrameUnit.MINUTE),
    testnet=True,  # Use testnet first!
    max_leverage=10.0,
    margin_type="ISOLATED",
)

env = BinanceFuturesTorchTradingEnv(config=config)
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
@dataclass
class BinanceFuturesTradingEnvConfig:
    api_key: str
    api_secret: str
    symbol: str
    timeframe: TimeFrame
    testnet: bool = True            # Testnet or mainnet
    max_leverage: float = 10.0      # Maximum leverage
    margin_type: str = "ISOLATED"   # ISOLATED or CROSS
    initial_margin: float = 1000.0
    window_size: int = 50
```

## Testnet vs Mainnet

**Testnet** (recommended for development):
```python
config = BinanceFuturesTradingEnvConfig(
    api_key=testnet_key,
    api_secret=testnet_secret,
    testnet=True,  # Fake funds
)
```

Get testnet API keys: https://testnet.binancefuture.com/

**Mainnet** (real money):
```python
config = BinanceFuturesTradingEnvConfig(
    api_key=mainnet_key,
    api_secret=mainnet_secret,
    testnet=False,  # Real trading!
)
```

## Margin Types

### Isolated Margin
- Each position has separate margin
- Liquidation affects only that position
- Lower risk, position-specific leverage

```python
config = BinanceFuturesTradingEnvConfig(
    margin_type="ISOLATED",
    max_leverage=10.0,
)
```

### Cross Margin
- All positions share account margin
- Liquidation affects entire account
- Higher leverage, shared risk

```python
config = BinanceFuturesTradingEnvConfig(
    margin_type="CROSS",
    max_leverage=20.0,
)
```

## Leverage

Set leverage per symbol:

```python
# Conservative leverage
config = BinanceFuturesTradingEnvConfig(
    max_leverage=3.0,  # 3x leverage
)

# Higher leverage (risky!)
config = BinanceFuturesTradingEnvConfig(
    max_leverage=50.0,  # 50x leverage
)
```

**Warning**: Higher leverage = higher liquidation risk!

## Funding Fees

Futures have periodic funding fees:
- **Rate**: ±0.01% typically
- **Frequency**: Every 8 hours (00:00, 08:00, 16:00 UTC)
- **Direction**: Longs pay shorts (or vice versa)

Environments simulate funding fees automatically.

## Example: Basic Futures Trading

```python
from torchtrade.envs import BinanceFuturesTorchTradingEnv, BinanceFuturesTradingEnvConfig

config = BinanceFuturesTradingEnvConfig(
    api_key=os.environ["BINANCE_TESTNET_KEY"],
    api_secret=os.environ["BINANCE_TESTNET_SECRET"],
    symbol="BTCUSDT",
    timeframe=TimeFrame(5, TimeFrameUnit.MINUTE),
    testnet=True,
    max_leverage=5.0,
    margin_type="ISOLATED",
)

env = BinanceFuturesTorchTradingEnv(config=config)
obs = env.reset()

# Go long with 5x leverage
action = {"direction": "long", "leverage": 5.0, "size": 0.5}
obs, reward, done, info = env.step(action)

print(f"Position: {info['position']}")
print(f"Unrealized PnL: ${info['unrealized_pnl']:.2f}")
```

## Example: With Risk Management

```python
from torchtrade.envs import BinanceFuturesSLTPTorchTradingEnv

config = BinanceFuturesSLTPTradingEnvConfig(
    api_key=os.environ["BINANCE_TESTNET_KEY"],
    api_secret=os.environ["BINANCE_TESTNET_SECRET"],
    symbol="ETHUSDT",
    timeframe=TimeFrame(1, TimeFrameUnit.MINUTE),
    testnet=True,
    max_leverage=3.0,
    sl_percent=0.02,  # 2% stop loss (important with leverage!)
    tp_percent=0.04,  # 4% take profit
)

env = BinanceFuturesSLTPTorchTradingEnv(config=config)
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
