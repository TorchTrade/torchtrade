# Bitget Futures Trading Environment

Live trading integration with Bitget for crypto futures markets (USDT-margined).

## Files

- **`base.py`**: Base Bitget Futures environment
- **`observation.py`**: Bitget-specific observation handling
- **`order_executor.py`**: Order execution for Bitget Futures API
- **`env.py`**: Main Bitget Futures environment
- **`env_sltp.py`**: Bitget environment with SL/TP
- **`utils.py`**: Helper functions

## Quick Start

```python
from torchtrade.envs import BitgetFuturesTorchTradingEnv, BitgetFuturesTradingEnvConfig
from torchtrade.envs.utils import TimeFrame, TimeFrameUnit

config = BitgetFuturesTradingEnvConfig(
    api_key="YOUR_KEY",
    api_secret="YOUR_SECRET",
    passphrase="YOUR_PASSPHRASE",
    symbol="BTCUSDT",
    timeframe=TimeFrame(1, TimeFrameUnit.MINUTE),
    testnet=True,  # Use testnet first!
    max_leverage=10.0,
    margin_mode="isolated",
)

env = BitgetFuturesTorchTradingEnv(config=config)
obs = env.reset()
```

## Features

- **Low Fees**: Competitive maker/taker fees
- **High Leverage**: Up to 125x leverage
- **Copy Trading Integration**: Compatible with Bitget copy trading
- **Isolated/Cross Margin**: Flexible margin modes
- **Testnet**: Safe testing with simulated funds

## Configuration

```python
@dataclass
class BitgetFuturesTradingEnvConfig:
    api_key: str
    api_secret: str
    passphrase: str              # Bitget requires passphrase
    symbol: str
    timeframe: TimeFrame
    testnet: bool = True         # Testnet or mainnet
    max_leverage: float = 10.0   # Maximum leverage
    margin_mode: str = "isolated"  # isolated or crossed
    initial_margin: float = 1000.0
    window_size: int = 50
```

**Note**: Bitget requires a passphrase in addition to API key/secret.

## Testnet vs Mainnet

**Testnet**:
```python
config = BitgetFuturesTradingEnvConfig(
    api_key=testnet_key,
    api_secret=testnet_secret,
    passphrase=testnet_passphrase,
    testnet=True,
)
```

Get testnet credentials: https://www.bitget.com/en/testnet/

**Mainnet**:
```python
config = BitgetFuturesTradingEnvConfig(
    api_key=mainnet_key,
    api_secret=mainnet_secret,
    passphrase=mainnet_passphrase,
    testnet=False,
)
```

## Margin Modes

### Isolated Margin
```python
config = BitgetFuturesTradingEnvConfig(
    margin_mode="isolated",
    max_leverage=10.0,
)
```

### Crossed Margin
```python
config = BitgetFuturesTradingEnvConfig(
    margin_mode="crossed",
    max_leverage=20.0,
)
```

## Example: Basic Trading

```python
from torchtrade.envs import BitgetFuturesTorchTradingEnv

config = BitgetFuturesTradingEnvConfig(
    api_key=os.environ["BITGET_TESTNET_KEY"],
    api_secret=os.environ["BITGET_TESTNET_SECRET"],
    passphrase=os.environ["BITGET_TESTNET_PASSPHRASE"],
    symbol="BTCUSDT",
    timeframe=TimeFrame(5, TimeFrameUnit.MINUTE),
    testnet=True,
    max_leverage=5.0,
)

env = BitgetFuturesTorchTradingEnv(config=config)
obs = env.reset()

# Long position
action = {"direction": "long", "leverage": 5.0, "size": 0.5}
obs, reward, done, info = env.step(action)
```

## Example: With SL/TP

```python
from torchtrade.envs import BitgetFuturesSLTPTorchTradingEnv

config = BitgetFuturesSLTPTradingEnvConfig(
    api_key=os.environ["BITGET_TESTNET_KEY"],
    api_secret=os.environ["BITGET_TESTNET_SECRET"],
    passphrase=os.environ["BITGET_TESTNET_PASSPHRASE"],
    symbol="ETHUSDT",
    timeframe=TimeFrame(1, TimeFrameUnit.MINUTE),
    testnet=True,
    max_leverage=3.0,
    sl_percent=0.02,
    tp_percent=0.04,
)

env = BitgetFuturesSLTPTorchTradingEnv(config=config)
obs = env.reset()
```

## Fees

**Trading Fees**:
- Maker: 0.02%
- Taker: 0.06%

Lower than most competitors!

**Funding Fees**:
- Rate: Varies (typically ±0.01%)
- Frequency: Every 8 hours

## Supported Symbols

USDT-margined perpetual futures:
- BTCUSDT, ETHUSDT, BNBUSDT
- Plus 200+ altcoins

Check symbols: https://api.bitget.com/api/mix/v1/market/contracts?productType=umcbl

## Best Practices

1. **Secure your passphrase**: Required for all API requests
2. **Start with testnet**: Test thoroughly before live trading
3. **Use conservative leverage**: 2-5x recommended
4. **Enable IP whitelist**: Restrict API access to known IPs
5. **Monitor fees**: Funding fees can add up

## API Rate Limits

- **REST API**: 20 requests/second per endpoint
- **WebSocket**: 100 subscriptions max

## Common Issues

**"Invalid passphrase"**: Check passphrase matches API key

**"Insufficient balance"**: Not enough margin for position

**"Symbol not found"**: Use correct symbol format (e.g., BTCUSDT)

**"Leverage exceeds limit"**: Reduce leverage setting

## Resources

- [Bitget API Docs](https://bitgetlimited.github.io/apidoc/en/mix/)
- [Testnet Platform](https://www.bitget.com/en/testnet/)
- [Fee Structure](https://www.bitget.com/en/rate/)

## See Also

- [Live Environments README](../README.md)
- [Core Base Classes](../../core/README.md)
