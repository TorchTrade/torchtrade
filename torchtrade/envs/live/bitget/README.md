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
import os

from torchtrade.envs.live.bitget.order_executor import MarginMode
from torchtrade.envs.live.bitget.env import BitgetFuturesTorchTradingEnv, BitgetFuturesTradingEnvConfig

config = BitgetFuturesTradingEnvConfig(
    symbol="BTCUSDT",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
    demo=True,  # Use testnet first!
    leverage=10.0,
    margin_mode=MarginMode.ISOLATED,
)

env = BitgetFuturesTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["BITGET_API_KEY"],
    api_secret=os.environ["BITGET_SECRET"],
    api_passphrase=os.environ["BITGET_PASSPHRASE"],
)
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
from torchtrade.envs.live.bitget.order_executor import MarginMode
from torchtrade.envs.core.live import ObservationFailurePolicy
from torchtrade.envs.live.bitget.order_executor import PositionMode
from dataclasses import dataclass

@dataclass
class BitgetFuturesTradingEnvConfig:
    symbol = 'BTCUSDT'
    time_frames = '1Hour'
    window_sizes = 10
    execute_on = '1Hour'
    product_type = 'USDT-FUTURES'
    leverage = 1
    margin_mode = MarginMode.ISOLATED
    position_mode = PositionMode.ONE_WAY
    action_levels = None
    done_on_bankruptcy = True
    bankrupt_threshold = 0.1
    demo = True
    seed = 42
    include_base_features = False
    close_position_on_init = True
    close_position_on_reset = False
    observation_failure_policy = ObservationFailurePolicy.HALT
    # Credentials: Env(config, api_key=..., api_secret=..., api_passphrase=...)
```

**Note**: Bitget requires a passphrase in addition to API key/secret.

## Testnet vs Mainnet

**Testnet**:
```python
from torchtrade.envs.live.bitget import BitgetFuturesTradingEnvConfig

config = BitgetFuturesTradingEnvConfig(
    demo=True,
)
```

Get testnet credentials: https://www.bitget.com/en/testnet/

**Mainnet**:
```python
from torchtrade.envs.live.bitget import BitgetFuturesTradingEnvConfig

config = BitgetFuturesTradingEnvConfig(
    demo=False,
)
```

## Margin Modes

### Isolated Margin
```python
from torchtrade.envs.live.bitget import BitgetFuturesTradingEnvConfig
from torchtrade.envs.live.bitget.order_executor import MarginMode

config = BitgetFuturesTradingEnvConfig(
    margin_mode=MarginMode.ISOLATED,
    leverage=10.0,
)
```

### Crossed Margin
```python
from torchtrade.envs.live.bitget import BitgetFuturesTradingEnvConfig
from torchtrade.envs.live.bitget.order_executor import MarginMode

config = BitgetFuturesTradingEnvConfig(
    margin_mode=MarginMode.CROSSED,
    leverage=20.0,
)
```

## Example: Basic Trading

```python
from torchtrade.envs.live.bitget import BitgetFuturesTradingEnvConfig

import os
import torch

from torchtrade.envs.live.bitget.env import BitgetFuturesTorchTradingEnv

config = BitgetFuturesTradingEnvConfig(
    symbol="BTCUSDT",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
    demo=True,
    leverage=5.0,
)

env = BitgetFuturesTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["BITGET_API_KEY"],
    api_secret=os.environ["BITGET_SECRET"],
    api_passphrase=os.environ["BITGET_PASSPHRASE"],
)
td = env.reset()

# Actions are CATEGORICAL indices into action_levels, which on Bitget
# defaults to [-1.0, -0.5, 0.0, 0.5, 1.0]:
# 0 = full short, 2 = flat, 4 = full long. Leverage and size come from the config,
# not from the action.
td["action"] = torch.tensor(4)  # go full long
td = env.step(td)
```

## Example: With SL/TP

```python
from torchtrade.envs.live.bitget import BitgetFuturesSLTPTradingEnvConfig

import os

from torchtrade.envs.live.bitget.env_sltp import BitgetFuturesSLTPTorchTradingEnv

config = BitgetFuturesSLTPTradingEnvConfig(
    symbol="ETHUSDT",
    time_frames=["1Min"],
    window_sizes=[10],
    execute_on="1Min",
    demo=True,
    leverage=3.0,
    stoploss_levels=[-0.02],
    takeprofit_levels=[0.04],
)

env = BitgetFuturesSLTPTorchTradingEnv(
    # Credentials are CONSTRUCTOR arguments, not config fields.
    config, api_key=os.environ["BITGET_API_KEY"],
    api_secret=os.environ["BITGET_SECRET"],
    api_passphrase=os.environ["BITGET_PASSPHRASE"],
)
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
