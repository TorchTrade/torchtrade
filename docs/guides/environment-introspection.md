# Environment Introspection

TorchTrade environments expose helper methods to programmatically understand their observation structure. This is useful for building neural network architectures, integrating with LLM actors, and debugging observation structure.

## Overview

All TorchTrade environments provide two methods for introspection:

- **`get_market_data_keys()`**: Returns list of market data observation keys based on configured timeframes
- **`get_account_state()`**: Returns list of account state field names

These methods allow you to dynamically adapt your code to different environment configurations without hardcoding observation structure.

> **⚠️ Important for Wrapped Environments:**
>
> When using `ParallelEnv` or `TransformedEnv` (as done in all training examples), access introspection methods via `.base_env`:
> ```python
> market_keys = env.base_env.get_market_data_keys()
> account_state = env.base_env.get_account_state()
> ```
> This ensures you access the underlying environment directly and avoid nested list returns from parallel workers.

## Methods

### get_market_data_keys()

Returns a list of market data observation keys in the format `market_data_{timeframe}_{window_size}`.

```python
from torchtrade.envs import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
import pandas as pd

# Create environment with multiple timeframes
config = SeqLongOnlyEnvConfig(
    symbol="BTC/USD",
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on="1min",
)
env = SeqLongOnlyEnv(df, config)

# Get market data keys
print(env.get_market_data_keys())
# Output: ['market_data_1Minute_12', 'market_data_5Minute_8', 'market_data_15Minute_8']
```

### get_account_state()

Returns a list of account state field names. The structure varies between environment types:

**Standard Environments (7 elements):**
```python
from torchtrade.envs import SeqLongOnlyEnv, SeqLongOnlyEnvConfig

env = SeqLongOnlyEnv(df, config)
print(env.get_account_state())
# Output: ['cash', 'position_size', 'position_value', 'entry_price',
#          'current_price', 'unrealized_pnlpct', 'holding_time']
```

**Futures Environments (10 elements):**
```python
from torchtrade.envs import SeqFuturesEnv, SeqFuturesEnvConfig

env = SeqFuturesEnv(df, config)
print(env.get_account_state())
# Output: ['cash', 'position_size', 'position_value', 'entry_price',
#          'current_price', 'unrealized_pnlpct', 'leverage',
#          'margin_ratio', 'liquidation_price', 'holding_time']
```

## Usage with Wrapped Environments

As noted above, wrapped environments require accessing methods via `.base_env`. Here's a complete example:

```python
from torchrl.envs import TransformedEnv, Compose, RewardSum

# Create and wrap environment
base_env = SeqLongOnlyEnv(df, config)
env = TransformedEnv(base_env, Compose(RewardSum()))

# Access via .base_env (see note above)
market_data_keys = env.base_env.get_market_data_keys()
account_state = env.base_env.get_account_state()
```

## Use Cases

### 1. Building Neural Network Architectures

Dynamically construct encoders for each timeframe:

```python
def make_policy(env, device):
    market_data_keys = env.get_market_data_keys()

    # Build encoders for each timeframe
    encoders = []
    for key in market_data_keys:
        encoder = SimpleCNNEncoder(
            input_shape=env.observation_spec[key].shape,
            output_shape=(1, 64),
        )
        encoders.append(encoder)

    # Account state encoder
    account_state_size = len(env.get_account_state())
    account_encoder = MLP(in_features=account_state_size, out_features=64)

    return PolicyNetwork(encoders, account_encoder)
```

### 2. LLM Actor Integration

Provide context to LLM-based trading agents:

```python
from torchtrade.actor import LLMTradingActor

env = AlpacaTorchTradingEnv(config)

# Initialize LLM actor with environment observation structure
actor = LLMTradingActor(
    api_key="...",
    market_data_keys=env.get_market_data_keys(),
    account_state_fields=env.get_account_state(),
)
```

### 3. Dynamic Observation Processing

Process observations without hardcoded keys:

```python
def process_observation(tensordict, env):
    # Extract all market data dynamically
    market_data = {}
    for key in env.get_market_data_keys():
        market_data[key] = tensordict[key]

    # Extract account state
    account_state_fields = env.get_account_state()
    account_state = tensordict["account_state"]

    # Create structured output
    return {
        "market_data": market_data,
        "account_state": dict(zip(account_state_fields, account_state))
    }
```

### 4. Debugging Observation Structure

Quickly inspect what observations are available:

```python
env = SeqFuturesSLTPEnv(df, config)

print("Environment Observation Structure")
print("=" * 50)
print(f"Market Data Keys: {env.get_market_data_keys()}")
print(f"Account State Fields: {env.get_account_state()}")
print(f"Total Observation Keys: {list(env.observation_spec.keys())}")
```

## Account State Structure

### Standard Environments (Spot/Long-Only)

| Field | Description | Type |
|-------|-------------|------|
| `cash` | Available cash balance | float |
| `position_size` | Number of units held | float |
| `position_value` | Current market value of position | float |
| `entry_price` | Price at which position was entered | float |
| `current_price` | Current market price | float |
| `unrealized_pnlpct` | Unrealized P&L as percentage | float |
| `holding_time` | Number of steps position has been held | int |

### Futures Environments

Futures environments include 3 additional fields for margin trading:

| Field | Description | Type |
|-------|-------------|------|
| `leverage` | Current leverage multiplier | float |
| `margin_ratio` | Margin balance / position value | float |
| `liquidation_price` | Price at which position would be liquidated | float |

## Best Practices

1. **Use for dynamic architectures**: When building networks that need to adapt to different timeframe configurations
2. **Document assumptions**: If your code expects specific account state fields, document which environment types are supported
3. **Validate in tests**: Test your code with different configurations to ensure it handles varying observation structures
4. **Remember `.base_env` pattern**: See the important note in the Overview section above

## Example: Complete Workflow

```python
from torchtrade.envs import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchrl.envs import ParallelEnv, TransformedEnv, Compose, RewardSum
import pandas as pd

# 1. Create base environment
df = pd.read_csv("data.csv", index_col=0, parse_dates=True)
config = SeqLongOnlyEnvConfig(
    symbol="BTC/USD",
    time_frames=["1min", "5min"],
    window_sizes=[12, 8],
    execute_on="1min",
)
base_env = SeqLongOnlyEnv(df, config)

# 2. Wrap environment for training
env = TransformedEnv(base_env, Compose(RewardSum()))

# 3. Introspect observation structure (use .base_env!)
market_data_keys = env.base_env.get_market_data_keys()
account_state_fields = env.base_env.get_account_state()

print(f"Market Data: {market_data_keys}")
print(f"Account State: {account_state_fields}")

# 4. Build policy using introspection
def make_policy(env):
    # Access base environment for introspection
    base = env.base_env if hasattr(env, 'base_env') else env

    market_keys = base.get_market_data_keys()
    account_size = len(base.get_account_state())

    # Build architecture dynamically
    encoders = [build_encoder(env.observation_spec[k]) for k in market_keys]
    account_net = MLP(in_features=account_size, out_features=64)

    return PolicyNetwork(encoders, account_net)

policy = make_policy(env)
```

## See Also

- **[Custom Environment Guide](custom-environment.md)**: Learn how to set `account_state` and `market_data_keys` in custom environments
- **[Offline Environments](../environments/offline.md)**: Offline environment configurations and observation structure
- **[Online Environments](../environments/online.md)**: Live trading environment observation structure
