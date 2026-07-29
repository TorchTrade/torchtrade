# Core Base Classes

This directory contains the fundamental base classes that all TorchTrade environments inherit from.

## Module Overview

### `base.py`
Root base class for all environments.

**Key Classes:**
- `TorchTradeBaseEnv`: Abstract base class defining the core environment interface
- `TorchTradeEnvConfig`: Configuration dataclass for base environment settings

**Core Responsibilities:**
- Environment lifecycle management (reset, step, close)
- Observation and action space definitions
- TorchRL integration (TensorDict support)
- Render and logging interfaces

### `offline_base.py`
Base class for all backtesting environments.

**Key Classes:**
- `TorchTradeOfflineEnv`: Extends `TorchTradeBaseEnv` for offline simulations

**Offline-Specific Features:**
- Historical data management
- Episode sampling and windowing
- Backtesting-specific metrics
- Fast-forward execution

### `live.py`
Base class for all live trading environments.

**Key Classes:**
- `TorchTradeLiveEnv`: Extends `TorchTradeBaseEnv` for live trading

**Live-Specific Features:**
- Real-time data streaming
- Order execution management
- Position synchronization
- Error handling and recovery

### `state.py`
Position state management.

**Key Classes:**
- `PositionState`: Tracks current position information

**State Attributes:**
- Position size and direction
- Entry price and timestamp
- Unrealized P&L
- Position metadata

### `reward.py`
Reward function abstractions.

**Key Classes:**
- `RewardFunction`: Abstract base for reward calculations
- `LogReturnReward`: Log return-based rewards
- `PercentReturnReward`: Percentage return rewards
- `RealizedPnLReward`: Realized profit/loss rewards
- `SharpeRatioReward`: Risk-adjusted return rewards

**Extensibility:**
Create custom reward functions by extending `RewardFunction` and implementing the `calculate()` method.

### `common.py`
Common types and enums.

**Key Types:**
- `ActionType`: Enum for discrete action types (BUY, SELL, HOLD)
- Shared constants and type definitions

## Class Hierarchy

```
TorchTradeBaseEnv (base.py)
├── TorchTradeOfflineEnv (offline_base.py)
│   ├── SeqLongOnlyEnv
│   ├── SeqFuturesEnv
│   └── ...
└── TorchTradeLiveEnv (live.py)
    ├── AlpacaBaseTorchTradingEnv
    ├── BinanceBaseTorchTradingEnv
    └── ...
```

## Usage Examples

### Extending the Base Environment

```python
from torchtrade.envs.core import TorchTradeOfflineEnv, TorchTradeEnvConfig
from dataclasses import dataclass

@dataclass
class MyEnvConfig(TorchTradeEnvConfig):
    custom_param: float = 1.0

class MyCustomEnv(TorchTradeOfflineEnv):
    def __init__(self, df, config: MyEnvConfig):
        super().__init__(config)
        self.df = df
        self.custom_param = config.custom_param

    def _reset(self, tensordict=None, **kwargs):
        # Custom reset logic
        return self._get_observation()

    def _step(self, tensordict):
        # Custom step logic
        return self._get_observation(), reward, done, info
```

### Using Position State

```python
from torchtrade.envs.core import PositionState

# Track position
position = PositionState(
    size=100.0,
    entry_price=50.0,
    direction="long",
)

# Update position
position.update(current_price=55.0)
unrealized_pnl = position.unrealized_pnl()
```

### Creating Custom Rewards

```python
from torchtrade.envs.core import RewardFunction

class CustomReward(RewardFunction):
    def calculate(self, state, action, next_state):
        # Custom reward logic
        pnl = next_state.portfolio_value - state.portfolio_value
        risk_penalty = self._calculate_risk(state)
        return pnl - risk_penalty

    def _calculate_risk(self, state):
        # Risk calculation logic
        return state.position_size * 0.01
```

## Design Patterns

### Template Method Pattern

Base classes define the overall algorithm structure:

```python
def reset(self, **kwargs):
    # Common setup
    self._initialize()

    # Call subclass-specific logic
    observation = self._reset(**kwargs)

    # Common cleanup
    self._finalize()
    return observation
```

Subclasses override `_reset()` to provide specific behavior.

### Strategy Pattern

Reward functions use the strategy pattern:

```python
env = MyEnv(
    config=config,
    reward_fn=SharpeRatioReward()  # Pluggable reward
)
```

### Observer Pattern

State changes notify observers:

```python
position.register_observer(logger)
position.update(price=new_price)  # Notifies logger
```

## Key Abstractions

### Observation Space

Environments must define their observation space:

```python
def _make_observation_space(self):
    return spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(self.window_size, self.n_features),
        dtype=np.float32
    )
```

### Action Space

Environments must define their action space:

```python
def _make_action_space(self):
    return spaces.Discrete(3)  # BUY, SELL, HOLD
```

### TensorDict Integration

All observations and actions use TensorDict:

```python
def _get_observation(self):
    return TensorDict({
        "observation": torch.tensor(self.current_obs),
        "position": torch.tensor([self.position]),
    }, batch_size=[])
```

## Best Practices

1. **Always call super().__init__()**: Ensure base class initialization
2. **Use dataclasses for configs**: Type-safe configuration management
3. **Implement abstract methods**: Don't skip required method overrides
4. **Handle errors gracefully**: Especially in live environments
5. **Log important events**: Use the built-in logging system
6. **Test thoroughly**: Write tests for custom environments

## Common Pitfalls

1. **Forgetting to reset state**: Always reset all stateful variables in `_reset()`
2. **Incorrect action space**: Ensure action space matches step() expectations
3. **Not handling edge cases**: Consider terminal states, missing data, etc.
4. **Mixing offline and live logic**: Keep concerns separated

## Testing Your Custom Environment

```python
import pytest
from torchrl.envs.utils import check_env_specs

def test_custom_env():
    env = MyCustomEnv(df=test_data, config=test_config)

    # Check environment specs
    check_env_specs(env)

    # Test reset
    obs = env.reset()
    assert obs is not None

    # Test step
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    assert next_obs is not None
```

## See Also

- [Utilities Documentation](../utils/README.md)
- [Offline Environments](../offline/README.md)
- [Live Environments](../live/README.md)
- [Main README](../README.md)
