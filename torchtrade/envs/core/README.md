# Core Base Classes

This directory contains the fundamental base classes that all TorchTrade environments inherit from.

## Module Overview

### `base.py`
Root base class for all environments.

**Key Classes:**
- `TorchTradeBaseEnv`: Abstract base class defining the core environment interface

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

### `default_rewards.py`
The shipped reward functions.

**Key Classes:**
- `default_rewards`: `log_return_reward`, `sharpe_ratio_reward` and
  `drawdown_penalty_reward`. Plain functions, not classes -- nothing to instantiate.

**Extensibility:**
A reward function is a plain callable passed to the env constructor; see `torchtrade/envs/core/default_rewards.py`.

### `common.py`
Common types and enums.

**Key Types:**
- `TradeMode` / `validate_trade_mode`: the three position-sizing modes
- Shared constants and type definitions

## Class Hierarchy

```
TorchTradeBaseEnv (base.py)
├── TorchTradeOfflineEnv (offline_base.py)
│   └── SequentialTradingEnv
│       └── SequentialTradingEnvSLTP
│           └── OneStepTradingEnv
└── TorchTradeLiveEnv (live.py)
    ├── AlpacaBaseTorchTradingEnv
    │   ├── AlpacaTorchTradingEnv
    │   └── AlpacaSLTPTorchTradingEnv
    └── TorchTradeFuturesLiveEnv (live/shared/futures_live_base.py)
        ├── BinanceBaseTorchTradingEnv
        ├── BitgetBaseTorchTradingEnv
        ├── BybitBaseTorchTradingEnv
        └── OKXBaseTorchTradingEnv
```

Each of the four futures bases has a plain and an SLTP leaf, elided here.

The offline side is a chain: `OneStepTradingEnv` inherits `SequentialTradingEnvSLTP`,
which inherits `SequentialTradingEnv`. A change to `SequentialTradingEnv` reaches all
three.

Two families sit outside this tree. `VectorizedSequentialTradingEnv` and
`PolymarketBetEnv` subclass TorchRL's `EnvBase` directly, and
`VectorizedSequentialTradingEnvSLTP` extends the plain vectorized env. None of them
inherit `TorchTradeOfflineEnv` or `TorchTradeLiveEnv`, so a change to those bases must be
applied to them by hand. The vectorized pair still shares `utils/` and the batched reward
in `core/default_rewards.py`; polymarket shares only `utils/termination.py`.

## Usage Examples

### Extending the Base Environment

```python
from dataclasses import dataclass

from torchtrade.envs.core import TorchTradeOfflineEnv
from torchtrade.envs.offline import SequentialTradingEnvConfig

# There is no exported TorchTradeEnvConfig base -- subclass the config of the env you
# are extending.
@dataclass
class MyEnvConfig(SequentialTradingEnvConfig):
    custom_param: float = 1.0

class MyCustomEnv(TorchTradeOfflineEnv):
    def __init__(self, df, config: MyEnvConfig):
        super().__init__(df, config)
        self.custom_param = config.custom_param

    # Filling the two abstract methods (_step, _get_portfolio_value) is necessary but
    # NOT sufficient: the base _reset also calls self._get_observation(), which the base
    # neither defines nor declares abstract, so a class that satisfies the ABC still
    # fails at reset(). Subclass a concrete env instead unless you intend to write all
    # three. And note _step is abstract all the way up on torchrl's EnvBase -- there is
    # no base implementation for super() to extend.
    def _get_portfolio_value(self, current_price=None) -> float:
        # `balance` -- there is no `cash` attribute on the offline envs.
        return self.balance

    def _step(self, tensordict):
        # TensorDict in, TensorDict out -- never the gym (obs, reward, done, info)
        # tuple. The outcome goes under "next".
        raise NotImplementedError("your logic here")
```

### Using Position State

```python
from torchtrade.envs.core import PositionState

# The fields are the ones the envs actually keep. There is no `size`/`direction` kwarg,
# no .update() and no .unrealized_pnl() -- the environment writes these directly.
position = PositionState(
    current_position=1,        # -1 short, 0 flat, +1 long
    position_size=100.0,
    entry_price=50.0,
)
position.unrealized_pnlpc = (55.0 - position.entry_price) / position.entry_price
```

### Creating Custom Rewards

```python
import math

# A reward function is a plain callable taking the HistoryTracker -- not a subclass, and
# not the env. See torchtrade/envs/core/default_rewards.py.
def drawdown_penalised_return(history) -> float:
    values = history.portfolio_values
    if len(values) < 2:
        return 0.0
    step_return = math.log(values[-1] / values[-2])
    drawdown = 1.0 - values[-1] / max(values)
    return step_return - 0.1 * drawdown
```

## Design Patterns

### Template Method Pattern

TorchRL's `EnvBase` owns the public `reset()`/`step()`; environments implement the
underscored hooks. There is no `_initialize`/`_finalize` pair -- neither exists in the
package:

```python
# EnvBase.reset() validates, calls YOUR _reset(), then checks the output against the
# specs. Subclasses override _reset and _step, never reset and step.
def _reset(self, tensordict=None, **kwargs):
    ...  # return a TensorDict matching observation_spec
```

### Strategy Pattern

Reward functions use the strategy pattern:

```python
from torchtrade.envs.core.default_rewards import sharpe_ratio_reward
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

# The kwarg is `reward_function` and it takes the function itself -- not `reward_fn`,
# and not an instance, since these are functions rather than classes.
env = SequentialTradingEnv(
    your_dataframe,
    SequentialTradingEnvConfig(time_frames=["1Min"], window_sizes=[10], execute_on="1Min"),
    reward_function=sharpe_ratio_reward,
)
```

## Key Abstractions

### Specs

There is no `_make_observation_space` hook and no gym `spaces` -- gym is not a dependency
of this package. Environments ASSIGN TorchRL specs, normally in `__init__`. The five LIVE
plain envs share that assignment via `TorchTradeLiveEnv._init_action_space`, and the five
SLTP ones via `SLTPMixin._init_bracket_action_space` (#288); the offline envs assign
directly:

```python
import torch
from torchrl.data import Categorical, Composite, Unbounded

observation_spec = Composite(
    market_data_1Minute_10=Unbounded(shape=(10, 5), dtype=torch.float32),
    account_state=Unbounded(shape=(6,), dtype=torch.float32),
)
action_spec = Categorical(3)  # index into action_levels
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
from torchrl.envs.utils import check_env_specs

from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

def test_env():
    # A CONCRETE env, not the MyCustomEnv sketch above: that one deliberately leaves
    # _get_observation unwritten, so it satisfies the ABC and still raises on reset().
    env = SequentialTradingEnv(
        test_data,
        SequentialTradingEnvConfig(
            time_frames=["1Min"], window_sizes=[10], execute_on="1Min"
        ),
    )

    # Check environment specs
    check_env_specs(env)

    # Test reset
    td = env.reset()

    # action_spec, not action_space -- and step takes/returns a TensorDict.
    td["action"] = env.action_spec.rand()
    transition, td = env.step_and_maybe_reset(td)
    # A tensor is never None, so `is not None` here would assert nothing.
    assert transition["next", "done"].numel() == 1
```

## See Also

- [Utilities Documentation](../utils/README.md)
- [Offline Environments](../../../docs/environments/offline.md)
- [Live Environments](../live/README.md)
- [Main README](../README.md)
