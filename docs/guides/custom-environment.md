# Building Custom Environments

This guide shows how to extend TorchTrade for custom trading environments.

## When to Build a Custom Environment

Build a custom environment when you need:

- **Custom asset types**: Options, forex, commodities
- **Complex order types**: Market-on-close, iceberg orders
- **Custom state**: Order book data, sentiment, news
- **Specific trading rules**: Pattern day trading, portfolio constraints
- **New exchange integrations**: Unsupported brokers/APIs

---

## Environment Architecture

TorchTrade environments inherit from TorchRL's `EnvBase`:

```
EnvBase (TorchRL)
    ↓
BaseTorchTradeEnv (Abstract base - optional)
    ↓
YourCustomEnv
```

### Required Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `_reset()` | Initialize episode state | TensorDict with initial observation |
| `_step(tensordict)` | Execute action, update state | TensorDict with next observation |
| `_set_seed(seed)` | Set random seed for reproducibility | None |
| `observation_spec` | Define observation space | CompositeSpec |
| `action_spec` | Define action space | DiscreteTensorSpec or ContinuousTensorSpec |
| `reward_spec` | Define reward space | UnboundedContinuousTensorSpec |

---

## Example 1: Simple Custom Environment

Minimal environment from scratch:

```python
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from tensordict import TensorDict
import torch

class SimpleCustomEnv(EnvBase):
    """
    Minimal custom trading environment.

    State: [price, position]
    Actions: 0 (HOLD), 1 (BUY), 2 (SELL)
    Reward: Log return
    """

    def __init__(self, prices: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.prices = prices
        self.current_step = 0
        self.position = 0  # 0 or 1
        self.entry_price = 0.0

        # Define specs
        self._observation_spec = CompositeSpec({
            "price": UnboundedContinuousTensorSpec(shape=(1,)),
            "position": UnboundedContinuousTensorSpec(shape=(1,)),
        })

        self._action_spec = DiscreteTensorSpec(n=3)
        self._reward_spec = UnboundedContinuousTensorSpec(shape=(1,))

    def _reset(self, tensordict=None, **kwargs):
        """Reset to initial state"""
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0

        return TensorDict({
            "price": torch.tensor([self.prices[0].item()]),
            "position": torch.tensor([0.0]),
        }, batch_size=self.batch_size)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Execute one step"""
        action = tensordict["action"].item()
        current_price = self.prices[self.current_step].item()

        # Execute action
        reward = 0.0
        if action == 1 and self.position == 0:  # BUY
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 1:  # SELL
            reward = (current_price - self.entry_price) / self.entry_price
            self.position = 0
            self.entry_price = 0.0

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        # Build output tensordict
        next_price = self.prices[self.current_step].item() if not done else current_price

        return TensorDict({
            "price": torch.tensor([next_price]),
            "position": torch.tensor([float(self.position)]),
            "reward": torch.tensor([reward]),
            "done": torch.tensor([done]),
        }, batch_size=self.batch_size)

    def _set_seed(self, seed: int):
        """Set random seed"""
        torch.manual_seed(seed)

# Usage
prices = torch.randn(1000).cumsum(0) + 100  # Random walk prices
env = SimpleCustomEnv(prices, batch_size=[])

obs = env.reset()
for _ in range(100):
    action = env.action_spec.rand()  # Random action
    obs = env.step(action)
    if obs["done"]:
        break
```

---

## Example 2: Extending Existing Environments

Extend `SeqLongOnlyEnv` to add custom features:

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from tensordict import TensorDict
import torch

class CustomLongOnlyEnv(SeqLongOnlyEnv):
    """
    Extended SeqLongOnlyEnv with sentiment data.
    """

    def __init__(self, df, config: SeqLongOnlyEnvConfig, sentiment_data: torch.Tensor):
        super().__init__(df, config)
        self.sentiment_data = sentiment_data  # Timeseries sentiment scores

        # Extend observation spec
        from torchrl.data import UnboundedContinuousTensorSpec
        self._observation_spec["sentiment"] = UnboundedContinuousTensorSpec(shape=(1,))

    def _reset(self, tensordict=None, **kwargs):
        """Add sentiment to observations"""
        obs = super()._reset(tensordict, **kwargs)

        # Add current sentiment
        sentiment_idx = self.sampler.reset_index
        obs["sentiment"] = torch.tensor([self.sentiment_data[sentiment_idx].item()])

        return obs

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Add sentiment to step observations"""
        obs = super()._step(tensordict)

        # Add current sentiment
        sentiment_idx = self.sampler.current_index
        obs["sentiment"] = torch.tensor([self.sentiment_data[sentiment_idx].item()])

        return obs

# Usage
import pandas as pd

df = pd.read_csv("prices.csv")
sentiment = torch.randn(len(df))  # Random sentiment scores

config = SeqLongOnlyEnvConfig(
    time_frames=["1min", "5min"],
    window_sizes=[12, 8],
    execute_on=(5, "Minute"),
)

env = CustomLongOnlyEnv(df, config, sentiment)

# Policy network sees sentiment in observations
obs = env.reset()
print(obs.keys())  # [..., 'sentiment']
```

---

## Design Patterns

### 1. Composition Over Inheritance

Prefer composing existing components:

```python
class CustomEnv(SeqLongOnlyEnv):
    def __init__(self, df, config, custom_component):
        super().__init__(df, config)
        self.custom_component = custom_component  # Inject custom logic

    def _step(self, tensordict):
        obs = super()._step(tensordict)
        # Modify obs with custom_component
        obs["custom_feature"] = self.custom_component.compute(obs)
        return obs
```

### 2. Observation Spec Extension

Always update observation specs when adding new fields:

```python
# In __init__
self._observation_spec["new_field"] = UnboundedContinuousTensorSpec(shape=(N,))
```

### 3. State Management

Keep environment state in clear instance variables:

```python
def __init__(self):
    self.current_step = 0      # Clear state
    self.portfolio_value = 1000
    self.position = 0
    # ... not scattered across methods
```

---

## Testing Custom Environments

### 1. Spec Compliance

Verify specs match actual outputs:

```python
env = CustomEnv(...)

# Check reset
obs = env.reset()
assert env.observation_spec.is_in(obs), "Reset observation doesn't match spec"

# Check step
action = env.action_spec.rand()
obs = env.step(action)
assert env.observation_spec.is_in(obs), "Step observation doesn't match spec"
assert env.reward_spec.is_in(obs["reward"]), "Reward doesn't match spec"
```

### 2. Episode Completion

Ensure episodes terminate correctly:

```python
env = CustomEnv(...)
obs = env.reset()

for i in range(10000):  # Safety limit
    action = env.action_spec.rand()
    obs = env.step(action)
    if obs["done"]:
        print(f"Episode ended at step {i}")
        break
else:
    raise AssertionError("Episode never ended!")
```

### 3. Reward Sanity

Check reward values are reasonable:

```python
rewards = []
for episode in range(100):
    obs = env.reset()
    episode_reward = 0
    while not obs["done"]:
        action = env.action_spec.rand()
        obs = env.step(action)
        episode_reward += obs["reward"].item()
    rewards.append(episode_reward)

print(f"Mean reward: {sum(rewards)/len(rewards):.2f}")
print(f"Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
```

---

## Common Pitfalls

| Issue | Problem | Solution |
|-------|---------|----------|
| **Spec mismatch** | Observation shape != spec shape | Update `_observation_spec` in `__init__` |
| **Forgotten batch_size** | TensorDict missing batch_size | Always pass `batch_size=self.batch_size` |
| **Missing done signal** | Episode never ends | Set `done=True` in terminal state |
| **Mutable state** | State persists across episodes | Reset ALL state variables in `_reset()` |
| **Incorrect device** | Tensors on wrong device | Use `self.device` for all tensors |

---

## Next Steps

- **[Offline Environments](../environments/offline.md)** - Understand existing environment architecture
- **[Reward Functions](reward-functions.md)** - Define custom reward logic
- **[Feature Engineering](custom-features.md)** - Add technical indicators
- **[TorchRL EnvBase](https://pytorch.org/rl/reference/envs.html#envbase)** - Base class documentation
