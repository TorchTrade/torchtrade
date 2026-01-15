# Building Custom Environments

This guide shows you how to extend TorchTrade to create custom trading environments for your specific needs.

## When to Build a Custom Environment

Consider building a custom environment when you need:

- **Custom asset types**: Options, forex, commodities, etc.
- **Complex order types**: Market-on-close, iceberg orders, etc.
- **Custom state representations**: Order book data, sentiment, news, etc.
- **Specific trading rules**: Pattern day trading rules, portfolio constraints, etc.
- **New exchange integrations**: Unsupported brokers or APIs

## Environment Architecture

TorchTrade environments inherit from TorchRL's `EnvBase`:

```
EnvBase (TorchRL)
    ↓
BaseTorchTradeEnv (Abstract base)
    ↓
YourCustomEnv
```

Key methods to implement:
- `_reset()`: Initialize episode state
- `_step(tensordict)`: Execute action and update state
- `_set_seed(seed)`: Set random seed
- `observation_spec`: Define observation space
- `action_spec`: Define action space
- `reward_spec`: Define reward space

## Example 1: Simple Custom Environment

Let's create a minimal custom environment from scratch:

```python
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from tensordict import TensorDict
import torch

class SimpleCustomEnv(EnvBase):
    """
    Minimal custom trading environment.

    State: [price]
    Actions: 0 (HOLD), 1 (BUY), 2 (SELL)
    Reward: Log return
    """

    def __init__(self, prices: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.prices = prices  # Historical prices
        self.current_step = 0
        self.position = 0  # 0 or 1 (no position or holding)
        self.entry_price = 0.0

        # Define specs
        self._observation_spec = CompositeSpec({
            "price": UnboundedContinuousTensorSpec(shape=(1,)),
            "position": UnboundedContinuousTensorSpec(shape=(1,)),
        })

        self._action_spec = DiscreteTensorSpec(n=3)  # 3 actions
        self._reward_spec = UnboundedContinuousTensorSpec(shape=(1,))

    def _reset(self, tensordict=None, **kwargs):
        """Reset environment to initial state"""
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0

        return TensorDict({
            "price": torch.tensor([self.prices[0].item()]),
            "position": torch.tensor([0.0]),
        }, batch_size=self.batch_size)

    def _step(self, tensordict: TensorDict):
        """Execute one step"""
        action = tensordict["action"].item()

        # Get current price
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

        # Next observation
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
prices = torch.linspace(100, 150, 1000)  # Simulated prices
env = SimpleCustomEnv(prices)

tensordict = env.reset()
tensordict["action"] = torch.tensor([1])  # BUY
tensordict = env.step(tensordict)
print(f"Reward: {tensordict['reward'].item()}")
```

## Example 2: Extending SeqLongOnlyEnv

The easiest way to create a custom environment is to extend an existing one:

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from tensordict import TensorDict
import torch

class PortfolioConstraintEnv(SeqLongOnlyEnv):
    """
    Custom environment with portfolio constraints.

    - Maximum position size: 50% of cash
    - Minimum holding time: 10 steps
    """

    def __init__(self, df, config, max_position_pct=0.5, min_hold_time=10):
        super().__init__(df, config)
        self.max_position_pct = max_position_pct
        self.min_hold_time = min_hold_time
        self.steps_since_buy = 0

    def _step(self, tensordict: TensorDict):
        """Override step to enforce constraints"""
        action = tensordict["action"].item()

        # Enforce minimum holding time
        if action == 0 and self.steps_since_buy < self.min_hold_time:
            # Force HOLD action
            tensordict["action"] = torch.tensor([1])

        # Call parent step
        tensordict = super()._step(tensordict)

        # Track holding time
        if action == 2:  # BUY
            self.steps_since_buy = 0
        elif self.position_size > 0:
            self.steps_since_buy += 1

        return tensordict

    def _calculate_position_size(self, action: int, current_price: float) -> float:
        """Override to limit position size"""
        if action == 2:  # BUY
            # Max 50% of cash
            max_cash = self.cash * self.max_position_pct
            return max_cash / current_price
        return 0.0


# Usage
config = SeqLongOnlyEnvConfig(...)
env = PortfolioConstraintEnv(df, config, max_position_pct=0.5, min_hold_time=10)
```

## Example 3: Custom Observation Space

Add custom observations like order book data or sentiment:

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDict
import torch

class SentimentAugmentedEnv(SeqLongOnlyEnv):
    """
    Environment with sentiment data in observations.
    """

    def __init__(self, df, config, sentiment_data: torch.Tensor):
        super().__init__(df, config)
        self.sentiment_data = sentiment_data  # [num_steps, sentiment_dim]

        # Update observation spec to include sentiment
        self._observation_spec = CompositeSpec({
            **self._observation_spec,  # Keep existing specs
            "sentiment": UnboundedContinuousTensorSpec(shape=(sentiment_data.shape[1],)),
        })

    def _get_observation(self, index: int) -> TensorDict:
        """Override to add sentiment data"""
        obs = super()._get_observation(index)

        # Add sentiment to observation
        obs["sentiment"] = self.sentiment_data[index]

        return obs


# Usage
sentiment = torch.randn(len(df), 10)  # 10-dim sentiment vectors
config = SeqLongOnlyEnvConfig(...)
env = SentimentAugmentedEnv(df, config, sentiment_data=sentiment)
```

## Example 4: Custom Action Space

Create complex action spaces (e.g., continuous position sizing):

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchrl.data import BoundedTensorSpec
from tensordict import TensorDict
import torch

class ContinuousPositionEnv(SeqLongOnlyEnv):
    """
    Environment with continuous position sizing.

    Action: float in [0, 1] representing % of cash to invest
    """

    def __init__(self, df, config):
        super().__init__(df, config)

        # Override action spec for continuous actions
        self._action_spec = BoundedTensorSpec(
            low=0.0,
            high=1.0,
            shape=(1,),
        )

    def _step(self, tensordict: TensorDict):
        """Override step for continuous actions"""
        action_value = tensordict["action"].item()  # float in [0, 1]

        current_price = self._get_current_price()

        # Calculate position size based on continuous action
        if action_value < 0.1:  # Close position (< 10%)
            if self.position_size > 0:
                self._sell_position(current_price)
        elif action_value > 0.1:  # Open/adjust position
            target_investment = self.cash * action_value
            target_position_size = target_investment / current_price

            # Adjust position
            if target_position_size > self.position_size:
                # Buy more
                amount_to_buy = target_position_size - self.position_size
                self._buy_position(current_price, amount_to_buy)
            elif target_position_size < self.position_size:
                # Sell some
                amount_to_sell = self.position_size - target_position_size
                self._partial_sell(current_price, amount_to_sell)

        # Calculate reward
        reward = self._calculate_reward()

        # Move to next step
        self.current_step += 1
        done = self._check_termination()

        # Get next observation
        next_obs = self._get_observation(self.current_step) if not done else self._get_observation(self.current_step - 1)

        return TensorDict({
            **next_obs,
            "reward": torch.tensor([reward]),
            "done": torch.tensor([done]),
        }, batch_size=self.batch_size)


# Usage
config = SeqLongOnlyEnvConfig(...)
env = ContinuousPositionEnv(df, config)
```

## Example 5: Multi-Asset Portfolio Environment

Manage a portfolio of multiple assets:

```python
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, BoundedTensorSpec
from tensordict import TensorDict
import torch
import pandas as pd

class MultiAssetPortfolioEnv(EnvBase):
    """
    Portfolio environment with multiple assets.

    State: [prices_asset1, prices_asset2, ..., portfolio_weights]
    Actions: [weight_asset1, weight_asset2, ...] (must sum to 1)
    """

    def __init__(self, dfs: List[pd.DataFrame], num_assets: int, **kwargs):
        super().__init__(**kwargs)
        self.dfs = dfs  # List of DataFrames (one per asset)
        self.num_assets = num_assets
        self.current_step = 0
        self.portfolio_value = 1.0  # Start with $1
        self.weights = torch.zeros(num_assets)  # Equal weight initially
        self.weights[:] = 1.0 / num_assets

        # Define specs
        self._observation_spec = CompositeSpec({
            "prices": UnboundedContinuousTensorSpec(shape=(num_assets,)),
            "weights": UnboundedContinuousTensorSpec(shape=(num_assets,)),
            "portfolio_value": UnboundedContinuousTensorSpec(shape=(1,)),
        })

        self._action_spec = BoundedTensorSpec(
            low=0.0,
            high=1.0,
            shape=(num_assets,),  # Weights for each asset
        )

        self._reward_spec = UnboundedContinuousTensorSpec(shape=(1,))

    def _reset(self, tensordict=None, **kwargs):
        """Reset environment"""
        self.current_step = 0
        self.portfolio_value = 1.0
        self.weights = torch.ones(self.num_assets) / self.num_assets

        prices = torch.tensor([df.iloc[0]["close"] for df in self.dfs])

        return TensorDict({
            "prices": prices,
            "weights": self.weights.clone(),
            "portfolio_value": torch.tensor([1.0]),
        }, batch_size=self.batch_size)

    def _step(self, tensordict: TensorDict):
        """Execute one step"""
        # Get new weights from action (normalize to sum to 1)
        new_weights = tensordict["action"]
        new_weights = new_weights / new_weights.sum()

        # Get current and next prices
        current_prices = torch.tensor([df.iloc[self.current_step]["close"] for df in self.dfs])
        self.current_step += 1
        next_prices = torch.tensor([df.iloc[self.current_step]["close"] for df in self.dfs])

        # Calculate returns for each asset
        returns = (next_prices - current_prices) / current_prices

        # Calculate portfolio return
        portfolio_return = (self.weights * returns).sum().item()

        # Update portfolio value
        old_value = self.portfolio_value
        self.portfolio_value *= (1 + portfolio_return)

        # Rebalance to new weights
        self.weights = new_weights

        # Reward: log return
        reward = torch.log(torch.tensor(self.portfolio_value / old_value))

        # Check done
        done = self.current_step >= min(len(df) for df in self.dfs) - 1

        return TensorDict({
            "prices": next_prices,
            "weights": self.weights.clone(),
            "portfolio_value": torch.tensor([self.portfolio_value]),
            "reward": reward.unsqueeze(0),
            "done": torch.tensor([done]),
        }, batch_size=self.batch_size)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)


# Usage
dfs = [btc_df, eth_df, sol_df]  # Multiple asset DataFrames
env = MultiAssetPortfolioEnv(dfs, num_assets=3)
```

## Best Practices

### 1. Use TensorDict Consistently

Always return `TensorDict` from `_reset()` and `_step()`:

```python
# ✅ Correct
return TensorDict({
    "observation": obs,
    "reward": reward,
    "done": done,
}, batch_size=self.batch_size)

# ❌ Wrong
return {"observation": obs, "reward": reward, "done": done}
```

### 2. Define Proper Specs

Specs tell TorchRL about your environment's structure:

```python
self._observation_spec = CompositeSpec({
    "price": UnboundedContinuousTensorSpec(shape=(1,)),
    "position": UnboundedContinuousTensorSpec(shape=(1,)),
})

self._action_spec = DiscreteTensorSpec(n=3)  # 3 discrete actions

self._reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
```

### 3. Handle Edge Cases

```python
def _step(self, tensordict: TensorDict):
    # Check for invalid actions
    action = tensordict["action"].item()
    if action < 0 or action >= self.num_actions:
        raise ValueError(f"Invalid action: {action}")

    # Check for NaN values
    if torch.isnan(tensordict["observation"]).any():
        raise ValueError("NaN values in observation")

    # Handle division by zero
    if self.portfolio_value <= 0:
        return self._terminate_episode()

    # ... rest of step logic
```

### 4. Document Your Environment

```python
class MyCustomEnv(EnvBase):
    """
    Custom trading environment for [specific use case].

    Observation Space:
        - field1: Description
        - field2: Description

    Action Space:
        - 0: Action description
        - 1: Action description

    Reward:
        Description of reward function

    Args:
        param1: Description
        param2: Description

    Example:
        >>> env = MyCustomEnv(...)
        >>> obs = env.reset()
        >>> obs = env.step(action)
    """
```

### 5. Test Thoroughly

```python
def test_custom_env():
    """Test suite for custom environment"""
    env = MyCustomEnv(...)

    # Test reset
    obs = env.reset()
    assert "observation" in obs
    assert not torch.isnan(obs["observation"]).any()

    # Test step
    action = torch.tensor([0])
    next_obs = env.step(TensorDict({"action": action}))
    assert "reward" in next_obs
    assert "done" in next_obs

    # Test episode completion
    for _ in range(1000):
        action = torch.randint(0, 3, (1,))
        next_obs = env.step(TensorDict({"action": action}))
        if next_obs["done"].item():
            break

    print("✅ All tests passed!")

test_custom_env()
```

## Debugging Tips

### Print State at Each Step

```python
def _step(self, tensordict: TensorDict):
    action = tensordict["action"].item()

    print(f"Step {self.current_step}:")
    print(f"  Action: {action}")
    print(f"  Current price: {self.current_price}")
    print(f"  Position: {self.position_size}")
    print(f"  Portfolio value: {self.portfolio_value}")

    # ... rest of step logic
```

### Visualize Episode Trajectory

```python
import matplotlib.pyplot as plt

def visualize_episode(env, policy):
    """Visualize a full episode"""
    prices = []
    actions = []
    rewards = []

    obs = env.reset()
    done = False

    while not done:
        action = policy(obs)
        obs = env.step(TensorDict({"action": action}))

        prices.append(env.current_price)
        actions.append(action.item())
        rewards.append(obs["reward"].item())
        done = obs["done"].item()

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

    ax1.plot(prices)
    ax1.set_title("Price")

    ax2.plot(actions)
    ax2.set_title("Actions")

    ax3.plot(rewards)
    ax3.set_title("Rewards")

    plt.tight_layout()
    plt.show()
```

## Next Steps

Now that you understand how to build custom environments:

- **[Custom Feature Engineering](custom-features.md)** - Add custom observations
- **[Custom Reward Functions](reward-functions.md)** - Design better rewards
- **[Understanding the Sampler](sampler.md)** - Use the sampler in custom environments
- **[Offline Environments](../environments/offline.md)** - Study existing implementations

## Resources

- **[TorchRL Documentation](https://pytorch.org/rl/)** - TorchRL EnvBase API
- **[Gymnasium Documentation](https://gymnasium.farama.org/)** - Environment design patterns
- **[TorchTrade Source Code](https://github.com/TorchTrade/torchtrade_envs)** - Reference implementations
