# Reward Functions

Reward functions shape your agent's behavior. TorchTrade provides a flexible system for defining custom reward functions that go beyond simple log returns.

## Default Behavior

By default, all environments use log returns:

```python
reward = log(portfolio_value_t / portfolio_value_t-1)
```

You can potentially improve learning and generalization by customizing this reward function.

---

## How It Works

Custom reward functions receive a `RewardContext` object and return a float reward:

```python
from torchtrade.envs.reward import RewardContext

def my_reward(ctx: RewardContext) -> float:
    """Custom reward function"""
    return reward_value
```

---

## RewardContext Fields

### Quick Reference Table

| Category | Field | Type | Description |
|----------|-------|------|-------------|
| **Core** | `old_portfolio_value` | float | Portfolio value before action |
| | `new_portfolio_value` | float | Portfolio value after action |
| | `action` | int | Action taken (index) |
| | `current_step` | int | Current episode step (0-indexed) |
| | `max_steps` | int | Maximum steps in episode |
| | `trade_executed` | bool | Whether a trade occurred |
| | `fee_paid` | float | Transaction fees this step |
| | `slippage_amount` | float | Slippage incurred this step |
| **Sequential Envs** | `portfolio_value_history` | list[float] | Past portfolio values |
| | `action_history` | list[int] | Past actions |
| | `reward_history` | list[float] | Past rewards |
| | `buy_and_hold_value` | float | Buy & hold benchmark (terminal only) |
| **Futures Envs** | `leverage` | float | Current leverage |
| | `margin_ratio` | float | Current margin ratio |
| | `liquidation_price` | float | Liquidation price |
| | `liquidated` | bool | Whether position was liquidated |
| **One-Step Envs** | `rollout_returns` | tensor | Returns during rollout |

---

## Basic Examples

### Example 1: Transaction Cost Penalty

```python
from torchtrade.envs.reward import RewardContext
import numpy as np

def cost_penalty_reward(ctx: RewardContext) -> float:
    """Penalize transaction costs to discourage overtrading"""
    if ctx.old_portfolio_value <= 0:
        return 0.0

    # Base reward: log return
    log_return = np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)

    # Penalty for fees and slippage (normalized by portfolio value)
    cost_penalty = -(ctx.fee_paid + ctx.slippage_amount) / ctx.old_portfolio_value

    return log_return + cost_penalty

# Use in environment
config = SeqLongOnlyEnvConfig(
    reward_fn=cost_penalty_reward,
    ...
)
```

### Example 2: Sharpe Ratio Reward

```python
def sharpe_ratio_reward(ctx: RewardContext) -> float:
    """Reward based on running Sharpe ratio"""
    # Access portfolio history
    history = ctx.metadata.get('portfolio_value_history', [])

    if len(history) < 10:  # Need minimum history
        if ctx.old_portfolio_value <= 0:
            return 0.0
        return np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)

    # Compute returns from history
    returns = [np.log(history[i] / history[i-1]) for i in range(1, len(history))]

    # Compute Sharpe ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-9
    sharpe = mean_return / std_return

    return np.clip(sharpe, -10.0, 10.0)
```

### Example 3: Sparse Terminal Reward

```python
def terminal_reward(ctx: RewardContext) -> float:
    """Reward only at episode end (for episodic optimization)"""
    # No reward during episode
    if ctx.current_step < ctx.max_steps - 1:
        return 0.0

    # Terminal reward: compare to buy & hold
    buy_hold = ctx.metadata.get('buy_and_hold_value', ctx.old_portfolio_value)

    if ctx.new_portfolio_value > buy_hold:
        return 1.0  # Beat buy & hold
    else:
        return -1.0  # Lost to buy & hold
```

---

## When to Use Each Reward

| Reward Type | Use Case | Pros | Cons |
|-------------|----------|------|------|
| **Log Return** (default) | General purpose | Simple, stable | Ignores costs, risk |
| **Cost Penalty** | Reduce overtrading | Discourages frequent trades | May miss opportunities |
| **Sharpe Ratio** | Risk-adjusted performance | Considers volatility | Requires history, unstable early |
| **Terminal Sparse** | Episodic tasks | Clear objective | Sparse signal, slow learning |
| **Drawdown Penalty** | Risk management | Controls max loss | May exit winners early |
| **Win Rate Bonus** | Consistency focus | Rewards reliability | May sacrifice returns |

---

## Design Principles

### 1. Dense vs Sparse Rewards

**Dense rewards** (every step):
- Faster learning
- More signal for gradient updates
- Risk: May optimize for short-term gains

**Sparse rewards** (terminal only):
- Aligns with true objective (final return)
- Less noise
- Risk: Slower learning, credit assignment problem

### 2. Normalize Rewards

Keep rewards in a consistent range (e.g., [-1, 1]) for stable learning:

```python
# Clip extreme values
reward = np.clip(reward, -10.0, 10.0)

# Or normalize by running statistics
reward = (reward - running_mean) / (running_std + 1e-9)
```

### 3. Avoid Lookahead Bias

Only use information available at decision time:

```python
# ❌ Lookahead bias
future_return = history[t+1] / history[t] - 1

# ✅ Correct - past data only
past_return = history[t] / history[t-1] - 1
```

### 4. Consider Environment Type

**Sequential Environments**: Can use history for complex rewards
**One-Step Environments**: Limited to immediate reward (no history)
**Futures Environments**: Consider liquidation risk in reward

---

## Advanced Patterns

### Multi-Objective Rewards

Combine multiple objectives with weighted sum:

```python
def multi_objective_reward(ctx: RewardContext) -> float:
    # Component 1: Returns
    log_return = np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)

    # Component 2: Cost penalty
    cost = -(ctx.fee_paid + ctx.slippage_amount) / ctx.old_portfolio_value

    # Component 3: Risk penalty (drawdown)
    history = ctx.metadata.get('portfolio_value_history', [])
    max_value = max(history) if history else ctx.old_portfolio_value
    drawdown = (ctx.new_portfolio_value - max_value) / max_value
    drawdown_penalty = min(0, drawdown) * 0.5  # Penalize only negative

    # Weighted combination
    return (
        1.0 * log_return +      # 100% weight on returns
        0.5 * cost +             # 50% weight on costs
        0.3 * drawdown_penalty   # 30% weight on drawdown
    )
```

### Adaptive Rewards

Adjust rewards based on training progress:

```python
class AdaptiveReward:
    def __init__(self):
        self.episode_count = 0

    def __call__(self, ctx: RewardContext) -> float:
        # Early training: Dense rewards for exploration
        if self.episode_count < 1000:
            return np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)

        # Later training: Sparse terminal rewards for optimization
        if ctx.current_step < ctx.max_steps - 1:
            return 0.0
        return ctx.new_portfolio_value - ctx.old_portfolio_value

    def on_episode_end(self):
        self.episode_count += 1

# Note: You'll need to manually call on_episode_end in your training loop
```

---

## Debugging Rewards

### Log Reward Statistics

```python
def debug_reward(ctx: RewardContext) -> float:
    reward = my_reward_fn(ctx)

    # Log statistics periodically
    if ctx.current_step % 100 == 0:
        print(f"Step {ctx.current_step}:")
        print(f"  Reward: {reward:.4f}")
        print(f"  Portfolio: {ctx.new_portfolio_value:.2f}")
        print(f"  Action: {ctx.action}")

    return reward
```

### Check Reward Distribution

```python
# Collect rewards during training
rewards = []

# After some episodes
import matplotlib.pyplot as plt

plt.hist(rewards, bins=50)
plt.title("Reward Distribution")
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.show()

# Check statistics
print(f"Mean: {np.mean(rewards):.4f}")
print(f"Std: {np.std(rewards):.4f}")
print(f"Min: {np.min(rewards):.4f}")
print(f"Max: {np.max(rewards):.4f}")
```

Good distributions:
- **Not constant**: Should vary with performance
- **Reasonable range**: Typically [-10, 10] for log returns
- **Some positive values**: Agent can achieve rewards

Bad patterns:
- **All zeros**: Reward function may have bug
- **Extreme outliers**: May need clipping
- **Constant value**: Reward not discriminating actions

---

## Next Steps

- **[Feature Engineering](custom-features.md)** - Engineer features that support your reward function
- **[Understanding the Sampler](sampler.md)** - How data flows through environments
- **[Loss Functions](../components/losses.md)** - Training objectives that work with rewards
- **[Offline Environments](../environments/offline.md)** - Apply custom rewards to environments
