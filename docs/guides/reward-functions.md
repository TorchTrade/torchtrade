# Custom Reward Functions

Reward functions are critical for training successful trading policies. TorchTrade provides a flexible system for defining custom reward functions that shape your agent's behavior.

## Default Behavior

By default, all environments use simple log returns:

```python
reward = log(portfolio_value_t / portfolio_value_t-1)
```

This is a reasonable starting point, but you can often improve learning by customizing the reward function.

## How It Works

Custom reward functions receive a `RewardContext` object containing all necessary state and return a float reward value:

```python
from torchtrade.envs.reward import RewardContext

def my_reward(ctx: RewardContext) -> float:
    """Custom reward function"""
    # Compute reward using ctx fields
    return reward_value
```

## RewardContext Fields

The `RewardContext` provides access to:

### Core Fields (Always Available)
- `old_portfolio_value`: Portfolio value before action
- `new_portfolio_value`: Portfolio value after action
- `action`: Action taken (integer index)
- `current_step`: Current episode step (0-indexed)
- `max_steps`: Maximum steps in episode
- `trade_executed`: Whether a trade occurred (bool)
- `fee_paid`: Transaction fees paid this step
- `slippage_amount`: Slippage incurred this step

### Metadata (Environment-Specific)
- `metadata`: Dict with environment-specific data

**Offline Sequential Environments:**
- `portfolio_value_history`: List of past portfolio values
- `action_history`: List of past actions
- `reward_history`: List of past rewards
- `buy_and_hold_value`: Buy & hold benchmark (terminal step only)

**Futures Environments:**
- `leverage`: Current leverage
- `margin_ratio`: Current margin ratio
- `liquidation_price`: Liquidation price
- `liquidated`: Whether position was liquidated (bool)

**One-Step Environments:**
- `rollout_returns`: Returns during rollout

## Basic Examples

### Example 1: Simple Log Return (Default)

```python
from torchtrade.envs.reward import RewardContext
import numpy as np

def log_return_reward(ctx: RewardContext) -> float:
    """Default reward: log return"""
    if ctx.old_portfolio_value <= 0:
        return 0.0

    return np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)
```

### Example 2: Transaction Cost Penalty

```python
def cost_penalty_reward(ctx: RewardContext) -> float:
    """Penalize transaction costs"""
    if ctx.old_portfolio_value <= 0:
        return 0.0

    # Base reward: log return
    log_return = np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)

    # Penalty for fees and slippage
    cost_penalty = -(ctx.fee_paid + ctx.slippage_amount) / ctx.old_portfolio_value

    return log_return + cost_penalty
```

### Example 3: Sparse Terminal Reward

```python
def terminal_reward(ctx: RewardContext) -> float:
    """Reward only at episode end"""
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

## Intermediate Examples

### Example 4: Sharpe Ratio Reward

```python
from torchtrade.envs.reward import RewardContext
import numpy as np

def sharpe_ratio_reward(ctx: RewardContext) -> float:
    """Reward based on running Sharpe ratio"""
    # Access portfolio history
    history = ctx.metadata.get('portfolio_value_history', [])

    if len(history) < 10:  # Need minimum history
        # Use log return for first few steps
        if ctx.old_portfolio_value <= 0:
            return 0.0
        return np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)

    # Compute returns from history
    returns = [
        np.log(history[i] / history[i-1])
        for i in range(1, len(history))
    ]

    # Compute Sharpe ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-9  # Avoid division by zero
    sharpe = mean_return / std_return

    # Clip to reasonable range
    return np.clip(sharpe, -10.0, 10.0)
```

### Example 5: Drawdown Penalty

```python
def drawdown_penalty_reward(ctx: RewardContext) -> float:
    """Penalize large drawdowns"""
    if ctx.old_portfolio_value <= 0:
        return 0.0

    # Base reward: log return
    log_return = np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)

    # Compute running max portfolio value
    history = ctx.metadata.get('portfolio_value_history', [ctx.old_portfolio_value])
    max_value = max(history)

    # Current drawdown
    drawdown = (ctx.new_portfolio_value - max_value) / max_value

    # Penalty for large drawdowns
    drawdown_penalty = 0.0
    if drawdown < -0.1:  # More than 10% drawdown
        drawdown_penalty = drawdown * 2  # 2x penalty

    return log_return + drawdown_penalty
```

### Example 6: Win Rate Bonus

```python
def win_rate_reward(ctx: RewardContext) -> float:
    """Bonus for maintaining high win rate"""
    if ctx.old_portfolio_value <= 0:
        return 0.0

    # Base reward
    log_return = np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)

    # Check win rate from history
    reward_history = ctx.metadata.get('reward_history', [])

    if len(reward_history) > 10:
        winning_trades = sum(1 for r in reward_history[-10:] if r > 0)
        win_rate = winning_trades / 10.0

        # Bonus for high win rate
        if win_rate > 0.6:
            log_return += 0.1  # +10% bonus

    return log_return
```

## Advanced Examples

### Example 7: Risk-Adjusted Profit

```python
def risk_adjusted_profit(ctx: RewardContext) -> float:
    """
    Comprehensive reward combining multiple factors:
    - Portfolio returns
    - Transaction costs
    - Trading frequency
    - Terminal performance vs buy & hold
    """
    if ctx.old_portfolio_value <= 0:
        return 0.0

    # 1. Base reward: log return
    log_return = np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)

    # 2. Transaction cost penalty
    fee_penalty = -ctx.fee_paid / ctx.old_portfolio_value

    # 3. Penalty for excessive trading
    trade_penalty = -0.001 if ctx.trade_executed else 0.0

    # 4. Terminal bonus for beating buy & hold
    terminal_bonus = 0.0
    if ctx.current_step >= ctx.max_steps - 1:
        buy_hold = ctx.metadata.get('buy_and_hold_value', ctx.old_portfolio_value)
        if ctx.new_portfolio_value > buy_hold:
            outperformance = (ctx.new_portfolio_value - buy_hold) / buy_hold
            terminal_bonus = np.clip(outperformance * 10, 0, 5.0)

    return log_return + fee_penalty + trade_penalty + terminal_bonus
```

### Example 8: Futures-Specific Reward

```python
def futures_margin_aware_reward(ctx: RewardContext) -> float:
    """
    Reward function for futures environments that considers:
    - Leverage
    - Margin ratio
    - Liquidation risk
    """
    if ctx.old_portfolio_value <= 0:
        return 0.0

    # Base reward
    log_return = np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)

    # Get futures-specific metadata
    leverage = ctx.metadata.get('leverage', 1.0)
    margin_ratio = ctx.metadata.get('margin_ratio', 1.0)
    liquidated = ctx.metadata.get('liquidated', False)

    # Heavy penalty for liquidation
    if liquidated:
        return -10.0

    # Penalty for risky margin levels
    margin_penalty = 0.0
    if margin_ratio < 0.3:  # Low margin ratio
        margin_penalty = -1.0 * (0.3 - margin_ratio)

    # Adjust reward by leverage (higher leverage = more risk)
    risk_adjusted_return = log_return / np.sqrt(leverage)

    return risk_adjusted_return + margin_penalty
```

### Example 9: Hybrid Dense + Sparse Reward

```python
def hybrid_dense_sparse_reward(ctx: RewardContext) -> float:
    """
    Combine dense (step-wise) and sparse (terminal) rewards.

    Dense rewards help learning, sparse rewards reduce noise.
    """
    # Dense component: Small step-wise reward
    if ctx.old_portfolio_value <= 0:
        dense_reward = 0.0
    else:
        dense_reward = 0.1 * np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)

    # Sparse component: Large terminal reward
    sparse_reward = 0.0
    if ctx.current_step >= ctx.max_steps - 1:
        buy_hold = ctx.metadata.get('buy_and_hold_value', ctx.old_portfolio_value)

        if ctx.new_portfolio_value > buy_hold * 1.05:  # Beat by 5%
            sparse_reward = 10.0
        elif ctx.new_portfolio_value > buy_hold:  # Beat slightly
            sparse_reward = 5.0
        elif ctx.new_portfolio_value > buy_hold * 0.95:  # Close
            sparse_reward = 0.0
        else:  # Lost significantly
            sparse_reward = -5.0

    return dense_reward + sparse_reward
```

### Example 10: One-Step Episodic Reward

```python
def onestep_rollout_reward(ctx: RewardContext) -> float:
    """
    Reward for one-step environments using rollout returns.

    Optimized for GRPO training.
    """
    # Get rollout returns from metadata
    rollout_returns = ctx.metadata.get('rollout_returns', [])

    if len(rollout_returns) == 0:
        return 0.0

    # Aggregate rollout performance
    total_return = sum(rollout_returns)
    mean_return = np.mean(rollout_returns)
    std_return = np.std(rollout_returns) + 1e-9

    # Sharpe-like metric
    sharpe = mean_return / std_return

    # Combine metrics
    reward = total_return + sharpe

    return np.clip(reward, -10.0, 10.0)
```

## Using Custom Rewards

### In Environment Config

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig

# Define custom reward
def my_reward(ctx: RewardContext) -> float:
    return ...  # Your reward logic

# Pass to config
config = SeqLongOnlyEnvConfig(
    reward_function=my_reward,  # ← Your custom reward
    time_frames=[1, 5, 15],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=1000
)

env = SeqLongOnlyEnv(df, config)
```

### Pre-Built Reward Functions

TorchTrade includes pre-built reward functions:

```python
from torchtrade.envs.reward import (
    sharpe_ratio_reward,
    drawdown_penalty_reward,
    terminal_comparison_reward,
    hybrid_dense_sparse_reward,
    realized_pnl_reward
)

# Use pre-built reward
config = SeqLongOnlyEnvConfig(
    reward_function=sharpe_ratio_reward,
    ...
)
```

## Best Practices

### 1. Keep Rewards Bounded

```python
def bounded_reward(ctx: RewardContext) -> float:
    reward = ...  # Compute reward

    # Clip to reasonable range
    return np.clip(reward, -10.0, 10.0)
```

### 2. Consider Scale

Rewards in range `[-10, 10]` work well with most RL algorithms:

```python
# ✅ Good scale
return np.clip(sharpe * 5.0, -10.0, 10.0)

# ❌ Too large - may cause instability
return sharpe * 1000.0

# ❌ Too small - may slow learning
return sharpe * 0.001
```

### 3. Balance Dense vs Sparse

```python
# Dense: Reward every step (helps learning)
dense = 0.1 * log_return

# Sparse: Reward at episode end (reduces noise)
sparse = 10.0 if terminal_condition else 0.0

# Combine both
return dense + sparse
```

### 4. Penalize Bad Behavior

```python
# Penalize excessive trading
if ctx.trade_executed:
    reward -= 0.001

# Penalize large drawdowns
if drawdown < -0.2:
    reward -= 5.0

# Penalize liquidations (futures)
if ctx.metadata.get('liquidated', False):
    reward -= 10.0
```

### 5. Test Extensively

```python
def test_reward_function():
    """Test reward function across scenarios"""
    # Test profit scenario
    ctx_profit = RewardContext(
        old_portfolio_value=1000,
        new_portfolio_value=1100,
        action=2,
        current_step=10,
        max_steps=100,
        trade_executed=True,
    )
    assert my_reward(ctx_profit) > 0, "Should reward profit"

    # Test loss scenario
    ctx_loss = RewardContext(
        old_portfolio_value=1000,
        new_portfolio_value=900,
        action=0,
        current_step=10,
        max_steps=100,
        trade_executed=True,
    )
    assert my_reward(ctx_loss) < 0, "Should penalize loss"

    print("✅ All reward tests passed!")

test_reward_function()
```

## Debugging Rewards

### Print Reward Components

```python
def debug_reward(ctx: RewardContext) -> float:
    """Reward function with debug prints"""
    if ctx.old_portfolio_value <= 0:
        return 0.0

    log_return = np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)
    fee_penalty = -ctx.fee_paid / ctx.old_portfolio_value

    total_reward = log_return + fee_penalty

    # Debug print every 10 steps
    if ctx.current_step % 10 == 0:
        print(f"Step {ctx.current_step}:")
        print(f"  Log return: {log_return:.4f}")
        print(f"  Fee penalty: {fee_penalty:.4f}")
        print(f"  Total reward: {total_reward:.4f}")

    return total_reward
```

### Log Reward Statistics

```python
from collections import deque

reward_buffer = deque(maxlen=100)

def tracked_reward(ctx: RewardContext) -> float:
    reward = ...  # Compute reward

    # Track rewards
    reward_buffer.append(reward)

    # Print statistics every 100 steps
    if ctx.current_step % 100 == 0 and len(reward_buffer) > 0:
        print(f"Last 100 rewards - Mean: {np.mean(reward_buffer):.4f}, "
              f"Std: {np.std(reward_buffer):.4f}, "
              f"Min: {np.min(reward_buffer):.4f}, "
              f"Max: {np.max(reward_buffer):.4f}")

    return reward
```

## Common Pitfalls

### Pitfall 1: Division by Zero

```python
# ❌ Can divide by zero
sharpe = mean_return / std_return

# ✅ Add small epsilon
sharpe = mean_return / (std_return + 1e-9)
```

### Pitfall 2: Unbounded Rewards

```python
# ❌ Can explode to infinity
return portfolio_value * 1000

# ✅ Clip rewards
return np.clip(log_return * 10, -10, 10)
```

### Pitfall 3: Lookahead Bias

```python
# ❌ Using future data
future_return = ctx.metadata.get('next_step_return')  # Don't do this!

# ✅ Only use past and current data
past_returns = ctx.metadata.get('reward_history', [])
```

### Pitfall 4: Too Complex

```python
# ❌ Over-engineered
def complex_reward(ctx):
    # 50 lines of complex logic...
    # Agent struggles to learn

# ✅ Start simple
def simple_reward(ctx):
    return np.log(ctx.new_portfolio_value / ctx.old_portfolio_value)
```

## Next Steps

- **[Custom Feature Engineering](custom-features.md)** - Add technical indicators
- **[Understanding the Sampler](sampler.md)** - How data sampling works
- **[Building Custom Environments](custom-environment.md)** - Extend TorchTrade
- **[Offline Environments](../environments/offline.md)** - Apply custom rewards

## Recommended Reading

- [Reward Shaping in RL](https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html)
- [Deep RL That Matters](https://arxiv.org/abs/1709.06560)
- [Policy Gradient Methods](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
