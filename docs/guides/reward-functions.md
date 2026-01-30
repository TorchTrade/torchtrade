# Reward Functions

Reward functions shape your agent's behavior. TorchTrade provides a flexible system for defining custom reward functions that go beyond simple log returns.

## Default Behavior

By default, all environments use log returns:

```python
reward = log(portfolio_value_t / portfolio_value_t-1)
```

You can potentially improve learning and generalization by customizing this reward function. See below for some references on reward function design.

---

## How It Works

Custom reward functions receive a `HistoryTracker` object and return a float reward:

```python
def my_reward(history) -> float:
    """Custom reward function."""
    return reward_value
```

Pass your reward function via the environment config:

```python
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

config = SequentialTradingEnvConfig(
    reward_function=my_reward,
    ...
)
env = SequentialTradingEnv(df, config)
```

---

## HistoryTracker Fields

The `history` object passed to your reward function is a `HistoryTracker` instance with these attributes:

| Field | Type | Description |
|-------|------|-------------|
| `portfolio_values` | list[float] | Portfolio value at each step |
| `base_prices` | list[float] | Asset price at each step |
| `actions` | list[float] | Actions taken at each step |
| `rewards` | list[float] | Rewards received at each step |
| `positions` | list[float] | Position sizes (positive=long, negative=short, 0=flat) |
| `action_types` | list[str] | Action types ("hold", "buy", "sell", "long", "short") |

All lists grow by one element per step. Use indexing (e.g., `history.portfolio_values[-1]`) to access recent values.

---

## Built-in Reward Functions

TorchTrade provides three built-in reward functions in `torchtrade.envs.core.default_rewards`:

```python
from torchtrade.envs.core.default_rewards import (
    log_return_reward,       # Default - log(value_t / value_t-1)
    sharpe_ratio_reward,     # Running Sharpe ratio
    drawdown_penalty_reward, # Log return with drawdown penalty
)

config = SequentialTradingEnvConfig(
    reward_function=sharpe_ratio_reward,
    ...
)
```

### `log_return_reward` (default)

```python
reward = log(portfolio_value[-1] / portfolio_value[-2])
```

- Scale-invariant, symmetric, handles compounding
- Returns `-10.0` on bankruptcy, `0.0` if insufficient history

### `sharpe_ratio_reward`

Running Sharpe ratio over all historical portfolio values. Encourages risk-adjusted returns. Clipped to `[-10.0, 10.0]`.

### `drawdown_penalty_reward`

Log return plus a penalty when drawdown from peak exceeds 10%. Discourages large drawdowns while rewarding gains.

---

## Custom Examples

### Example 1: Transaction Cost Penalty

```python
import numpy as np

def cost_aware_reward(history) -> float:
    """Log return scaled by trade frequency."""
    if len(history.portfolio_values) < 2:
        return 0.0

    old_value = history.portfolio_values[-2]
    new_value = history.portfolio_values[-1]

    if old_value <= 0:
        return -10.0

    log_return = np.log(new_value / old_value)

    # Penalize frequent action changes
    if len(history.actions) >= 2 and history.actions[-1] != history.actions[-2]:
        log_return -= 0.001  # Small penalty for switching

    return float(log_return)
```

### Example 2: Sparse Terminal Reward

```python
def terminal_reward(history) -> float:
    """Only reward at episode end based on total return."""
    if len(history.portfolio_values) < 2:
        return 0.0

    # Give zero reward during episode
    # At terminal step, the environment handles this naturally
    # Use total return as final reward
    initial_value = history.portfolio_values[0]
    current_value = history.portfolio_values[-1]

    if initial_value <= 0:
        return -10.0

    return float(np.log(current_value / initial_value))
```

### Example 3: Multi-Objective Reward

```python
def multi_objective_reward(history) -> float:
    """Combine returns with drawdown penalty."""
    if len(history.portfolio_values) < 2:
        return 0.0

    old_value = history.portfolio_values[-2]
    new_value = history.portfolio_values[-1]

    if old_value <= 0 or new_value <= 0:
        return -10.0

    # Component 1: Log return
    log_ret = float(np.log(new_value / old_value))

    # Component 2: Drawdown from peak
    peak = max(history.portfolio_values)
    drawdown = (peak - new_value) / peak if peak > 0 else 0.0
    dd_penalty = -5.0 * drawdown if drawdown > 0.1 else 0.0

    return log_ret + dd_penalty
```

---

## When to Use Each Reward

| Reward Type | Use Case | Pros | Cons |
|-------------|----------|------|------|
| **Log Return** (default) | General purpose | Simple, stable | Ignores costs, risk |
| **Sharpe Ratio** | Risk-adjusted performance | Considers volatility | Requires history, unstable early |
| **Drawdown Penalty** | Risk management | Controls max loss | May exit winners early |
| **Terminal Sparse** | Episodic tasks | Clear objective | Sparse signal, slow learning |
| **Cost Penalty** | Reduce overtrading | Discourages frequent trades | May miss opportunities |

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

Keep rewards in a consistent range for stable learning:

```python
# Clip extreme values
reward = np.clip(reward, -10.0, 10.0)
```

### 3. Avoid Lookahead Bias

Only use information available at decision time — use past values from `history`, never future data.

### 4. Consider Environment Type

**Sequential Environments**: Full history available for complex rewards
**One-Step Environments**: Limited to immediate reward (minimal history)
**Futures Environments**: Consider liquidation risk in reward

---

## References

### Research Papers on Reward Design

- **[Reward Shaping for Reinforcement Learning in Financial Trading](https://arxiv.org/html/2506.04358v1)** - Comprehensive study on reward function design for trading applications
- **[Deep Reinforcement Learning for Trading](https://inria.hal.science/hal-05449819v1/document)** - Research on RL approaches and reward engineering for financial markets
- **[Reward Function Design in Deep Reinforcement Learning for Financial Trading](https://arno.uvt.nl/show.cgi?fid=174684)** - Analysis of different reward formulations and their impact on trading performance

---

## Next Steps

- **[Feature Engineering](custom-features.md)** - Engineer features that support your reward function
- **[Understanding the Sampler](sampler.md)** - How data flows through environments
- **[Loss Functions](../components/losses.md)** - Training objectives that work with rewards
- **[Offline Environments](../environments/offline.md)** - Apply custom rewards to environments
