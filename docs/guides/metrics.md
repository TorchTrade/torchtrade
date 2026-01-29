# Performance Metrics

TorchTrade provides a comprehensive set of trading performance metrics to evaluate agent performance. These metrics help assess risk-adjusted returns, drawdown characteristics, and trading efficiency.

## Overview

The `torchtrade.metrics` module provides standard trading metrics including:

| Metric | Description | Typical Good Value |
|--------|-------------|-------------------|
| **Total Return** | Overall portfolio return | > 0% |
| **Sharpe Ratio** | Risk-adjusted return (volatility) | > 1.0 (> 2.0 excellent) |
| **Sortino Ratio** | Risk-adjusted return (downside volatility) | > 1.5 |
| **Calmar Ratio** | Return per unit of maximum drawdown | > 1.0 |
| **Max Drawdown** | Largest peak-to-trough decline | < 20% |
| **Max DD Duration** | Length of maximum drawdown period | Shorter is better |
| **Win Rate** | Percentage of profitable periods | > 50% |
| **Profit Factor** | Total profits / total losses | > 1.5 |
| **Number of Trades** | Total trades executed | Depends on strategy |

---

## Available Metrics

### 1. Maximum Drawdown

Measures the largest peak-to-trough decline in portfolio value.

```python
from torchtrade.metrics import compute_max_drawdown
import torch

# Portfolio values over time
portfolio_values = torch.tensor([1000, 1100, 1050, 900, 950, 1200])

dd_metrics = compute_max_drawdown(portfolio_values)

print(f"Max Drawdown: {dd_metrics['max_drawdown']:.2%}")
print(f"DD Duration: {dd_metrics['max_drawdown_duration']} periods")
print(f"Current Drawdown: {dd_metrics['current_drawdown']:.2%}")
```

**Returns:**
```python
{
    'max_drawdown': -0.1818,        # -18.18% drawdown
    'max_drawdown_duration': 2,      # 2 periods from peak to trough
    'current_drawdown': 0.0,         # Currently at new peak
    'peak_value': 1100.0,            # Peak before drawdown
    'trough_value': 900.0            # Lowest point in drawdown
}
```

### 2. Sharpe Ratio

Measures risk-adjusted returns by comparing excess returns to volatility.

```python
from torchtrade.metrics import compute_sharpe_ratio
import torch

# Per-period returns
returns = torch.tensor([0.01, -0.02, 0.03, -0.01, 0.02, 0.04])

# For 1-minute crypto data: 525,600 periods per year
sharpe = compute_sharpe_ratio(
    returns,
    periods_per_year=525600,
    rf_annual=0.0  # Risk-free rate (default: 0%)
)

print(f"Sharpe Ratio: {sharpe:.2f}")
```

**Interpretation:**
- **< 1.0**: Poor risk-adjusted returns
- **1.0-2.0**: Good performance
- **> 2.0**: Very good performance
- **> 3.0**: Excellent performance

### 3. Sortino Ratio

Similar to Sharpe, but only penalizes downside volatility (returns below target).

```python
from torchtrade.metrics import compute_sortino_ratio

sortino = compute_sortino_ratio(
    returns,
    periods_per_year=525600,
    rf_annual=0.0,
    target_return=0.0  # Target return threshold
)

print(f"Sortino Ratio: {sortino:.2f}")
```

**Why use Sortino?**
- More relevant for trading strategies where upside volatility is desirable
- Only penalizes downside risk below target
- Often higher than Sharpe for profitable strategies

### 4. Calmar Ratio

Measures annualized return per unit of maximum drawdown.

```python
from torchtrade.metrics import compute_calmar_ratio

calmar = compute_calmar_ratio(
    portfolio_values,
    periods_per_year=525600
)

print(f"Calmar Ratio: {calmar:.2f}")
```

**Interpretation:**
- **< 0.5**: Poor
- **0.5-1.0**: Acceptable
- **1.0-3.0**: Good
- **> 3.0**: Excellent

### 5. Win Rate and Profit Metrics

**Note**: Due to the possibility of customized reward functions, we define win rate as the percentage of steps where `reward > 0` rather than using traditional profit/loss definitions.

```python
from torchtrade.metrics import compute_win_rate

# Rewards from episode
rewards = torch.tensor([0.01, -0.02, 0.03, -0.01, 0.02, 0.04])

win_metrics = compute_win_rate(rewards)

print(f"Win Rate: {win_metrics['win_rate (reward>0)']:.2%}")
print(f"Average Win: {win_metrics['avg_win']:.4f}")
print(f"Average Loss: {win_metrics['avg_loss']:.4f}")
print(f"Profit Factor: {win_metrics['profit_factor']:.2f}")
```

**Returns:**
```python
{
    'win_rate (reward>0)': 0.6667,   # 66.67% winning periods
    'avg_win': 0.025,                 # Average winning return
    'avg_loss': -0.015,               # Average losing return
    'profit_factor': 2.5              # 2.5x more profits than losses
}
```

---

## Computing All Metrics at Once

Use `compute_all_metrics()` to calculate all standard metrics in one call:

```python
from torchtrade.metrics import compute_all_metrics
import torch

# Episode data
portfolio_values = torch.tensor([1000, 1050, 1100, 1080, 1150, 1200])
rewards = torch.tensor([0.05, 0.048, -0.018, 0.065, 0.043])
action_history = [2, 1, 0, 2, 1]  # Actions taken during episode

# Compute all metrics
# For 1-minute crypto: 525,600 periods/year
# For 5-minute: 105,120 periods/year
# For 1-hour: 8,760 periods/year
# For daily: 365 periods/year
metrics = compute_all_metrics(
    portfolio_values=portfolio_values,
    rewards=rewards,
    action_history=action_history,
    periods_per_year=525600
)

# Print results
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value}")
```

**Output:**
```
total_return: 0.2
sharpe_ratio: 1.85
sortino_ratio: 2.34
calmar_ratio: 3.12
max_drawdown: -0.0182
max_dd_duration: 1
num_trades: 2
win_rate (reward>0): 0.8
avg_win: 0.052
avg_loss: -0.018
profit_factor: 5.78
```

---

## Integration with Training Loops

### Example 1: Evaluating Policy Performance

```python
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.metrics import compute_all_metrics
from torchrl.collectors import SyncDataCollector
import torch

# Create evaluation environment
eval_env = SequentialTradingEnv(test_df, config)

# Collect evaluation rollouts
collector = SyncDataCollector(
    eval_env,
    policy,
    frames_per_batch=10000,
    total_frames=10000,
    device="cuda"
)

# Collect episode data
for batch in collector:
    # Extract episode data
    rewards = batch["reward"]
    # Note: Portfolio values need to be tracked in environment or computed from rewards

    # Compute metrics
    metrics = compute_all_metrics(
        portfolio_values=portfolio_values,
        rewards=rewards,
        action_history=batch["action"].tolist(),
        periods_per_year=525600  # Adjust based on execute_on frequency
    )

    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate (reward>0)']:.2%}")
```

### Example 2: Logging Metrics to Weights & Biases

```python
import wandb
from torchtrade.metrics import compute_all_metrics

# Initialize W&B
wandb.init(project="torchtrade-training")

# During training loop
for epoch in range(num_epochs):
    # ... training code ...

    # Evaluate on test set
    metrics = compute_all_metrics(
        portfolio_values=eval_portfolio_values,
        rewards=eval_rewards,
        action_history=eval_actions,
        periods_per_year=525600
    )

    # Log to W&B
    wandb.log({
        f"eval/{k}": v for k, v in metrics.items()
    })
```

---

## Creating Custom Metrics

You can create custom metrics by following the same pattern as built-in functions:

### Example: Maximum Consecutive Wins

```python
import torch

def compute_max_consecutive_wins(rewards: torch.Tensor) -> int:
    """
    Compute the maximum number of consecutive winning periods.

    Args:
        rewards: 1D torch.Tensor of per-period rewards

    Returns:
        Maximum consecutive wins
    """
    if len(rewards) == 0:
        return 0

    # Create boolean mask for wins
    wins = rewards > 0

    # Count consecutive wins
    max_streak = 0
    current_streak = 0

    for is_win in wins:
        if is_win:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak

# Usage
rewards = torch.tensor([0.01, 0.02, 0.03, -0.01, 0.02, 0.01])
max_wins = compute_max_consecutive_wins(rewards)
print(f"Max Consecutive Wins: {max_wins}")  # 3
```

### Example: Average Trade Duration

```python
def compute_avg_trade_duration(action_history: list) -> float:
    """
    Compute average duration of trades (time in position).

    Args:
        action_history: List of actions (0=sell/close, 1=hold, 2=buy/long)

    Returns:
        Average trade duration in periods
    """
    trade_durations = []
    current_duration = 0
    in_position = False

    for action in action_history:
        if action == 2:  # Enter position
            in_position = True
            current_duration = 1
        elif action == 1 and in_position:  # Hold position
            current_duration += 1
        elif action == 0 and in_position:  # Exit position
            trade_durations.append(current_duration)
            in_position = False
            current_duration = 0

    if len(trade_durations) == 0:
        return 0.0

    return sum(trade_durations) / len(trade_durations)

# Usage
actions = [2, 1, 1, 0, 2, 1, 0, 1, 1]
avg_duration = compute_avg_trade_duration(actions)
print(f"Average Trade Duration: {avg_duration:.1f} periods")
```

---

## Periods Per Year Configuration

Choose `periods_per_year` based on your `execute_on` frequency:

| Execute On | Periods Per Year | Calculation |
|------------|-----------------|-------------|
| 1 minute | 525,600 | 60 × 24 × 365 |
| 5 minutes | 105,120 | 12 × 24 × 365 |
| 15 minutes | 35,040 | 4 × 24 × 365 |
| 1 hour | 8,760 | 24 × 365 |
| 4 hours | 2,190 | 6 × 365 |
| 1 day | 365 | 365 |

```python
# Example: 5-minute execution
config = SequentialTradingEnvConfig(
    execute_on=(5, "Minute"),
    # ... other config
)

# Use 105,120 periods per year for metrics
metrics = compute_all_metrics(
    portfolio_values=portfolio_values,
    rewards=rewards,
    action_history=action_history,
    periods_per_year=105120  # 5-minute periods
)
```

---

## Best Practices

### 1. Choose Appropriate Benchmarks

Compare metrics to relevant benchmarks:

```python
# Compute buy & hold baseline
buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]

print(f"Agent Return: {metrics['total_return']:.2%}")
print(f"Buy & Hold: {buy_hold_return:.2%}")
print(f"Outperformance: {metrics['total_return'] - buy_hold_return:.2%}")
```

### 2. Consider Multiple Metrics

Don't rely on a single metric. Look at:
- **Return metrics**: Total return, Sharpe, Sortino
- **Risk metrics**: Max drawdown, drawdown duration
- **Efficiency metrics**: Win rate, profit factor, number of trades

### 3. Account for Transaction Costs

Ensure your environment includes realistic transaction fees and slippage:

```python
config = SequentialTradingEnvConfig(
    transaction_fee=0.0025,  # 0.25% per trade
    slippage=0.001           # 0.1% slippage
)
```

### 4. Use Rolling Windows for Stability

For long episodes, compute metrics over rolling windows:

```python
def compute_rolling_sharpe(portfolio_values, window_size=100, periods_per_year=525600):
    """Compute Sharpe ratio over rolling window."""
    returns = torch.diff(portfolio_values) / portfolio_values[:-1]

    rolling_sharpes = []
    for i in range(len(returns) - window_size + 1):
        window_returns = returns[i:i+window_size]
        sharpe = compute_sharpe_ratio(window_returns, periods_per_year)
        rolling_sharpes.append(sharpe)

    return rolling_sharpes
```

---

## Interpreting Metrics Together

### Example: Evaluating a Strategy

```python
metrics = compute_all_metrics(portfolio_values, rewards, actions, periods_per_year=525600)

# Check profitability
if metrics['total_return'] > 0:
    print("✓ Strategy is profitable")
else:
    print("✗ Strategy is unprofitable")

# Check risk-adjusted returns
if metrics['sharpe_ratio'] > 1.5:
    print(f"✓ Good risk-adjusted returns (Sharpe: {metrics['sharpe_ratio']:.2f})")
else:
    print(f"✗ Poor risk-adjusted returns (Sharpe: {metrics['sharpe_ratio']:.2f})")

# Check drawdown risk
if abs(metrics['max_drawdown']) < 0.2:  # Less than 20%
    print(f"✓ Acceptable drawdown ({metrics['max_drawdown']:.2%})")
else:
    print(f"✗ High drawdown risk ({metrics['max_drawdown']:.2%})")

# Check trade efficiency
if metrics['win_rate (reward>0)'] > 0.5 and metrics['profit_factor'] > 1.5:
    print(f"✓ Efficient trading (Win rate: {metrics['win_rate (reward>0)']:.2%}, PF: {metrics['profit_factor']:.2f})")
else:
    print(f"✗ Inefficient trading (Win rate: {metrics['win_rate (reward>0)']:.2%}, PF: {metrics['profit_factor']:.2f})")
```

---

## Common Issues

### Issue 1: Extreme Sharpe Ratios

**Problem**: Sharpe ratio is unrealistically high (> 10) or low (< -10)

**Solution**:
- Check that `periods_per_year` matches your execution frequency
- Ensure returns are not scaled incorrectly
- Verify no division by zero in volatility calculation

### Issue 2: Zero Max Drawdown

**Problem**: Max drawdown is 0 even with losses

**Solution**:
- Ensure portfolio values are monotonically increasing at some point
- Check that portfolio values are computed correctly
- Verify you're passing portfolio values, not returns

### Issue 3: Inconsistent Metrics

**Problem**: High Sharpe but low Calmar, or vice versa

**Explanation**:
- **High Sharpe, Low Calmar**: Strategy has consistent small gains but occasional large drawdowns
- **Low Sharpe, High Calmar**: Strategy has volatile returns but recovers quickly from drawdowns

Both are valid - consider which risk profile suits your use case.

---

## Next Steps

- **[Reward Functions](reward-functions.md)** - Design rewards to optimize for specific metrics
- **[Feature Engineering](custom-features.md)** - Add features that correlate with performance
- **[Offline Environments](../environments/offline.md)** - Backtest strategies with historical data
- **[Examples](../examples.md)** - See metrics in complete training scripts

---

## References

- **[Sharpe Ratio](https://en.wikipedia.org/wiki/Sharpe_ratio)** - Original paper and interpretation
- **[Sortino Ratio](https://en.wikipedia.org/wiki/Sortino_ratio)** - Downside risk-adjusted returns
- **[Calmar Ratio](https://en.wikipedia.org/wiki/Calmar_ratio)** - Return per unit of drawdown
- **[Maximum Drawdown](https://en.wikipedia.org/wiki/Drawdown_(economics))** - Peak-to-trough decline
