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

### Individual Metrics

```python
from torchtrade.metrics import (
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_calmar_ratio,
    compute_win_rate,
)
import torch

portfolio_values = torch.tensor([1000, 1100, 1050, 900, 950, 1200])
returns = torch.tensor([0.01, -0.02, 0.03, -0.01, 0.02, 0.04])

# Max Drawdown - largest peak-to-trough decline
dd = compute_max_drawdown(portfolio_values)
# Returns: max_drawdown, max_drawdown_duration, current_drawdown, peak_value, trough_value

# Sharpe Ratio - risk-adjusted return (excess return / volatility)
sharpe = compute_sharpe_ratio(returns, periods_per_year=525600)

# Sortino Ratio - like Sharpe but only penalizes downside volatility
sortino = compute_sortino_ratio(returns, periods_per_year=525600)

# Calmar Ratio - annualized return / max drawdown
calmar = compute_calmar_ratio(portfolio_values, periods_per_year=525600)

# Win Rate - percentage of steps where reward > 0
win_metrics = compute_win_rate(returns)
# Returns: win_rate, avg_win, avg_loss, profit_factor
```

---

## Usage with HistoryTracker

Every TorchTrade environment records episode data in a `HistoryTracker` (see [State Management](../environments/offline.md#state-management)). After a rollout, extract the history and feed it to the metrics:

```python
import torch
from torchtrade.metrics import compute_all_metrics

# Run evaluation rollout
obs = env.reset()
done = False
while not done:
    action = policy(obs)
    obs = env.step(action)
    done = obs["done"].item()

# Get history from environment
history = env.history  # HistoryTracker instance

# Compute all metrics
metrics = compute_all_metrics(
    portfolio_values=torch.tensor(history.portfolio_values),
    rewards=torch.tensor(history.rewards),
    action_history=history.actions,
    periods_per_year=525600,  # See periods table below
)
# Returns dict with: total_return, sharpe_ratio, sortino_ratio, calmar_ratio,
# max_drawdown, max_dd_duration, num_trades, win_rate, avg_win, avg_loss, profit_factor
```

### Logging to Weights & Biases

```python
import wandb

wandb.init(project="torchtrade-training")

for epoch in range(num_epochs):
    # ... training code ...

    # Evaluate and compute metrics from history (as above)
    history = eval_env.history
    metrics = compute_all_metrics(
        portfolio_values=torch.tensor(history.portfolio_values),
        rewards=torch.tensor(history.rewards),
        action_history=history.actions,
        periods_per_year=525600,
    )

    wandb.log({f"eval/{k}": v for k, v in metrics.items()})
```

---

## Custom Metrics

Create custom metrics following the same pattern — accept `torch.Tensor` rewards or portfolio values, return a scalar:

```python
def compute_max_consecutive_wins(rewards: torch.Tensor) -> int:
    """Maximum number of consecutive winning periods."""
    wins = rewards > 0
    max_streak = current = 0
    for is_win in wins:
        current = current + 1 if is_win else 0
        max_streak = max(max_streak, current)
    return max_streak
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

Match `periods_per_year` to your `execute_on` config setting.

---

## Best Practices

- **Compare to benchmarks**: Always compute buy & hold return as a baseline
- **Use multiple metrics**: Don't rely on a single metric — check returns (Sharpe, Sortino), risk (max drawdown), and efficiency (win rate, profit factor)
- **Include costs**: Set realistic `transaction_fee` and `slippage` in your environment config
- **Watch for extreme Sharpe**: If Sharpe > 10 or < -10, check that `periods_per_year` matches your execution frequency

---

## Next Steps

- **[Reward Functions](reward-functions.md)** - Design rewards to optimize for specific metrics
- **[Feature Engineering](custom-features.md)** - Add features that correlate with performance
- **[Offline Environments](../environments/offline.md)** - Backtest strategies with historical data
- **[Examples](../examples/index.md)** - See metrics in complete training scripts

---

## References

- **[Sharpe Ratio](https://en.wikipedia.org/wiki/Sharpe_ratio)** - Original paper and interpretation
- **[Sortino Ratio](https://en.wikipedia.org/wiki/Sortino_ratio)** - Downside risk-adjusted returns
- **[Calmar Ratio](https://en.wikipedia.org/wiki/Calmar_ratio)** - Return per unit of drawdown
- **[Maximum Drawdown](https://en.wikipedia.org/wiki/Drawdown_(economics))** - Peak-to-trough decline
