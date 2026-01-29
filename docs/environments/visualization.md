# Visualizing Episode History

All offline environments support **`render_history()`** to visualize episode performance after training or evaluation. The method automatically detects the environment type and renders appropriate plots.

## Overview

The `render_history()` method is inherited from the `TorchTradeOfflineEnv` base class and provides consistent visualization across all 3 offline environments:

- **SequentialTradingEnv**
- **SequentialTradingEnvSLTP**
- **OneStepTradingEnv**

The visualization automatically adapts based on the trading mode (spot vs futures).

## Usage

```python
# After running an episode
env.reset()
for _ in range(episode_length):
    action = policy(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

# Visualize the episode
env.render_history()  # Display plots with buy-and-hold baseline

# Without buy-and-hold baseline
env.render_history(plot_bh_baseline=False)

# Or save to a variable for later
fig = env.render_history(return_fig=True, plot_bh_baseline=True)
fig.savefig("episode_history.png")
```

## Visualization Types

The rendered plots automatically adapt based on environment type:

### Spot Mode (`leverage=1`)

**2 subplots**:

1. **Price History with Actions**: Shows the asset price over time with buy/sell actions marked as green upward triangles (buy) and red downward triangles (sell)
2. **Portfolio Value vs Buy-and-Hold**: Compares your agent's portfolio value against a simple buy-and-hold baseline strategy

### Futures Mode (`leverage > 1`)

**3 subplots**:

1. **Price History with Actions**: Shows the asset price over time with long/short actions marked as green upward triangles (long) and red downward triangles (short)
2. **Portfolio Value vs Buy-and-Hold**: Compares your agent's portfolio value against a simple buy-and-hold baseline strategy
3. **Position History**: Visualizes position size over time with green areas for long positions, red areas for short positions, and flat sections for no position

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `return_fig` | bool | `False` | Return matplotlib figure instead of displaying |
| `plot_bh_baseline` | bool | `True` | Show buy-and-hold baseline comparison |

Implemented in `torchtrade/envs/offline/base.py`. Requires `matplotlib`.
