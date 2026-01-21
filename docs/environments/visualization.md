# Visualizing Episode History

All offline environments support **`render_history()`** to visualize episode performance after training or evaluation. The method automatically detects the environment type and renders appropriate plots.

## Overview

The `render_history()` method is inherited from the `TorchTradeOfflineEnv` base class and provides consistent visualization across all 6 offline environments:

- **SeqLongOnlyEnv**
- **SeqLongOnlySLTPEnv**
- **LongOnlyOneStepEnv**
- **SeqFuturesEnv**
- **SeqFuturesSLTPEnv**
- **FuturesOneStepEnv**

The visualization automatically adapts based on the environment type (long-only vs futures).

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
env.render_history()  # Display plots

# Or save to a variable for later
fig = env.render_history(return_fig=True)
fig.savefig("episode_history.png")
```

## Visualization Types

The rendered plots automatically adapt based on environment type:

### Long-Only Environments

`SeqLongOnlyEnv`, `SeqLongOnlySLTPEnv`, `LongOnlyOneStepEnv`

**2 subplots**:

1. **Price History with Actions**: Shows the asset price over time with buy/sell actions marked as green upward triangles (buy) and red downward triangles (sell)
2. **Portfolio Value vs Buy-and-Hold**: Compares your agent's portfolio value against a simple buy-and-hold baseline strategy

### Futures Environments

`SeqFuturesEnv`, `SeqFuturesSLTPEnv`, `FuturesOneStepEnv`

**3 subplots**:

1. **Price History with Actions**: Shows the asset price over time with long/short actions marked as green upward triangles (long) and red downward triangles (short)
2. **Portfolio Value vs Buy-and-Hold**: Compares your agent's portfolio value against a simple buy-and-hold baseline strategy
3. **Position History**: Visualizes position size over time with green areas for long positions, red areas for short positions, and flat sections for no position

## Example Output

<!-- TODO: Insert actual rendered plots here -->
*Example plots will be added here showing typical visualization output for long-only and futures environments.*

### Long-Only Environment Example
<!-- TODO: Add 2-subplot image for long-only environment -->

### Futures Environment Example
<!-- TODO: Add 3-subplot image for futures environment -->

## Implementation Details

- **Automatic detection**: Checks for `positions` key in history to determine environment type
- **Matplotlib-based**: Requires `matplotlib` to be installed
- **Buy-and-hold baseline**: Calculated from initial portfolio value (buys asset at t=0 and holds)
- **Action markers**: All actions are marked on the price chart for easy analysis
- **Consistent API**: Same method works across all 6 offline environments

## Example with Different Environments

```python
import pandas as pd
from torchtrade.envs.offline import (
    SeqLongOnlyEnv,
    SeqLongOnlyEnvConfig,
    SeqFuturesEnv,
    SeqFuturesEnvConfig
)

# Long-only environment
config = SeqLongOnlyEnvConfig(...)
env = SeqLongOnlyEnv(df, config)
# ... run episode ...
env.render_history()  # Shows 2 subplots

# Futures environment
config = SeqFuturesEnvConfig(...)
env = SeqFuturesEnv(df, config)
# ... run episode ...
env.render_history()  # Shows 3 subplots

# One-step environments work too!
from torchtrade.envs.offline import LongOnlyOneStepEnv, FuturesOneStepEnv

env = LongOnlyOneStepEnv(df, config)
# ... run episode ...
env.render_history()  # Shows 2 subplots

env = FuturesOneStepEnv(df, config)
# ... run episode ...
env.render_history()  # Shows 3 subplots
```

## Technical Notes

### Base Class Implementation

The method is implemented in `TorchTradeOfflineEnv` base class at `torchtrade/envs/offline/base.py:457-614`.

### Detection Logic

```python
def render_history(self, return_fig=False):
    history_dict = self.history.to_dict()

    # Automatically detect environment type
    is_futures = 'positions' in history_dict

    if is_futures:
        # Render 3-subplot layout (price, portfolio, position)
    else:
        # Render 2-subplot layout (price, portfolio)
```

### Dependencies

- `matplotlib.pyplot` - For plotting
- `datetime` - For timestamp formatting (if using datetime indices)

### History Tracking

All offline environments use `HistoryTracker` (or `FuturesHistoryTracker` for futures environments) to record:

- Price at each step
- Action taken
- Reward received
- Portfolio value
- Position size (futures only)

See `torchtrade/envs/state.py:33-148` for history tracking implementation details.

## Next Steps

- **[Offline Environments](offline.md)** - Learn about the 6 offline environments
- **[History Tracking](offline.md#history-tracking)** - Understanding episode history data
- **[Metrics](../guides/metrics.md)** - Computing performance metrics from history
