# Understanding the Sampler

The `MarketDataObservationSampler` is the core component that handles multi-timeframe data sampling in TorchTrade's offline environments. This guide explains how it works and how to use it effectively.

## What Is the Sampler?

The sampler is responsible for:

1. **Resampling** 1-minute OHLCV data to multiple timeframes (5m, 15m, 1h, etc.)
2. **Applying feature preprocessing** to each timeframe
3. **Creating sliding windows** of market data for observations
4. **Preventing lookahead bias** by indexing bars correctly

```
1-Minute OHLCV Data
    ↓
MarketDataObservationSampler
    ├── Resample to 5-minute bars
    ├── Resample to 15-minute bars
    ├── Resample to 1-hour bars
    ├── Apply feature preprocessing
    └── Create sliding windows
    ↓
Multi-Timeframe Observations
    ├── market_data_1Minute: [12, features]
    ├── market_data_5Minute: [8, features]
    └── market_data_15Minute: [8, features]
```

## Basic Usage

The sampler is created automatically when you initialize an offline environment:

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
import pandas as pd

# Load 1-minute OHLCV data
df = pd.read_csv("btcusdt_1m.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Configure multi-timeframe sampling
config = SeqLongOnlyEnvConfig(
    time_frames=[1, 5, 15, 60],        # Minutes: 1m, 5m, 15m, 1h
    window_sizes=[12, 8, 8, 24],       # Lookback per timeframe
    execute_on=(5, "Minute"),          # Execute every 5 minutes
)

env = SeqLongOnlyEnv(df, config)

# The sampler is created internally:
# env.sampler = MarketDataObservationSampler(...)
```

## How Resampling Works

### Basic Resampling

The sampler resamples 1-minute base data to higher timeframes using pandas `resample()`:

```python
# 1-minute → 5-minute
resampled_5m = df.resample('5T').agg({
    'open': 'first',    # First price in 5-min window
    'high': 'max',      # Highest price in 5-min window
    'low': 'min',       # Lowest price in 5-min window
    'close': 'last',    # Last price in 5-min window
    'volume': 'sum'     # Total volume in 5-min window
})
```

### Lookahead Bias Prevention (CRITICAL!)

**Issue #10 Fix**: By default, pandas `resample()` indexes bars by their START time, but aggregates data through their END time. This creates lookahead bias!

Example:
- A 5-minute bar at `00:25:00` contains data from `00:25:00` to `00:29:59`
- An agent executing at `00:27:00` would see minute 29 data (future!)

**Solution**: TorchTrade shifts higher timeframe bars forward by their period, indexing them by END time:

```python
# After resampling
if tf > execute_on:  # Only shift coarser timeframes
    offset = pd.Timedelta(tf.to_pandas_freq())
    resampled.index = resampled.index + offset
```

This ensures:
- Higher timeframe bars are indexed by END time (when complete)
- Only completed bars are visible to the agent
- No lookahead bias in backtests

**Example:**
```
Original (START time indexing):
00:25:00 → [00:25:00 - 00:29:59] ❌ Visible at 00:27:00 (lookahead!)

Fixed (END time indexing):
00:30:00 → [00:25:00 - 00:29:59] ✅ Only visible at 00:30:00+
```

## Multi-Timeframe Configuration

### Time Frames

Specify timeframes in minutes:

```python
config = SeqLongOnlyEnvConfig(
    time_frames=[1, 5, 15, 60],        # 1m, 5m, 15m, 1h
    ...
)
```

### Window Sizes

Each timeframe has a lookback window:

```python
config = SeqLongOnlyEnvConfig(
    time_frames=[1, 5, 15, 60],
    window_sizes=[12, 8, 8, 24],       # Bars per timeframe
    ...
)
```

This creates observations:
- `market_data_1Minute`: Last 12 one-minute bars (12 minutes)
- `market_data_5Minute`: Last 8 five-minute bars (40 minutes)
- `market_data_15Minute`: Last 8 fifteen-minute bars (2 hours)
- `market_data_60Minute`: Last 24 one-hour bars (24 hours)

### Execute On

Controls trade execution frequency:

```python
config = SeqLongOnlyEnvConfig(
    execute_on=(5, "Minute"),          # Execute every 5 minutes
    ...
)
```

The agent receives observations and can trade every 5 minutes, even though it observes 1m, 5m, 15m, and 1h data.

## Feature Preprocessing

The sampler applies custom feature preprocessing to each timeframe after resampling:

```python
import ta

def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators"""
    df["features_close"] = df["close"]
    df["features_rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df.fillna(0, inplace=True)
    return df

config = SeqLongOnlyEnvConfig(
    feature_preprocessing_fn=custom_preprocessing,  # Applied to each timeframe
    time_frames=[1, 5, 15],
    ...
)
```

The function is called separately for each resampled timeframe:
1. Resample 1m → 1m (no change)
2. Apply `custom_preprocessing(df_1m)`
3. Resample 1m → 5m
4. Apply `custom_preprocessing(df_5m)`
5. Resample 1m → 15m
6. Apply `custom_preprocessing(df_15m)`

## Sliding Windows

The sampler creates sliding windows for each timeframe:

```python
# Example: window_size=8 for 5-minute timeframe
# At execution time 01:00:00, the window contains:
[
    [00:25:00],  # 5m bar from 00:20:00-00:24:59
    [00:30:00],  # 5m bar from 00:25:00-00:29:59
    [00:35:00],  # 5m bar from 00:30:00-00:34:59
    [00:40:00],  # 5m bar from 00:35:00-00:39:59
    [00:45:00],  # 5m bar from 00:40:00-00:44:59
    [00:50:00],  # 5m bar from 00:45:00-00:49:59
    [00:55:00],  # 5m bar from 00:50:00-00:54:59
    [01:00:00],  # 5m bar from 00:55:00-00:59:59
]
```

At the next execution step (01:05:00), the window slides forward:
```python
[
    [00:30:00],  # Oldest bar dropped
    [00:35:00],
    ...
    [01:00:00],
    [01:05:00],  # New bar added
]
```

## Advanced Usage

### Episode Start Randomization

Episodes can start at random positions in the dataset:

```python
# During env.reset()
start_idx = env.sampler.get_random_start_index()

# Or specify max trajectory length
config = SeqLongOnlyEnvConfig(
    max_traj_length=1000,  # Limit episode to 1000 steps
    ...
)
```

### Accessing the Sampler

You can access the sampler directly from the environment:

```python
env = SeqLongOnlyEnv(df, config)

# Access sampler
sampler = env.sampler

# Get observation at specific index
obs = sampler.get_observation(index=100)

print(obs.keys())  # ['market_data_1Minute', 'market_data_5Minute', ...]
```

### Data Requirements

The sampler requires sufficient historical data for the largest lookback window:

```python
config = SeqLongOnlyEnvConfig(
    time_frames=[1, 5, 15, 60],
    window_sizes=[12, 8, 8, 24],       # 24 hours max lookback
    execute_on=(5, "Minute"),
)

# Minimum data required: 24 hours + 1 hour buffer = 25 hours
# At 1-minute resolution: 25 * 60 = 1500 rows
```

If insufficient data, the sampler raises an error:

```
ValueError: Resampled dataframe for timeframe 60Minute is empty
```

## Performance Optimization

### Pre-compute Resampled DataFrames

The sampler pre-computes all resampled DataFrames during initialization:

```python
# Happens once during env creation
env = SeqLongOnlyEnv(df, config)  # ← Resampling happens here

# Subsequent resets and steps are fast
env.reset()  # ← No resampling, uses pre-computed data
env.step(tensordict)  # ← Fast window lookup
```

### Named Tuples for Speed

The sampler uses named tuples for fast OHLCV access:

```python
from collections import namedtuple

OHLCV = namedtuple('OHLCV', ['open', 'high', 'low', 'close', 'volume'])

# Faster than dict access
ohlcv = OHLCV(open=100, high=105, low=99, close=103, volume=1000)
print(ohlcv.close)  # Fast attribute access
```

## Building Custom Samplers

You can create custom samplers by subclassing `MarketDataObservationSampler`:

```python
from torchtrade.envs.offline.sampler import MarketDataObservationSampler

class CustomSampler(MarketDataObservationSampler):
    def __init__(self, df, time_frames, window_sizes, execute_on, **kwargs):
        super().__init__(df, time_frames, window_sizes, execute_on, **kwargs)

    def get_observation(self, index: int) -> Dict[str, torch.Tensor]:
        """Override to customize observation structure"""
        obs = super().get_observation(index)

        # Add custom fields
        obs["custom_feature"] = torch.tensor([...])

        return obs
```

Then use in your environment:

```python
class CustomEnv(SeqLongOnlyEnv):
    def __init__(self, df, config):
        super().__init__(df, config)

        # Replace sampler with custom one
        self.sampler = CustomSampler(
            df,
            config.time_frames,
            config.window_sizes,
            config.execute_on,
            feature_processing_fn=config.feature_preprocessing_fn
        )
```

## Common Issues

### Issue 1: Empty Resampled DataFrame

**Error:**
```
ValueError: Resampled dataframe for timeframe 60Minute is empty
```

**Cause:** Insufficient data for the requested timeframe.

**Solution:** Ensure you have enough historical data:
```python
# For 1-hour bars with window_size=24, you need:
# 24 hours + buffer ≈ 30 hours of 1-minute data
# = 30 * 60 = 1800 rows minimum
```

### Issue 2: NaN Values in Observations

**Error:** Policy receives NaN values.

**Cause:** Feature preprocessing created NaNs.

**Solution:** Fill NaNs in preprocessing function:
```python
def preprocessing(df):
    df["features_rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df.fillna(0, inplace=True)  # ← Important!
    return df
```

### Issue 3: Slow Environment Creation

**Issue:** `SeqLongOnlyEnv(df, config)` takes a long time.

**Cause:** Large dataset with complex feature preprocessing.

**Solution:** Cache pre-processed data:
```python
import joblib

# Save preprocessed data
env = SeqLongOnlyEnv(df, config)
joblib.dump(env.sampler, "sampler_cache.pkl")

# Load cached sampler
sampler = joblib.load("sampler_cache.pkl")
env.sampler = sampler
```

## Debugging Tips

### Print Sampler Info

```python
env = SeqLongOnlyEnv(df, config)

print(f"Resampled timeframes: {env.sampler.resampled_dfs.keys()}")
print(f"Max lookback: {env.sampler.max_lookback}")

for tf_key, df_resampled in env.sampler.resampled_dfs.items():
    print(f"{tf_key}: {len(df_resampled)} bars, "
          f"first={df_resampled.index[0]}, "
          f"last={df_resampled.index[-1]}")
```

### Visualize Observations

```python
import matplotlib.pyplot as plt

# Get observation at specific step
obs = env.sampler.get_observation(index=100)

# Plot 5-minute close prices
close_prices = obs["market_data_5Minute"][:, -1]  # Assuming close is last feature

plt.plot(close_prices.numpy())
plt.title("5-Minute Close Prices (8-bar window)")
plt.xlabel("Bar")
plt.ylabel("Price")
plt.show()
```

## Next Steps

- **[Custom Feature Engineering](custom-features.md)** - Add technical indicators
- **[Custom Reward Functions](reward-functions.md)** - Design better rewards
- **[Building Custom Environments](custom-environment.md)** - Extend TorchTrade
- **[Offline Environments](../environments/offline.md)** - Apply sampler knowledge

## Technical Reference

### Source Code

The sampler implementation is in:
```
torchtrade/envs/offline/sampler.py
```

Key methods:
- `__init__()`: Initialize and pre-compute resampled data
- `get_observation(index)`: Get multi-timeframe observation at index
- `get_random_start_index()`: Get random valid starting position
- `reset()`: Reset sampler state

### TimeFrame Class

```python
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit

# Create timeframe
tf = TimeFrame(5, TimeFrameUnit.Minute)

# Convert to pandas frequency
tf.to_pandas_freq()  # → "5T"

# Get observation key
tf.obs_key_freq()  # → "5Minute"
```
