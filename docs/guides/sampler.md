# Understanding the Sampler

The `MarketDataObservationSampler` (found in `torchtrade/envs/offline/sampler.py`) handles multi-timeframe data sampling in TorchTrade's offline environments. It resamples high-frequency data (1-minute bars) to multiple timeframes and creates synchronized observation windows while preventing lookahead bias. This allows RL agents to observe market patterns across different time scales simultaneously, from short-term momentum to long-term trends.

## What Is the Sampler?

The sampler:

1. **Resamples** 1-minute OHLCV to multiple timeframes (5m, 15m, 1h)
2. **Applies feature preprocessing** to each timeframe
3. **Creates sliding windows** of market data
4. **Prevents lookahead bias** by correct bar indexing

```
1-Minute Data → Sampler → Multi-Timeframe Observations
                  ├── Resample to timeframes
                  ├── Apply preprocessing
                  └── Create windows
```

---

## Basic Usage

### How It Works

The sampler takes your 1-minute OHLCV data, resamples it to multiple timeframes, and provides synchronized observation windows at each execution step. Here's a direct example of using the sampler:

```python
import pandas as pd
from torchtrade.envs.offline.sampler import MarketDataObservationSampler
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit

# Load your OHLCV data
df = pd.read_csv("btcusdt_1m.csv")
# Required columns: timestamp, open, high, low, close, volume

# Create sampler
sampler = MarketDataObservationSampler(
    df=df,
    time_frames=[
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
        TimeFrame(15, TimeFrameUnit.Minute),
    ],
    window_sizes=[12, 8, 8],
    execute_on=TimeFrame(5, TimeFrameUnit.Minute),
)

# Get observations
obs_dict, timestamp, truncated = sampler.get_sequential_observation()

# obs_dict contains:
# {
#   "market_data_1Minute": torch.Tensor([12, num_features]),
#   "market_data_5Minute": torch.Tensor([8, num_features]),
#   "market_data_15Minute": torch.Tensor([8, num_features]),
# }

# Reset for new episode
sampler.reset(random_start=True)
```

### Usage in Offline Environments

The sampler is used in all offline environments (SeqLongOnlyEnv, SeqFuturesEnv, etc.) and allows flexible selection of timeframes through the environment configuration:

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig

# Configure multi-timeframe sampling
config = SeqLongOnlyEnvConfig(
    time_frames=["1min", "5min", "15min", "1hour"],
    window_sizes=[12, 8, 8, 24],
    execute_on=(5, "Minute"),
)

env = SeqLongOnlyEnv(df, config)

# Observations contain all timeframes
obs = env.reset()
# obs["market_data_1Minute"]: (12, features)
# obs["market_data_5Minute"]: (8, features)
# obs["market_data_15Minute"]: (8, features)
# obs["market_data_1Hour"]: (24, features)
```

---

## How Resampling Works

### Timeframe Alignment

The sampler resamples your 1-minute OHLCV data to multiple timeframes (5min, 15min, 1hour, etc.) and ensures all observations are synchronized at each execution step.

**Example: execute_on=5Minute**

```
Time (minutes):     0    5    10   15   20   25   30
                    |----|----|----|----|----|----|
1-minute bars:      60 bars available
5-minute bars:      |  A  |  B  |  C  |  D  |  E  |  F  |
15-minute bars:     |      X      |      Y      |      Z      |

Execute at:              ↑         ↑         ↑
                        t=5      t=10      t=15
```

At t=10 (executing on 5-minute bar B):
- **1-minute data**: Last 12 bars (from recent history)
- **5-minute data**: Bar A (completed at t=5)
- **15-minute data**: Bar X (completed at t=0)

### Lookahead Bias Prevention

**The Problem**: In real trading, you can't use a bar's data until it has fully closed. A 15-minute bar spanning 0-15 minutes isn't complete until minute 15.

**The Solution**: The sampler indexes higher timeframe bars by their **END time**, not their START time:

```python
# Without fix (WRONG - causes lookahead bias):
# 15-min bar covering [0-15] is indexed at t=0
# At t=10, agent could see bar [0-15] before it closes at t=15 ❌

# With fix (CORRECT - in sampler.py lines 71-77):
# 15-min bar covering [0-15] is indexed at t=15 (its END time)
# At t=10, agent can only see bars that closed BEFORE t=10 ✅
```

**Detailed Example at t=10**:

When your agent executes at minute 10, here's what data is available:

```
✅ CAN use (completed bars only):
  - 1-min bars: [1, 2, 3, ..., 9] (bar 10 is still forming)
  - 5-min bar A [0-5]: Closed at t=5, fully complete
  - 15-min bar covering previous period: Only if it ended before t=10

❌ CANNOT use (incomplete bars):
  - 5-min bar B [5-10]: Still forming, closes at t=10
  - 15-min bar X [0-15]: Still forming, closes at t=15
```

**Why This Matters**: Without this protection, your agent would train on future information (looking into bars that haven't closed yet), leading to unrealistic backtest results that won't work in live trading.

**Implementation Detail** (from `sampler.py:71-77`):

Higher timeframes (coarser than `execute_on`) are shifted forward by their period during resampling. This ensures bars are indexed by their END time. When the agent queries data at execution time, `searchsorted` automatically excludes any bars that haven't closed yet.

---

## Common Configuration Patterns

| Pattern | time_frames | window_sizes | execute_on | Use Case |
|---------|-------------|--------------|------------|----------|
| **Single Timeframe** | `["1min"]` | `[100]` | `(1, "Minute")` | High-frequency, simple features |
| **Multi-Timeframe** | `["1min", "5min", "15min"]` | `[12, 8, 8]` | `(5, "Minute")` | Capture multiple market rhythms |
| **Hierarchical** | `["1min", "5min", "15min", "60min", "240min"]` | `[12, 8, 8, 24, 48]` | `(5, "Minute")` | Complex strategies, trend analysis |
| **Long-Term** | `["60min", "240min", "1440min"]` | `[24, 24, 30]` | `(60, "Minute")` | Position trading, low frequency |

---

## Key Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `time_frames` | list[str] | Timeframes as strings (e.g., "1min", "5min", "1h") | `["1min", "5min", "15min"]` |
| `window_sizes` | list[int] | Lookback window per timeframe | `[12, 8, 8]` |
| `execute_on` | tuple | (value, "Minute"/"Hour") | `(5, "Minute")` |
| `feature_preprocessing_fn` | callable | Transform OHLCV before windowing | `add_indicators` |

---

## Window Size Selection

Choose window sizes based on the information needed:

**For 1-minute timeframe**:
- **12 bars** = 12 minutes of data (short-term)
- **60 bars** = 1 hour of data (medium-term)
- **240 bars** = 4 hours of data (long-term)

**For 5-minute timeframe**:
- **8 bars** = 40 minutes
- **12 bars** = 1 hour
- **24 bars** = 2 hours

**Rule of thumb**: Higher timeframes need fewer bars (they already capture more history per bar).

---

## Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **NaN values in observations** | Training crashes | Fill NaN in `feature_preprocessing_fn` with `df.fillna(0)` |
| **Episode too short** | Episode ends after few steps | Check data length covers `max(window_sizes) * max(time_frames) + episode_length` |
| **Misaligned timeframes** | Unexpected data patterns | Use `execute_on` that's a multiple of all `time_frames` |
| **Memory issues** | OOM errors | Reduce `window_sizes` or number of `time_frames` |
| **Slow sampling** | Environment init takes long | Cache preprocessing results or simplify indicator calculations |

---

## Performance Tips

### 1. Efficient Preprocessing

```python
# ❌ Slow - recalculating indicators
def slow_preprocessing(df):
    for i in range(len(df)):
        df.loc[i, "sma"] = df["close"][i-20:i].mean()
    return df

# ✅ Fast - vectorized operations
def fast_preprocessing(df):
    df["sma"] = df["close"].rolling(20).mean()
    return df
```

### 2. Appropriate Window Sizes

Larger windows = more memory and computation:

```python
# Memory usage ≈ batch_size × num_envs × sum(window_sizes) × num_features × 4 bytes

# Example: 32 batch × 8 envs × (12+8+8) windows × 10 features × 4 bytes ≈ 290 KB
```

Keep `sum(window_sizes) × num_features` reasonable (< 1000 total values per observation).

---

## Technical Reference

- **Source**: [`torchtrade/envs/offline/sampler.py`](https://github.com/TorchTrade/TorchTrade/blob/main/torchtrade/envs/offline/sampler.py)
- **Resampling Logic**: Uses pandas `resample().agg()` with OHLCV aggregation rules
- **Indexing**: Execution times mapped to 1-minute bar indices, then resampled timeframes aligned

---

## Next Steps

- **[Feature Engineering](custom-features.md)** - Add technical indicators via preprocessing
- **[Reward Functions](reward-functions.md)** - Design rewards that work with your sampled data
- **[Offline Environments](../environments/offline.md)** - Apply sampler configuration to environments
