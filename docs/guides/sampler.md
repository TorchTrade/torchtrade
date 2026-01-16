# Understanding the Sampler

The `MarketDataObservationSampler` handles multi-timeframe data sampling in TorchTrade's offline environments.

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

The sampler is created automatically:

```python
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig

# Configure multi-timeframe sampling
config = SeqLongOnlyEnvConfig(
    time_frames=["1min", "5min", "15min", "60min"],        # All values in minutes: 1min, 5min, 15min, 60min
    window_sizes=[12, 8, 8, 24],       # Lookback per timeframe
    execute_on=(5, "Minute"),          # Trade every 5min
)

env = SeqLongOnlyEnv(df, config)

# Observations contain all timeframes
obs = env.reset()
# obs["market_data_1Minute"]: (12, features)
# obs["market_data_5Minute"]: (8, features)
# obs["market_data_15Minute"]: (8, features)
# obs["market_data_60Minute"]: (24, features)
```

---

## How Resampling Works

### Timeframe Alignment

The sampler ensures all timeframes align with the execution timeframe:

**Example: execute_on=5Minute**

```
1-minute bars:  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
5-minute bars:  |    A    |    B    |    C    |    D    |
Execute at:            ↑         ↑         ↑         ↑
                    t=5       t=10      t=15      t=20
```

At each execution timestep:
- All timeframes show data **up to and including** the current bar
- No future data is leaked
- Lower timeframes have more recent bars

### Lookahead Bias Prevention

**Key principle**: At time t, only use data from bars that have **closed** by time t.

```python
# At t=10 (executing on 5-minute bar B)
# ✅ Can use: 5-min bar A (closed at t=5)
# ❌ Cannot use: 5-min bar B (closes at t=10, not fully formed yet)

# Implementation: Sampler indexes one bar behind execution time
window = data[exec_index - window_size : exec_index]  # Excludes current forming bar
```

This is why you see a shift between execution time and visible data.

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

## Advanced: Custom Resampling

If you need non-standard resampling (e.g., tick bars, volume bars), pass pre-resampled data:

```python
# Prepare custom timeframes externally
df_1m = raw_data  # Already 1-minute bars
df_5m = resample_custom(raw_data, method="volume_bars", threshold=1000)
df_15m = resample_custom(raw_data, method="tick_bars", threshold=100)

# Environment handles them as standard timeframes
config = SeqLongOnlyEnvConfig(
    time_frames=["1min", "5min", "15min"],  # Interpreted as provided timeframes
    # ... rest of config
)
```

**Note**: This is advanced usage. Standard time-based resampling works for most cases.

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
