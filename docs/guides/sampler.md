# Understanding the Sampler

The `MarketDataObservationSampler` (found in `torchtrade/envs/offline/sampler.py`) handles multi-timeframe data sampling in TorchTrade's offline environments. It resamples high-frequency data (1-minute bars) to multiple timeframes and creates synchronized observation windows while preventing lookahead bias. This allows RL agents to observe market patterns across different time scales simultaneously, from short-term momentum to long-term trends.

## What Is the Sampler?

The sampler:

1. **Resamples** 1-minute OHLCV (+ optional auxiliary columns) to multiple timeframes (5m, 15m, 1h)
2. **Applies feature preprocessing** to each timeframe
3. **Creates sliding windows** of market data
4. **Prevents lookahead bias** by correct bar indexing

```
1-Minute Data (OHLCV + optional aux) → Sampler → Multi-Timeframe Observations
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
# Optional: additional columns (funding_rate, basis, etc.) are passed through automatically

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

The sampler is used in all offline environments (SequentialTradingEnv, SequentialTradingEnvSLTP, OneStepTradingEnv) and allows flexible selection of timeframes through the environment configuration:

```python
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

# Configure multi-timeframe sampling
config = SequentialTradingEnvConfig(
    time_frames=["1min", "5min", "15min", "1hour"],
    window_sizes=[12, 8, 8, 24],
    execute_on=(5, "Minute"),
)

env = SequentialTradingEnv(df, config)

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

## Auxiliary Data Columns

The sampler accepts DataFrames with extra columns beyond OHLCV. This lets you include data from external sources — funding rates, basis, open interest, sentiment scores — alongside price data without needing a `feature_preprocessing_fn`.

### How It Works

```
DataFrame columns:  [timestamp, open, high, low, close, volume, funding_rate, basis]
                                                                 ^^^^^^^^^^^^  ^^^^^
                                                                 auxiliary columns
After reorder:      [open, high, low, close, volume, funding_rate, basis]
                     ───────── OHLCV ─────────  ──── auxiliary ────
                     positions 0-4 (guaranteed)   positions 5+
```

1. **Validation**: The sampler checks that OHLCV + timestamp columns are present. Extra columns are accepted.
2. **Column reordering**: OHLCV is always placed at positions 0-4, auxiliary columns follow after. This preserves internal positional contracts.
3. **Resampling**: When resampling to higher timeframes, auxiliary columns use `"last"` aggregation (the value at bar close), while OHLCV uses canonical rules (open=first, high=max, low=min, close=last, volume=sum).
4. **Sparse data handling**: Auxiliary NaN values are forward-filled after resampling (see below).
5. **Observation tensors**: Shape becomes `(window_size, 5 + n_aux)` instead of `(window_size, 5)`.

### Sparse Auxiliary Data

!!! warning "Auxiliary data is forward-filled automatically"
    Auxiliary data often has a **different frequency** than OHLCV. For example, funding rates update every 8 hours while price data is 1-minute. When resampled, most bars will have no auxiliary value (NaN).

    **The sampler handles this automatically:** after resampling, auxiliary NaN values are forward-filled — the last known value persists until the next update. Any leading NaN (before the first known value) is filled with 0.

    ```
    Raw funding_rate (1h updates on 1m data):
    t=0: 0.001, t=1: NaN, t=2: NaN, ..., t=59: NaN, t=60: 0.002, ...

    After resampling to 5min bars:
    [0-5]:  NaN → 0.001 (last of [0.001, NaN, NaN, NaN, NaN])
    [5-10]: NaN → forward-filled to 0.001
    [10-15]: NaN → forward-filled to 0.001
    ...
    [60-65]: 0.002 (new value arrives)
    ```

    **This means:**

    - You do **not** need to pre-fill auxiliary data before passing it to the sampler
    - Bars are never dropped due to missing auxiliary data (only missing OHLCV drops a bar)
    - The agent always sees the **most recent known value** for each auxiliary column

### Example: Futures Data with Funding Rate

```python
import pandas as pd
from torchtrade.envs.offline.infrastructure.sampler import MarketDataObservationSampler
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit

# Load OHLCV + auxiliary data
df = pd.DataFrame({
    "timestamp": timestamps,
    "open": open_prices,
    "high": high_prices,
    "low": low_prices,
    "close": close_prices,
    "volume": volumes,
    "funding_rate": funding_rates,  # extra column
    "basis": basis_values,          # extra column
})

sampler = MarketDataObservationSampler(
    df=df,
    time_frames=[TimeFrame(1, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute)],
    window_sizes=[12, 8],
    execute_on=TimeFrame(1, TimeFrameUnit.Minute),
)

# Observations now include auxiliary data
sampler.reset()
obs, ts, truncated = sampler.get_sequential_observation()
# obs["1Minute"].shape = (12, 7)  — 5 OHLCV + 2 auxiliary
# obs["5Minute"].shape = (8, 7)
```

### Combining with Feature Processing

Auxiliary columns are available inside `feature_preprocessing_fn`, letting you derive features from both OHLCV and auxiliary data:

```python
def futures_features(df: pd.DataFrame) -> pd.DataFrame:
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df["features_funding_rate"] = df["funding_rate"]  # auxiliary column
    df["features_basis_norm"] = df["basis"] / df["close"]  # derived from aux + OHLCV
    df.fillna(0, inplace=True)
    return df
```

### When to Use Auxiliary Columns vs Feature Processing

| Approach | Use For | Example |
|----------|---------|---------|
| **Auxiliary columns** | Raw external data that exists in your dataset | Funding rate, open interest, basis, sentiment |
| **Feature processing** | Derived indicators computed from data | RSI, MACD, Bollinger Bands, moving averages |
| **Both together** | External data + derived features | Funding rate (aux) + funding rate change (derived) |

Without `feature_preprocessing_fn`, auxiliary columns flow through directly as raw features in the observation tensor. With `feature_preprocessing_fn`, you control exactly which columns become `features_*` outputs.

---

## Key Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `time_frames` | list[str] | Timeframes as strings (e.g., "1min", "5min", "1h") | `["1min", "5min", "15min"]` |
| `window_sizes` | list[int] | Lookback window per timeframe | `[12, 8, 8]` |
| `execute_on` | tuple | (value, "Minute"/"Hour") | `(5, "Minute")` |
| `feature_preprocessing_fn` | callable | Transform OHLCV (+ aux) before windowing | `add_indicators` |

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
| **Stale auxiliary data** | Agent sees outdated values | Expected behavior — sparse aux data is forward-filled. Ensure your data source updates frequently enough for your use case |
| **Episode too short** | Episode ends after few steps | Check data length covers `max(window_sizes) * max(time_frames) + episode_length` |
| **Misaligned timeframes** | Unexpected data patterns | Use `execute_on` that's a multiple of all `time_frames` |
| **Memory issues** | OOM errors | Reduce `window_sizes` or number of `time_frames` |
| **Slow sampling** | Environment init takes long | Cache preprocessing results or simplify indicator calculations |

---

## Performance Tips

- **Vectorize preprocessing**: Use pandas operations (`df["close"].rolling(20).mean()`) instead of loops.
- **Appropriate window sizes**: Larger windows = more memory. Keep `sum(window_sizes) × num_features` < 1000 total values per observation.

---

## Technical Reference

- **Source**: [`torchtrade/envs/offline/infrastructure/sampler.py`](https://github.com/TorchTrade/TorchTrade/blob/main/torchtrade/envs/offline/infrastructure/sampler.py)
- **Resampling Logic**: Uses pandas `resample().agg()` with OHLCV aggregation rules (auxiliary columns use `"last"`)
- **Column Order**: OHLCV always occupies positions 0-4 regardless of input column order
- **Indexing**: Execution times mapped to 1-minute bar indices, then resampled timeframes aligned

---

## Next Steps

- **[Feature Engineering](custom-features.md)** - Add technical indicators via preprocessing
- **[Reward Functions](reward-functions.md)** - Design rewards that work with your sampled data
- **[Offline Environments](../environments/offline.md)** - Apply sampler configuration to environments
