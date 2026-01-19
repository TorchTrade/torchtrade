# Feature Engineering

TorchTrade allows you to add custom technical indicators and features to your market observations. This guide shows you how to preprocess your OHLCV data with custom features before it's fed to your policy.

## How It Works

The `feature_preprocessing_fn` parameter in environment configs transforms raw OHLCV data into custom features. This function is called on each resampled timeframe during environment initialization.

**IMPORTANT**: All feature columns must start with `features_` prefix (e.g., `features_close`, `features_rsi_14`). Only columns with this prefix will be included in the observation space.

---

## Basic Usage

### Example 1: Adding Technical Indicators

```python
import pandas as pd
import ta  # Technical Analysis library
from torchtrade.envs.offline import SeqLongOnlyEnv, SeqLongOnlyEnvConfig

def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators as features.

    IMPORTANT: All feature columns must start with 'features_' prefix.
    """
    # Basic OHLCV features (always include these)
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]

    # RSI (Relative Strength Index)
    df["features_rsi_14"] = ta.momentum.RSIIndicator(
        df["close"], window=14
    ).rsi()

    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df["close"])
    df["features_macd"] = macd.macd()
    df["features_macd_signal"] = macd.macd_signal()
    df["features_macd_histogram"] = macd.macd_diff()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["features_bb_high"] = bollinger.bollinger_hband()
    df["features_bb_mid"] = bollinger.bollinger_mavg()
    df["features_bb_low"] = bollinger.bollinger_lband()

    # Fill NaN values (important!)
    df.fillna(0, inplace=True)

    return df

# Use in environment config
config = SeqLongOnlyEnvConfig(
    feature_preprocessing_fn=custom_preprocessing,
    time_frames=["1min", "5min", "15min"],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=1000
)

env = SeqLongOnlyEnv(df, config)
```

### Example 2: Normalized Features

Feature normalization can be done at multiple levels:

- **Feature preprocessing** (shown below) - Normalize in the preprocessing function

- **TorchRL transforms** - Use [VecNorm](https://docs.pytorch.org/rl/main/reference/generated/torchrl.envs.transforms.VecNorm.html) or [ObservationNorm](https://docs.pytorch.org/rl/main/reference/generated/torchrl.envs.transforms.ObservationNorm.html) transforms

- **Network level** - Use BatchNorm, LayerNorm, or other normalization layers in your policy network

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalized_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features using StandardScaler for stable training.
    """
    # Basic OHLCV
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]

    # Price changes (returns)
    df["features_return"] = df["close"].pct_change()

    # Normalize features
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col.startswith("features_")]

    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Fill NaN values
    df.fillna(0, inplace=True)

    return df

config = SeqLongOnlyEnvConfig(
    feature_preprocessing_fn=normalized_preprocessing,
    ...
)
```

---

## Important Rules

### 1. Feature Column Naming

**All feature columns MUST start with `features_` prefix:**

```python
# ✅ Correct
df["features_rsi_14"] = ...
df["features_macd"] = ...
df["features_close"] = ...

# ❌ Wrong - will be ignored
df["rsi_14"] = ...
df["macd"] = ...
df["close"] = ...
```

### 2. Handle NaN Values

Technical indicators often produce NaN values at the beginning. **Always fill NaN values:**

```python
# Option 1: Fill with 0 (recommended for most indicators)
df.fillna(0, inplace=True)

# Option 2: Forward fill (use for prices)
df.fillna(method='ffill', inplace=True)

# Option 3: Backward fill
df.fillna(method='bfill', inplace=True)
```

### 3. Return the Modified DataFrame

Your function must return the modified DataFrame:

```python
def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # ... add features ...
    return df  # ✅ Return modified DataFrame
```

### 4. Avoid Lookahead Bias

**Do NOT use future data in your features:**

```python
# ❌ Lookahead bias - using future data
df["features_future_return"] = df["close"].pct_change().shift(-1)

# ✅ Correct - only past data
df["features_past_return"] = df["close"].pct_change()
```

---

## Common Technical Indicators

### Quick Reference Table

| Category | Indicator | ta Library Code | Use Case |
|----------|-----------|-----------------|----------|
| **Momentum** | RSI | `ta.momentum.RSIIndicator(close, window=14).rsi()` | Overbought/oversold detection |
| | Stochastic | `ta.momentum.StochasticOscillator(high, low, close).stoch()` | Momentum confirmation |
| | Williams %R | `ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()` | Short-term overbought/oversold |
| **Trend** | SMA | `ta.trend.SMAIndicator(close, window=20).sma_indicator()` | Trend direction |
| | EMA | `ta.trend.EMAIndicator(close, window=20).ema_indicator()` | Responsive trend following |
| | MACD | `ta.trend.MACD(close).macd()` | Trend changes |
| | ADX | `ta.trend.ADXIndicator(high, low, close, window=14).adx()` | Trend strength |
| **Volatility** | Bollinger Bands | `ta.volatility.BollingerBands(close, window=20)` | Volatility and price bounds |
| | ATR | `ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()` | Volatility measurement |
| | Keltner | `ta.volatility.KeltnerChannel(high, low, close)` | Alternative to Bollinger |
| **Volume** | OBV | `ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()` | Accumulation/distribution |
| | VPT | `ta.volume.VolumePriceTrendIndicator(close, volume).volume_price_trend()` | Volume-price confirmation |
| | ADI | `ta.volume.AccDistIndexIndicator(high, low, close, volume).acc_dist_index()` | Money flow |

### Usage Pattern

```python
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Basic OHLCV
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    # ... other OHLCV ...

    # Pick indicators from table above
    df["features_rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["features_sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    df["features_atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    df.fillna(0, inplace=True)
    return df
```

---

## Performance Considerations

### Vectorize Operations

Use pandas vectorized operations instead of loops for faster computation:

```python
# ❌ Slow - using loops
for i in range(len(df)):
    df.loc[i, "features_return"] = (df.loc[i, "close"] / df.loc[i-1, "close"]) - 1

# ✅ Fast - vectorized (100x faster)
df["features_return"] = df["close"].pct_change()
```

**General principle**: If you're using a loop, there's probably a pandas method that does it faster.

---

## Debugging Tips

### Check for NaN Values

```python
def safe_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df["features_close"] = df["close"]
    # ... add features ...

    # Check for NaN values before filling
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print("⚠️ Warning: NaN values detected:")
        print(nan_counts[nan_counts > 0])

    df.fillna(0, inplace=True)
    return df
```

This helps catch issues with indicator configuration or missing data.

---

## Recommended Libraries

| Library | Indicators | Installation | Best For |
|---------|-----------|--------------|----------|
| **[ta](https://github.com/bukosabino/ta)** | 40+ | `pip install ta` | Standard indicators, easy API |
| **[pandas-ta](https://github.com/twopirllc/pandas-ta)** | 130+ | `pip install pandas-ta` | Comprehensive collection |
| **[TA-Lib](https://github.com/mrjbq7/ta-lib)** | 150+ | `pip install TA-Lib` | Performance, industry standard |
| **[sklearn](https://scikit-learn.org/)** | N/A | `pip install scikit-learn` | Feature scaling, normalization |

**Recommendation**: Start with `ta` for simplicity, use `TA-Lib` if you need maximum performance.

---

## Next Steps

- **[Reward Functions](reward-functions.md)** - Design reward signals that work with your features
- **[Understanding the Sampler](sampler.md)** - How multi-timeframe sampling works
- **[Transforms](../components/transforms.md)** - Alternative feature engineering with Chronos embeddings
- **[Offline Environments](../environments/offline.md)** - Apply custom features to environments
