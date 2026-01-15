# Custom Feature Engineering

TorchTrade allows you to add custom technical indicators and features to your market observations. This guide shows you how to preprocess your OHLCV data with custom features before it's fed to your policy.

## How It Works

The `feature_preprocessing_fn` parameter in environment configs allows you to transform raw OHLCV data into custom features. This function is called on each resampled timeframe during environment initialization.

```
Raw OHLCV Data
    ↓
Resample to Multiple Timeframes
    ↓
Apply feature_preprocessing_fn  ← Your custom function
    ↓
Create Sliding Windows
    ↓
Feed to Policy Network
```

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
    time_frames=[1, 5, 15],
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=1000
)

env = SeqLongOnlyEnv(df, config)
```

### Example 2: Normalized Features

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalized_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features using StandardScaler.
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

### Example 3: Volume-Based Features

```python
import pandas as pd

def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based trading features.
    """
    # Basic OHLCV
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]

    # Volume Moving Average
    df["features_volume_ma_20"] = df["volume"].rolling(window=20).mean()

    # Volume Ratio (current / average)
    df["features_volume_ratio"] = df["volume"] / (df["features_volume_ma_20"] + 1e-9)

    # On-Balance Volume (OBV)
    df["features_obv"] = ta.volume.OnBalanceVolumeIndicator(
        df["close"], df["volume"]
    ).on_balance_volume()

    # Volume-Weighted Average Price (VWAP)
    df["features_vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

    # Fill NaN values
    df.fillna(0, inplace=True)

    return df

config = SeqLongOnlyEnvConfig(
    feature_preprocessing_fn=volume_features,
    ...
)
```

## Advanced Usage

### Example 4: Multi-Timeframe Features with Context

Sometimes you want different features for different timeframes. The sampler calls your function separately for each timeframe after resampling.

```python
import pandas as pd
import ta

def adaptive_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapt features based on timeframe frequency.

    The function receives resampled data for each timeframe independently.
    """
    # Basic OHLCV
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]

    # Infer timeframe from data frequency
    # (Note: This is approximate - you might want to pass timeframe explicitly)
    freq = pd.infer_freq(df.index)

    if freq in ['T', '1T']:  # 1-minute data
        # Short-term indicators for 1m data
        df["features_rsi_7"] = ta.momentum.RSIIndicator(df["close"], window=7).rsi()
        df["features_ema_10"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()

    elif freq in ['5T', '5min']:  # 5-minute data
        # Medium-term indicators for 5m data
        df["features_rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["features_ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()

    else:  # Higher timeframes (15m, 1h, etc.)
        # Long-term indicators
        df["features_rsi_21"] = ta.momentum.RSIIndicator(df["close"], window=21).rsi()
        df["features_ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()

    # Fill NaN values
    df.fillna(0, inplace=True)

    return df

config = SeqLongOnlyEnvConfig(
    feature_preprocessing_fn=adaptive_preprocessing,
    time_frames=[1, 5, 15, 60],
    ...
)
```

### Example 5: Combining Multiple Indicator Libraries

```python
import pandas as pd
import ta
import pandas_ta as pta

def comprehensive_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine features from multiple technical analysis libraries.
    """
    # Basic OHLCV
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]

    # From ta library
    df["features_rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["features_adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()

    # From pandas_ta library
    df["features_cci"] = pta.cci(df["high"], df["low"], df["close"], length=20)
    df["features_atr"] = pta.atr(df["high"], df["low"], df["close"], length=14)

    # Custom calculations
    df["features_price_momentum_5"] = df["close"].pct_change(periods=5)
    df["features_volume_momentum_5"] = df["volume"].pct_change(periods=5)

    # Fill NaN values
    df.fillna(0, inplace=True)

    return df

config = SeqLongOnlyEnvConfig(
    feature_preprocessing_fn=comprehensive_preprocessing,
    ...
)
```

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
# Option 1: Fill with 0
df.fillna(0, inplace=True)

# Option 2: Forward fill
df.fillna(method='ffill', inplace=True)

# Option 3: Backward fill
df.fillna(method='bfill', inplace=True)

# Option 4: Drop NaN rows (use carefully!)
df.dropna(inplace=True)
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

## Common Technical Indicators

### Momentum Indicators

```python
import ta

# RSI (Relative Strength Index)
df["features_rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

# Stochastic Oscillator
stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
df["features_stoch_k"] = stoch.stoch()
df["features_stoch_d"] = stoch.stoch_signal()

# Williams %R
df["features_williams_r"] = ta.momentum.WilliamsRIndicator(
    df["high"], df["low"], df["close"], lbp=14
).williams_r()
```

### Trend Indicators

```python
import ta

# Moving Averages
df["features_sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
df["features_ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()

# MACD
macd = ta.trend.MACD(df["close"])
df["features_macd"] = macd.macd()
df["features_macd_signal"] = macd.macd_signal()
df["features_macd_hist"] = macd.macd_diff()

# ADX (Average Directional Index)
df["features_adx"] = ta.trend.ADXIndicator(
    df["high"], df["low"], df["close"], window=14
).adx()
```

### Volatility Indicators

```python
import ta

# Bollinger Bands
bollinger = ta.volatility.BollingerBands(df["close"], window=20)
df["features_bb_high"] = bollinger.bollinger_hband()
df["features_bb_mid"] = bollinger.bollinger_mavg()
df["features_bb_low"] = bollinger.bollinger_lband()
df["features_bb_width"] = bollinger.bollinger_wband()

# ATR (Average True Range)
df["features_atr"] = ta.volatility.AverageTrueRange(
    df["high"], df["low"], df["close"], window=14
).average_true_range()

# Keltner Channels
keltner = ta.volatility.KeltnerChannel(df["high"], df["low"], df["close"])
df["features_keltner_high"] = keltner.keltner_channel_hband()
df["features_keltner_low"] = keltner.keltner_channel_lband()
```

### Volume Indicators

```python
import ta

# OBV (On-Balance Volume)
df["features_obv"] = ta.volume.OnBalanceVolumeIndicator(
    df["close"], df["volume"]
).on_balance_volume()

# Volume Price Trend
df["features_vpt"] = ta.volume.VolumePriceTrendIndicator(
    df["close"], df["volume"]
).volume_price_trend()

# Accumulation/Distribution
df["features_adi"] = ta.volume.AccDistIndexIndicator(
    df["high"], df["low"], df["close"], df["volume"]
).acc_dist_index()
```

## Performance Considerations

### Cache Expensive Computations

If your preprocessing is expensive, consider caching results:

```python
import joblib
from pathlib import Path

def cached_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    cache_file = Path("cache/preprocessed_data.pkl")

    if cache_file.exists():
        return joblib.load(cache_file)

    # Expensive computations
    df["features_close"] = df["close"]
    # ... more features ...

    # Save to cache
    cache_file.parent.mkdir(exist_ok=True)
    joblib.dump(df, cache_file)

    return df
```

### Vectorize Operations

Use pandas vectorized operations instead of loops:

```python
# ❌ Slow - using loops
for i in range(len(df)):
    df.loc[i, "features_return"] = (df.loc[i, "close"] / df.loc[i-1, "close"]) - 1

# ✅ Fast - vectorized
df["features_return"] = df["close"].pct_change()
```

## Debugging Tips

### Print Feature Shapes

```python
def debug_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Input shape: {df.shape}")

    df["features_close"] = df["close"]
    # ... add more features ...

    feature_cols = [col for col in df.columns if col.startswith("features_")]
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")

    return df
```

### Check for NaN Values

```python
def safe_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df["features_close"] = df["close"]
    # ... add features ...

    # Check for NaN values
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print("⚠️ Warning: NaN values detected:")
        print(nan_counts[nan_counts > 0])

    df.fillna(0, inplace=True)
    return df
```

## Next Steps

- **[Custom Reward Functions](reward-functions.md)** - Design better reward signals
- **[Understanding the Sampler](sampler.md)** - How multi-timeframe sampling works
- **[Building Custom Environments](custom-environment.md)** - Extend TorchTrade
- **[Offline Environments](../environments/offline.md)** - Apply custom features to environments

## Recommended Libraries

- **[ta](https://github.com/bukosabino/ta)** - Technical Analysis library
- **[pandas-ta](https://github.com/twopirllc/pandas-ta)** - 130+ indicators
- **[TA-Lib](https://github.com/mrjbq7/ta-lib)** - Popular C library with Python bindings
- **[sklearn](https://scikit-learn.org/)** - Feature scaling and normalization
