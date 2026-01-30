# Feature Engineering

TorchTrade allows you to add custom technical indicators and features to your market observations. This guide shows you how to preprocess your OHLCV data with custom features before it's fed to your policy.

## How It Works

The `feature_preprocessing_fn` parameter in environment configs transforms raw OHLCV data into custom features. This function is called on each resampled timeframe during environment initialization.

**IMPORTANT**: All feature columns must start with `features_` prefix (e.g., `features_close`, `features_rsi_14`). Only columns with this prefix will be included in the observation space.

!!! warning "Timeframe Format Matters"
    When specifying `time_frames`, use **canonical forms** to avoid confusion:

    - ✅ **Use**: `"1hour"`, `"2hours"`, `"1day"`
    - ❌ **Avoid**: `"60min"`, `"120min"`, `"24hour"`, `"1440min"`

    **Why?** Different formats create different observation keys:

    - `time_frames=["60min"]` → observation key: `"market_data_60Minute"`
    - `time_frames=["1hour"]` → observation key: `"market_data_1Hour"`

    These are treated as **DIFFERENT timeframes**. Models trained with one format won't work with the other. The framework will issue a warning if you use non-canonical forms like `"60min"` to guide you toward cleaner observation keys.

---

## Basic Usage

### Example 1: Adding Technical Indicators

```python
import pandas as pd
import ta  # Technical Analysis library
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

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
config = SequentialTradingEnvConfig(
    feature_preprocessing_fn=custom_preprocessing,
    time_frames=["1min", "5min", "15min"],  # Note: use "1hour" not "60min"
    window_sizes=[12, 8, 8],
    execute_on=(5, "Minute"),
    initial_cash=1000
)

env = SequentialTradingEnv(df, config)
```

### Example 2: Normalized Features (Recommended)

**Feature normalization is critical for stable RL training.** The recommended approach is to normalize features during preprocessing using sklearn's StandardScaler, which avoids device-related issues with TorchRL's VecNorm transforms.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalized_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features using StandardScaler for stable training.

    This approach is preferred over VecNormV2/ObservationNorm transforms
    which can have device compatibility issues on GPU.
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

config = SequentialTradingEnvConfig(
    feature_preprocessing_fn=normalized_preprocessing,
    ...
)
```

**Alternative approaches:**
- **TorchRL transforms** - VecNormV2 and ObservationNorm are available but may have device compatibility issues
- **Network level** - Use BatchNorm, LayerNorm, or other normalization layers in your policy network

!!! tip "Advanced Normalization"
    StandardScaler uses fixed statistics from training data. For data with **regime changes**, consider rolling window normalization or per-regime scalers. For most use cases, StandardScaler is sufficient.

---

## Important Rules

1. **Feature prefix**: All columns MUST start with `features_` (e.g., `features_rsi_14`). Columns without this prefix are ignored.
2. **Handle NaN**: Technical indicators produce NaN at the start. Always call `df.fillna(0, inplace=True)` (or `ffill`/`bfill`).
3. **Return the DataFrame**: Your function must `return df`.
4. **No lookahead bias**: Only use past data. Never use `.shift(-1)` or future values.

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

## Performance Tips

- **Vectorize**: Use pandas operations (`df["close"].pct_change()`) instead of loops — 100x faster.
- **Check NaN**: Add `df.isna().sum()` during development to catch indicator issues before `fillna`.

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
