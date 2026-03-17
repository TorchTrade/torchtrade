# Feature Engineering

TorchTrade allows you to add custom technical indicators and features to your market observations. This guide shows you how to preprocess your OHLCV data with custom features before it's fed to your policy.

## How It Works

There are two ways to enrich observations:

1. **Feature processing functions** (`feature_preprocessing_fn`) — compute derived features from OHLCV data (RSI, MACD, etc.). Covered in this guide.
2. **Auxiliary data columns** — pass extra columns alongside OHLCV in your DataFrame (funding rate, basis, etc.). See [Understanding the Sampler](sampler.md#auxiliary-data-columns).

The `feature_preprocessing_fn` parameter in environment configs transforms raw OHLCV data (plus any auxiliary columns) into custom features. This function is called on each resampled timeframe during environment initialization.

**IMPORTANT**: When using `feature_preprocessing_fn`, all feature columns must start with `features_` prefix (e.g., `features_close`, `features_rsi_14`). Only columns with this prefix will be included in the observation space. Without `feature_preprocessing_fn`, raw columns (OHLCV + any auxiliary columns) are used directly as features.

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

## Per-Timeframe Feature Processing

When using multiple timeframes, you can apply **different feature processing functions** to each timeframe. This is useful when you want:

- Different indicators for different timeframes (e.g., fast indicators for 1min, trend indicators for 1hour)
- Different feature counts per timeframe
- Raw OHLCV for some timeframes, processed features for others

### Example: Different Features Per Timeframe

```python
def process_1min(df: pd.DataFrame) -> pd.DataFrame:
    """Fast timeframe: price action features (3 features)."""
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df["features_range"] = df["high"] - df["low"]
    df.fillna(0, inplace=True)
    return df

def process_1hour(df: pd.DataFrame) -> pd.DataFrame:
    """Slow timeframe: trend features (5 features)."""
    df["features_close"] = df["close"]
    df["features_sma_10"] = df["close"].rolling(10).mean()
    df["features_sma_20"] = df["close"].rolling(20).mean()
    df["features_volatility"] = df["close"].pct_change().rolling(10).std()
    df["features_volume_ma"] = df["volume"].rolling(10).mean()
    df.fillna(0, inplace=True)
    return df

config = SequentialTradingEnvConfig(
    time_frames=["1min", "1hour"],
    window_sizes=[30, 10],
    execute_on="1min",
    feature_preprocessing_fn=[process_1min, process_1hour],  # List of functions
    initial_cash=10000,
)

env = SequentialTradingEnv(df, config)

# Observation specs will have different shapes:
# - market_data_1Minute_30: shape (30, 3)
# - market_data_1Hour_10: shape (10, 5)
```

### Mixing Processed and Raw Data

Use `None` in the list to skip processing for a timeframe (keeps raw OHLCV):

```python
config = SequentialTradingEnvConfig(
    time_frames=["1min", "5min"],
    window_sizes=[30, 10],
    feature_preprocessing_fn=[process_1min, None],  # 1min processed, 5min raw OHLCV
)
```

### Backward Compatibility

A single function still works and applies to all timeframes:

```python
# These are equivalent:
feature_preprocessing_fn=my_function
feature_preprocessing_fn=[my_function, my_function]
```

---

## Using Auxiliary Data in Feature Processing

If your DataFrame contains auxiliary columns (funding rate, basis, open interest, etc.), they are available inside `feature_preprocessing_fn` alongside OHLCV:

```python
# DataFrame with auxiliary columns
df = pd.DataFrame({
    "timestamp": ..., "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...,
    "funding_rate": ...,  # auxiliary
    "open_interest": ..., # auxiliary
})

def futures_features(df: pd.DataFrame) -> pd.DataFrame:
    # Standard OHLCV features
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]

    # Features from auxiliary columns
    df["features_funding_rate"] = df["funding_rate"]
    df["features_oi_change"] = df["open_interest"].pct_change()

    # Features combining OHLCV + auxiliary
    df["features_basis_norm"] = df["basis"] / df["close"]

    df.fillna(0, inplace=True)
    return df
```

**Two modes of operation:**

- **Without `feature_preprocessing_fn`**: All columns (OHLCV + auxiliary) flow through directly as raw features. The `features_*` prefix is not required.
- **With `feature_preprocessing_fn`**: Only columns starting with `features_*` are included. You control exactly which columns become features.

See [Auxiliary Data Columns](sampler.md#auxiliary-data-columns) for details on how auxiliary columns are resampled and handled.

---

## Exchange-Specific Kline Fields (Live Environments)

Live environments fetch kline data directly from exchange APIs. Some exchanges return additional fields beyond standard OHLCV that you can use in your `feature_preprocessing_fn`.

### Binance

Binance klines include extra market microstructure data:

| Column | Type | Description |
|--------|------|-------------|
| `open`, `high`, `low`, `close` | float | Standard price data |
| `volume` | float | Base asset volume |
| `quote_volume` | float | Quote asset volume (e.g., USDT volume) |
| `trades` | int | Number of trades in the candle |
| `taker_buy_base` | float | Taker buy volume (base asset) |
| `taker_buy_quote` | float | Taker buy volume (quote asset) |

These allow you to derive sentiment and microstructure features without additional API calls:

```python
def binance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Taker buy ratio: proportion of volume from aggressive buyers
    # >0.5 means buyers dominate, <0.5 means sellers dominate
    df["features_taker_buy_ratio"] = df["taker_buy_base"] / df["volume"]
    # Quote volume change (captures dollar-volume momentum)
    df["features_quote_vol_pct"] = df["quote_volume"].pct_change().fillna(0)
    # Average trade size (large = institutional, small = retail)
    df["features_avg_trade_size"] = df["volume"] / df["trades"]
    # Standard price features
    df["features_close"] = df["close"].pct_change().fillna(0)
    df.fillna(0, inplace=True)
    return df

env = BinanceFuturesTorchTradingEnv(
    config=config,
    feature_preprocessing_fn=binance_features,
)
```

### Bitget and Bybit

Bitget (via CCXT) and Bybit (via pybit) kline APIs return only standard OHLCV columns (`open`, `high`, `low`, `close`, `volume`). To get equivalent sentiment data on these exchanges, you would need to fetch it via separate API calls outside the observation class. Adding built-in support for auxiliary data fetching (funding rate, taker buy/sell ratio, open interest) across all exchanges is planned for a future release.

---

## Important Rules

1. **Feature prefix** (when using `feature_preprocessing_fn`): All output columns MUST start with `features_` (e.g., `features_rsi_14`). Columns without this prefix are ignored. Without a preprocessing function, raw columns (OHLCV + auxiliary) are used directly.
2. **Handle NaN**: Technical indicators produce NaN at the start. Always call `df.fillna(0, inplace=True)` (or `ffill`/`bfill`).
3. **Return the DataFrame**: Your function must `return df`.
4. **No lookahead bias**: Only use past data. Never use `.shift(-1)` or future values.
5. **List length must match**: When using a list of functions, it must have the same length as `time_frames`.

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
