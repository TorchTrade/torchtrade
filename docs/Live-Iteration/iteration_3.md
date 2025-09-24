# Updated Account State - Iteration 3


## TLDR: Adding  entry point, unrealized profit and holding time to the account state.



___________________________

## Observation Setup

```python

    total_farming_steps = 10000
    save_buffer_every = 10
    max_rollout_steps = 72 #  72 steps a 5min -> 6h per episode -> 4 episodes per day
    policy_type = "random"

    torch.manual_seed(42)
    np.random.seed(42)
    # Create environment configuration
    config = AlpacaTradingEnvConfig(
        symbol="BTC/USD",
        paper=True,
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
            TimeFrame(5, TimeFrameUnit.Minute),
            TimeFrame(15, TimeFrameUnit.Minute),
            TimeFrame(1, TimeFrameUnit.Hour),
        ],
        window_sizes=[12, 8, 8, 24],  # ~12m, 40m, 2h, 1d
        execute_on=TimeFrame(5, TimeFrameUnit.Minute), # Try 15min
    )

```

## Feature Preprocessing Function
with the feature preprocessing function:

```python

def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess OHLCV dataframe with engineered features for RL trading.

    Expected columns: ["open", "high", "low", "close", "volume"]
    Index can be datetime or integer.
    """

    df = df.copy().reset_index(drop=True)

    # --- Basic features ---
    # Log returns
    df["features_return_log"] = np.log(df["close"]).diff()

    # Rolling volatility (5-period)
    df["features_volatility"] = df["features_return_log"].rolling(window=5).std()

    # ATR (14) normalized
    df["features_atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range() / df["close"]

    # --- Momentum & trend ---
    ema_12 = ta.trend.EMAIndicator(close=df["close"], window=12).ema_indicator()
    ema_24 = ta.trend.EMAIndicator(close=df["close"], window=24).ema_indicator()
    df["features_ema_12"] = ema_12
    df["features_ema_24"] = ema_24
    df["features_ema_slope"] = ema_12.diff()

    macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["features_macd_hist"] = macd.macd_diff()

    df["features_rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    # --- Volatility bands ---
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["features_bb_pct"] = bb.bollinger_pband()

    # --- Volume / flow ---
    df["features_volume_z"] = (
        (df["volume"] - df["volume"].rolling(20).mean()) /
        df["volume"].rolling(20).std()
    )
    df["features_vwap_dev"] = df["close"] - (
        (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    )

    # --- Candle structure ---
    df["features_body_ratio"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    df["features_upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (
        df["high"] - df["low"] + 1e-9
    )
    df["features_lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (
        df["high"] - df["low"] + 1e-9
    )

    # Drop rows with NaN from indicators
    df.dropna(inplace=True)

    return df

```
## New Account State

In this iteration we changed the account state to include the following information:

- cash
- position_size
- position_value
- entry_price
- unrealized_pnlpc
- holding_time

Before it was:

- cash
- position_size
- position_value

And the agent could not know at which price it entered the position or how long it held the position and how the unrealized pnl was (even though this is in a way redundant to the current price and entry price).

```python 

    if position_status is None:
        self.position_hold_counter = 0
        position_size = 0.0
        position_value = 0.0
        entry_price = 0.0
        unrealized_pnlpc = 0.0
        holding_time = self.position_hold_counter

    else:
        self.position_hold_counter += 1
        position_size = position_status.qty
        position_value = position_status.market_value
        entry_price = position_status.avg_entry_price
        unrealized_pnlpc = position_status.unrealized_plpc
        holding_time = self.position_hold_counter

    account_state = torch.tensor(
        [cash, position_size, position_value, entry_price, unrealized_pnlpc, holding_time], dtype=torch.float
    )

```

## Inspection of the Collected Data

We have collected 15 episodes on colony2 with 2 positive trades. Both colony1 and colony2 did errored out due to some weird API error.

Colony1:

```bash

requests.exceptions.ConnectionError: HTTPSConnectionPool(host='paper-api.alpaca.markets', port=443): Max retries exceeded with url: /v2/account (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0xffff673686d0>: Failed to resolve 'paper-api.alpaca.markets' ([Errno -3] Temporary failure in name resolution)"))

```

Colony2:

```bash

  File "/home/colony2/miniconda3/envs/alpaca_env/lib/python3.9/site-packages/requests/adapters.py", line 659, in send
    raise ConnectionError(err, request=request)
requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))


```

But besides that overall results look good. Account state seems more informative now. Ready for some tests to train a policy on the new colony2 data even though it has few positive trades. But just to see how a deployed policy would perform.

Positive trades from colony2:

```python

time step 502
account tensor([     6.3800,      0.0027,    312.1673, 115623.7031,     -0.0032,
             2.0000])
action tensor(0)
next_account tensor([318.5800,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000])
reward tensor([0.0001])


time step 698
account tensor([     5.1600,      0.0023,    256.9382, 112426.6250,     -0.0019,
             1.0000])
action tensor(0)
next_account tensor([262.1900,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000])
reward tensor([0.0004])

```

Results from colony1 can be neglected as it only collected 182 steps due to the error.



However, we need to use these samples from iteration3 to test and compare the OneStepEnv and the regular sequential env!

