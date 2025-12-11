from torchtrade.envs.offline.sampler import MarketDataObservationSampler, MarketDataObservationSampler_old
import pandas as pd
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit
import ta
import torch



# Before: Full: Sampling 72300 observations took 12.228289 seconds.
#         1000 Steps: Sampling 1000 observations took 0.166234 seconds.
#   
# AFTER: Full: Sampling 72299 observations took 0.750082 seconds.
#        1000 Steps: Sampling 1000 observations took 0.006994 seconds.


def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess OHLCV dataframe with engineered features for RL trading.

    Expected columns: ["open", "high", "low", "close", "volume"]
    Index can be datetime or integer.
    """

    df = df.copy().reset_index(drop=False)

    # --- Basic features ---
    # Log returns
    # df["features_return_log"] = np.log(df["close"]).diff()
    df["features_close"] = df["close"]

    # # Rolling volatility (5-period)
    # df["features_volatility"] = df["features_return_log"].rolling(window=5).std()

    # # ATR (14) normalized
    # df["features_atr"] = ta.volatility.AverageTrueRange(
    #     high=df["high"], low=df["low"], close=df["close"], window=14
    # ).average_true_range() / df["close"]

    # --- Momentum & trend ---
    ema_12 = ta.trend.EMAIndicator(close=df["close"], window=12).ema_indicator()
    ema_24 = ta.trend.EMAIndicator(close=df["close"], window=24).ema_indicator()
    df["features_ema_12"] = ema_12
    df["features_ema_24"] = ema_24
    #df["features_ema_slope"] = ema_12.diff()

    # macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    # df["features_macd_hist"] = macd.macd_diff()

    # df["features_rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    # # --- Volatility bands ---
    # bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    # df["features_bb_pct"] = bb.bollinger_pband()

    # --- Volume / flow ---
    df["features_volume_z"] = (
        (df["volume"] - df["volume"].rolling(20).mean()) /
        df["volume"].rolling(20).std()
    )
    # df["features_vwap_dev"] = df["close"] - (
    #     (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    # )

    # # --- Candle structure ---
    # df["features_body_ratio"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    # df["features_upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (
    #     df["high"] - df["low"] + 1e-9
    # )
    # df["features_lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (
    #     df["high"] - df["low"] + 1e-9
    #)

    # Drop rows with NaN from indicators
    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    df = pd.read_csv("/home/sebastian/Documents/TorchTrade/torchrl_alpaca_env/torchtrade/data/binance_spot_1m_cleaned/btcusdt_spot_1m_12_2024_to_09_2025.csv")
    test_df = df[0:(1440 *21)] # 14 days
    train_df = df[(1440 * 21):]

    time_frames=[
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
        TimeFrame(15, TimeFrameUnit.Minute),
        TimeFrame(1, TimeFrameUnit.Hour),
    ]
    window_sizes=[12, 8, 8, 24]  # ~12m, 40m, 2h, 1d
    execute_on=TimeFrame(5, TimeFrameUnit.Minute) # Try 15min


    sampler = MarketDataObservationSampler(
        train_df,
        time_frames=time_frames,
        window_sizes=window_sizes,
        execute_on=execute_on,
        feature_processing_fn=custom_preprocessing,
        features_start_with="features_",
        max_traj_length=1000,
    )

    sampler_old = MarketDataObservationSampler_old(
        train_df,
        time_frames=time_frames,
        window_sizes=window_sizes,
        execute_on=execute_on,
        feature_processing_fn=custom_preprocessing,
        features_start_with="features_",
        max_traj_length=1000,
    )


    max_len = sampler.reset(random_start=False)
    obs_keys = sampler.get_observation_keys()
    feature_keys = sampler.get_feature_keys()

    max_len_old = sampler_old.reset(random_start=False)
    obs_keys_old = sampler_old.get_observation_keys()
    feature_keys_old = sampler_old.get_feature_keys()

    assert max_len == max_len_old
    assert obs_keys == obs_keys_old
    assert feature_keys == feature_keys_old



    # for i in range(max_len):
    #     obs, time_stamp, trunc = sampler.get_sequential_observation()
    #     obs_old, time_stamp_old, trunc_old = sampler_old.get_sequential_observation()

    #     for k, v in obs.items():
    #         assert torch.isclose(torch.from_numpy(obs_old[k]).float(), obs[k]).all()
    #     assert time_stamp == time_stamp_old
    #     assert trunc == trunc_old

    import time
    start = time.time()
    for i in range(max_len):
        obs, time_stamp, trunc = sampler.get_sequential_observation()
        obs = sampler.get_base_features(time_stamp)
    end = time.time()
    print(f"Sampling {max_len} observations took {end - start:.6f} seconds.")