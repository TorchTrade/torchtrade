"""
"""
from __future__ import annotations

import numpy as np
import torch
import tqdm
from torchrl._utils import timeit
torch.set_float32_matmul_precision("high")
from trading_envs.alpaca.torch_env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import torch
import os
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    TensorDictReplayBuffer,
)
import functools
from dotenv import load_dotenv
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    TransformedEnv,
)
from torchrl.collectors import SyncDataCollector
# Load environment variables
load_dotenv(dotenv_path=".env")


import pandas as pd
import numpy as np
import ta  # pip install ta

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

def make_collector(train_env, frames_per_batch=1, total_frames=10000, policy=None, compile_mode=False, device="cpu"):
    """Make collector."""
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    collector = SyncDataCollector(
        train_env,
        policy,
        init_random_frames=0,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )
    collector.set_seed(42)
    return collector

def main():
    device = torch.device("cpu")

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

    # Create environment
    env = AlpacaTorchTradingEnv(
        config, api_key=os.getenv("API_KEY"),
        api_secret=os.getenv("SECRET_KEY"),
        feature_preprocessing_fn=custom_preprocessing
    )

    def apply_env_transforms(env, max_episode_steps=1000):
        transformed_env = TransformedEnv(
            env,
            Compose(
                InitTracker(),
                StepCounter(max_episode_steps),
                DoubleToFloat(),
                RewardSum(),
            ),
        )
        return transformed_env

    env = apply_env_transforms(env, max_rollout_steps)
    scratch_dir = None
    storage_cls = (
        functools.partial(LazyTensorStorage, device=device)
        if not scratch_dir
        else functools.partial(LazyMemmapStorage, device="cpu", scratch_dir=scratch_dir)
    )
    # Create replay buffer
    replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=3,
            storage=storage_cls(
                1000000,
            ),
            batch_size=1,
            shared=False,
        )
    collector = make_collector(env, policy=None, frames_per_batch=1, total_frames=total_farming_steps)


    # Run Farming
    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=total_farming_steps)
    collector_iter = iter(collector)
    total_iter = len(collector)
    for i in range(total_iter):
        timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)

        with timeit("collect"):
            tensordict = next(collector_iter)

        current_frames = tensordict.numel()
        pbar.update(current_frames)

        with timeit("rb - extend"):
            # Add to replay buffer
            tensordict = tensordict.reshape(-1)
            replay_buffer.extend(tensordict)

        collected_frames += current_frames

        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

        # Logging
        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean()
            metrics_to_log["train/episode_length"] = episode_length.sum() / len(
                episode_length
            )
        print(metrics_to_log)
        if collected_frames % save_buffer_every == 0:
            replay_buffer.dumps(f"./replay_buffer_{policy_type}.pt")


    

    

if __name__ == "__main__":
    main()