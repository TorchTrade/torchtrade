"""
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
from torchrl._utils import timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
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
# Load environment variables
load_dotenv()


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



def main():
    device = torch.device("cpu")

    total_eval_steps = 1000
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


    # Run Evaluation
    total_collected = 0
    pbar = tqdm(total=total_eval_steps, desc="Evaluating", unit="steps")
    for i in range(1000):
        with set_exploration_type(
            ExplorationType.DETERMINISTIC
        ), torch.no_grad(), timeit("eval"):
            eval_rollout = env.rollout(
                max_rollout_steps,
                auto_cast_to_device=True,
                break_when_any_done=True, # we want to continue sample until we reach the required steps
                #set_truncated=True,
            )

        episode_end = (
            eval_rollout["next", "done"]
            if eval_rollout["next", "done"].any()
            else eval_rollout["next", "truncated"]
        )
        episode_rewards = eval_rollout["next", "episode_reward"][episode_end]
        episode_length = eval_rollout["next", "step_count"][episode_end]
        print("*** Evaluation Stats: ***")
        print(f"Episode rewards: {episode_rewards.mean()}")
        print(f"Episode rewards std: {episode_rewards.std()}")
        print(f"Episode count: {len(episode_rewards)}")
        print(f"Episode length: {episode_length.sum() / len(episode_length)}")
        # could do some preprocessing here
        eval_rollout = eval_rollout.cpu().reshape(-1)
        steps_collected = eval_rollout.batch_size[0]
        total_collected += steps_collected
        pbar.update(steps_collected)
        pbar.set_postfix({
            'collected': f'{total_collected}/{total_eval_steps}'
        })
        replay_buffer.extend(eval_rollout)
        replay_buffer.dumps(f"./replay_buffer_{policy_type}.pt")
        if total_collected >= total_eval_steps:
            break

    pbar.close()
    

    

if __name__ == "__main__":
    main()