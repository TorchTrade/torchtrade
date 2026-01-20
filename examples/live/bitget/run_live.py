"""
Live data collection example using AlpacaTorchTradingEnv.

This script demonstrates how to collect live trading data from Alpaca
using a trained policy or random actions. Data is stored in a replay buffer
for offline training.

Usage:
    python examples/live/alpaca/collect_live.py

Requirements:
    - .env file with API_KEY and SECRET_KEY
    - Alpaca paper trading account
"""
from __future__ import annotations

import functools
import os

import numpy as np
import pandas as pd
import ta
import torch
import tqdm
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv
from torchrl._utils import timeit
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    TensorDictReplayBuffer,
)
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    TransformedEnv,
)
from torchrl.envs.transforms import (
    InitTracker,
    RewardSum,
    SqueezeTransform,
    StepCounter,
    UnsqueezeTransform,
)

from torchtrade.envs.bitget import BitgetFuturesTorchTradingEnv, BitgetFuturesTradingEnvConfig

torch.set_float32_matmul_precision("high")

# Load environment variables
load_dotenv(dotenv_path=".env")


def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """

    """
    df = df.copy().reset_index(drop=True)

    # --- Basic features ---
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]

    # Drop rows with NaN from indicators
    df.dropna(inplace=True)

    return df


def make_collector(train_env, frames_per_batch=1, total_frames=10000, policy=None, device="cpu"):
    """Make data collector."""
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


def apply_env_transforms(env, max_episode_steps=1000):
    """Apply standard transforms to the environment."""
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_episode_steps),
            DoubleToFloat(),
            RewardSum(),
            # UnsqueezeTransform(
            #     dim=0,
            #     allow_positive_dim=True,
            #     in_keys=[
            #         "market_data_1Minute_12",
            #         "market_data_5Minute_8",
            #         "market_data_15Minute_8",
            #         "market_data_1Hour_24",
            #         "account_state",
            #     ],
            # ),
        ),
    )
    return transformed_env


def main():
    device = torch.device("cpu")

    total_farming_steps = 10000
    save_buffer_every = 1
    max_rollout_steps = 10  # 72 steps @ 5min -> 6h per episode -> 4 episodes per day
    policy_type = "random"

    torch.manual_seed(42)
    np.random.seed(42)

    # Create environment configuration
    config = BitgetFuturesTradingEnvConfig(
        symbol="BTC/USD",
        demo=True,
        time_frames=["5min", "15min"],
        window_sizes=[6, 32], 
        execute_on="5min",
        include_base_features=True,
        quantity_per_trade=0.002,  # Adjust based on Bitget minimums
        leverage=5,
    )

    # Create environment
    env = BitgetFuturesTorchTradingEnv(
        config,
        api_key=os.getenv("BITGETACCESSAPIKEY"),
        api_secret=os.getenv("BITGETSECRETKEY"),
        api_passphrase=os.getenv("BITGETPASSPHRASE"),
        feature_preprocessing_fn=custom_preprocessing,
    )

    env = apply_env_transforms(env, max_rollout_steps)

    # Create replay buffer
    scratch_dir = None
    storage_cls = (
        functools.partial(LazyTensorStorage, device=device)
        if not scratch_dir
        else functools.partial(LazyMemmapStorage, device="cpu", scratch_dir=scratch_dir)
    )
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=3,
        storage=storage_cls(10_000),
        batch_size=1,
        shared=False,
    )

    # Use random policy for data collection
    policy = None  # Random actions

    collector = make_collector(
        env,
        policy=policy,
        frames_per_batch=1,
        total_frames=total_farming_steps,
    )

    # squeezer = SqueezeTransform(
    #     dim=-3,
    #     in_keys=[
    #         "market_data_1Minute_12",
    #         "market_data_5Minute_8",
    #         "market_data_15Minute_8",
    #         "market_data_1Hour_24",
    #         "account_state",
    #         ("next", "market_data_1Minute_12"),
    #         ("next", "market_data_5Minute_8"),
    #         ("next", "market_data_15Minute_8"),
    #         ("next", "market_data_1Hour_24"),
    #         ("next", "account_state"),
    #     ],
    # )

    # Main collection loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=total_farming_steps)
    collector_iter = iter(collector)
    total_iter = len(collector)

    for i in range(total_iter):
        timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)

        with timeit("collect"):
            tensordict = next(collector_iter)

        # Remove intermediate encoding keys if present
        for key in ["encoding5min", "encoding15min", "encoding1h", "encoding1min", "encoding_account_state"]:
            if key in tensordict.keys():
                tensordict.pop(key)

        squeezer(tensordict)
        current_frames = tensordict.numel()
        pbar.update(current_frames)

        with timeit("rb - extend"):
            tensordict = tensordict.reshape(-1)
            replay_buffer.extend(tensordict)

        collected_frames += current_frames

        # Log episode metrics
        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean()
            metrics_to_log["train/episode_length"] = episode_length.sum() / len(episode_length)
        print(metrics_to_log)

        if collected_frames % save_buffer_every == 0:
            replay_buffer.dumps(f"./replay_buffer_{policy_type}.pt")

    pbar.close()
    print(f"Collection complete. Total frames: {collected_frames}")


if __name__ == "__main__":
    main()