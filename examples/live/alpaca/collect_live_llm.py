"""
"""
from __future__ import annotations

import numpy as np
import torch
import tqdm
from torchrl._utils import timeit
torch.set_float32_matmul_precision("high")
from torchtrade.envs.alpaca.torch_env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
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
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter, UnsqueezeTransform, SqueezeTransform
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    TransformedEnv,
)
from torchtrade.actor.llm_actor import LLMActor
from torchrl.collectors import SyncDataCollector
# Load environment variables
load_dotenv(dotenv_path=".env")

import pandas as pd
import numpy as np

def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess OHLCV dataframe with engineered features for RL trading.

    Expected columns: ["open", "high", "low", "close", "volume"]
    Index can be datetime or integer.
    """

    df = df.copy().reset_index(drop=True)

    # --- Basic features ---
    # Log returns
    df["feature_close"] = df["close"]
    df["feature_open"] = df["open"]
    df["feature_high"] = df["high"]
    df["feature_low"] = df["low"]
    df["feature_volume"] = df["volume"]

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
        trust_policy=True,
    )
    collector.set_seed(42)
    return collector

def main():
    device = torch.device("cpu")

    total_farming_steps = 10000
    save_buffer_every = 10
    max_rollout_steps = 72 #  72 steps a 5min -> 6h per episode -> 4 episodes per day


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
        include_base_features=True,
    )

    # Create environment
    env = AlpacaTorchTradingEnv(
        config, api_key=os.getenv("API_KEY"),
        api_secret=os.getenv("SECRET_KEY"),
        feature_preprocessing_fn=custom_preprocessing,
    )

    def apply_env_transforms(env, max_episode_steps=1000):
        transformed_env = TransformedEnv(
            env,
            Compose(
                InitTracker(),
                StepCounter(max_episode_steps),
                DoubleToFloat(),
                RewardSum(),
                UnsqueezeTransform(dim=0, allow_positive_dim=True, in_keys=["market_data_1Minute_12",
                                                                            "market_data_5Minute_8",
                                                                            "market_data_15Minute_8",
                                                                            "market_data_1Hour_24",
                                                                            "account_state"]),
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
                10_000,
            ),
            batch_size=1,
            shared=False,
        )
    policy = LLMActor(model="gpt-5-mini", debug=True)
    policy_type="gpt5mini"
    collector = make_collector(env, policy=policy, frames_per_batch=1, total_frames=total_farming_steps)

    squeezer = SqueezeTransform(dim=-3, in_keys=["market_data_1Minute_12",
                                                                "market_data_5Minute_8",
                                                                "market_data_15Minute_8",
                                                                "market_data_1Hour_24",
                                                                "account_state",
                                                                ("next", "market_data_1Minute_12"),
                                                                ("next", "market_data_5Minute_8"),
                                                                ("next", "market_data_15Minute_8"),
                                                                ("next", "market_data_1Hour_24"),
                                                                ("next", "account_state")])

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

        squeezer(tensordict)
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
