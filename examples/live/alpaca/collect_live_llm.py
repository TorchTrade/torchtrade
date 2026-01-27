"""
Live trading data collection with LLM actor on Alpaca.

This script collects live trading data using an LLM-based policy (GPT models)
on the Alpaca exchange. The collected trajectories are stored in a replay buffer
for offline analysis or imitation learning.
"""
from __future__ import annotations

import os
import functools

import numpy as np
import pandas as pd
import torch
import tqdm

from dotenv import load_dotenv

from torchrl._utils import timeit
from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    TensorDictReplayBuffer,
)
from torchrl.envs.transforms import (
    InitTracker,
    RewardSum,
    StepCounter,
    UnsqueezeTransform,
    SqueezeTransform,
)
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    TransformedEnv,
)
from torchtrade.actor.llm_actor import LLMActor
from torchrl.collectors import SyncDataCollector

# Load environment variables
load_dotenv(dotenv_path=".env")

torch.set_float32_matmul_precision("high")

def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess OHLCV dataframe with basic features for LLM trading.

    For LLM-based trading, we keep features simple and interpretable.
    The LLM receives raw OHLCV data as input for reasoning.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: ["open", "high", "low", "close", "volume"]

    Returns
    -------
    pd.DataFrame
        Processed dataframe with feature columns prefixed with "feature_"
    """
    df = df.copy().reset_index(drop=True)

    # Add basic OHLCV features
    df["feature_open"] = df["open"]
    df["feature_high"] = df["high"]
    df["feature_low"] = df["low"]
    df["feature_close"] = df["close"]
    df["feature_volume"] = df["volume"]

    # Drop rows with NaN
    df.dropna(inplace=True)

    return df
        


def make_collector(
    train_env,
    frames_per_batch=1,
    total_frames=10000,
    policy=None,
    compile_mode=False,
    device="cpu"
):
    """
    Create a synchronous data collector for live trading.

    Parameters
    ----------
    train_env : TransformedEnv
        The trading environment
    frames_per_batch : int
        Number of frames to collect per batch (default: 1)
    total_frames : int
        Total frames to collect before stopping (default: 10000)
    policy : nn.Module
        The trading policy (LLMActor in this case)
    compile_mode : bool
        Whether to use torch.compile (default: False)
    device : str or torch.device
        Device for data collection (default: "cpu")

    Returns
    -------
    SyncDataCollector
        Configured data collector
    """
    if device in ("", None):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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
    save_buffer_every = 4
    max_rollout_steps = 360

    torch.manual_seed(42)
    np.random.seed(42)

    # Create environment configuration
    config = AlpacaTradingEnvConfig(
        symbol="BTC/USD",
        paper=True,
        time_frames=["15Min"],
        window_sizes=[16],
        execute_on="15Min",
        include_base_features=True,
    )

    # Create environment
    env = AlpacaTorchTradingEnv(
        config, api_key=os.getenv("API_KEY"),
        api_secret=os.getenv("SECRET_KEY"),
        feature_preprocessing_fn=custom_preprocessing,
    )

    def apply_env_transforms(env, max_episode_steps=1000):
        """Apply TorchRL transforms to the environment."""
        unsqueeze_keys = env.market_data_keys + ["account_state"]

        transformed_env = TransformedEnv(
            env,
            Compose(
                InitTracker(),
                StepCounter(max_episode_steps),
                DoubleToFloat(),
                RewardSum(),
                UnsqueezeTransform(
                    dim=0,
                    allow_positive_dim=True,
                    in_keys=unsqueeze_keys
                ),
            ),
        )
        return transformed_env

    # Apply environment transforms
    env = apply_env_transforms(env, max_rollout_steps)

    # Unwrap TransformedEnv to access base environment
    market_data_keys = env.base_env.market_data_keys
    account_state = env.base_env.account_state_key
    transform_keys = market_data_keys + [account_state]

    # Configure replay buffer storage
    scratch_dir = None
    storage_cls = functools.partial(
        LazyTensorStorage if not scratch_dir else LazyMemmapStorage,
        device=device if not scratch_dir else "cpu",
        **({"scratch_dir": scratch_dir} if scratch_dir else {})
    )

    # Create replay buffer for storing trajectories
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=3,
        storage=storage_cls(10_000),
        batch_size=1,
        shared=False,
    )

    # Initialize LLM-based policy
    policy = LLMActor(
        market_data_keys=market_data_keys,
        account_state=env.base_env.ACCOUNT_STATE,
        model="gpt-5-mini",
        debug=True,
        symbol=config.symbol,
        feature_keys=["open", "high", "low", "close", "volume"],
        execute_on="15Minute",
    )
    policy_type = "gpt5mini"

    # Create data collector
    collector = make_collector(
        env,
        policy=policy,
        frames_per_batch=1,
        total_frames=total_farming_steps
    )

    # Build squeeze transform for removing batch dimension after collection
    squeeze_keys = transform_keys + [("next", key) for key in transform_keys]
    squeezer = SqueezeTransform(dim=-3, in_keys=squeeze_keys)

    # Main data collection loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=total_farming_steps, desc="Collecting live data")
    collector_iter = iter(collector)
    total_iter = len(collector)

    for i in range(total_iter):
        timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)

        # Collect a batch of transitions from the environment
        with timeit("collect"):
            tensordict = next(collector_iter)

        # Squeeze dimensions and update progress
        squeezer(tensordict)
        current_frames = tensordict.numel()
        pbar.update(current_frames)

        # Store transitions in replay buffer
        with timeit("rb - extend"):
            tensordict = tensordict.reshape(-1)
            replay_buffer.extend(tensordict)

        collected_frames += current_frames

        # Extract episode statistics for completed episodes
        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

        # Log episode metrics
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log = {
                "train/reward": episode_rewards.mean().item(),
                "train/episode_length": (episode_length.sum() / len(episode_length)).item(),
            }
            print(f"Episode completed: {metrics_to_log}")

        # Periodically save replay buffer
        if collected_frames % save_buffer_every == 0:
            replay_buffer.dumps(f"./replay_buffer_{policy_type}.pt")
            print(f"Saved replay buffer at {collected_frames} frames")


if __name__ == "__main__":
    main()
