"""
Live trading with LocalLLMActor on Alpaca (paper trading).

Runs a local LLM as the trading policy on Alpaca's paper trading API,
collecting trajectories into a replay buffer.

Usage:
    python examples/llm/local/online.py

Requirements:
    pip install -e ".[llm_local]"
    # Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env
"""

import os

import numpy as np
import pandas as pd
import torch
import tqdm
from dotenv import load_dotenv

from torchtrade.actor import LocalLLMActor
from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import Compose, DoubleToFloat, TransformedEnv
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter, UnsqueezeTransform

load_dotenv(dotenv_path=".env")
torch.set_float32_matmul_precision("high")


def simple_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["features_close"] = df["close"]
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_volume"] = df["volume"]
    df.dropna(inplace=True)
    return df


def main():
    print("=" * 80)
    print("LocalLLMActor - Live Trading (Alpaca Paper)")
    print("=" * 80)

    device = torch.device("cpu")
    total_frames = 1000
    max_episode_steps = 360
    torch.manual_seed(42)
    np.random.seed(42)

    # Create Alpaca environment
    print("\nCreating Alpaca paper trading environment...")
    config = AlpacaTradingEnvConfig(
        symbol="BTC/USD",
        paper=True,
        time_frames=["15Min"],
        window_sizes=[16],
        execute_on="15Min",
        include_base_features=True,
    )

    env = AlpacaTorchTradingEnv(
        config,
        api_key=os.getenv("ALPACA_API_KEY"),
        api_secret=os.getenv("ALPACA_SECRET_KEY"),
        feature_preprocessing_fn=simple_preprocessing,
    )

    # Apply transforms
    unsqueeze_keys = env.market_data_keys + ["account_state"]
    env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_episode_steps),
            DoubleToFloat(),
            RewardSum(),
            UnsqueezeTransform(dim=0, allow_positive_dim=True, in_keys=unsqueeze_keys),
        ),
    )

    # Create LocalLLMActor
    print("Initializing LocalLLMActor (Qwen/Qwen2.5-0.5B-Instruct)...")
    policy = LocalLLMActor(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        backend="vllm",
        device="cuda" if torch.cuda.is_available() else "cpu",
        debug=True,
        action_space_type="standard",
    )

    # Create collector and replay buffer
    collector = SyncDataCollector(
        env,
        policy,
        init_random_frames=0,
        frames_per_batch=1,
        total_frames=total_frames,
        device=device,
        trust_policy=True,
    )
    collector.set_seed(42)

    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=3,
        storage=LazyTensorStorage(10_000, device=device),
        batch_size=1,
        shared=False,
    )

    # Collection loop
    print(f"\nCollecting {total_frames} frames...")
    print("=" * 80)

    collected_frames = 0
    pbar = tqdm.tqdm(total=total_frames, desc="Collecting live data")

    for tensordict in collector:
        current_frames = tensordict.numel()
        pbar.update(current_frames)

        tensordict = tensordict.reshape(-1)
        replay_buffer.extend(tensordict)
        collected_frames += current_frames

        # Log completed episodes
        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            print(f"\nEpisode completed - reward: {episode_rewards.mean().item():.4f}, "
                  f"length: {(episode_length.sum() / len(episode_length)).item():.0f}")

    pbar.close()
    collector.shutdown()
    if not env.is_closed:
        env.close()

    print("\n" + "=" * 80)
    print(f"Collection complete. {collected_frames} frames in replay buffer.")
    print("=" * 80)


if __name__ == "__main__":
    main()
