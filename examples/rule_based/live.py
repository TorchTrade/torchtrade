"""
Live trading with MeanReversionActor on Alpaca (paper trading).

Runs a rule-based mean reversion strategy on Alpaca's paper trading API,
collecting trajectories into a replay buffer.

Usage:
    python examples/rule_based/live.py

Requirements:
    Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env
"""

import os

import numpy as np
import pandas as pd
import torch
import tqdm
from dotenv import load_dotenv

from torchtrade.actor import MeanReversionActor
from torchtrade.envs.live.alpaca.env import AlpacaTorchTradingEnv, AlpacaTradingEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import Compose, DoubleToFloat, TransformedEnv
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter, UnsqueezeTransform

load_dotenv(dotenv_path=".env")


def main():
    print("=" * 80)
    print("MeanReversionActor - Live Trading (Alpaca Paper)")
    print("=" * 80)

    device = torch.device("cpu")
    total_frames = 1000
    max_episode_steps = 360
    torch.manual_seed(42)
    np.random.seed(42)

    # Create actor
    execute_tf = TimeFrame(1, TimeFrameUnit.Minute)
    actor = MeanReversionActor(
        bb_window=20,
        bb_std=2.0,
        stoch_rsi_window=14,
        oversold_threshold=20.0,
        overbought_threshold=80.0,
        execute_timeframe=execute_tf,
        debug=True,
    )

    # Create Alpaca environment with actor's preprocessing
    print("\nCreating Alpaca paper trading environment...")
    config = AlpacaTradingEnvConfig(
        symbol="BTC/USD",
        paper=True,
        time_frames=["1Min"],
        window_sizes=[48],
        execute_on="1Min",
        include_base_features=False,
    )

    env = AlpacaTorchTradingEnv(
        config,
        api_key=os.getenv("ALPACA_API_KEY"),
        api_secret=os.getenv("ALPACA_SECRET_KEY"),
        feature_preprocessing_fn=actor.get_preprocessing_fn(),
    )

    # Set actor's market_data_keys and features_order from env
    actor.market_data_keys = env.market_data_keys
    actor.features_order = env.feature_keys
    actor.feature_idx = {feat: i for i, feat in enumerate(actor.features_order)}

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

    # Create collector and replay buffer
    collector = SyncDataCollector(
        env,
        actor,
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
