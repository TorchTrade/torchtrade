"""PPO live trading on Alpaca.

Self-contained script that creates an Alpaca live environment,
builds the PPO actor (reusing model creation from utils.py), loads
pre-trained weights, and runs the policy live.

Usage:
    # Run with pre-trained weights
    python live.py --weights ppo_policy_100.pth

    # Run with random policy (no weights)
    python live.py

Requirements:
    - .env file with ALPACA_API_KEY and ALPACA_SECRET_KEY
    - Alpaca paper trading account (paper=True by default)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import tqdm
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from torchrl._utils import timeit
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    TransformedEnv,
)
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter

from torchtrade.envs.live.alpaca.env import (
    AlpacaTorchTradingEnv,
    AlpacaTradingEnvConfig,
)

load_dotenv(dotenv_path=".env")


# ------------------------------------------------------------------
# Preprocessing (must match training preprocessing exactly)
# ------------------------------------------------------------------

def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise OHLCV features with StandardScaler (matches training)."""
    df = df.copy().reset_index(drop=False)

    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]

    scaler = StandardScaler()
    feature_cols = [c for c in df.columns if c.startswith("features_")]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    df.fillna(0, inplace=True)
    return df


# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

def make_live_env(
    symbol: str = "BTC/USD",
    time_frames: list[str] | None = None,
    window_sizes: list[int] | None = None,
    execute_on: str = "1Hour",
    action_levels: list[float] | None = None,
    paper: bool = True,
    max_episode_steps: int = 1000,
):
    """Create a transformed Alpaca live environment."""
    if time_frames is None:
        time_frames = ["1Hour"]
    if window_sizes is None:
        window_sizes = [24]
    if action_levels is None:
        action_levels = [0.0, 1.0]

    config = AlpacaTradingEnvConfig(
        symbol=symbol,
        time_frames=time_frames,
        window_sizes=window_sizes,
        execute_on=execute_on,
        action_levels=action_levels,
        paper=paper,
        include_base_features=False,
    )

    env = AlpacaTorchTradingEnv(
        config,
        api_key=os.getenv("ALPACA_API_KEY", ""),
        api_secret=os.getenv("ALPACA_SECRET_KEY", ""),
        feature_preprocessing_fn=custom_preprocessing,
    )

    env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
            StepCounter(max_steps=max_episode_steps),
        ),
    )
    return env


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPO live trading on Alpaca")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pth weights file")
    parser.add_argument("--symbol", type=str, default="BTC/USD")
    parser.add_argument("--time_frames", nargs="+", default=["1Hour"])
    parser.add_argument("--window_sizes", nargs="+", type=int, default=[24])
    parser.add_argument("--execute_on", type=str, default="1Hour")
    parser.add_argument("--action_levels", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--paper", action="store_true", default=True)
    parser.add_argument("--no_paper", action="store_false", dest="paper")
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--network_type", type=str, default="cnn", choices=["cnn", "batchnorm_mlp"])
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--save_buffer", type=str, default="live_replay_buffer_ppo")
    parser.add_argument("--save_every", type=int, default=10, help="Save replay buffer every N frames")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(args.device)

    # Build live environment
    env = make_live_env(
        symbol=args.symbol,
        time_frames=args.time_frames,
        window_sizes=args.window_sizes,
        execute_on=args.execute_on,
        action_levels=args.action_levels,
        paper=args.paper,
    )

    # Build actor (reuse PPO model builder from utils)
    from utils import make_ppo_models

    # Create a minimal config namespace that make_ppo_models expects
    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.model = _Cfg()
    cfg.model.network_type = args.network_type
    cfg.model.hidden_size = args.hidden_size
    cfg.model.dropout = args.dropout
    cfg.model.activation = "tanh"
    cfg.model.num_layers = 4
    cfg.env = _Cfg()
    cfg.env.time_frames = args.time_frames
    cfg.env.window_sizes = args.window_sizes

    actor, critic = make_ppo_models(env, device=device, cfg=cfg)

    # Load weights if provided
    if args.weights:
        state_dict = torch.load(args.weights, map_location=device, weights_only=True)
        actor.load_state_dict(state_dict)
        print(f"Loaded weights from {args.weights}")

    actor.eval()
    policy = actor

    # Replay buffer to store live transitions
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(args.buffer_size, device=device),
        batch_size=1,
    )

    # Run live
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=1,
        total_frames=args.total_steps,
        device=device,
        init_random_frames=0,
    )

    collected_frames = 0
    pbar = tqdm.tqdm(total=args.total_steps)

    for tensordict in collector:
        current_frames = tensordict.numel()
        collected_frames += current_frames
        pbar.update(current_frames)

        replay_buffer.extend(tensordict.reshape(-1))

        if collected_frames % args.save_every == 0:
            replay_buffer.dumps(args.save_buffer)

        # Log episode metrics
        episode_end = tensordict["next", "done"] | tensordict["next", "truncated"]
        episode_rewards = tensordict["next", "episode_reward"][episode_end]
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            print(
                f"Episode reward: {episode_rewards.mean().item():.4f}, "
                f"length: {episode_length.float().mean().item():.0f}"
            )

    pbar.close()
    collector.shutdown()

    replay_buffer.dumps(args.save_buffer)
    print(f"Live run complete. Total frames: {collected_frames}")
    print(f"Replay buffer saved to {args.save_buffer}")


if __name__ == "__main__":
    main()
