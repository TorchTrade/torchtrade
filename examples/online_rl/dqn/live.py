"""DQN live trading on Binance Futures.

Self-contained script that creates a Binance futures live environment,
builds the DQN agent (reusing model creation from utils.py), loads
pre-trained weights, and runs the policy live.

Usage:
    # Run with pre-trained weights
    python live.py --weights dqn_policy_100.pth

    # Run with random policy (no weights)
    python live.py

Requirements:
    - .env file with BINANCE_API_KEY and BINANCE_SECRET_KEY
    - Binance testnet account (demo=True by default)
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
    FlattenObservation,
    TransformedEnv,
)
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter

from torchtrade.envs.live.binance.env import (
    BinanceFuturesTorchTradingEnv,
    BinanceFuturesTradingEnvConfig,
)
from torchtrade.envs.live.binance.order_executor import MarginType

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
    symbol: str = "BTCUSDT",
    time_frames: list[str] | None = None,
    window_sizes: list[int] | None = None,
    execute_on: str = "1Hour",
    leverage: int = 2,
    action_levels: list[float] | None = None,
    demo: bool = True,
    max_episode_steps: int = 1000,
):
    """Create a transformed Binance futures live environment."""
    if time_frames is None:
        time_frames = ["1Hour"]
    if window_sizes is None:
        window_sizes = [24]
    if action_levels is None:
        action_levels = [-1.0, 0.0, 1.0]

    config = BinanceFuturesTradingEnvConfig(
        symbol=symbol,
        time_frames=time_frames,
        window_sizes=window_sizes,
        execute_on=execute_on,
        leverage=leverage,
        action_levels=action_levels,
        demo=demo,
        include_base_features=False,
    )

    env = BinanceFuturesTorchTradingEnv(
        config,
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_SECRET", ""),
        feature_preprocessing_fn=custom_preprocessing,
    )

    # Flatten market obs for TDQN (single timeframe flat vector)
    obs_keys = list(env.observation_spec.keys())
    market_keys = [k for k in obs_keys if k.startswith("market_")]
    flatten_transforms = [
        FlattenObservation(in_keys=[k], first_dim=-2, last_dim=-1)
        for k in market_keys
    ]

    env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
            StepCounter(max_steps=max_episode_steps),
            *flatten_transforms,
        ),
    )
    return env


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DQN live trading on Binance")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pth weights file")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--time_frames", nargs="+", default=["1Min"])
    parser.add_argument("--window_sizes", nargs="+", type=int, default=[24])
    parser.add_argument("--execute_on", type=str, default="1Min")
    parser.add_argument("--leverage", type=int, default=2)
    parser.add_argument("--action_levels", nargs="+", type=float, default=[-1.0, 0.0, 1.0])
    parser.add_argument("--demo", action="store_true", default=True)
    parser.add_argument("--no_demo", action="store_false", dest="demo")
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--save_buffer", type=str, default="live_replay_buffer_dqn")
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
        leverage=args.leverage,
        action_levels=args.action_levels,
        demo=args.demo,
    )

    # Build actor (reuse TDQN agent builder from utils)
    from utils import make_tdqn_agent

    # Create a minimal config namespace that make_tdqn_agent expects
    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.model = _Cfg()
    cfg.model.hidden_size = args.hidden_size
    cfg.model.dropout = args.dropout
    cfg.collector = _Cfg()
    cfg.collector.annealing_frames = 1  # no exploration in live
    cfg.collector.eps_start = 0.0
    cfg.collector.eps_end = 0.0

    actor, exploration_policy = make_tdqn_agent(cfg, env, device=device)

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
