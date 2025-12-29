"""
Live data collection example using AlpacaSLTPTorchTradingEnv.

This script demonstrates how to collect live trading data from Alpaca
using the Stop Loss / Take Profit environment with bracket orders.
Data is stored in a replay buffer for offline training.

The SL/TP environment uses a combinatorial action space:
- Action 0: HOLD (do nothing)
- Actions 1..N: BUY with specific (stop_loss_pct, take_profit_pct) combinations

Example with stoploss_levels=(-0.02, -0.05) and takeprofit_levels=(0.05, 0.10):
- Action 0: HOLD
- Action 1: BUY with SL=-2%, TP=+5%
- Action 2: BUY with SL=-2%, TP=+10%
- Action 3: BUY with SL=-5%, TP=+5%
- Action 4: BUY with SL=-5%, TP=+10%

Usage:
    python examples/live/alpaca/collect_live_sltp.py

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

from torchtrade.envs.alpaca import AlpacaSLTPTorchTradingEnv, AlpacaSLTPTradingEnvConfig

torch.set_float32_matmul_precision("high")

# Load environment variables
load_dotenv(dotenv_path=".env")


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
            UnsqueezeTransform(
                dim=0,
                allow_positive_dim=True,
                in_keys=[
                    "market_data_1Minute_12",
                    "market_data_5Minute_8",
                    "market_data_15Minute_8",
                    "market_data_1Hour_24",
                    "account_state",
                ],
            ),
        ),
    )
    return transformed_env


def main():
    device = torch.device("cpu")

    total_farming_steps = 10000
    save_buffer_every = 10
    max_rollout_steps = 72  # 72 steps @ 5min -> 6h per episode -> 4 episodes per day
    policy_type = "random_sltp"

    torch.manual_seed(42)
    np.random.seed(42)

    # Define SL/TP levels
    # Stop loss: -2%, -5%, -10%
    # Take profit: +5%, +10%, +20%
    stoploss_levels = (-0.02, -0.05, -0.10)
    takeprofit_levels = (0.05, 0.10, 0.20)

    # Create environment configuration
    config = AlpacaSLTPTradingEnvConfig(
        symbol="BTC/USD",
        paper=True,
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
            TimeFrame(5, TimeFrameUnit.Minute),
            TimeFrame(15, TimeFrameUnit.Minute),
            TimeFrame(1, TimeFrameUnit.Hour),
        ],
        window_sizes=[12, 8, 8, 24],  # ~12m, 40m, 2h, 1d
        execute_on=TimeFrame(5, TimeFrameUnit.Minute),
        stoploss_levels=stoploss_levels,
        takeprofit_levels=takeprofit_levels,
        include_base_features=True,
    )

    # Create environment
    env = AlpacaSLTPTorchTradingEnv(
        config,
        api_key=os.getenv("API_KEY"),
        api_secret=os.getenv("SECRET_KEY"),
        feature_preprocessing_fn=custom_preprocessing,
    )

    # Print action space info
    print(f"Action space size: {env.action_spec.n}")
    print(f"Action mapping:")
    for action_idx, (sl, tp) in env.action_map.items():
        if sl is None:
            print(f"  Action {action_idx}: HOLD")
        else:
            print(f"  Action {action_idx}: BUY with SL={sl*100:.1f}%, TP={tp*100:.1f}%")

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

    squeezer = SqueezeTransform(
        dim=-3,
        in_keys=[
            "market_data_1Minute_12",
            "market_data_5Minute_8",
            "market_data_15Minute_8",
            "market_data_1Hour_24",
            "account_state",
            ("next", "market_data_1Minute_12"),
            ("next", "market_data_5Minute_8"),
            ("next", "market_data_15Minute_8"),
            ("next", "market_data_1Hour_24"),
            ("next", "account_state"),
        ],
    )

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

            # Log action distribution
            actions = tensordict["action"]
            action_counts = torch.bincount(actions.flatten(), minlength=env.action_spec.n)
            metrics_to_log["actions/hold_pct"] = (action_counts[0] / actions.numel()).item()
            metrics_to_log["actions/buy_pct"] = (action_counts[1:].sum() / actions.numel()).item()

        print(metrics_to_log)

        if collected_frames % save_buffer_every == 0:
            replay_buffer.dumps(f"./replay_buffer_{policy_type}.pt")

    pbar.close()
    print(f"Collection complete. Total frames: {collected_frames}")


if __name__ == "__main__":
    main()
