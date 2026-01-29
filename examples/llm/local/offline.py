"""
Offline backtesting with LocalLLMActor on SequentialTradingEnv.

Runs a local LLM through a sequential trading environment using
historical data, printing actions, reasoning traces, and episode metrics.

Usage:
    python examples/llm/local/offline.py

Requirements:
    pip install -e ".[llm_local]"
"""

import datasets
import pandas as pd
import torch

from torchtrade.actor import LocalLLMActor
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


def simple_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.dropna(inplace=True)
    return df


def main():
    print("=" * 80)
    print("LocalLLMActor - Offline Backtesting")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    dataset = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025")
    df = dataset["train"].to_pandas()
    df = df.head(1000)
    print(f"  Loaded {len(df)} rows")

    # Create environment
    print("Creating environment...")
    config = SequentialTradingEnvConfig(
        symbol="BTC/USD",
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
            TimeFrame(5, TimeFrameUnit.Minute),
        ],
        window_sizes=[12, 8],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        initial_cash=10000,
        transaction_fee=0.001,
        slippage=0.001,
        include_base_features=False,
        random_start=False,
    )
    env = SequentialTradingEnv(df, config, feature_preprocessing_fn=simple_preprocessing)
    print(f"  Max steps: {env.max_steps}")

    # Create actor
    print("Initializing LocalLLMActor (Qwen/Qwen2.5-0.5B-Instruct)...")
    actor = LocalLLMActor(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        backend="vllm",
        device="cuda" if torch.cuda.is_available() else "cpu",
        debug=True,
        action_space_type="standard",
    )

    # Run rollout
    max_steps = 10
    print(f"\nRunning rollout ({max_steps} steps)...")
    print("=" * 80)

    td = env.reset()
    action_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
    episode_reward = 0.0

    for step in range(max_steps):
        td = actor(td)
        action_idx = td["action"].item()
        print(f"\nStep {step + 1}: {action_names.get(action_idx, 'UNKNOWN')} (idx={action_idx})")

        if "thinking" in td.keys():
            print(f"  Reasoning: {td['thinking'][:200]}...")

        td = env.step(td)
        reward = td["next", "reward"].item()
        episode_reward += reward
        print(f"  Reward: {reward:.4f} | Cumulative: {episode_reward:.4f}")

        if td["next", "done"].item():
            print("\nEpisode ended (done=True)")
            break
        td = td["next"]

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print(f"  Steps: {step + 1}")
    print(f"  Total reward: {episode_reward:.4f}")
    if hasattr(env, "cash"):
        print(f"  Final cash: ${env.cash:.2f}")
    if hasattr(env, "position_value"):
        print(f"  Position value: ${env.position_value:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
