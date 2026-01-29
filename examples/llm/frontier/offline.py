"""
Offline backtesting with LLMActor (OpenAI API) on SequentialTradingEnv.

Runs an OpenAI model through a sequential trading environment using
historical data, printing actions, reasoning traces, and episode metrics.

Usage:
    python examples/llm/api/offline.py

Requirements:
    pip install openai python-dotenv
    # Set OPENAI_API_KEY in .env
"""

import datasets
import pandas as pd
import torch

from torchtrade.actor import FrontierLLMActor
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig


def simple_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=False)
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.dropna(inplace=True)
    return df


def main():
    print("=" * 80)
    print("LLMActor (OpenAI API) - Offline Backtesting")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    dataset = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025")
    df = dataset["train"].to_pandas()

    # Convert timestamp column to datetime for proper filtering
    df['0'] = pd.to_datetime(df['0'])

    # Create environment
    print("Creating environment...")
    config = SequentialTradingEnvConfig(
        symbol="BTC/USD",
        time_frames=["1Hour"],
        window_sizes=[48],
        execute_on="1Hour",
        initial_cash=10000,
        transaction_fee=0.0,
        slippage=0.0,
        include_base_features=False,
        random_start=False,
    )
    env = SequentialTradingEnv(df, config, feature_preprocessing_fn=simple_preprocessing)
    print(f"  Max steps: {env.max_steps}")

    # Create actor
    print("Initializing LLMActor (OpenAI API)...")
    actor = FrontierLLMActor(
        market_data_keys=env.market_data_keys,
        account_state_labels=env.account_state,
        action_levels=env.action_levels,
        model="gpt-4o-mini",
        symbol=config.symbol,
        execute_on=config.execute_on,
        feature_keys=["open", "high", "low", "close", "volume"],
        debug=True,
    )

    # Run rollout
    max_steps = 10
    print(f"\nRunning rollout ({max_steps} steps)...")
    print("=" * 80)

    td = env.reset()
    episode_reward = 0.0
    # NOTE: We use manual rollouts for demonstration here
    # but you can use a collector and a replay buffer to store transitions
    for step in range(max_steps):
        td = actor(td)
        action_idx = td["action"].item()
        print(f"\nStep {step + 1}: action={action_idx}")

        if "thinking" in td.keys():
            thinking = td["thinking"]
            if isinstance(thinking, str):
                print(f"  Reasoning: {thinking[:200]}...")

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
