"""
Simple demonstration of LocalLLMActor for offline backtesting.

This example shows:
1. Loading a small dataset from CSV
2. Creating a SequentialTradingEnv
3. Running LocalLLMActor with a configurable local LLM
4. Printing actions and reasoning traces

Usage:
    python examples/llm/local/inference_example.py

Requirements:
    pip install -e ".[llm_local]"
"""

import pandas as pd
import torch

from torchtrade.actor import LocalLLMActor
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


def simple_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple preprocessing function that adds basic OHLCV features.

    Expected columns: ["open", "high", "low", "close", "volume"]
    """
    df = df.copy().reset_index(drop=True)

    # Add features with required prefix
    df["features_close"] = df["close"]
    df["features_open"] = df["open"]
    df["features_high"] = df["high"]
    df["features_low"] = df["low"]
    df["features_volume"] = df["volume"]

    # Drop NaN if any
    df.dropna(inplace=True)

    return df


def main():
    print("=" * 80)
    print("LocalLLMActor Inference Example")
    print("=" * 80)

    # Load data (small subset for quick demo)
    print("\n[1/5] Loading data...")
    # Option 1: Use local CSV file (download from your data source)
    # df = pd.read_csv("btcusdt_spot_1m_01_2020_to_12_2025.csv")

    # Option 2: Use HuggingFace dataset (recommended)
    import datasets
    dataset = datasets.load_dataset("Torch-Trade/BTCUSD_sport_1m_12_2024_to_09_2025")
    df = dataset["train"].to_pandas()

    # Take a small sample (1000 rows ~ 16 hours of 1-minute data)
    df = df.head(1000)
    print(f"  Loaded {len(df)} rows")

    # Create environment configuration
    print("\n[2/5] Creating environment...")
    config = SequentialTradingEnvConfig(
        symbol="BTC/USD",
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
            TimeFrame(5, TimeFrameUnit.Minute),
        ],
        window_sizes=[12, 8],  # 12 minutes, 40 minutes
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        initial_cash=10000,
        transaction_fee=0.001,  # 0.1% fee
        slippage=0.001,  # 0.1% slippage
        include_base_features=False,
        random_start=False,  # Start from beginning for reproducibility
    )

    # Create environment
    env = SequentialTradingEnv(df, config, feature_preprocessing_fn=simple_preprocessing)
    print(f"  Environment created with {env.max_steps} max steps")

    # Create LocalLLMActor
    print("\n[3/5] Initializing LocalLLMActor...")
    print("  Model: Qwen/Qwen2.5-0.5B-Instruct")
    print("  Backend: vllm (with transformers fallback)")
    print("  This may take a moment to download the model...")

    actor = LocalLLMActor(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        backend="vllm",
        device="cuda" if torch.cuda.is_available() else "cpu",
        debug=True,  # Enable debug output to see prompts and responses
        action_space_type="standard",  # 3-action: buy/sell/hold
    )

    # Run a short rollout
    print("\n[4/5] Running rollout (10 steps)...")
    print("=" * 80)

    td = env.reset()
    episode_reward = 0.0

    for step in range(10):
        print(f"\n--- Step {step + 1} ---")

        # Get action from LLM
        td = actor(td)

        # Map action to string
        action_idx = td["action"].item()
        action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        action_str = action_map.get(action_idx, "UNKNOWN")

        print(f"\nAction chosen: {action_str} (index={action_idx})")

        # Print thinking if available
        if "thinking" in td.keys():
            print(f"Reasoning: {td['thinking'][:200]}...")  # Print first 200 chars

        # Take step in environment
        td = env.step(td)

        # Get reward and done status
        reward = td["next", "reward"].item()
        done = td["next", "done"].item()
        episode_reward += reward

        print(f"Reward: {reward:.4f} | Cumulative: {episode_reward:.4f}")

        # Move to next state
        td = td["next"]

        if done:
            print("\nEpisode ended (done=True)")
            break

        print("=" * 80)

    # Print summary
    print("\n[5/5] Summary")
    print("=" * 80)
    print(f"Steps completed: {step + 1}")
    print(f"Total reward: {episode_reward:.4f}")

    # Get environment metrics if available
    if hasattr(env, "cash"):
        print(f"Final cash: ${env.cash:.2f}")
    if hasattr(env, "position_value"):
        print(f"Position value: ${env.position_value:.2f}")

    print("\nExample completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
