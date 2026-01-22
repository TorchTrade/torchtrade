"""
Demonstration of LocalLLMActor on SeqFuturesSLTPEnv.

This example shows:
1. Futures trading with leverage and bracket orders (stop-loss/take-profit)
2. Multi-timeframe observations
3. Handling combinatorial action space
4. Episode metrics and performance analysis

Usage:
    python examples/llm/local/futures_sltp_example.py

Requirements:
    pip install -e ".[llm_local]"
"""

import pandas as pd
import torch

from torchtrade.actor import LocalLLMActor
from torchtrade.envs.offline import SeqFuturesSLTPEnv, SeqFuturesSLTPEnvConfig
from torchtrade.envs.offline.seqfuturessltp import futures_sltp_action_map
from torchtrade.envs.offline.infrastructure.utils import TimeFrame, TimeFrameUnit


def preprocessing_with_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing function that adds OHLCV features.

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
    print("LocalLLMActor + SeqFuturesSLTPEnv Example")
    print("Futures Trading with Leverage and Bracket Orders")
    print("=" * 80)

    # Load data
    print("\n[1/6] Loading data...")
    # Option 1: Use local CSV file (download from your data source)
    # df = pd.read_csv("btcusdt_spot_1m_01_2020_to_12_2025.csv")

    # Option 2: Use HuggingFace dataset (recommended)
    import datasets
    dataset = datasets.load_dataset("Torch-Trade/BTCUSD_sport_1m_12_2024_to_09_2025")
    df = dataset["train"].to_pandas()

    # Use more data for futures trading (5000 rows ~ 3.5 days)
    df = df.head(5000)
    print(f"  Loaded {len(df)} rows")

    # Create environment configuration
    print("\n[2/6] Creating SeqFuturesSLTPEnv...")
    config = SeqFuturesSLTPEnvConfig(
        symbol="BTC/USD",
        # Multi-timeframe observations
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
            TimeFrame(5, TimeFrameUnit.Minute),
            TimeFrame(15, TimeFrameUnit.Minute),
            TimeFrame(1, TimeFrameUnit.Hour),
        ],
        window_sizes=[12, 8, 8, 24],  # ~12m, 40m, 2h, 1d
        execute_on=TimeFrame(5, TimeFrameUnit.Minute),
        # Futures settings
        initial_cash=10000,
        leverage=5,  # 5x leverage
        # Stop-loss and take-profit levels
        stoploss_levels=(-0.02, -0.05),  # -2%, -5%
        takeprofit_levels=(0.05, 0.1),  # +5%, +10%
        # Trading costs
        transaction_fee=0.0004,  # 0.04% (typical futures fee)
        slippage=0.001,  # 0.1%
        # Environment settings
        include_base_features=False,
        random_start=False,  # Deterministic start
    )

    # Build action map
    action_map = futures_sltp_action_map(
        config.stoploss_levels,
        config.takeprofit_levels
    )

    print(f"  Action space size: {len(action_map)}")
    print(f"  Actions breakdown:")
    print(f"    - 0: Hold/Close")
    print(f"    - 1-{len(config.stoploss_levels) * len(config.takeprofit_levels)}: Long positions")
    print(f"    - {len(config.stoploss_levels) * len(config.takeprofit_levels) + 1}-{len(action_map) - 1}: Short positions")

    # Create environment
    env = SeqFuturesSLTPEnv(df, config, feature_preprocessing_fn=preprocessing_with_features)
    print(f"  Max steps: {env.max_steps}")

    # Create LocalLLMActor with futures_sltp support
    print("\n[3/6] Initializing LocalLLMActor...")
    print("  Model: Qwen/Qwen2.5-0.5B-Instruct")
    print("  Action space: futures_sltp (combinatorial with SL/TP)")
    print("  This may take a moment to download the model...")

    actor = LocalLLMActor(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        backend="vllm",
        device="cuda" if torch.cuda.is_available() else "cpu",
        debug=False,  # Set to True to see full prompts/responses
        action_space_type="futures_sltp",
        action_map=action_map,
        temperature=0.7,
    )

    # Run episode
    print("\n[4/6] Running episode...")
    print("=" * 80)

    td = env.reset()
    step = 0
    max_steps = 50  # Limit steps for demo

    actions_taken = []
    rewards = []

    while step < max_steps and step < env.max_steps:
        # Get action from LLM
        td = actor(td)

        # Record action
        action_idx = td["action"].item()
        actions_taken.append(action_idx)

        # Get action details
        side, sl, tp = action_map[action_idx]

        # Print step info (every 10 steps)
        if step % 10 == 0 or step < 5:
            print(f"\nStep {step + 1}/{max_steps}")
            if side is None:
                print(f"  Action: Hold/Close (idx={action_idx})")
            else:
                print(f"  Action: {side.upper()} SL={sl*100:.1f}% TP={tp*100:.1f}% (idx={action_idx})")

            # Print account state
            account = td["account_state"]
            if account.dim() > 1:
                account = account.squeeze()
            print(f"  Cash: ${account[0].item():.2f}")
            print(f"  Position size: {account[1].item():.4f}")
            print(f"  Leverage: {account[6].item():.1f}x")
            if account[1].item() != 0:
                print(f"  Liquidation price: ${account[8].item():.2f}")

        # Take step
        td = env.step(td)

        # Record reward
        reward = td["next", "reward"].item()
        rewards.append(reward)

        # Check done/truncated
        done = td["next", "done"].item()
        truncated = td["next", "truncated"].item() if "truncated" in td["next"].keys() else False

        if done or truncated:
            print(f"\nEpisode ended at step {step + 1}")
            if done:
                print("  Reason: done=True")
            if truncated:
                print("  Reason: truncated=True")
            break

        # Move to next state
        td = td["next"]
        step += 1

    print("=" * 80)

    # Print episode summary
    print("\n[5/6] Episode Summary")
    print("=" * 80)
    print(f"Steps completed: {step + 1}")
    print(f"Total reward: {sum(rewards):.4f}")
    print(f"Mean reward: {sum(rewards) / len(rewards):.4f}")
    print(f"Max reward: {max(rewards):.4f}")
    print(f"Min reward: {min(rewards):.4f}")

    # Action distribution
    print("\nAction distribution:")
    unique_actions = set(actions_taken)
    for action_idx in sorted(unique_actions):
        count = actions_taken.count(action_idx)
        pct = count / len(actions_taken) * 100
        side, sl, tp = action_map[action_idx]
        if side is None:
            action_desc = "Hold/Close"
        else:
            action_desc = f"{side.upper()} SL={sl*100:.1f}% TP={tp*100:.1f}%"
        print(f"  {action_idx}: {action_desc:30s} - {count:3d} times ({pct:5.1f}%)")

    # Environment metrics
    print("\nFinal state:")
    print(f"  Cash: ${env.cash:.2f}")
    print(f"  Position size: {env.position_size:.4f}")
    print(f"  Position value: ${env.position_value:.2f}")
    print(f"  Total portfolio value: ${env.cash + env.position_value:.2f}")

    # Calculate return
    initial_value = config.initial_cash[0] if isinstance(config.initial_cash, tuple) else config.initial_cash
    final_value = env.cash + env.position_value
    total_return = (final_value - initial_value) / initial_value * 100

    print(f"\n  Initial capital: ${initial_value:.2f}")
    print(f"  Final capital: ${final_value:.2f}")
    print(f"  Total return: {total_return:.2f}%")

    # Visualization (optional)
    print("\n[6/6] Visualization")
    print("  To visualize trading history, uncomment env.render_history()")
    # env.render_history()

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
