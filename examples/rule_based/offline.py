"""
Offline backtesting with MeanReversionActor on SequentialTradingEnv.

Runs a rule-based mean reversion strategy through a sequential trading
environment using historical data, printing actions and episode metrics.
Parameters are not tuned — this serves as an example of how to wire a
rule-based actor to an offline environment.

Usage:
    python examples/rule_based/offline.py
"""

import datasets
import pandas as pd

from torchtrade.actor import MeanReversionActor
from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


def main():
    print("=" * 80)
    print("MeanReversionActor - Offline Backtesting")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    dataset = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025")
    df = dataset["train"].to_pandas()
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create actor first to get its preprocessing function
    execute_tf = TimeFrame(1, TimeFrameUnit.Hour)
    actor = MeanReversionActor(
        bb_window=14,
        bb_std=1.2,
        stoch_rsi_window=10,
        oversold_threshold=35.0,
        overbought_threshold=65.0,
        volume_multiplier=1.0,
        execute_timeframe=execute_tf,
        debug=False,
    )
    preprocessing_fn = actor.get_preprocessing_fn()

    # Create environment with actor's preprocessing
    print("Creating environment...")
    config = SequentialTradingEnvConfig(
        symbol="BTC/USD",
        time_frames=["1Hour"],
        window_sizes=[48],
        execute_on="1Hour",
        initial_cash=10000,
        transaction_fee=0.0,
        slippage=0.0,
        action_levels=[-1, 0, 1],
        include_base_features=False,
        random_start=False,
    )
    env = SequentialTradingEnv(df, config, feature_preprocessing_fn=preprocessing_fn)
    print(f"  Max steps: {env.max_steps}")

    # Set actor's market_data_keys and features_order from env
    actor.market_data_keys = env.market_data_keys
    actor.features_order = env.sampler.get_feature_keys()
    actor.feature_idx = {feat: i for i, feat in enumerate(actor.features_order)}

    # Run rollout
    max_steps = 5000
    print(f"\nRunning rollout ({max_steps} steps)...")
    print("=" * 80)

    td = env.reset()
    episode_reward = 0.0

    for step in range(max_steps):
        td = actor(td)
        action_idx = td["action"].item()
        target_pct = env.action_levels[action_idx] * 100
        print(f"\nStep {step + 1}: action={action_idx} (target {target_pct:+.0f}%)")

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
