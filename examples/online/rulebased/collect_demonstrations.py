"""
Collect expert demonstrations from rule-based actors for imitation learning.

This script demonstrates how to:
1. Create rule-based expert actors
2. Collect state-action demonstrations from experts
3. Evaluate expert performance on backtesting data
4. Save demonstrations for behavioral cloning

Usage:
    python collect_demonstrations.py --num_episodes 100 --expert momentum
    python collect_demonstrations.py --num_episodes 1000 --expert all --save_path demos.pt

For more information, see:
    https://github.com/TorchTrade/torchtrade_envs/issues/54
"""

import argparse
from pathlib import Path
from typing import List, Optional

import datasets
import pandas as pd
import torch
from tensordict import TensorDict
from tqdm import tqdm

from torchtrade.actor import MomentumActor, MeanReversionActor, BreakoutActor, create_expert_ensemble
from torchtrade.envs import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit


def create_env(df: pd.DataFrame, config_overrides: Optional[dict] = None) -> SeqLongOnlyEnv:
    """Create a SeqLongOnlyEnv with default or custom configuration."""
    config = SeqLongOnlyEnvConfig(
        symbol="BTC/USD",
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
            TimeFrame(5, TimeFrameUnit.Minute),
            TimeFrame(15, TimeFrameUnit.Minute),
        ],
        window_sizes=[12, 8, 8],
        execute_on=TimeFrame(5, TimeFrameUnit.Minute),
        initial_cash=1000,
        transaction_fee=0.0025,
        slippage=0.001,
        seed=42,
        random_start=True,
    )

    # Apply any overrides
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)

    return SeqLongOnlyEnv(df, config)


def collect_demonstrations_from_expert(
    env: SeqLongOnlyEnv,
    expert,
    num_episodes: int = 100,
    verbose: bool = True,
) -> TensorDict:
    """
    Collect state-action demonstrations from a single expert.

    Args:
        env: Trading environment
        expert: Rule-based actor (e.g., MomentumActor)
        num_episodes: Number of episodes to collect
        verbose: Whether to show progress bar

    Returns:
        TensorDict containing:
            - observations: Market data and account state
            - actions: Expert actions
            - rewards: Environment rewards
            - episode_id: Episode number for each transition
    """
    demonstrations = []

    iterator = range(num_episodes)
    if verbose:
        iterator = tqdm(iterator, desc=f"Collecting from {expert.__class__.__name__}")

    for episode_id in iterator:
        obs = env.reset()
        done = False
        step = 0

        while not done:
            # Expert selects action
            obs_with_action = expert(obs.clone())
            action = obs_with_action["action"].item()

            # Store demonstration
            demo = {
                "observation": obs.clone(),
                "action": torch.tensor([action], dtype=torch.long),
                "episode_id": torch.tensor([episode_id], dtype=torch.long),
                "step": torch.tensor([step], dtype=torch.long),
            }

            # Step environment
            obs = env.step(obs_with_action)
            done = obs.get("done", torch.tensor([False])).item()

            # Store reward
            demo["reward"] = obs.get("reward", torch.tensor([0.0]))

            demonstrations.append(demo)
            step += 1

    # Stack all demonstrations into a single TensorDict
    stacked_demos = torch.stack([TensorDict(d, batch_size=[]) for d in demonstrations])

    if verbose:
        print(f"\nCollected {len(demonstrations)} transitions from {num_episodes} episodes")
        print(f"Average episode length: {len(demonstrations) / num_episodes:.1f}")

    return stacked_demos


def evaluate_expert(
    env: SeqLongOnlyEnv,
    expert,
    num_episodes: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Evaluate expert performance on the environment.

    Args:
        env: Trading environment
        expert: Rule-based actor
        num_episodes: Number of evaluation episodes
        verbose: Whether to print results

    Returns:
        Dictionary with performance metrics:
            - mean_return: Average episode return
            - std_return: Standard deviation of returns
            - mean_length: Average episode length
            - action_distribution: Frequency of each action
    """
    episode_returns = []
    episode_lengths = []
    all_actions = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            obs_with_action = expert(obs.clone())
            action = obs_with_action["action"].item()
            all_actions.append(action)

            obs = env.step(obs_with_action)
            reward = obs.get("reward", torch.tensor([0.0])).item()
            done = obs.get("done", torch.tensor([False])).item()

            episode_return += reward
            episode_length += 1

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    # Calculate action distribution
    action_counts = torch.bincount(torch.tensor(all_actions), minlength=env.action_spec.n)
    action_distribution = (action_counts.float() / action_counts.sum()).tolist()

    results = {
        "mean_return": sum(episode_returns) / len(episode_returns),
        "std_return": torch.tensor(episode_returns).std().item(),
        "mean_length": sum(episode_lengths) / len(episode_lengths),
        "action_distribution": action_distribution,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Expert: {expert.__class__.__name__}")
        print(f"{'='*60}")
        print(f"Mean Return: {results['mean_return']:.4f} ± {results['std_return']:.4f}")
        print(f"Mean Episode Length: {results['mean_length']:.1f}")
        print(f"Action Distribution:")
        for i, freq in enumerate(action_distribution):
            action_name = ["Sell", "Hold", "Buy"][i] if env.action_spec.n == 3 else f"Action {i}"
            print(f"  {action_name}: {freq*100:.1f}%")
        print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect expert demonstrations")
    parser.add_argument(
        "--expert",
        type=str,
        default="all",
        choices=["momentum", "mean_reversion", "breakout", "all"],
        help="Which expert to use (default: all)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to collect per expert (default: 100)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation (default: 10)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save demonstrations (default: None, don't save)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Torch-Trade/btcusdt_spot_1m_01_2020_to_12_2025",
        help="HuggingFace dataset path (default: Torch-Trade/btcusdt_spot_1m_01_2020_to_12_2025)",
    )
    parser.add_argument(
        "--test_split_start",
        type=str,
        default="2025-01-01",
        help="Test split start date (default: 2025-01-01)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for actors",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}...")
    df = datasets.load_dataset(args.data_path)["train"].to_pandas()
    df['0'] = pd.to_datetime(df['0'])

    # Split data
    test_split_date = pd.to_datetime(args.test_split_start)
    train_df = df[df['0'] < test_split_date]

    print(f"Train data: {len(train_df)} rows from {train_df['0'].min()} to {train_df['0'].max()}")

    # Create environment
    env = create_env(train_df)
    print(f"Environment: {env.__class__.__name__}")
    print(f"Action space size: {env.action_spec.n}")

    # Determine market data keys for actors
    # This assumes the environment has market_data_* keys in its observation spec
    market_data_keys = [
        key for key in env.observation_spec.keys() if key.startswith("market_data_")
    ]
    print(f"Market data keys: {market_data_keys}")

    # Create experts
    expert_kwargs = {
        "market_data_keys": market_data_keys,
        "action_space_size": env.action_spec.n,
        "debug": args.debug,
    }

    if args.expert == "all":
        experts = create_expert_ensemble(**expert_kwargs)
        expert_names = ["momentum", "mean_reversion", "breakout"]
    else:
        expert_map = {
            "momentum": MomentumActor,
            "mean_reversion": MeanReversionActor,
            "breakout": BreakoutActor,
        }
        experts = [expert_map[args.expert](**expert_kwargs)]
        expert_names = [args.expert]

    # Collect demonstrations and evaluate
    all_demonstrations = []

    for expert, name in zip(experts, expert_names):
        print(f"\n{'#'*60}")
        print(f"# Expert: {name.upper()}")
        print(f"{'#'*60}\n")

        # Evaluate expert
        results = evaluate_expert(env, expert, num_episodes=args.eval_episodes)

        # Collect demonstrations
        demos = collect_demonstrations_from_expert(
            env, expert, num_episodes=args.num_episodes
        )
        all_demonstrations.append(demos)

    # Combine all demonstrations
    if len(all_demonstrations) > 1:
        combined_demos = torch.cat(all_demonstrations, dim=0)
    else:
        combined_demos = all_demonstrations[0]

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total transitions collected: {len(combined_demos)}")
    print(f"Total episodes: {args.num_episodes * len(experts)}")

    # Calculate combined action distribution
    all_actions = combined_demos["action"].squeeze()
    action_counts = torch.bincount(all_actions, minlength=env.action_spec.n)
    action_dist = (action_counts.float() / action_counts.sum()).tolist()

    print(f"\nCombined Action Distribution:")
    for i, freq in enumerate(action_dist):
        action_name = ["Sell", "Hold", "Buy"][i] if env.action_spec.n == 3 else f"Action {i}"
        print(f"  {action_name}: {freq*100:.1f}%")

    # Save demonstrations if requested
    if args.save_path:
        save_path = Path(args.save_path)
        print(f"\nSaving demonstrations to {save_path}...")
        torch.save(combined_demos, save_path)
        print(f"Saved {len(combined_demos)} transitions")

    print(f"\n{'='*60}\n")
    print("Next steps:")
    print("1. Use these demonstrations for behavioral cloning (BC)")
    print("2. Train a policy to imitate expert actions")
    print("3. Fine-tune with PPO for better performance")
    print("\nSee Issue #54 for implementation details:")
    print("https://github.com/TorchTrade/torchtrade_envs/issues/54")


if __name__ == "__main__":
    main()
