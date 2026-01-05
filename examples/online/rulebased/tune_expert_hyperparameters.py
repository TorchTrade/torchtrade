"""
Hyperparameter tuning for rule-based expert actors.

This script demonstrates how to:
1. Load historical data from HuggingFace
2. Split into train/test sets
3. Perform grid search for optimal hyperparameters on train data
4. Evaluate best configuration on test data with visualization

Since the strategies are deterministic, each configuration is evaluated once per dataset.

Usage:
    # Tune MomentumActor parameters on full dataset
    python tune_expert_hyperparameters.py --expert momentum

    # Tune with custom data and save results
    python tune_expert_hyperparameters.py --expert mean_reversion \
        --data_path Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025 \
        --test_split_start "2025-01-01" \
        --save_config best_config.json \
        --save_plot trading_history.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from itertools import product

import datasets
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchtrade.actor import MomentumActor, MeanReversionActor, BreakoutActor
from torchtrade.envs import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit


# ============================================================================
# Hyperparameter Search Spaces
# ============================================================================

HYPERPARAMETER_GRIDS = {
    "momentum": {
        "momentum_window": [5, 10, 15, 20],
        "volatility_window": [10, 20, 30],
        "momentum_threshold": [0.005, 0.01, 0.015, 0.02],
        "volatility_threshold": [0.015, 0.02, 0.025, 0.03],
    },
    "mean_reversion": {
        "ma_window": [10, 15, 20, 25, 30],
        "deviation_threshold": [0.01, 0.015, 0.02, 0.025, 0.03],
    },
    "breakout": {
        "bb_window": [15, 20, 25, 30],
        "bb_std": [1.5, 2.0, 2.5, 3.0],
    },
}


# ============================================================================
# Helper Functions
# ============================================================================


def create_env(df: pd.DataFrame, seed: int = 42) -> SeqLongOnlyEnv:
    """Create a SeqLongOnlyEnv for evaluation.

    For deterministic strategies, we run ONE full episode through the entire dataset.

    Args:
        df: DataFrame with OHLCV data
        seed: Random seed
    """
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
        seed=seed,
    )
    return SeqLongOnlyEnv(df, config)


def create_expert(expert_name: str, hyperparams: Dict, market_data_keys: List[str]):
    """Create an expert actor with given hyperparameters."""
    base_kwargs = {
        "market_data_keys": market_data_keys,
        "action_space_size": 3,
    }

    if expert_name == "momentum":
        return MomentumActor(**base_kwargs, **hyperparams)
    elif expert_name == "mean_reversion":
        return MeanReversionActor(**base_kwargs, **hyperparams)
    elif expert_name == "breakout":
        return BreakoutActor(**base_kwargs, **hyperparams)
    else:
        raise ValueError(f"Unknown expert: {expert_name}")


def evaluate_expert(
    env: SeqLongOnlyEnv,
    expert,
    num_episodes: int = 1,
    render_history: bool = False,
) -> Dict[str, float]:
    """
    Evaluate expert performance on the environment.

    Since strategies are deterministic, num_episodes=1 is sufficient for a single evaluation.
    Multiple episodes only make sense if you want to test across different time periods.

    Returns:
        Dictionary with performance metrics:
            - mean_return: Average total return per episode
            - std_return: Standard deviation of returns
            - sharpe_ratio: Sharpe ratio (mean / std)
            - max_return: Best episode return
            - min_return: Worst episode return
            - win_rate: Fraction of profitable episodes
            - mean_length: Average episode length
    """
    episode_returns = []
    episode_lengths = []
    all_portfolio_values = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        portfolio_values = []

        while not (done or truncated):
            obs_with_action = expert(obs.clone())

            # Step environment and handle running out of data
            try:
                obs = env.step(obs_with_action)
            except ValueError as e:
                if "No more timestamps available" in str(e):
                    # Environment ran out of data - episode is truncated
                    truncated = True
                    break
                else:
                    raise

            reward = obs.get("reward", torch.tensor([0.0])).item()
            done = obs.get("done", torch.tensor([False])).item()
            truncated = obs.get("truncated", torch.tensor([False])).item()

            episode_return += reward
            episode_length += 1

            # Track portfolio value for visualization
            if render_history and episode == 0:  # Only first episode
                account_state = obs.get("account_state", torch.zeros(7))
                cash = account_state[0, 0].item()
                position_value = account_state[0, 2].item()
                portfolio_values.append(cash + position_value)

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        if render_history and episode == 0:
            all_portfolio_values = portfolio_values

    returns = np.array(episode_returns)

    metrics = {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "sharpe_ratio": float(returns.mean() / returns.std()) if returns.std() > 0 else 0.0,
        "max_return": float(returns.max()),
        "min_return": float(returns.min()),
        "win_rate": float((returns > 0).sum() / len(returns)),
        "mean_length": float(np.mean(episode_lengths)),
    }

    if render_history:
        metrics["portfolio_values"] = all_portfolio_values

    return metrics


def grid_search(
    env: SeqLongOnlyEnv,
    expert_name: str,
    hyperparameter_grid: Dict[str, List],
    market_data_keys: List[str],
    num_episodes: int = 1,
    verbose: bool = True,
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    Perform grid search over hyperparameters.

    Since strategies are deterministic, num_episodes=1 evaluates each config once.

    Returns:
        - best_hyperparams: Best hyperparameter configuration
        - best_metrics: Performance metrics for best config
        - all_results: List of all configurations and their metrics
    """
    # Generate all combinations
    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())
    param_combinations = list(product(*param_values))

    total_combinations = len(param_combinations)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Grid Search: {total_combinations} configurations")
        print(f"{'='*60}\n")

    all_results = []
    best_sharpe = -float('inf')
    best_hyperparams = None
    best_metrics = None

    iterator = tqdm(param_combinations, desc="Tuning") if verbose else param_combinations

    for param_combo in iterator:
        # Create hyperparameter dict
        hyperparams = dict(zip(param_names, param_combo))

        # Create expert with these hyperparameters
        expert = create_expert(expert_name, hyperparams, market_data_keys)

        # Evaluate
        metrics = evaluate_expert(env, expert, num_episodes=num_episodes)

        # Store result
        result = {
            "hyperparams": hyperparams,
            "metrics": metrics,
        }
        all_results.append(result)

        # Update best
        if metrics["sharpe_ratio"] > best_sharpe:
            best_sharpe = metrics["sharpe_ratio"]
            best_hyperparams = hyperparams
            best_metrics = metrics

            if verbose:
                iterator.set_postfix({
                    "best_sharpe": f"{best_sharpe:.3f}",
                    "mean_return": f"{metrics['mean_return']:.2f}",
                })

    return best_hyperparams, best_metrics, all_results


def render_trading_history(
    portfolio_values: List[float],
    expert_name: str,
    hyperparams: Dict,
    save_path: str = None,
):
    """Visualize portfolio value over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    steps = list(range(len(portfolio_values)))
    initial_value = portfolio_values[0]

    # Plot portfolio value
    ax.plot(steps, portfolio_values, linewidth=2, label="Portfolio Value", color='blue')
    ax.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.5, label="Initial Value")

    # Calculate and show final return
    final_value = portfolio_values[-1]
    total_return = ((final_value - initial_value) / initial_value) * 100

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_title(
        f"{expert_name.title()} Expert - Trading History\n"
        f"Total Return: {total_return:+.2f}%",
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add hyperparameters text
    hyperparam_text = "Hyperparameters:\n" + "\n".join(
        [f"{k}: {v}" for k, v in hyperparams.items()]
    )
    ax.text(
        0.02, 0.98, hyperparam_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trading history to {save_path}")
    else:
        plt.show()

    plt.close()


def print_results(
    expert_name: str,
    best_hyperparams: Dict,
    best_metrics: Dict,
    test_metrics: Dict = None,
):
    """Print tuning results in a nice format."""
    print(f"\n{'='*60}")
    print(f"TUNING RESULTS: {expert_name.upper()} EXPERT")
    print(f"{'='*60}\n")

    print("Best Hyperparameters (Train Set):")
    for param, value in best_hyperparams.items():
        print(f"  {param}: {value}")

    print(f"\nTrain Set Performance:")
    print(f"  Mean Return:    {best_metrics['mean_return']:>8.2f}")
    print(f"  Std Return:     {best_metrics['std_return']:>8.2f}")
    print(f"  Sharpe Ratio:   {best_metrics['sharpe_ratio']:>8.3f}")
    print(f"  Win Rate:       {best_metrics['win_rate']:>8.1%}")
    print(f"  Max Return:     {best_metrics['max_return']:>8.2f}")
    print(f"  Min Return:     {best_metrics['min_return']:>8.2f}")
    print(f"  Avg Length:     {best_metrics['mean_length']:>8.1f}")

    if test_metrics:
        print(f"\nTest Set Performance:")
        print(f"  Mean Return:    {test_metrics['mean_return']:>8.2f}")
        print(f"  Std Return:     {test_metrics['std_return']:>8.2f}")
        print(f"  Sharpe Ratio:   {test_metrics['sharpe_ratio']:>8.3f}")
        print(f"  Win Rate:       {test_metrics['win_rate']:>8.1%}")
        print(f"  Max Return:     {test_metrics['max_return']:>8.2f}")
        print(f"  Min Return:     {test_metrics['min_return']:>8.2f}")
        print(f"  Avg Length:     {test_metrics['mean_length']:>8.1f}")

        # Compare train vs test
        print(f"\nGeneralization (Test vs Train):")
        sharpe_diff = test_metrics['sharpe_ratio'] - best_metrics['sharpe_ratio']
        print(f"  Sharpe Difference: {sharpe_diff:+.3f}")
        if sharpe_diff > 0.1:
            print("  ✅ Better performance on test set!")
        elif sharpe_diff < -0.2:
            print("  ⚠️  Significant performance drop on test set")
        else:
            print("  ✓ Similar performance on test set")

    print(f"\n{'='*60}\n")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Tune rule-based expert hyperparameters")
    parser.add_argument(
        "--expert",
        type=str,
        default="momentum",
        choices=["momentum", "mean_reversion", "breakout"],
        help="Which expert to tune (default: momentum)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025",
        help="HuggingFace dataset path",
    )
    parser.add_argument(
        "--test_split_start",
        type=str,
        default="2025-01-01",
        help="Test split start date (default: 2025-01-01)",
    )
    parser.add_argument(
        "--train_episodes",
        type=int,
        default=1,
        help="Number of episodes for training evaluation (default: 1, since strategies are deterministic)",
    )
    parser.add_argument(
        "--test_episodes",
        type=int,
        default=1,
        help="Number of episodes for test evaluation (default: 1, since strategies are deterministic)",
    )
    parser.add_argument(
        "--save_config",
        type=str,
        default=None,
        help="Path to save best configuration JSON (optional)",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="Path to save trading history plot (optional)",
    )
    parser.add_argument(
        "--custom_grid",
        type=str,
        default=None,
        help="Path to custom hyperparameter grid JSON (optional)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}...")
    df = datasets.load_dataset(args.data_path)["train"].to_pandas()
    df['0'] = pd.to_datetime(df['0'])

    # Split data
    test_split_date = pd.to_datetime(args.test_split_start)
    train_df = df[df['0'] < test_split_date]
    test_df = df[df['0'] >= test_split_date]

    print(f"\nData Split:")
    print(f"  Train: {len(train_df):,} rows ({train_df['0'].min()} to {train_df['0'].max()})")
    print(f"  Test:  {len(test_df):,} rows ({test_df['0'].min()} to {test_df['0'].max()})")

    # Create environments
    print("\nCreating environments...")
    train_env = create_env(train_df, seed=args.seed)
    test_env = create_env(test_df, seed=args.seed)

    # Get market data keys
    market_data_keys = [
        key for key in train_env.observation_spec.keys()
        if key.startswith("market_data_")
    ]
    print(f"Market data keys: {market_data_keys}")

    # Load hyperparameter grid
    if args.custom_grid:
        print(f"\nLoading custom hyperparameter grid from {args.custom_grid}...")
        with open(args.custom_grid, 'r') as f:
            hyperparameter_grid = json.load(f)
    else:
        hyperparameter_grid = HYPERPARAMETER_GRIDS[args.expert]

    print(f"\nHyperparameter Grid:")
    for param, values in hyperparameter_grid.items():
        print(f"  {param}: {values}")

    # Perform grid search on training data
    best_hyperparams, best_metrics, all_results = grid_search(
        train_env,
        args.expert,
        hyperparameter_grid,
        market_data_keys,
        num_episodes=args.train_episodes,
        verbose=True,
    )

    # Evaluate best configuration on test data
    print(f"\nEvaluating best configuration on test set...")
    best_expert = create_expert(args.expert, best_hyperparams, market_data_keys)
    test_metrics = evaluate_expert(
        test_env,
        best_expert,
        num_episodes=args.test_episodes,
        render_history=True,
    )

    # Print results
    print_results(args.expert, best_hyperparams, best_metrics, test_metrics)

    # Render trading history
    if "portfolio_values" in test_metrics:
        save_plot_path = args.save_plot or f"{args.expert}_trading_history.png"
        render_trading_history(
            test_metrics["portfolio_values"],
            args.expert,
            best_hyperparams,
            save_path=save_plot_path,
        )

    # Save best configuration
    if args.save_config:
        config_data = {
            "expert": args.expert,
            "best_hyperparams": best_hyperparams,
            "train_metrics": best_metrics,
            "test_metrics": {k: v for k, v in test_metrics.items() if k != "portfolio_values"},
            "data_info": {
                "data_path": args.data_path,
                "test_split_start": args.test_split_start,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
            },
        }

        with open(args.save_config, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"Saved best configuration to {args.save_config}")

    # Print top 5 configurations
    print(f"\nTop 5 Configurations (by Sharpe Ratio):")
    sorted_results = sorted(
        all_results,
        key=lambda x: x["metrics"]["sharpe_ratio"],
        reverse=True
    )[:5]

    for i, result in enumerate(sorted_results, 1):
        hyperparams = result["hyperparams"]
        metrics = result["metrics"]
        print(f"\n  {i}. Sharpe: {metrics['sharpe_ratio']:.3f}, "
              f"Return: {metrics['mean_return']:.2f}, "
              f"Win Rate: {metrics['win_rate']:.1%}")
        for param, value in hyperparams.items():
            print(f"     {param}: {value}")


if __name__ == "__main__":
    main()
