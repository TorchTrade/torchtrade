"""
Hyperparameter tuning for Mean Reversion Actor.

This script performs grid search to find optimal hyperparameters for the mean reversion
strategy using Bollinger Bands and Stochastic RSI.

Usage:
    # Basic usage with defaults
    python tune_hyperparameters.py

    # Custom data and save results
    python tune_hyperparameters.py \
        --data_path Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025 \
        --test_split_start "2025-01-01" \
        --save_config best_config.json \
        --save_plot trading_history.png

    # Custom hyperparameter grid
    python tune_hyperparameters.py --custom_grid my_grid.json
"""

import argparse
import json
from itertools import product
from typing import Callable, Dict, List, Tuple

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from torchtrade.actor.rulebased.meanreversion.actor import MeanReversionActor
from torchtrade.envs import SeqFuturesEnv, SeqFuturesEnvConfig
from torchtrade.envs.offline.infrastructure.utils import TimeFrame, TimeFrameUnit


# ============================================================================
# Hyperparameter Search Space
# ============================================================================

DEFAULT_HYPERPARAMETER_GRID = {
    "bb_window": [15, 20, 25],
    "bb_std": [2.0],
    "stoch_rsi_window": [10, 12, 14, 16],
    "stoch_k_window": [3],
    "stoch_d_window": [3],
    "oversold_threshold": [15, 20, 25],
    "overbought_threshold": [70, 75, 80, 85],
    "volume_confirmation": [1.1, 1.2, 1.3],
}


# ============================================================================
# Helper Functions
# ============================================================================


def create_env(df: pd.DataFrame, preprocessing_fn: Callable = None, seed: int = 42, time_frames: List[TimeFrame] = None, execute_timeframe: TimeFrame = TimeFrame(15, TimeFrameUnit.Minute)) -> SeqFuturesEnv:
    """
    Create a SeqFuturesEnv for evaluation.

    For deterministic strategies, we run ONE full episode through the entire dataset.

    Args:
        df: DataFrame with OHLCV data
        preprocessing_fn: Optional preprocessing function from actor
        seed: Random seed

    Returns:
        SeqFuturesEnv instance
    """
    config = SeqFuturesEnvConfig(
        symbol="BTC/USD",
        time_frames=time_frames,
        window_sizes=[2, 2],  # Small windows OK - features computed on full dataset
        execute_on=execute_timeframe,
        initial_cash=1000,
        transaction_fee=0.0025,
        slippage=0.001,
        seed=seed,
        random_start=False,
    )
    return SeqFuturesEnv(df, config, feature_preprocessing_fn=preprocessing_fn)


def create_actor(hyperparams: Dict, market_data_keys: List[str], execute_timeframe: TimeFrame) -> MeanReversionActor:
    """
    Create a MeanReversionActor with given hyperparameters.

    Args:
        hyperparams: Dictionary of hyperparameters
        market_data_keys: List of market data keys from environment

    Returns:
        MeanReversionActor instance
    """
    features_order = [
        "features_bb_middle", "features_bb_std", "features_bb_upper",
        "features_bb_lower", "features_bb_position",
        "features_stoch_rsi_k", "features_stoch_rsi_d", "features_volume", "features_avg_volume"
    ]

    return MeanReversionActor(
        market_data_keys=market_data_keys,
        features_order=features_order,
        action_space_size=3,
        execute_timeframe=execute_timeframe,
        **hyperparams
    )


def evaluate_actor(
    env: SeqFuturesEnv,
    actor: MeanReversionActor,
    render_history: bool = False,
) -> Dict[str, float]:
    """
    Evaluate actor performance on the environment.

    Since strategies are deterministic, num_episodes=1 is sufficient for a single evaluation.

    Args:
        env: Trading environment
        actor: Mean reversion actor
        render_history: Whether to track portfolio values for visualization

    Returns:
        Dictionary with performance metrics:
            - mean_return: Average total return per episode
            - std_return: Standard deviation of returns
            - sharpe_ratio: Sharpe ratio (mean / std)
            - max_return: Best episode return
            - min_return: Worst episode return
            - win_rate: Fraction of profitable episodes
            - mean_length: Average episode length
            - portfolio_values: (if render_history=True) List of portfolio values
    """
    obs = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        obs_with_action = actor(obs.clone())


        obs = env.step(obs_with_action)
        done = obs.get(("next", "done"), torch.tensor([False])).item()
        truncated = obs.get(("next", "truncated"), torch.tensor([False])).item()

        if done:
            break

    metrics = env.get_metrics()
    if 1 not in env.action_history or -1 not in env.action_history:
        metrics["sharpe_ratio"] = -float('inf')
    if render_history:
        env.render_history()
            
    return metrics


def grid_search(
    df: pd.DataFrame,
    hyperparameter_grid: Dict[str, List],
    market_data_keys: List[str],
    execute_timeframe: TimeFrame,
    time_frames: List[TimeFrame],
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    Perform grid search over hyperparameters.

    Since strategies are deterministic, num_episodes=1 evaluates each config once.

    IMPORTANT: Each hyperparameter combination gets its own environment because
    the preprocessing function depends on the hyperparameters (e.g., bb_window,
    stoch_rsi_window).

    Args:
        df: Raw DataFrame with OHLCV data
        hyperparameter_grid: Dictionary of hyperparameter lists
        market_data_keys: List of market data keys
        execute_timeframe: TimeFrame for execution
        time_frames: List of TimeFrames to use in environment
        seed: Random seed for environment
        num_episodes: Number of episodes per evaluation
        verbose: Whether to print progress

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

        # Create actor with these hyperparameters
        actor = create_actor(hyperparams, market_data_keys, execute_timeframe)

        # Get preprocessing function from actor
        preprocessing_fn = actor.get_preprocessing_fn()

        # Create environment with actor's preprocessing function
        env = create_env(df, preprocessing_fn=preprocessing_fn, seed=seed, time_frames=time_frames, execute_timeframe=execute_timeframe)

        # Evaluate
        metrics = evaluate_actor(env, actor)

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
                    "mean_return": f"{metrics['total_return']:.2f}",
                })
        if best_sharpe == -float('inf'):
            best_hyperparams = hyperparams
            best_metrics = metrics
            print("Warning: No good hyperparameters found, returning last parameter combination")
    return best_hyperparams, best_metrics, all_results




def print_results(
    best_hyperparams: Dict,
    best_metrics: Dict,
    test_metrics: Dict = None,
):
    """Print tuning results in a nice format."""
    print(f"\n{'='*60}")
    print(f"TUNING RESULTS: MEAN REVERSION ACTOR")
    print(f"{'='*60}\n")

    print("Best Hyperparameters (Train Set):")
    for param, value in best_hyperparams.items():
        print(f"  {param}: {value}")

    print(f"\nTrain Set Performance:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v}")

    if test_metrics:
        print(f"\nTest Set Performance:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")

    print(f"\n{'='*60}\n")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Tune Mean Reversion Actor hyperparameters")
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
        help="Number of episodes for training evaluation (default: 1, since strategy is deterministic)",
    )
    parser.add_argument(
        "--test_episodes",
        type=int,
        default=1,
        help="Number of episodes for test evaluation (default: 1, since strategy is deterministic)",
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

    time_frames = [TimeFrame(5, TimeFrameUnit.Minute), TimeFrame(15, TimeFrameUnit.Minute)]
    execute_timeframe = TimeFrame(15, TimeFrameUnit.Minute)

    # Create temporary environment to get market data keys
    print("\nCreating temporary environment to infer structure...")
    temp_env = create_env(train_df.head(1000), seed=args.seed, time_frames=time_frames, execute_timeframe=execute_timeframe)
    market_data_keys = [
        key for key in temp_env.observation_spec.keys()
        if key.startswith("market_data_")
    ]
    print(f"Market data keys: {market_data_keys}")

    # Load hyperparameter grid
    if args.custom_grid:
        print(f"\nLoading custom hyperparameter grid from {args.custom_grid}...")
        with open(args.custom_grid, 'r') as f:
            hyperparameter_grid = json.load(f)
    else:
        hyperparameter_grid = DEFAULT_HYPERPARAMETER_GRID

    print(f"\nHyperparameter Grid:")
    for param, values in hyperparameter_grid.items():
        print(f"  {param}: {values}")

    # Perform grid search on training data
    print("\nStarting grid search on training data...")
    best_hyperparams, best_metrics, all_results = grid_search(
        train_df,
        hyperparameter_grid,
        market_data_keys,
        execute_timeframe,
        time_frames,
        seed=args.seed,
        verbose=True,
    )

    # Evaluate best configuration on test data
    print(f"\nEvaluating best configuration on test set...")
    best_actor = create_actor(best_hyperparams, market_data_keys, execute_timeframe)
    preprocessing_fn = best_actor.get_preprocessing_fn()

    train_env = create_env(train_df, preprocessing_fn=preprocessing_fn, seed=args.seed, time_frames=time_frames, execute_timeframe=execute_timeframe)
    print("Rendering training set with best actor.")
    _ = evaluate_actor(
        train_env,
        best_actor,
        render_history=True,
    )
    
    print("Running test set with best actor.")
    test_env = create_env(test_df, preprocessing_fn=preprocessing_fn, seed=args.seed, time_frames=time_frames, execute_timeframe=execute_timeframe)

    test_metrics = evaluate_actor(
        test_env,
        best_actor,
        render_history=True,
    )

    # Print results
    print_results(best_hyperparams, best_metrics, test_metrics)


    # Save best configuration
    if args.save_config:
        config_data = {
            "actor": "mean_reversion",
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
