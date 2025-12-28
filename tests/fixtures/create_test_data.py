"""
Generate synthetic test data for example testing.

This module creates minimal OHLCV datasets that can be used to test
training examples without requiring real market data.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict


def create_test_ohlcv_csv(
    output_path: str,
    n_rows: int = 5000,
    initial_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a synthetic OHLCV CSV file for testing.

    Args:
        output_path: Path to save the CSV file
        n_rows: Number of rows to generate
        initial_price: Starting price
        seed: Random seed for reproducibility

    Returns:
        The generated DataFrame
    """
    np.random.seed(seed)

    # Generate timestamps (1-minute intervals)
    start_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq="1min")

    # Generate price data with random walk
    returns = np.random.normal(0, 0.001, n_rows)
    close_prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.002, n_rows)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.002, n_rows)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price

    # Ensure OHLC consistency
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))

    # Generate volume
    volume = np.random.lognormal(10, 1, n_rows)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created test OHLCV data: {output_path} ({n_rows} rows)")

    return df


def create_test_replay_buffer(
    output_path: str,
    n_transitions: int = 1000,
    obs_dim: int = 4,
    window_size: int = 10,
    n_actions: int = 3,
    seed: int = 42,
) -> TensorDict:
    """
    Create a synthetic replay buffer TensorDict for offline RL testing.

    Args:
        output_path: Path to save the TensorDict
        n_transitions: Number of transitions to generate
        obs_dim: Observation feature dimension
        window_size: Observation window size
        n_actions: Number of discrete actions
        seed: Random seed for reproducibility

    Returns:
        The generated TensorDict
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate synthetic observations
    observation = torch.randn(n_transitions, window_size, obs_dim)

    # Generate random actions (discrete)
    action = torch.randint(0, n_actions, (n_transitions,))

    # Generate rewards with some structure
    rewards = torch.randn(n_transitions) * 0.01

    # Generate next observations
    next_observation = torch.randn(n_transitions, window_size, obs_dim)

    # Generate done flags (sparse)
    done = torch.zeros(n_transitions, dtype=torch.bool)
    # Mark some episodes as done
    episode_length = 100
    done_indices = torch.arange(episode_length - 1, n_transitions, episode_length)
    done[done_indices] = True

    # Create TensorDict
    td = TensorDict({
        "observation": observation,
        "action": action,
        "next": TensorDict({
            "observation": next_observation,
            "reward": rewards,
            "done": done,
            "terminated": done,
        }, batch_size=[n_transitions]),
    }, batch_size=[n_transitions])

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(td, output_path)
    print(f"Created test replay buffer: {output_path} ({n_transitions} transitions)")

    return td


def create_all_test_data(base_dir: str = None):
    """Create all test data files needed for example testing."""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "data"

    base_dir = Path(base_dir)

    # Create OHLCV CSV for offline/online examples
    create_test_ohlcv_csv(
        output_path=str(base_dir / "test_btcusd_1m.csv"),
        n_rows=5000,  # ~3.5 days of minute data
    )

    # Create replay buffer for offline IQL
    create_test_replay_buffer(
        output_path=str(base_dir / "test_replay_buffer.pt"),
        n_transitions=1000,
    )

    print(f"\nAll test data created in: {base_dir}")


if __name__ == "__main__":
    create_all_test_data()
