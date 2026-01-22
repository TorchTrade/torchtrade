"""Benchmark script for SeqFuturesSLTPEnv performance optimizations.

This script measures the environment step throughput after
the performance optimizations applied in the feature/perf-seq-futures-sltp branch.
"""

import time
import torch
import pandas as pd
import numpy as np

from torchtrade.envs.offline import SeqFuturesSLTPEnv, SeqFuturesSLTPEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


def generate_synthetic_ohlcv(n_rows: int = 100000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for benchmarking."""
    np.random.seed(seed)

    # Generate timestamps (1 minute frequency)
    timestamps = pd.date_range("2023-01-01", periods=n_rows, freq="1min")

    # Generate price data with random walk
    base_price = 50000.0
    returns = np.random.randn(n_rows) * 0.001  # 0.1% volatility per step
    close_prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV from close prices
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": close_prices * (1 + np.random.randn(n_rows) * 0.0005),
        "high": close_prices * (1 + np.abs(np.random.randn(n_rows)) * 0.001),
        "low": close_prices * (1 - np.abs(np.random.randn(n_rows)) * 0.001),
        "close": close_prices,
        "volume": np.abs(np.random.randn(n_rows)) * 100 + 50,
    })

    # Ensure high >= open, close, low and low <= open, close
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    return df


def benchmark_env_steps(env, n_episodes: int = 100, n_steps_per_episode: int = 50) -> dict:
    """Benchmark environment stepping throughput."""

    step_times = []
    reset_times = []
    total_steps = 0

    for ep in range(n_episodes):
        # Time reset
        t0 = time.perf_counter()
        td = env.reset()
        reset_times.append(time.perf_counter() - t0)

        for step in range(n_steps_per_episode):
            # Sample random action (avoid always hold)
            action = torch.randint(1, env.action_spec.n, ())
            td.set("action", action)

            # Time step
            t0 = time.perf_counter()
            td = env.step(td)
            step_times.append(time.perf_counter() - t0)

            total_steps += 1
            td = td["next"]

            if td.get("done", False) or td.get("truncated", False):
                break

    return {
        "total_steps": total_steps,
        "mean_step_time_ms": np.mean(step_times) * 1000,
        "std_step_time_ms": np.std(step_times) * 1000,
        "mean_reset_time_ms": np.mean(reset_times) * 1000,
        "steps_per_second": total_steps / sum(step_times),
        "total_time_s": sum(step_times) + sum(reset_times),
    }


def main():
    print("=" * 70)
    print("SeqFuturesSLTPEnv Performance Benchmark")
    print("=" * 70)

    # Generate data
    print("\nGenerating synthetic OHLCV data (100k rows)...")
    df = generate_synthetic_ohlcv(n_rows=100000)
    print(f"  Data shape: {df.shape}")

    # Create environment config
    config = SeqFuturesSLTPEnvConfig(
        symbol="BTC/USD",
        time_frames=[TimeFrame(15, TimeFrameUnit.Minute)],
        window_sizes=[32],
        execute_on=TimeFrame(15, TimeFrameUnit.Minute),
        leverage=5,
        transaction_fee=0.0004,
        slippage=0.001,
        stoploss_levels=[-0.02, -0.05],
        takeprofit_levels=[0.05, 0.1],
        initial_cash=5000,
        seed=42,
        random_start=True,
        max_traj_length=500,  # Limit episode length for benchmarking
    )

    # Create environment
    print("\nCreating SeqFuturesSLTPEnv...")
    env = SeqFuturesSLTPEnv(df, config)
    print(f"  Action space: {env.action_spec.n} actions")
    print(f"  Max steps: {env.max_steps}")

    # Warmup
    print("\nWarming up (10 episodes)...")
    _ = benchmark_env_steps(env, n_episodes=10, n_steps_per_episode=50)

    # Benchmark
    print("\nRunning benchmark (500 episodes, up to 100 steps each)...")
    results = benchmark_env_steps(env, n_episodes=500, n_steps_per_episode=100)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Total steps:        {results['total_steps']:,}")
    print(f"  Mean step time:     {results['mean_step_time_ms']:.3f} ms")
    print(f"  Std step time:      {results['std_step_time_ms']:.3f} ms")
    print(f"  Mean reset time:    {results['mean_reset_time_ms']:.3f} ms")
    print(f"  Steps per second:   {results['steps_per_second']:,.0f}")
    print(f"  Total time:         {results['total_time_s']:.2f} s")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
