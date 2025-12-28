"""
Shared pytest fixtures for torchtrade tests.
"""

import numpy as np
import pandas as pd
import pytest

from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit


@pytest.fixture
def sample_ohlcv_df():
    """
    Create a synthetic OHLCV DataFrame for testing.

    Generates 1440 minutes (1 day) of minute-level data with realistic
    price movements and volume patterns.
    """
    np.random.seed(42)
    n_minutes = 1440  # 1 day of minute data

    # Generate timestamps
    start_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

    # Generate price data with random walk
    initial_price = 100.0
    returns = np.random.normal(0, 0.001, n_minutes)  # Small random returns
    close_prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.002, n_minutes)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.002, n_minutes)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price

    # Ensure OHLC consistency: low <= open, close <= high
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))

    # Generate volume
    volume = np.random.lognormal(10, 1, n_minutes)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })

    return df


@pytest.fixture
def large_ohlcv_df():
    """
    Create a larger synthetic OHLCV DataFrame for stress testing.

    Generates 10080 minutes (7 days) of data.
    """
    np.random.seed(42)
    n_minutes = 10080  # 7 days

    start_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

    initial_price = 100.0
    returns = np.random.normal(0, 0.001, n_minutes)
    close_prices = initial_price * np.exp(np.cumsum(returns))

    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.002, n_minutes)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.002, n_minutes)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price

    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))

    volume = np.random.lognormal(10, 1, n_minutes)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })

    return df


@pytest.fixture
def default_timeframes():
    """Default timeframe configuration for testing."""
    return [
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
    ]


@pytest.fixture
def default_window_sizes():
    """Default window sizes matching default_timeframes."""
    return [10, 5]


@pytest.fixture
def execute_timeframe():
    """Default execution timeframe for testing."""
    return TimeFrame(1, TimeFrameUnit.Minute)


