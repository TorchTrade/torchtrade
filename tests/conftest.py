"""
Shared pytest fixtures for torchtrade tests.
"""

import os
import numpy as np
import pandas as pd
import pytest

from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


# Fix MKL threading issue for ParallelEnv tests
@pytest.fixture(scope="session", autouse=True)
def set_mkl_threading_layer():
    """Set MKL threading layer to GNU to avoid conflicts with multiprocessing.

    This prevents EOFError in ParallelEnv tests caused by MKL incompatibility
    with libgomp when spawning worker processes.
    """
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    yield


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


@pytest.fixture
def trending_up_df():
    """Create synthetic data with clear upward trend for TP testing."""
    np.random.seed(42)
    n_minutes = 500

    start_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

    # Strong upward trend
    initial_price = 100.0
    trend = np.linspace(0, 0.3, n_minutes)  # 30% increase
    noise = np.random.normal(0, 0.001, n_minutes)
    close_prices = initial_price * (1 + trend + np.cumsum(noise))

    high_prices = close_prices * 1.002
    low_prices = close_prices * 0.998
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price

    volume = np.random.lognormal(10, 1, n_minutes)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })


@pytest.fixture
def trending_down_df():
    """Create synthetic data with clear downward trend for SL testing."""
    np.random.seed(42)
    n_minutes = 500

    start_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

    # Strong downward trend
    initial_price = 100.0
    trend = np.linspace(0, -0.2, n_minutes)  # 20% decrease
    noise = np.random.normal(0, 0.001, n_minutes)
    close_prices = initial_price * (1 + trend + np.cumsum(noise))

    high_prices = close_prices * 1.002
    low_prices = close_prices * 0.998
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price

    volume = np.random.lognormal(10, 1, n_minutes)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })


@pytest.fixture
def price_gap_df():
    """
    Create synthetic data with price gaps that skip past SL/TP levels.

    Simulates gap-down and gap-up scenarios where the price jumps
    past the SL/TP threshold in a single candle.
    """
    np.random.seed(42)
    n_minutes = 200

    start_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

    initial_price = 100.0
    close_prices = np.full(n_minutes, initial_price)

    # Create a gap-down at minute 50 (10% drop)
    close_prices[50:100] = initial_price * 0.90

    # Create a gap-up at minute 150 (15% jump from current level)
    close_prices[150:] = initial_price * 0.90 * 1.15

    # Add small noise
    close_prices = close_prices * (1 + np.random.normal(0, 0.001, n_minutes))

    high_prices = close_prices * 1.001
    low_prices = close_prices * 0.999
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price

    # Make the gap visible in open/close
    open_prices[50] = initial_price
    open_prices[150] = initial_price * 0.90

    volume = np.random.lognormal(10, 1, n_minutes)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })


# ============================================================================
# UNIFIED ENVIRONMENT FIXTURES (for consolidated tests)
# ============================================================================

@pytest.fixture(params=["spot", "futures"])
def trading_mode(request):
    """Parametrized fixture for testing both spot and futures configurations.

    Returns leverage value for each mode:
    - spot: leverage=1
    - futures: leverage=10
    """
    return {"spot": 1, "futures": 10}[request.param]


@pytest.fixture
def unified_config_spot():
    """Config for spot-like trading (leverage=1, no shorts).

    Default action_levels: [0, 1] (flat, long)
    """
    from torchtrade.envs.offline import SequentialTradingEnvConfig

    return SequentialTradingEnvConfig(
        leverage=1,  # No liquidation
        # action_levels auto-generated: [0, 1]
        initial_cash=1000,
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        transaction_fee=0.01,
        slippage=0.0,
        seed=42,
        max_traj_length=100,
        random_start=False,
    )


@pytest.fixture
def unified_config_futures():
    """Config for futures-like trading (leverage>1, bidirectional).

    Default action_levels: [-1, 0, 1] (short, flat, long)
    """
    from torchtrade.envs.offline import SequentialTradingEnvConfig

    return SequentialTradingEnvConfig(
        leverage=10,  # Enables liquidation
        # action_levels auto-generated: [-1, 0, 1]
        initial_cash=1000,
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        transaction_fee=0.01,
        slippage=0.0,
        seed=42,
        max_traj_length=100,
        random_start=False,
    )


def simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature processing function for testing."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.fillna(0, inplace=True)
    return df


def validate_account_state(account_state, leverage):
    """
    Shared validation helper for account state structure.

    Args:
        account_state: Account state tensor [6]
            [exposure_pct, position_direction, unrealized_pnl_pct,
             holding_time, leverage, distance_to_liquidation]
        leverage: Leverage value (1 for spot, >1 for futures)
    """
    import torch

    assert isinstance(account_state, torch.Tensor)
    assert account_state.shape[-1] == 6, "Account state should have 6 elements"

    # Extract components (new 6-element structure)
    exposure_pct = account_state[..., 0]
    position_direction = account_state[..., 1]
    unrealized_pnl_pct = account_state[..., 2]
    holding_time = account_state[..., 3]
    state_leverage = account_state[..., 4]
    distance_to_liquidation = account_state[..., 5]

    # Common validations
    assert (exposure_pct >= 0).all(), "Exposure percentage should be non-negative"
    assert (holding_time >= 0).all(), "Holding time should be non-negative"
    assert (state_leverage >= 1.0).all(), "Leverage should be >= 1.0"
    assert (distance_to_liquidation >= 0).all(), "Distance to liquidation should be non-negative"

    # Leverage-specific validations
    if leverage == 1:
        # Spot: position_direction should be 0 or +1 (no shorts)
        assert (position_direction >= 0).all(), "Spot position direction should be >= 0 (no shorts)"
        assert ((position_direction == 0) | (position_direction == 1)).all(), \
            "Spot position direction should be 0 or 1"
        # Spot: leverage = 1.0
        assert (state_leverage == 1.0).all(), "Spot leverage should be 1.0"
        # Spot: distance_to_liquidation should be 1.0 (no liquidation risk)
        assert (distance_to_liquidation == 1.0).all(), "Spot distance_to_liquidation should be 1.0"
    else:  # leverage > 1
        # Futures: position_direction should be -1, 0, or +1
        assert ((position_direction == -1) | (position_direction == 0) | (position_direction == 1)).all(), \
            "Futures position direction should be -1, 0, or +1"
        # Futures: leverage >= 1.0 (already checked above)
        assert (state_leverage >= 1.0).all(), "Futures leverage should be >= 1.0"


