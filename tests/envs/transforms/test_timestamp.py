"""Tests for TimestampTransform."""

import time

import numpy as np
import pandas as pd
import pytest
from tensordict import TensorDict
from torchrl.envs import TransformedEnv

from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.transforms import TimestampTransform
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


@pytest.fixture
def simple_df():
    """Create a simple DataFrame for testing."""
    np.random.seed(42)
    n = 1000

    dates = pd.date_range("2024-01-01", periods=n, freq="1min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": close + np.random.randn(n) * 0.05,
            "high": close + np.abs(np.random.randn(n) * 0.1),
            "low": close - np.abs(np.random.randn(n) * 0.1),
            "close": close,
            "volume": np.random.randint(100, 1000, n),
        }
    )


@pytest.fixture
def base_env(simple_df):
    """Create base environment for testing."""
    config = SequentialTradingEnvConfig(
        symbol="TEST/USD",
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        execute_on=TimeFrame(5, TimeFrameUnit.Minute),
        include_base_features=False,
        initial_cash=100000,
        transaction_fee=0.0,
        slippage=0.0,
        random_start=False,
        max_traj_length=50,
        seed=42,
    )
    return SequentialTradingEnv(simple_df, config)


@pytest.mark.parametrize("out_key", ["timestamp", "custom_ts"])
def test_reset_and_step_add_timestamps(base_env, out_key):
    """Verify reset and step add valid Unix timestamps with custom keys."""
    transform = TimestampTransform(out_key=out_key)
    env = TransformedEnv(base_env, transform)

    # Test reset
    before_reset = time.time()
    td = env.reset()
    after_reset = time.time()

    assert out_key in td.keys()
    reset_ts = td[out_key]
    assert isinstance(reset_ts, float)
    assert reset_ts > 1700000000  # Sanity check: valid Unix timestamp (after 2023)
    assert before_reset <= reset_ts <= after_reset

    # Test step (with small delay to ensure different timestamp)
    time.sleep(0.01)
    before_step = time.time()
    td = td.set("action", env.action_spec.rand())
    td = env.step(td)
    after_step = time.time()

    step_ts = td["next", out_key]
    assert isinstance(step_ts, float)
    assert before_step <= step_ts <= after_step
    assert step_ts > reset_ts


def test_rollout_timestamps_monotonic(base_env):
    """Verify timestamps increase monotonically during rollout."""
    env = TransformedEnv(base_env, TimestampTransform())
    rollout = env.rollout(max_steps=20)

    assert "timestamp" in rollout.keys()

    timestamps = [rollout[i]["next", "timestamp"] for i in range(rollout.shape[0])]
    assert all(isinstance(ts, float) for ts in timestamps)
    assert timestamps == sorted(timestamps)


def test_timestamp_survives_memmap(base_env, tmp_path):
    """Verify timestamps survive memmap serialization (critical for dataset creation)."""
    env = TransformedEnv(base_env, TimestampTransform())

    td = env.reset()
    original_ts = td["timestamp"]

    # Save and load via memmap
    memmap_path = tmp_path / "tensordict"
    td.memmap_(memmap_path)
    loaded_td = TensorDict.load_memmap(memmap_path)

    assert loaded_td["timestamp"] == original_ts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
