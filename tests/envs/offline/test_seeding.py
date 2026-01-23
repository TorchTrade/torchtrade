"""
Tests for seeding and reproducibility in offline environments.

This module tests that all offline environments produce reproducible
trajectories when given the same seed, covering:
- MarketDataObservationSampler seeding
- InitialBalanceSampler seeding
- End-to-end environment reproducibility
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline.infrastructure.sampler import MarketDataObservationSampler
from torchtrade.envs.offline.infrastructure.utils import InitialBalanceSampler
from torchtrade.envs.offline.longonly.sequential import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.offline.longonly.sequential_sltp import SeqLongOnlySLTPEnv, SeqLongOnlySLTPEnvConfig
from torchtrade.envs.offline.longonly.onestep import LongOnlyOneStepEnv, LongOnlyOneStepEnvConfig
from torchtrade.envs.offline.futures.sequential import SeqFuturesEnv, SeqFuturesEnvConfig
from torchtrade.envs.offline.futures.sequential_sltp import SeqFuturesSLTPEnv, SeqFuturesSLTPEnvConfig
from torchtrade.envs.offline.futures.onestep import FuturesOneStepEnv, FuturesOneStepEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


# Test configuration constants
FUTURES_CONFIG = {
    "leverage": 10.0,
    "maintenance_margin_rate": 0.05,
}

FUTURES_SLTP_CONFIG = {
    **FUTURES_CONFIG,
    "stoploss_levels": [-0.05, -0.1],
    "takeprofit_levels": [0.05, 0.1],
}

LONGONLY_SLTP_CONFIG = {
    "stoploss_levels": [-0.05, -0.1],
    "takeprofit_levels": [0.05, 0.1],
}


def simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature processing function for testing."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.fillna(0, inplace=True)
    return df


class TestSamplerSeeding:
    """Tests for MarketDataObservationSampler seeding."""

    def test_sampler_accepts_seed_parameter(
        self, large_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """Sampler should initialize successfully with seed parameter."""
        sampler = MarketDataObservationSampler(
            df=large_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
            max_traj_length=100,
            seed=42,
        )
        assert sampler is not None
        assert hasattr(sampler, 'np_rng')

    def test_random_start_reproducible_with_same_seed(
        self, large_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """Same seed should produce identical starting positions."""
        # Create two samplers with same seed
        sampler1 = MarketDataObservationSampler(
            df=large_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
            max_traj_length=100,
            seed=42,
        )
        sampler2 = MarketDataObservationSampler(
            df=large_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
            max_traj_length=100,
            seed=42,
        )

        # Reset with random start multiple times
        positions1 = []
        positions2 = []
        for _ in range(10):
            sampler1.reset(random_start=True)
            _, ts1, _ = sampler1.get_sequential_observation()
            positions1.append(ts1)

            sampler2.reset(random_start=True)
            _, ts2, _ = sampler2.get_sequential_observation()
            positions2.append(ts2)

        # All positions should match
        assert positions1 == positions2, "Same seed should produce identical starting positions"

    def test_random_start_different_with_different_seeds(
        self, large_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """Different seeds should produce different starting positions."""
        # Create two samplers with different seeds
        sampler1 = MarketDataObservationSampler(
            df=large_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
            max_traj_length=100,
            seed=42,
        )
        sampler2 = MarketDataObservationSampler(
            df=large_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
            max_traj_length=100,
            seed=99,
        )

        # Reset with random start multiple times
        positions1 = []
        positions2 = []
        for _ in range(10):
            sampler1.reset(random_start=True)
            _, ts1, _ = sampler1.get_sequential_observation()
            positions1.append(ts1)

            sampler2.reset(random_start=True)
            _, ts2, _ = sampler2.get_sequential_observation()
            positions2.append(ts2)

        # At least some positions should differ
        assert positions1 != positions2, "Different seeds should produce different starting positions"


class TestInitialBalanceSamplerSeeding:
    """Tests for InitialBalanceSampler seeding and modern RNG usage."""

    def test_fixed_balance_always_returns_same_value(self):
        """Fixed balance should always return the same value."""
        sampler = InitialBalanceSampler(initial_cash=10000, seed=42)

        # Sample multiple times
        samples = [sampler.sample() for _ in range(10)]

        # All samples should be identical
        assert all(s == 10000.0 for s in samples), "Fixed balance should always return same value"

    def test_range_sampling_reproducible_with_same_seed(self):
        """Same seed should produce identical sampling sequences."""
        sampler1 = InitialBalanceSampler(initial_cash=[5000, 15000], seed=42)
        sampler2 = InitialBalanceSampler(initial_cash=[5000, 15000], seed=42)

        # Sample multiple times from both
        samples1 = [sampler1.sample() for _ in range(20)]
        samples2 = [sampler2.sample() for _ in range(20)]

        # All samples should match
        assert samples1 == samples2, "Same seed should produce identical sampling sequences"

    def test_range_sampling_different_with_different_seeds(self):
        """Different seeds should produce different sampling sequences."""
        sampler1 = InitialBalanceSampler(initial_cash=[5000, 15000], seed=42)
        sampler2 = InitialBalanceSampler(initial_cash=[5000, 15000], seed=99)

        # Sample multiple times from both
        samples1 = [sampler1.sample() for _ in range(20)]
        samples2 = [sampler2.sample() for _ in range(20)]

        # Sequences should differ
        assert samples1 != samples2, "Different seeds should produce different sampling sequences"

    def test_does_not_pollute_global_rng_state(self):
        """InitialBalanceSampler should not affect global NumPy RNG state."""
        # Set global state
        np.random.seed(123)
        baseline = [np.random.random() for _ in range(5)]

        # Reset and create sampler
        np.random.seed(123)
        sampler = InitialBalanceSampler(initial_cash=[5000, 15000], seed=42)
        _ = [sampler.sample() for _ in range(10)]  # Sample multiple times
        after_sampler = [np.random.random() for _ in range(5)]

        # Global state should be unchanged
        assert baseline == after_sampler, "InitialBalanceSampler should not pollute global RNG state"


class TestEnvironmentSeeding:
    """Integration tests for environment-level seeding across all offline environments."""

    @pytest.mark.parametrize("env_class,config_class,config_kwargs", [
        (SeqLongOnlyEnv, SeqLongOnlyEnvConfig, {}),
        (SeqLongOnlySLTPEnv, SeqLongOnlySLTPEnvConfig, LONGONLY_SLTP_CONFIG),
        (LongOnlyOneStepEnv, LongOnlyOneStepEnvConfig, {}),
        (SeqFuturesEnv, SeqFuturesEnvConfig, FUTURES_CONFIG),
        (SeqFuturesSLTPEnv, SeqFuturesSLTPEnvConfig, FUTURES_SLTP_CONFIG),
        (FuturesOneStepEnv, FuturesOneStepEnvConfig, FUTURES_CONFIG),
    ])
    def test_env_reset_reproducible_with_same_seed(
        self, large_ohlcv_df, env_class, config_class, config_kwargs
    ):
        """Environment resets should be reproducible with same seed."""
        # Base config
        base_config = {
            "symbol": "TEST/USD",
            "time_frames": [TimeFrame(1, TimeFrameUnit.Minute)],
            "window_sizes": [10],
            "execute_on": TimeFrame(1, TimeFrameUnit.Minute),
            "initial_cash": 10000,
            "transaction_fee": 0.001,
            "slippage": 0.0,
            "max_traj_length": 100,
            "seed": 42,
            "random_start": True,
        }
        base_config.update(config_kwargs)

        # Create two environments with same seed
        config1 = config_class(**base_config)
        config2 = config_class(**base_config)

        env1 = env_class(large_ohlcv_df, config1, feature_preprocessing_fn=simple_feature_fn)
        env2 = env_class(large_ohlcv_df, config2, feature_preprocessing_fn=simple_feature_fn)

        # Reset multiple times and compare initial observations
        for i in range(5):
            obs1 = env1.reset()
            obs2 = env2.reset()

            # Account states should match
            assert torch.allclose(obs1['account_state'], obs2['account_state']), \
                f"Reset {i}: Account states should match with same seed"

            # Market data should match for all timeframes
            for key in env1.market_data_keys:
                assert torch.allclose(obs1[key], obs2[key]), \
                    f"Reset {i}: Market data '{key}' should match with same seed"

            # Coverage indices should match (if present)
            if "reset_index" in obs1.keys():
                assert obs1["reset_index"] == obs2["reset_index"], \
                    f"Reset {i}: reset_index should match with same seed"
                assert obs1["state_index"] == obs2["state_index"], \
                    f"Reset {i}: state_index should match with same seed"

    @pytest.mark.parametrize("env_class,config_class,config_kwargs", [
        (SeqLongOnlyEnv, SeqLongOnlyEnvConfig, {}),
        (SeqFuturesEnv, SeqFuturesEnvConfig, FUTURES_CONFIG),
    ])
    def test_full_episode_reproducible_with_same_seed(
        self, large_ohlcv_df, env_class, config_class, config_kwargs
    ):
        """Full episode trajectories should be reproducible with same seed."""
        # Base config
        base_config = {
            "symbol": "TEST/USD",
            "time_frames": [TimeFrame(1, TimeFrameUnit.Minute)],
            "window_sizes": [10],
            "execute_on": TimeFrame(1, TimeFrameUnit.Minute),
            "initial_cash": 10000,
            "transaction_fee": 0.001,
            "slippage": 0.0,
            "max_traj_length": 50,  # Shorter for faster test
            "seed": 42,
            "random_start": True,
        }
        base_config.update(config_kwargs)

        # Create two environments with same seed
        config1 = config_class(**base_config)
        config2 = config_class(**base_config)

        env1 = env_class(large_ohlcv_df, config1, feature_preprocessing_fn=simple_feature_fn)
        env2 = env_class(large_ohlcv_df, config2, feature_preprocessing_fn=simple_feature_fn)

        # Run full episodes with same actions
        td1 = env1.reset()
        td2 = env2.reset()

        # Choose a deterministic action sequence
        num_actions = env1.action_spec.space.n
        action_sequence = [i % num_actions for i in range(50)]

        rewards1 = []
        rewards2 = []

        for action in action_sequence:
            # Step both environments
            action_tensor = torch.tensor(action, dtype=torch.long)

            td1.set("action", action_tensor)
            td2.set("action", action_tensor)

            result1 = env1.step(td1)
            result2 = env2.step(td2)

            # Rewards should match
            reward1 = result1['next', 'reward'].item()
            reward2 = result2['next', 'reward'].item()
            rewards1.append(reward1)
            rewards2.append(reward2)

            # Update td for next step
            td1 = result1["next"]
            td2 = result2["next"]

            # Break if either environment is done
            if td1.get('done', False).item() or td2.get('done', False).item():
                # Both should terminate at same time
                assert td1.get('done', False).item() == td2.get('done', False).item(), \
                    "Both environments should terminate at same step"
                break

        # All rewards should match
        assert rewards1 == rewards2, "Full episode rewards should match with same seed"

    @pytest.mark.parametrize("env_class,config_class,config_kwargs", [
        (SeqLongOnlyEnv, SeqLongOnlyEnvConfig, {}),
        (SeqFuturesEnv, SeqFuturesEnvConfig, FUTURES_CONFIG),
    ])
    def test_different_seeds_produce_different_trajectories(
        self, large_ohlcv_df, env_class, config_class, config_kwargs
    ):
        """Different seeds should produce different trajectories."""
        # Base config
        base_config = {
            "symbol": "TEST/USD",
            "time_frames": [TimeFrame(1, TimeFrameUnit.Minute)],
            "window_sizes": [10],
            "execute_on": TimeFrame(1, TimeFrameUnit.Minute),
            "initial_cash": [8000, 12000],  # Use range for more randomness
            "transaction_fee": 0.001,
            "slippage": 0.0,
            "max_traj_length": 100,
            "random_start": True,
        }
        base_config.update(config_kwargs)

        # Create two environments with different seeds
        config1 = config_class(**base_config, seed=42)
        config2 = config_class(**base_config, seed=99)

        env1 = env_class(large_ohlcv_df, config1, feature_preprocessing_fn=simple_feature_fn)
        env2 = env_class(large_ohlcv_df, config2, feature_preprocessing_fn=simple_feature_fn)

        # Collect starting positions from multiple resets
        positions1 = []
        positions2 = []

        for _ in range(10):
            obs1 = env1.reset()
            obs2 = env2.reset()

            # Extract initial cash (first element of account_state)
            cash1 = obs1['account_state'][0].item()
            cash2 = obs2['account_state'][0].item()

            positions1.append(cash1)
            positions2.append(cash2)

        # At least some initial conditions should differ
        assert positions1 != positions2, "Different seeds should produce different trajectories"
