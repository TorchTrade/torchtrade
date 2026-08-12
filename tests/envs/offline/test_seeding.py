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
from torchtrade.envs.offline.sequential import SequentialTradingEnv, SequentialTradingEnvConfig
from torchtrade.envs.offline.sequential_sltp import SequentialTradingEnvSLTP, SequentialTradingEnvSLTPConfig
from torchtrade.envs.offline.onestep import OneStepTradingEnv, OneStepTradingEnvConfig
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

    def test_float_initial_cash_works(self):
        """Float initial_cash should not crash."""
        sampler = InitialBalanceSampler(initial_cash=10000.0, seed=42)
        result = sampler.sample()
        assert result == 10000.0

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
        (SequentialTradingEnv, SequentialTradingEnvConfig, {}),
        (SequentialTradingEnvSLTP, SequentialTradingEnvSLTPConfig, LONGONLY_SLTP_CONFIG),
        (OneStepTradingEnv, OneStepTradingEnvConfig, {}),
        (SequentialTradingEnv, SequentialTradingEnvConfig, FUTURES_CONFIG),
        (SequentialTradingEnvSLTP, SequentialTradingEnvSLTPConfig, FUTURES_SLTP_CONFIG),
        (OneStepTradingEnv, OneStepTradingEnvConfig, FUTURES_CONFIG),
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
        (SequentialTradingEnv, SequentialTradingEnvConfig, {}),
        (SequentialTradingEnv, SequentialTradingEnvConfig, FUTURES_CONFIG),
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
        (SequentialTradingEnv, SequentialTradingEnvConfig, {}),
        (SequentialTradingEnv, SequentialTradingEnvConfig, FUTURES_CONFIG),
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

        # Collect starting conditions from multiple resets
        balances1 = []
        balances2 = []

        for _ in range(10):
            obs1 = env1.reset()
            obs2 = env2.reset()

            # Extract initial balance (stored in env.balance, not in account_state)
            # Account state has exposure_pct at index 0, not cash
            cash1 = env1.balance
            cash2 = env2.balance

            balances1.append(cash1)
            balances2.append(cash2)

        # At least some initial conditions should differ due to different seeds
        assert balances1 != balances2, "Different seeds should produce different trajectories"


def test_initial_cash_randomisation_can_draw_its_configured_maximum():
    """np_rng.integers is half-open, so the configured max was unreachable (#290).

    A user asking for [1000, 2000] got [1000, 1999]. Invisible at wide ranges, and the
    whole point of the parameter at narrow ones.
    """
    drawn = {int(InitialBalanceSampler([1000, 1005], seed=s).sample()) for s in range(200)}
    assert drawn == {1000, 1001, 1002, 1003, 1004, 1005}, (
        f"drew {sorted(drawn)} -- the configured bounds must both be reachable"
    )


class TestSetSeedReachesTheEpisodeRNGs:
    """#273: set_seed() reseeded the global streams, which decide nothing about an episode.

    The two RNGs that determine what an episode IS -- the sampler's start index and the
    initial cash -- are built once from `config.seed` and were never touched again. The
    consequence was not "set_seed is a no-op": SerialEnv/ParallelEnv seed their workers
    `seed, seed+1, ...` precisely to decorrelate them, so every worker replayed the same
    start indices and the same starting cash. A batch of N workers had a diversity of 1.
    """

    @staticmethod
    def _episodes(df, seed, k=6):
        config = SequentialTradingEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=(5000, 15000),
            max_traj_length=20,
            slippage=0.01,
        )
        env = SequentialTradingEnv(df.copy(), config, feature_preprocessing_fn=simple_feature_fn)
        env.set_seed(seed)
        out = []
        for _ in range(k):
            env.reset()
            out.append((int(env.sampler._sequential_idx), float(env.balance)))
        return out

    def test_different_seeds_give_different_episodes(self, large_ohlcv_df):
        """The measurement from the issue: these two were byte-identical."""
        assert self._episodes(large_ohlcv_df, 1) != self._episodes(large_ohlcv_df, 999)

    def test_adjacent_seeds_give_different_episodes(self, large_ohlcv_df):
        """Adjacent, because that is what ParallelEnv hands its workers.

        A fix that only decorrelates distant seeds would leave the actual failure -- N
        workers collecting the same trajectories -- exactly as broken.
        """
        assert self._episodes(large_ohlcv_df, 1) != self._episodes(large_ohlcv_df, 2)

    def test_same_seed_still_reproduces(self, large_ohlcv_df):
        """The point of the fix is reproducibility, not merely difference."""
        assert self._episodes(large_ohlcv_df, 7) == self._episodes(large_ohlcv_df, 7)

    def _rng_states(self, df, seed):
        config = SequentialTradingEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=(5000, 15000),
        )
        env = SequentialTradingEnv(df.copy(), config, feature_preprocessing_fn=simple_feature_fn)
        env.set_seed(seed)
        return (
            env.sampler.np_rng.bit_generator.state,
            env.initial_cash_sampler.np_rng.bit_generator.state,
        )

    def test_the_two_episode_streams_are_independent(self, large_ohlcv_df):
        """Asserted on the bit-generator state, because the drawn VALUES hide this.

        Point both RNGs at one seed and the start index and the cash still look unrelated
        -- they draw different ranges off the same bits -- so a test comparing episodes
        passes while the two quantities are locked together. Comparing the streams
        themselves is the only way to see it.
        """
        sampler_state, cash_state = self._rng_states(large_ohlcv_df, 5)
        assert sampler_state != cash_state

    def test_streams_do_not_collide_across_adjacent_workers(self, large_ohlcv_df):
        """ParallelEnv hands worker N `seed + N`, so a fixed per-stream offset re-couples
        them: with `seed` / `seed + 1`, worker 1's cash stream IS worker 2's start-index
        stream. SeedSequence.spawn is what avoids that."""
        w1_sampler, w1_cash = self._rng_states(large_ohlcv_df, 1)
        w2_sampler, w2_cash = self._rng_states(large_ohlcv_df, 2)

        assert w1_cash != w2_sampler
        assert w1_sampler != w2_cash
        assert w1_sampler != w2_sampler

    def test_slippage_does_not_consume_the_global_torch_stream(self, large_ohlcv_df):
        """Slippage shared the global stream with the policy's own exploration sampling.

        So what a saved seed reproduced depended on how many times the policy had sampled
        in between. The control is drawn from the same seed WITHOUT an env step in the
        way: if stepping consumes global entropy, the second draw diverges from it.
        """
        config = SequentialTradingEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=10000,
            slippage=0.01,
            leverage=2,
            action_levels=[-1, 0, 1],
        )

        torch.manual_seed(1234)
        control = torch.rand(3)

        env = SequentialTradingEnv(
            large_ohlcv_df.copy(), config, feature_preprocessing_fn=simple_feature_fn
        )
        env.set_seed(3)
        torch.manual_seed(1234)
        td = env.reset()
        td["action"] = torch.tensor(2)
        env.step(td)
        after = torch.rand(3)

        assert torch.equal(control, after), (
            "stepping the env moved the global torch stream, so slippage still shares it "
            "with the policy's exploration"
        )

    def test_seedless_call_falls_back_to_the_configured_seed(self, large_ohlcv_df):
        """`set_seed()` with no argument is documented to fall back to `config.seed`.

        The first version of this override returned early on None and silently dropped
        that contract -- making the no-argument form a true no-op, which is the bug the
        override exists to fix, reintroduced for one call shape.
        """
        config = SequentialTradingEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            initial_cash=(5000, 15000),
            seed=4242,
        )

        def episodes():
            env = SequentialTradingEnv(
                large_ohlcv_df.copy(), config, feature_preprocessing_fn=simple_feature_fn
            )
            env.set_seed()
            out = []
            for _ in range(4):
                env.reset()
                out.append((int(env.sampler._sequential_idx), float(env.balance)))
            return out

        # Reproducible, and reaching the episode RNGs rather than only the globals.
        assert episodes() == episodes()
        explicit = SequentialTradingEnv(
            large_ohlcv_df.copy(), config, feature_preprocessing_fn=simple_feature_fn
        )
        explicit.set_seed(4242)
        explicit.reset()
        assert int(explicit.sampler._sequential_idx) == episodes()[0][0]
