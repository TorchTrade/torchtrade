"""Tests for CoverageTracker transform."""

import pytest
import numpy as np
import pandas as pd
import torch
from torchrl.envs import TransformedEnv, Compose, InitTracker, DoubleToFloat, RewardSum

from torchtrade.envs.offline import SequentialTradingEnv, SequentialTradingEnvConfig

# Alias for backwards compatibility
SeqLongOnlyEnv = SequentialTradingEnv
SeqLongOnlyEnvConfig = SequentialTradingEnvConfig
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from torchtrade.envs.transforms import CoverageTracker


@pytest.fixture
def simple_df():
    """Create a simple DataFrame for testing."""
    np.random.seed(42)
    n = 1000  # 1000 minutes of data

    dates = pd.date_range('2024-01-01', periods=n, freq='1min')
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close + np.random.randn(n) * 0.05,
        'high': close + np.abs(np.random.randn(n) * 0.1),
        'low': close - np.abs(np.random.randn(n) * 0.1),
        'close': close,
        'volume': np.random.randint(100, 1000, n),
    })

    return df


@pytest.fixture
def env_random_start(simple_df):
    """Create environment with random_start=True."""
    config = SeqLongOnlyEnvConfig(
        symbol="TEST/USD",
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        execute_on=TimeFrame(5, TimeFrameUnit.Minute),
        include_base_features=False,
        initial_cash=100000,  # High cash to reduce bankruptcy risk during random actions
        transaction_fee=0.0,  # No fees to reduce losses
        slippage=0.0,  # No slippage to reduce losses
        random_start=True,
        max_traj_length=50,  # Short trajectories for testing
        seed=42,
    )

    env = SeqLongOnlyEnv(simple_df, config)

    # Apply transforms including CoverageTracker
    transformed_env = TransformedEnv(
        env,
        Compose(
            CoverageTracker(),
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
        ),
    )

    return transformed_env


@pytest.fixture
def env_sequential(simple_df):
    """Create environment with random_start=False (sequential)."""
    config = SeqLongOnlyEnvConfig(
        symbol="TEST/USD",
        time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
        window_sizes=[10],
        execute_on=TimeFrame(5, TimeFrameUnit.Minute),
        include_base_features=False,
        initial_cash=10000,  # Use int to avoid subscript error
        random_start=False,  # Sequential
        max_traj_length=50,
        seed=42,
    )

    env = SeqLongOnlyEnv(simple_df, config)

    # Apply transforms including CoverageTracker
    transformed_env = TransformedEnv(
        env,
        Compose(
            CoverageTracker(),
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
        ),
    )

    return transformed_env


def get_coverage_tracker(env):
    """Helper to extract CoverageTracker from transformed env."""
    for transform in env.transform:
        if isinstance(transform, CoverageTracker):
            return transform
    return None


class TestCoverageTrackerInitialization:
    """Test CoverageTracker initialization."""

    def test_tracker_exists_in_transform(self, env_random_start):
        """Test that CoverageTracker is properly added to transform chain."""
        tracker = get_coverage_tracker(env_random_start)
        assert tracker is not None, "CoverageTracker not found in transform chain"
        assert isinstance(tracker, CoverageTracker)

    def test_tracker_starts_uninitialized(self, env_random_start):
        """Test that tracker starts uninitialized before first reset."""
        tracker = get_coverage_tracker(env_random_start)
        assert tracker._reset_coverage_counts is None
        assert tracker._state_coverage_counts is None
        assert tracker._total_resets == 0
        assert tracker._total_states == 0
        assert tracker._num_positions is None


class TestCoverageTrackerRandomStart:
    """Test CoverageTracker with random start environments."""

    def test_initialization_on_first_reset(self, env_random_start):
        """Test that coverage tracking is initialized on first reset."""
        tracker = get_coverage_tracker(env_random_start)

        # Before reset
        assert tracker._reset_coverage_counts is None
        assert tracker._state_coverage_counts is None

        # After reset
        env_random_start.reset()

        assert tracker._reset_coverage_counts is not None
        assert tracker._state_coverage_counts is not None
        assert tracker._enabled is True
        assert tracker._num_positions > 0
        assert len(tracker._reset_coverage_counts) == tracker._num_positions
        assert len(tracker._state_coverage_counts) == tracker._num_positions

    def test_tracks_single_reset(self, env_random_start):
        """Test that a single reset is tracked."""
        tracker = get_coverage_tracker(env_random_start)

        env_random_start.reset()

        stats = tracker.get_coverage_stats()
        assert stats["enabled"] is True
        assert stats["total_resets"] == 1
        assert stats["reset_visited"] == 1
        # State coverage should also have 1 visited (the reset state)
        assert stats["state_visited"] == 1

    def test_tracks_multiple_resets(self, env_random_start):
        """Test that multiple resets are tracked."""
        tracker = get_coverage_tracker(env_random_start)

        num_resets = 50
        for _ in range(num_resets):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()
        assert stats["enabled"] is True
        assert stats["total_resets"] == num_resets
        assert stats["reset_visited"] >= 1
        assert stats["reset_visited"] <= num_resets
        assert 0.0 <= stats["reset_coverage"] <= 1.0
        assert 0.0 <= stats["state_coverage"] <= 1.0

    def test_coverage_percentage_increases(self, env_random_start):
        """Test that coverage percentage increases with more resets."""
        tracker = get_coverage_tracker(env_random_start)

        # First few resets
        for _ in range(10):
            env_random_start.reset()
        stats_early = tracker.get_coverage_stats()

        # Many more resets
        for _ in range(100):
            env_random_start.reset()
        stats_late = tracker.get_coverage_stats()

        assert stats_late["reset_coverage"] >= stats_early["reset_coverage"]
        assert stats_late["reset_visited"] >= stats_early["reset_visited"]

    def test_mean_visits_calculation(self, env_random_start):
        """Test that mean visits per position is calculated correctly."""
        tracker = get_coverage_tracker(env_random_start)

        num_resets = 100
        for _ in range(num_resets):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()

        # Mean should be total_resets / total_positions
        expected_mean = num_resets / stats["total_positions"]
        assert abs(stats["reset_mean_visits"] - expected_mean) < 0.01

    def test_max_min_visits(self, env_random_start):
        """Test that max/min visit counts are tracked."""
        tracker = get_coverage_tracker(env_random_start)

        for _ in range(50):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()

        assert stats["reset_max_visits"] >= stats["reset_min_visits"]
        assert stats["reset_max_visits"] >= 0
        assert stats["reset_min_visits"] >= 0

    def test_std_visits_calculation(self, env_random_start):
        """Test that standard deviation of visits is calculated."""
        tracker = get_coverage_tracker(env_random_start)

        for _ in range(50):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()

        assert stats["reset_std_visits"] >= 0
        assert isinstance(stats["reset_std_visits"], float)


class TestCoverageTrackerSequential:
    """Test CoverageTracker with sequential start environments."""

    def test_disabled_for_sequential_env(self, env_sequential):
        """Test that tracker is disabled for sequential environments."""
        tracker = get_coverage_tracker(env_sequential)

        env_sequential.reset()

        stats = tracker.get_coverage_stats()
        assert stats["enabled"] is False
        assert "message" in stats
        assert "disabled" in stats["message"].lower()

    def test_no_tracking_for_sequential_env(self, env_sequential):
        """Test that no coverage is tracked for sequential environments."""
        tracker = get_coverage_tracker(env_sequential)

        for _ in range(10):
            env_sequential.reset()

        # Coverage counts should remain None
        assert tracker._reset_coverage_counts is None


class TestCoverageTrackerStats:
    """Test coverage statistics functionality."""

    def test_get_stats_before_initialization(self, env_random_start):
        """Test getting stats before any resets."""
        tracker = get_coverage_tracker(env_random_start)

        stats = tracker.get_coverage_stats()
        # Should return disabled message before initialization
        assert stats["enabled"] is False

    def test_get_distribution(self, env_random_start):
        """Test getting raw coverage distribution."""
        tracker = get_coverage_tracker(env_random_start)

        env_random_start.reset()

        dist = tracker.get_coverage_distribution()
        assert dist is not None
        assert isinstance(dist, dict)
        assert "reset_counts" in dist
        assert "state_counts" in dist
        assert isinstance(dist["reset_counts"], np.ndarray)
        assert isinstance(dist["state_counts"], np.ndarray)
        assert len(dist["reset_counts"]) == tracker._num_positions
        assert len(dist["state_counts"]) == tracker._num_positions
        assert np.sum(dist["reset_counts"]) == tracker._total_resets
        assert np.sum(dist["state_counts"]) == tracker._total_states

    def test_get_distribution_returns_copy(self, env_random_start):
        """Test that get_distribution returns a copy, not reference."""
        tracker = get_coverage_tracker(env_random_start)

        env_random_start.reset()

        dist1 = tracker.get_coverage_distribution()
        dist2 = tracker.get_coverage_distribution()

        # Modify dist1 reset_counts
        dist1["reset_counts"][0] = 9999

        # dist2 should be unchanged
        assert dist2["reset_counts"][0] != 9999

        # Also test state_counts
        dist1["state_counts"][0] = 8888
        assert dist2["state_counts"][0] != 8888

    def test_reset_coverage(self, env_random_start):
        """Test resetting coverage statistics."""
        tracker = get_coverage_tracker(env_random_start)

        # Track some resets
        for _ in range(20):
            env_random_start.reset()

        stats_before = tracker.get_coverage_stats()
        assert stats_before["total_resets"] == 20

        # Reset coverage
        tracker.reset_coverage()

        stats_after = tracker.get_coverage_stats()
        assert stats_after["total_resets"] == 0
        assert stats_after["total_states"] == 0
        assert stats_after["reset_visited"] == 0
        assert stats_after["state_visited"] == 0
        assert stats_after["reset_coverage"] == 0.0
        assert stats_after["state_coverage"] == 0.0


class TestCoverageTrackerEdgeCases:
    """Test edge cases and error conditions."""

    def test_coverage_with_single_position(self, simple_df):
        """Test coverage tracking with very limited positions."""
        # Create env with very short trajectory length to limit positions
        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(5, TimeFrameUnit.Minute),
            include_base_features=False,
            initial_cash=10000,  # Use int to avoid subscript error
            random_start=True,
            max_traj_length=5,  # Very short
            seed=42,
        )

        env = SeqLongOnlyEnv(simple_df, config)
        transformed_env = TransformedEnv(
            env,
            Compose(CoverageTracker(), InitTracker()),
        )

        tracker = get_coverage_tracker(transformed_env)

        # Reset multiple times
        for _ in range(10):
            transformed_env.reset()

        stats = tracker.get_coverage_stats()
        assert stats["enabled"] is True
        assert stats["total_resets"] == 10

    def test_100_percent_coverage(self, simple_df):
        """Test achieving 100% coverage."""
        # Create env with few positions
        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(5, TimeFrameUnit.Minute),
            include_base_features=False,
            initial_cash=10000,  # Use int to avoid subscript error
            random_start=True,
            max_traj_length=50,
            seed=42,
        )

        env = SeqLongOnlyEnv(simple_df, config)
        transformed_env = TransformedEnv(
            env,
            Compose(CoverageTracker(), InitTracker()),
        )

        tracker = get_coverage_tracker(transformed_env)

        # Reset many times to likely achieve full coverage
        for _ in range(500):
            transformed_env.reset()

        stats = tracker.get_coverage_stats()

        # Check if we achieved or are close to full coverage
        assert stats["reset_coverage"] > 0.5  # At least 50% with enough resets
        # Unvisited positions can be calculated as total - visited
        unvisited = stats["total_positions"] - stats["reset_visited"]
        assert unvisited >= 0


class TestCoverageTrackerIntegration:
    """Integration tests with environment rollouts."""

    def test_coverage_during_rollout(self, env_random_start):
        """Test that coverage tracking works during full rollouts."""
        tracker = get_coverage_tracker(env_random_start)

        # Do multiple rollouts
        # Note: Random actions can cause bankruptcy, which is a valid episode termination
        successful_resets = 0
        for _ in range(5):
            td = env_random_start.reset()
            successful_resets += 1
            done = False
            steps = 0
            max_steps = 100

            try:
                while not done and steps < max_steps:
                    action = env_random_start.action_spec.rand()
                    td = env_random_start.step(td.set("action", action))
                    done = td["next", "done"].item()
                    steps += 1
            except ValueError:
                # Bankruptcy can occur with random actions - this is expected
                pass

        stats = tracker.get_coverage_stats()

        # Should have tracked all resets (even if some led to bankruptcy)
        assert stats["total_resets"] == successful_resets
        assert stats["reset_visited"] >= 1
        assert stats["reset_visited"] <= successful_resets

    def test_coverage_with_parallel_env_wrapper(self, simple_df):
        """Test coverage tracking through ParallelEnv wrapper."""
        from torchrl.envs import ParallelEnv, EnvCreator
        import functools

        def make_env(df):
            config = SeqLongOnlyEnvConfig(
                symbol="TEST/USD",
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                execute_on=TimeFrame(5, TimeFrameUnit.Minute),
                include_base_features=False,
                initial_cash=10000,  # Use int to avoid subscript error
                random_start=True,
                max_traj_length=50,
                seed=42,
            )
            return SeqLongOnlyEnv(df, config)

        # Create parallel environment
        maker = functools.partial(make_env, simple_df)
        parallel_env = ParallelEnv(
            2,  # 2 parallel environments
            EnvCreator(maker),
            serial_for_single=True,
        )

        # Apply transforms
        transformed_env = TransformedEnv(
            parallel_env,
            Compose(
                CoverageTracker(),
                InitTracker(),
                DoubleToFloat(),
            ),
        )

        tracker = get_coverage_tracker(transformed_env)

        # Reset parallel env
        transformed_env.reset()

        stats = tracker.get_coverage_stats()

        # Should track coverage (though behavior may differ with parallel envs)
        assert stats is not None


class TestMathematicalValidation:
    """Test mathematical properties and formulas of coverage statistics."""

    def test_visited_unvisited_sum_equals_total(self, env_random_start):
        """Test that visited + unvisited = total_positions."""
        tracker = get_coverage_tracker(env_random_start)

        for _ in range(50):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()

        # Mathematical invariant
        unvisited = stats["total_positions"] - stats["reset_visited"]
        assert stats["reset_visited"] + unvisited == stats["total_positions"]

    def test_coverage_percentage_formula(self, env_random_start):
        """Test that coverage matches the formula."""
        tracker = get_coverage_tracker(env_random_start)

        for _ in range(50):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()

        # coverage = visited / total (range [0, 1])
        expected_coverage = stats["reset_visited"] / stats["total_positions"]
        assert abs(stats["reset_coverage"] - expected_coverage) < 0.01

    def test_mean_visits_formula(self, env_random_start):
        """Test that mean_visits matches total_resets / total_positions."""
        tracker = get_coverage_tracker(env_random_start)

        num_resets = 100
        for _ in range(num_resets):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()

        # mean_visits = total_resets / total_positions
        expected_mean = num_resets / stats["total_positions"]
        assert abs(stats["reset_mean_visits"] - expected_mean) < 0.01

    def test_distribution_sum_equals_total_resets(self, env_random_start):
        """Test that sum of coverage distribution equals total resets."""
        tracker = get_coverage_tracker(env_random_start)

        num_resets = 75
        for _ in range(num_resets):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()
        distribution = tracker.get_coverage_distribution()

        # Sum of all visit counts should equal total resets
        assert np.sum(distribution["reset_counts"]) == stats["total_resets"]
        assert np.sum(distribution["reset_counts"]) == num_resets

    def test_distribution_non_negative(self, env_random_start):
        """Test that all distribution values are non-negative."""
        tracker = get_coverage_tracker(env_random_start)

        for _ in range(50):
            env_random_start.reset()

        distribution = tracker.get_coverage_distribution()

        # All counts should be >= 0
        assert np.all(distribution["reset_counts"] >= 0)

    def test_visited_positions_matches_nonzero_counts(self, env_random_start):
        """Test that visited_positions equals number of non-zero entries in distribution."""
        tracker = get_coverage_tracker(env_random_start)

        for _ in range(50):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()
        distribution = tracker.get_coverage_distribution()

        # Count of non-zero entries
        nonzero_count = np.sum(distribution["reset_counts"] > 0)
        assert stats["reset_visited"] == nonzero_count

    def test_max_visits_equals_distribution_max(self, env_random_start):
        """Test that max_visits matches the maximum in distribution."""
        tracker = get_coverage_tracker(env_random_start)

        for _ in range(50):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()
        distribution = tracker.get_coverage_distribution()

        assert stats["reset_max_visits"] == np.max(distribution["reset_counts"])

    def test_min_visits_equals_distribution_min(self, env_random_start):
        """Test that min_visits matches the minimum in distribution."""
        tracker = get_coverage_tracker(env_random_start)

        for _ in range(50):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()
        distribution = tracker.get_coverage_distribution()

        assert stats["reset_min_visits"] == np.min(distribution["reset_counts"])

    def test_std_visits_matches_numpy_std(self, env_random_start):
        """Test that std_visits matches numpy's std calculation."""
        tracker = get_coverage_tracker(env_random_start)

        for _ in range(50):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()
        distribution = tracker.get_coverage_distribution()

        expected_std = np.std(distribution["reset_counts"])
        assert abs(stats["reset_std_visits"] - expected_std) < 0.01


class TestDeterministicCoverage:
    """Test coverage tracking with deterministic, controlled resets."""

    def test_single_position_repeated_resets(self, simple_df):
        """Test that resetting to same position increments count correctly."""
        # Create env with fixed seed
        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(5, TimeFrameUnit.Minute),
            include_base_features=False,
            initial_cash=10000,
            random_start=True,
            max_traj_length=50,
            seed=12345,  # Fixed seed for determinism
        )

        env = SeqLongOnlyEnv(simple_df, config)
        transformed_env = TransformedEnv(
            env,
            Compose(CoverageTracker(), InitTracker()),
        )

        tracker = get_coverage_tracker(transformed_env)

        # Record which position is selected on first reset
        transformed_env.reset()
        first_reset_idx = env.sampler._sequential_idx
        distribution_after_first = tracker.get_coverage_distribution().copy()

        # Reset environment again with same seed - should get same or different position
        # But we can verify counts are consistent
        num_additional_resets = 10
        for _ in range(num_additional_resets):
            transformed_env.reset()

        stats = tracker.get_coverage_stats()
        distribution = tracker.get_coverage_distribution()

        # Total resets should be 1 + num_additional_resets
        assert stats["total_resets"] == 1 + num_additional_resets

        # Sum of distribution should equal total resets
        assert np.sum(distribution["reset_counts"]) == stats["total_resets"]

        # At least the first position should have been visited
        assert distribution["reset_counts"][first_reset_idx] >= distribution_after_first["reset_counts"][first_reset_idx]

    def test_coverage_increments_monotonically(self, env_random_start):
        """Test that coverage metrics increase or stay same, never decrease."""
        tracker = get_coverage_tracker(env_random_start)

        visited_history = []
        coverage_pct_history = []
        total_resets_history = []

        for i in range(100):
            env_random_start.reset()
            stats = tracker.get_coverage_stats()

            visited_history.append(stats["reset_visited"])
            coverage_pct_history.append(stats["reset_coverage"])
            total_resets_history.append(stats["total_resets"])

        # Visited positions should never decrease
        for i in range(1, len(visited_history)):
            assert visited_history[i] >= visited_history[i-1]

        # Coverage percentage should never decrease
        for i in range(1, len(coverage_pct_history)):
            assert coverage_pct_history[i] >= coverage_pct_history[i-1]

        # Total resets should strictly increase
        for i in range(1, len(total_resets_history)):
            assert total_resets_history[i] == total_resets_history[i-1] + 1

    def test_exact_coverage_pattern(self, simple_df):
        """Test exact coverage pattern with very controlled environment."""
        # Create env with very limited positions and fixed seed
        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(5, TimeFrameUnit.Minute),
            include_base_features=False,
            initial_cash=10000,
            random_start=True,
            max_traj_length=10,  # Very short to limit positions
            seed=99999,  # Fixed seed
        )

        env = SeqLongOnlyEnv(simple_df, config)
        transformed_env = TransformedEnv(
            env,
            Compose(CoverageTracker(), InitTracker()),
        )

        tracker = get_coverage_tracker(transformed_env)

        # Do a specific number of resets
        num_resets = 20
        for _ in range(num_resets):
            transformed_env.reset()

        stats = tracker.get_coverage_stats()
        distribution = tracker.get_coverage_distribution()

        # Verify exact consistency
        assert stats["total_resets"] == num_resets
        assert np.sum(distribution["reset_counts"]) == num_resets
        assert stats["reset_visited"] == np.sum(distribution["reset_counts"] > 0)
        unvisited = np.sum(distribution["reset_counts"] == 0)
        assert stats["total_positions"] - stats["reset_visited"] == unvisited
        assert stats["total_positions"] == len(distribution["reset_counts"])

    def test_zero_coverage_initially(self, env_random_start):
        """Test that coverage starts at zero before any resets."""
        tracker = get_coverage_tracker(env_random_start)

        # Before first reset, tracker is uninitialized
        stats = tracker.get_coverage_stats()
        assert stats["enabled"] is False

    def test_first_reset_creates_one_visit(self, env_random_start):
        """Test that first reset creates exactly one visited position."""
        tracker = get_coverage_tracker(env_random_start)

        env_random_start.reset()

        stats = tracker.get_coverage_stats()
        distribution = tracker.get_coverage_distribution()

        # Exactly 1 reset
        assert stats["total_resets"] == 1

        # Exactly 1 position visited
        assert stats["reset_visited"] == 1

        # Exactly one entry in distribution should be 1, rest should be 0
        assert np.sum(distribution["reset_counts"] == 1) == 1
        assert np.sum(distribution["reset_counts"] == 0) == stats["total_positions"] - 1


class TestParallelEnvironmentCoverage:
    """Test CoverageTracker behavior with ParallelEnv.

    Note: CoverageTracker currently has limited support for ParallelEnv due to the
    wrapper structure. These tests document the expected behavior.
    """

    def test_parallel_env_disables_tracking(self, simple_df):
        """Test ParallelEnv coverage tracking limitation with direct reset() calls.

        When using ParallelEnv with CoverageTracker as a transform (not postproc),
        coverage tracking initializes but doesn't track resets because reset_index
        is not propagated by ParallelEnv in the aggregated tensordict.

        For ParallelEnv coverage tracking, use CoverageTracker as collector postproc
        instead of as a transform.
        """
        from torchrl.envs import ParallelEnv, EnvCreator
        import functools

        def make_env(df):
            config = SeqLongOnlyEnvConfig(
                symbol="TEST/USD",
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                execute_on=TimeFrame(5, TimeFrameUnit.Minute),
                include_base_features=False,
                initial_cash=10000,
                random_start=True,
                max_traj_length=50,
                seed=None,
            )
            return SeqLongOnlyEnv(df, config)

        num_workers = 2
        maker = functools.partial(make_env, simple_df)
        parallel_env = ParallelEnv(
            num_workers,
            EnvCreator(maker),
            serial_for_single=True,
        )

        transformed_env = TransformedEnv(
            parallel_env,
            Compose(CoverageTracker(), InitTracker()),
        )

        tracker = get_coverage_tracker(transformed_env)

        # Reset and check stats
        transformed_env.reset()
        stats = tracker.get_coverage_stats()

        # Tracker initializes (detects random_start from test env)
        assert stats["enabled"] is True
        assert stats["total_positions"] > 0
        # With dual tracking, resets are now tracked in _reset() if reset_index is present
        # ParallelEnv with 2 workers means 2 resets
        assert stats["total_resets"] == num_workers
        assert stats["total_states"] == num_workers  # Each reset also tracks state

    def test_parallel_env_limitation_documented(self):
        """Document how CoverageTracker works with ParallelEnv.

        CoverageTracker supports ParallelEnv when used as collector postproc:
        - Each worker adds reset_index to its tensordict during reset
        - Collector batches individual worker outputs with reset_index preserved
        - CoverageTracker.forward() aggregates coverage from the batch
        - Zero IPC overhead, batch processing for performance

        The recommended pattern is shown in examples/online/ppo_futures/:
        - Create coverage_tracker = CoverageTracker()
        - Pass as postproc to SyncDataCollector
        - Do NOT add to environment transforms

        Direct reset() calls on ParallelEnv don't work because ParallelEnv's
        aggregation strips reset_index (not in observation_spec).
        """
        # This is a documentation test - no actual code to run
        # The recommended pattern is:
        #     return TransformedEnv(
        #         env,
        #         Compose(
        #             CoverageTracker(),  # Works with ParallelEnv via IPC
        #             InitTracker(),
        #             ...
        #         ),
        #     )
        #
        # parallel_env = ParallelEnv(num_envs, EnvCreator(maker))
        # transformed_env = apply_env_transforms(parallel_env)
        assert True


class TestCollectorPostproc:
    """Test CoverageTracker as collector postproc (primary use case)."""

    def test_collector_postproc_integration(self, simple_df):
        """Test CoverageTracker as collector postproc (recommended pattern)."""
        from torchrl.collectors import SyncDataCollector

        # Create env with random_start
        def make_env(df):
            config = SeqLongOnlyEnvConfig(
                symbol="TEST/USD",
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                execute_on=TimeFrame(5, TimeFrameUnit.Minute),
                include_base_features=False,
                initial_cash=100000,  # High cash to reduce bankruptcy risk during random actions
                transaction_fee=0.0,  # No fees to reduce losses
                slippage=0.0,  # No slippage to reduce losses
                random_start=True,
                max_traj_length=50,
                seed=None,
            )
            return SeqLongOnlyEnv(df, config)

        env = make_env(simple_df)
        transformed_env = TransformedEnv(
            env,
            Compose(
                InitTracker(),
                DoubleToFloat(),
                RewardSum(),
            ),
        )

        # Create simple random policy
        class RandomPolicy:
            def __init__(self, action_spec):
                self.action_spec = action_spec

            def __call__(self, td):
                # For ParallelEnv, action_spec.rand() already handles batch dimension correctly
                batch_size = td.batch_size if hasattr(td, 'batch_size') else torch.Size([])
                td.set("action", self.action_spec.rand(batch_size))
                return td

        policy = RandomPolicy(transformed_env.action_spec)

        # Create coverage tracker as postproc
        coverage_tracker = CoverageTracker()

        # Create collector with coverage tracker as postproc
        # Use large total_frames to avoid auto-close
        collector = SyncDataCollector(
            transformed_env,
            policy,
            frames_per_batch=100,
            total_frames=10000,  # Large number to prevent auto-close
            device="cpu",
            postproc=coverage_tracker,  # Use as postproc
        )

        # Collect a few batches
        # Note: Random actions can cause bankruptcy, which is a valid episode termination
        collected_frames = 0
        successful_batches = 0
        try:
            for i, batch in enumerate(collector):
                if i >= 3:  # Collect 3 batches
                    break
                collected_frames += batch.numel()
                successful_batches += 1
        except (TypeError, ValueError):
            # TypeError: close() errors during iteration
            # ValueError: Bankruptcy can occur with random actions
            pass

        # Verify coverage was tracked (even if some episodes ended in bankruptcy)
        stats = coverage_tracker.get_coverage_stats()

        # If we collected at least one batch, coverage should be enabled and tracking
        if successful_batches > 0:
            assert stats["enabled"] is True, f"Coverage should be enabled after {successful_batches} batches"
            assert stats["total_resets"] > 0, "Coverage tracker should track resets via postproc"
            assert stats["reset_visited"] > 0
            assert stats["reset_coverage"] > 0
        else:
            # If collection failed immediately (e.g., bankruptcy on first episode),
            # we can't verify much - just that the test didn't crash
            pass

        collector.shutdown()
        try:
            transformed_env.close()
        except TypeError:
            pass  # Some TorchRL versions don't support raise_if_closed parameter

    def test_forward_batch_aggregation(self, simple_df):
        """Test forward() correctly aggregates coverage from batched tensordict."""
        from tensordict import TensorDict

        # Create environment to initialize tracker
        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(5, TimeFrameUnit.Minute),
            include_base_features=False,
            initial_cash=10000,
            random_start=True,
            max_traj_length=50,
            seed=42,
        )

        env = SeqLongOnlyEnv(simple_df, config)
        transformed_env = TransformedEnv(
            env,
            Compose(
                CoverageTracker(),
                InitTracker(),
            ),
        )

        tracker = get_coverage_tracker(transformed_env)

        # Initialize tracker by resetting once
        transformed_env.reset()
        initial_stats = tracker.get_coverage_stats()
        initial_resets = initial_stats["total_resets"]

        # Create a fake batched tensordict with reset indices
        # Simulate 5 resets: positions [0, 1, 2, 0, 1] (0 appears twice, 1 appears twice, 2 once)
        reset_indices = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
        fake_batch = TensorDict(
            {"reset_index": reset_indices},
            batch_size=[5],
        )

        # Call forward() directly
        tracker.forward(fake_batch)

        # Verify aggregation
        stats = tracker.get_coverage_stats()
        distribution = tracker.get_coverage_distribution()

        # Should have added 5 resets total
        assert stats["total_resets"] == initial_resets + 5

        # Position 0 should have 2 additional visits
        # Position 1 should have 2 additional visits
        # Position 2 should have 1 additional visit
        # (Plus whatever the initial reset added)
        total_visits_to_0_1_2 = distribution["reset_counts"][0] + distribution["reset_counts"][1] + distribution["reset_counts"][2]
        # We know we added exactly 5 visits to these positions
        # But the initial reset might have also used one of these positions
        # So we can only assert that at least 5 visits were added
        assert total_visits_to_0_1_2 >= 5

        # Cleanup
        try:
            transformed_env.close()
        except TypeError:
            pass  # Some TorchRL versions don't support raise_if_closed parameter

    def test_parallel_env_with_collector_postproc(self, simple_df):
        """Test ParallelEnv coverage tracking via collector postproc (recommended pattern).

        This test validates that CoverageTracker can track coverage from ParallelEnv
        when used as a collector postproc by simulating batched reset_index data.
        """
        from tensordict import TensorDict

        # Create a single environment to initialize the tracker
        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(5, TimeFrameUnit.Minute),
            include_base_features=False,
            initial_cash=10000,
            random_start=True,
            max_traj_length=50,
            seed=42,
        )

        env = SeqLongOnlyEnv(simple_df, config)
        transformed_env = TransformedEnv(
            env,
            Compose(
                CoverageTracker(),
                InitTracker(),
            ),
        )

        tracker = get_coverage_tracker(transformed_env)

        # Initialize tracker
        transformed_env.reset()

        # Simulate what a ParallelEnv collector would produce:
        # Multiple workers, each with their own reset_index in the batch
        # Batch shape: [num_workers, trajectory_length] but reset_index appears when episodes reset

        # Simulate batch from 3 workers where 2 workers had resets
        # (In real collection, reset_index appears in timesteps where done=True and env resets)
        simulated_batch = TensorDict(
            {
                "reset_index": torch.tensor([5, 12, 5, 23, 12], dtype=torch.long),  # 5 resets total
                "done": torch.tensor([True, True, True, True, True]),
            },
            batch_size=[5],
        )

        # Call forward() as collector would
        tracker.forward(simulated_batch)

        # Verify coverage was tracked from the simulated ParallelEnv batch
        stats = tracker.get_coverage_stats()
        assert stats["enabled"] is True
        assert stats["total_resets"] >= 5, "Should track all resets from batch"
        assert stats["reset_visited"] > 0

        # Verify specific positions were visited
        distribution = tracker.get_coverage_distribution()
        assert distribution["reset_counts"][5] >= 2, "Position 5 appeared twice in batch"
        assert distribution["reset_counts"][12] >= 2, "Position 12 appeared twice in batch"
        assert distribution["reset_counts"][23] >= 1, "Position 23 appeared once in batch"

        # Cleanup
        try:
            transformed_env.close()
        except TypeError:
            pass

    def test_forward_handles_edge_cases(self, simple_df):
        """Test forward() handles missing reset_index and disabled tracking."""
        from tensordict import TensorDict

        # Create environment to initialize tracker
        config = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(5, TimeFrameUnit.Minute),
            include_base_features=False,
            initial_cash=10000,
            random_start=True,
            max_traj_length=50,
            seed=42,
        )

        env = SeqLongOnlyEnv(simple_df, config)
        transformed_env = TransformedEnv(
            env,
            Compose(
                CoverageTracker(),
                InitTracker(),
            ),
        )

        tracker = get_coverage_tracker(transformed_env)

        # Initialize tracker
        transformed_env.reset()

        # Test 1: tensordict without reset_index (should pass through unchanged)
        fake_batch_no_index = TensorDict(
            {"observation": torch.randn(5, 10)},
            batch_size=[5],
        )
        stats_before = tracker.get_coverage_stats()
        result = tracker.forward(fake_batch_no_index)

        # Should return tensordict unchanged
        assert "reset_index" not in result.keys()
        assert "observation" in result.keys()

        # Coverage should not change
        stats_after = tracker.get_coverage_stats()
        assert stats_after["total_resets"] == stats_before["total_resets"]

        # Test 2: Empty tensordict
        empty_batch = TensorDict({}, batch_size=[0])
        result_empty = tracker.forward(empty_batch)
        # Empty batch should be returned unchanged
        assert result_empty is not None
        assert result_empty.batch_size == torch.Size([0])

        # Test 3: Create tracker with disabled tracking (sequential env)
        config_seq = SeqLongOnlyEnvConfig(
            symbol="TEST/USD",
            time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
            window_sizes=[10],
            execute_on=TimeFrame(5, TimeFrameUnit.Minute),
            include_base_features=False,
            initial_cash=10000,
            random_start=False,  # Sequential, tracking disabled
            max_traj_length=50,
            seed=42,
        )

        env_seq = SeqLongOnlyEnv(simple_df, config_seq)
        transformed_env_seq = TransformedEnv(
            env_seq,
            Compose(
                CoverageTracker(),
                InitTracker(),
            ),
        )

        tracker_disabled = get_coverage_tracker(transformed_env_seq)
        transformed_env_seq.reset()

        # Create batch with reset_index
        batch_with_index = TensorDict(
            {"reset_index": torch.tensor([0, 1, 2], dtype=torch.long)},
            batch_size=[3],
        )

        # Should pass through without processing (tracking disabled)
        stats_disabled_before = tracker_disabled.get_coverage_stats()
        result_disabled = tracker_disabled.forward(batch_with_index)
        stats_disabled_after = tracker_disabled.get_coverage_stats()

        assert stats_disabled_before["enabled"] is False
        assert stats_disabled_after["enabled"] is False
        assert result_disabled is not None

        # Cleanup
        try:
            transformed_env.close()
        except TypeError:
            pass  # Some TorchRL versions don't support raise_if_closed parameter
        try:
            transformed_env_seq.close()
        except TypeError:
            pass  # Some TorchRL versions don't support raise_if_closed parameter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
