"""Tests for CoverageTracker transform."""

import pytest
import numpy as np
import pandas as pd
import torch
from torchrl.envs import TransformedEnv, Compose, InitTracker, DoubleToFloat, RewardSum

from torchtrade.envs.offline.seqlongonly import SeqLongOnlyEnv, SeqLongOnlyEnvConfig
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit
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
        initial_cash=10000,  # Use int to avoid subscript error
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
        assert tracker._coverage_counts is None
        assert tracker._total_resets == 0
        assert tracker._num_positions is None


class TestCoverageTrackerRandomStart:
    """Test CoverageTracker with random start environments."""

    def test_initialization_on_first_reset(self, env_random_start):
        """Test that coverage tracking is initialized on first reset."""
        tracker = get_coverage_tracker(env_random_start)

        # Before reset
        assert tracker._coverage_counts is None

        # After reset
        env_random_start.reset()

        assert tracker._coverage_counts is not None
        assert tracker._enabled is True
        assert tracker._num_positions > 0
        assert len(tracker._coverage_counts) == tracker._num_positions

    def test_tracks_single_reset(self, env_random_start):
        """Test that a single reset is tracked."""
        tracker = get_coverage_tracker(env_random_start)

        env_random_start.reset()

        stats = tracker.get_coverage_stats()
        assert stats["enabled"] is True
        assert stats["total_resets"] == 1
        assert stats["visited_positions"] == 1
        assert stats["unvisited_positions"] == stats["total_positions"] - 1

    def test_tracks_multiple_resets(self, env_random_start):
        """Test that multiple resets are tracked."""
        tracker = get_coverage_tracker(env_random_start)

        num_resets = 50
        for _ in range(num_resets):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()
        assert stats["enabled"] is True
        assert stats["total_resets"] == num_resets
        assert stats["visited_positions"] >= 1
        assert stats["visited_positions"] <= num_resets

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

        assert stats_late["coverage_percentage"] >= stats_early["coverage_percentage"]
        assert stats_late["visited_positions"] >= stats_early["visited_positions"]

    def test_mean_visits_calculation(self, env_random_start):
        """Test that mean visits per position is calculated correctly."""
        tracker = get_coverage_tracker(env_random_start)

        num_resets = 100
        for _ in range(num_resets):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()

        # Mean should be total_resets / total_positions
        expected_mean = num_resets / stats["total_positions"]
        assert abs(stats["mean_visits_per_position"] - expected_mean) < 0.01

    def test_max_min_visits(self, env_random_start):
        """Test that max/min visit counts are tracked."""
        tracker = get_coverage_tracker(env_random_start)

        for _ in range(50):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()

        assert stats["max_visits"] >= stats["min_visits"]
        assert stats["max_visits"] >= 0
        assert stats["min_visits"] >= 0

    def test_std_visits_calculation(self, env_random_start):
        """Test that standard deviation of visits is calculated."""
        tracker = get_coverage_tracker(env_random_start)

        for _ in range(50):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()

        assert stats["std_visits"] >= 0
        assert isinstance(stats["std_visits"], float)

    def test_entropy_calculation(self, env_random_start):
        """Test that coverage entropy is calculated."""
        tracker = get_coverage_tracker(env_random_start)

        for _ in range(50):
            env_random_start.reset()

        stats = tracker.get_coverage_stats()

        # Entropy should be positive for non-uniform distribution
        assert stats["coverage_entropy"] >= 0
        assert isinstance(stats["coverage_entropy"], float)


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
        assert tracker._coverage_counts is None


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
        assert isinstance(dist, np.ndarray)
        assert len(dist) == tracker._num_positions
        assert np.sum(dist) == tracker._total_resets

    def test_get_distribution_returns_copy(self, env_random_start):
        """Test that get_distribution returns a copy, not reference."""
        tracker = get_coverage_tracker(env_random_start)

        env_random_start.reset()

        dist1 = tracker.get_coverage_distribution()
        dist2 = tracker.get_coverage_distribution()

        # Modify dist1
        dist1[0] = 9999

        # dist2 should be unchanged
        assert dist2[0] != 9999

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
        assert stats_after["visited_positions"] == 0
        assert stats_after["coverage_percentage"] == 0.0


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
        assert stats["coverage_percentage"] > 50  # At least 50% with enough resets
        assert stats["unvisited_positions"] >= 0


class TestCoverageTrackerIntegration:
    """Integration tests with environment rollouts."""

    def test_coverage_during_rollout(self, env_random_start):
        """Test that coverage tracking works during full rollouts."""
        tracker = get_coverage_tracker(env_random_start)

        # Do multiple rollouts
        for _ in range(5):
            td = env_random_start.reset()
            done = False
            steps = 0
            max_steps = 100

            while not done and steps < max_steps:
                action = env_random_start.action_spec.rand()
                td = env_random_start.step(td.set("action", action))
                done = td["next", "done"].item()
                steps += 1

        stats = tracker.get_coverage_stats()

        # Should have tracked 5 resets
        assert stats["total_resets"] == 5
        assert stats["visited_positions"] >= 1
        assert stats["visited_positions"] <= 5

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
