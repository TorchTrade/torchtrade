"""
Tests for MarketDataObservationSampler.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline.sampler import MarketDataObservationSampler
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit


def simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature processing function for testing."""
    df = df.copy().reset_index(drop=False)
    df["features_close"] = df["close"]
    df["features_volume"] = df["volume"]
    df.dropna(inplace=True)
    return df


class TestSamplerInitialization:
    """Tests for sampler initialization."""

    def test_sampler_initializes_with_valid_data(
        self, sample_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """Sampler should initialize without errors with valid data."""
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
        )
        assert sampler is not None
        assert sampler.max_steps > 0

    def test_sampler_with_feature_processing(
        self, sample_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """Sampler should work with custom feature processing function."""
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
            feature_processing_fn=simple_feature_fn,
            features_start_with="features_",
        )
        feature_keys = sampler.get_feature_keys()
        assert "features_close" in feature_keys
        assert "features_volume" in feature_keys

    def test_sampler_raises_on_mismatched_window_sizes(self, sample_ohlcv_df, execute_timeframe):
        """Sampler should raise error when window_sizes length != time_frames length."""
        time_frames = [
            TimeFrame(1, TimeFrameUnit.Minute),
            TimeFrame(5, TimeFrameUnit.Minute),
        ]
        window_sizes = [10]  # Only one size for two timeframes

        with pytest.raises(ValueError, match="window_sizes must be"):
            MarketDataObservationSampler(
                df=sample_ohlcv_df,
                time_frames=time_frames,
                window_sizes=window_sizes,
                execute_on=execute_timeframe,
            )

    def test_sampler_with_single_timeframe(self, sample_ohlcv_df, execute_timeframe):
        """Sampler should work with a single TimeFrame (not a list)."""
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            execute_on=execute_timeframe,
        )
        assert len(sampler.get_observation_keys()) == 1


class TestSamplerReset:
    """Tests for sampler reset functionality."""

    def test_reset_restores_unseen_timestamps(
        self, sample_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """Reset should restore all unseen timestamps."""
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
        )
        initial_count = sampler.reset(random_start=False)

        # Consume some observations
        for _ in range(10):
            sampler.get_sequential_observation()

        # Reset and verify count is restored
        reset_count = sampler.reset(random_start=False)
        assert reset_count == initial_count

    def test_reset_with_max_traj_length(self, sample_ohlcv_df, execute_timeframe):
        """Reset should respect max_traj_length."""
        max_traj = 100
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            execute_on=execute_timeframe,
            max_traj_length=max_traj,
        )
        count = sampler.reset(random_start=False)
        assert count == max_traj

    def test_reset_with_random_start(
        self, large_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """Reset with random_start should produce varying start positions."""
        sampler = MarketDataObservationSampler(
            df=large_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
            max_traj_length=100,
            seed=42,
        )

        # Get first timestamp after multiple random resets
        first_timestamps = []
        for _ in range(5):
            sampler.reset(random_start=True)
            obs, ts, _ = sampler.get_sequential_observation()
            first_timestamps.append(ts)

        # Should have some variation in start times
        unique_timestamps = len(set(first_timestamps))
        assert unique_timestamps > 1


class TestSamplerObservations:
    """Tests for observation sampling."""

    def test_sequential_observation_shape(
        self, sample_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """Sequential observations should have correct tensor shapes."""
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
        )
        sampler.reset(random_start=False)

        obs, timestamp, truncated = sampler.get_sequential_observation()

        for tf, ws in zip(default_timeframes, default_window_sizes):
            key = tf.obs_key_freq()
            assert key in obs
            assert obs[key].shape[0] == ws
            assert isinstance(obs[key], torch.Tensor)

    def test_sequential_observation_chronological_order(
        self, sample_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """Sequential observations should return timestamps in chronological order."""
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
            max_traj_length=50,
        )
        sampler.reset(random_start=False)

        timestamps = []
        for _ in range(50):
            _, ts, _ = sampler.get_sequential_observation()
            timestamps.append(ts)

        # Verify chronological order
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    def test_truncated_flag_on_last_observation(
        self, sample_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """Truncated flag should be True only on the last observation."""
        max_traj = 10
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
            max_traj_length=max_traj,
        )
        sampler.reset(random_start=False)

        for i in range(max_traj):
            _, _, truncated = sampler.get_sequential_observation()
            if i < max_traj - 1:
                assert not truncated
            else:
                assert truncated

    def test_observation_values_are_finite(
        self, sample_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """All observation values should be finite (no NaN or Inf)."""
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
            max_traj_length=50,
        )
        sampler.reset(random_start=False)

        for _ in range(50):
            obs, _, _ = sampler.get_sequential_observation()
            for key, tensor in obs.items():
                assert torch.isfinite(tensor).all(), f"Non-finite values in {key}"


class TestSamplerBaseFeatures:
    """Tests for base feature retrieval."""

    def test_get_base_features_returns_ohlcv(
        self, sample_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """get_base_features should return OHLCV dict."""
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
        )
        sampler.reset(random_start=False)

        _, timestamp, _ = sampler.get_sequential_observation()
        base_features = sampler.get_base_features(timestamp)

        assert "open" in base_features
        assert "high" in base_features
        assert "low" in base_features
        assert "close" in base_features
        assert "volume" in base_features

    def test_base_features_are_valid_numbers(
        self, sample_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """Base features should be valid positive numbers."""
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
        )
        sampler.reset(random_start=False)

        _, timestamp, _ = sampler.get_sequential_observation()
        base_features = sampler.get_base_features(timestamp)

        assert base_features["open"] > 0
        assert base_features["high"] >= base_features["low"]
        assert base_features["close"] > 0
        assert base_features["volume"] >= 0


class TestSamplerHelperMethods:
    """Tests for helper methods."""

    def test_get_observation_keys(
        self, sample_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """get_observation_keys should return correct timeframe keys."""
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
        )

        keys = sampler.get_observation_keys()
        assert len(keys) == len(default_timeframes)
        assert "1Minute" in keys
        assert "5Minute" in keys

    def test_get_max_steps(
        self, sample_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """get_max_steps should return a positive integer."""
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=default_timeframes,
            window_sizes=default_window_sizes,
            execute_on=execute_timeframe,
        )

        max_steps = sampler.get_max_steps()
        assert isinstance(max_steps, int)
        assert max_steps > 0

    def test_get_max_steps_respects_max_traj_length(self, sample_ohlcv_df, execute_timeframe):
        """get_max_steps should not exceed max_traj_length."""
        max_traj = 50
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            execute_on=execute_timeframe,
            max_traj_length=max_traj,
        )

        assert sampler.get_max_steps() <= max_traj
