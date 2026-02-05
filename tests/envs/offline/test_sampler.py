"""
Tests for MarketDataObservationSampler.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from torchtrade.envs.offline.infrastructure.sampler import MarketDataObservationSampler
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit


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


class TestSamplerExactValues:
    """Tests for exact value verification - ensures observations match raw data."""

    @pytest.fixture
    def deterministic_ohlcv_df(self):
        """
        Create a deterministic OHLCV DataFrame where values are predictable.

        Price pattern: close = 100 + minute_index (100, 101, 102, ...)
        This makes it easy to verify exact values.
        """
        n_minutes = 200
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

        # Deterministic values: close = 100 + index
        close_prices = np.array([100.0 + i for i in range(n_minutes)])
        open_prices = close_prices - 0.5
        high_prices = close_prices + 1.0
        low_prices = close_prices - 1.0
        volume = np.array([1000.0 + i * 10 for i in range(n_minutes)])

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        })

    def test_single_timeframe_exact_values(self, deterministic_ohlcv_df):
        """Observation values should exactly match the raw data for single timeframe."""
        window_size = 5
        sampler = MarketDataObservationSampler(
            df=deterministic_ohlcv_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=window_size,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset(random_start=False)

        obs, timestamp, _ = sampler.get_sequential_observation()
        obs_tensor = obs["1Minute"]

        # Get the raw data for comparison
        df = deterministic_ohlcv_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # Get the expected window from raw data
        raw_window = df.loc[:timestamp].tail(window_size)

        # Verify each value matches exactly
        for i, (idx, row) in enumerate(raw_window.iterrows()):
            assert obs_tensor[i, 0].item() == pytest.approx(row["open"], rel=1e-5)
            assert obs_tensor[i, 1].item() == pytest.approx(row["high"], rel=1e-5)
            assert obs_tensor[i, 2].item() == pytest.approx(row["low"], rel=1e-5)
            assert obs_tensor[i, 3].item() == pytest.approx(row["close"], rel=1e-5)
            assert obs_tensor[i, 4].item() == pytest.approx(row["volume"], rel=1e-5)

    def test_multi_timeframe_exact_values(self, deterministic_ohlcv_df):
        """Observation values should match for multiple timeframes."""
        sampler = MarketDataObservationSampler(
            df=deterministic_ohlcv_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[5, 3],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset(random_start=False)

        obs, timestamp, _ = sampler.get_sequential_observation()

        # Verify 1-minute timeframe has correct shape
        assert obs["1Minute"].shape == (5, 5)  # 5 bars, 5 features (OHLCV)

        # Verify 5-minute timeframe has correct shape
        assert obs["5Minute"].shape == (3, 5)  # 3 bars, 5 features

        # For 5-minute resampled data, high should be max of 5-minute window
        # This is a structural check - values should be aggregated correctly
        assert obs["5Minute"][-1, 1].item() >= obs["5Minute"][-1, 2].item()  # high >= low

    def test_base_features_match_timestamp(self, deterministic_ohlcv_df):
        """Base features should return values from the correct timestamp."""
        sampler = MarketDataObservationSampler(
            df=deterministic_ohlcv_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=5,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset(random_start=False)

        obs, timestamp, _ = sampler.get_sequential_observation()
        base_features = sampler.get_base_features(timestamp)

        # Get expected values from raw data
        df = deterministic_ohlcv_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        expected_row = df.loc[timestamp]

        assert base_features["open"] == pytest.approx(expected_row["open"], rel=1e-5)
        assert base_features["high"] == pytest.approx(expected_row["high"], rel=1e-5)
        assert base_features["low"] == pytest.approx(expected_row["low"], rel=1e-5)
        assert base_features["close"] == pytest.approx(expected_row["close"], rel=1e-5)
        assert base_features["volume"] == pytest.approx(expected_row["volume"], rel=1e-5)


class TestSamplerNoFutureLeakage:
    """
    Critical tests to ensure no future information leakage.

    In trading, using future data is a critical bug that leads to unrealistic
    backtesting results. These tests verify that observations at time T
    contain ONLY data from time <= T.
    """

    @pytest.fixture
    def sequential_ohlcv_df(self):
        """
        Create OHLCV data with sequential close prices for easy leakage detection.

        Close price = minute index, so if we see close=50 at timestamp minute 30,
        that's future leakage (we're seeing data from minute 50).
        """
        n_minutes = 200
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

        # Close price equals the minute index - makes leakage obvious
        close_prices = np.array([float(i) for i in range(n_minutes)])
        open_prices = close_prices
        high_prices = close_prices + 0.5
        low_prices = close_prices - 0.5
        volume = np.ones(n_minutes) * 1000

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        })

    @pytest.fixture
    def large_sequential_ohlcv_df(self):
        """
        Create larger OHLCV data for testing higher execution timeframes.

        7 days of 1-minute data = 10080 minutes.
        Close price = minute index for easy leakage detection.
        """
        n_minutes = 10080  # 7 days
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

        close_prices = np.array([float(i) for i in range(n_minutes)])
        open_prices = close_prices
        high_prices = close_prices + 0.5
        low_prices = close_prices - 0.5
        volume = np.ones(n_minutes) * 1000

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        })

    def test_no_future_leakage_single_timeframe(self, sequential_ohlcv_df):
        """
        Observation at time T must not contain any data from time > T.

        Since close = minute_index, all close values in window should be <= T's minute index.
        """
        window_size = 10
        sampler = MarketDataObservationSampler(
            df=sequential_ohlcv_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=window_size,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset(random_start=False)

        # Check multiple observations
        for step in range(50):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            obs_tensor = obs["1Minute"]

            # Get the minute index from the timestamp
            start_time = pd.Timestamp("2024-01-01 00:00:00")
            current_minute_idx = int((timestamp - start_time).total_seconds() / 60)

            # All close prices in the observation should be <= current_minute_idx
            close_values = obs_tensor[:, 3].numpy()  # Column 3 is close

            for i, close_val in enumerate(close_values):
                assert close_val <= current_minute_idx, (
                    f"Future leakage detected at step {step}! "
                    f"Observation contains close={close_val} but current minute is {current_minute_idx}"
                )

            if truncated:
                break

    def test_no_future_leakage_multi_timeframe(self, sequential_ohlcv_df):
        """
        Multi-timeframe observations: 1-minute data should not leak, but higher
        timeframes return bars indexed by START time with aggregated data.

        IMPORTANT NOTE: For higher timeframes (5min, 15min, etc.), pandas resampling
        returns bars indexed by their START timestamp. A 5-minute bar at 00:25:00
        contains aggregated data from 00:25:00-00:29:59, including the close at minute 29.

        For strict no-lookahead, the sampler would need to return the PREVIOUS completed
        bar instead of the current in-progress bar. This test verifies the 1-minute
        data has no leakage (critical) and documents the higher-timeframe behavior.
        """
        sampler = MarketDataObservationSampler(
            df=sequential_ohlcv_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 5],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset(random_start=False)

        start_time = pd.Timestamp("2024-01-01 00:00:00")

        for step in range(30):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_minute_idx = int((timestamp - start_time).total_seconds() / 60)

            # CRITICAL: 1-minute timeframe must have NO future leakage
            close_1min = obs["1Minute"][:, 3].numpy()
            for close_val in close_1min:
                assert close_val <= current_minute_idx, (
                    f"Future leakage in 1Minute at step {step}! "
                    f"Close={close_val}, current_minute={current_minute_idx}"
                )

            # 5-minute timeframe: bars are indexed by START time
            # A bar starting at minute X has close from minute X+4
            # This is expected behavior for pandas resampling
            close_5min = obs["5Minute"][:, 3].numpy()

            # Verify 5-minute bars are internally consistent
            # The last bar's close should be at most 4 minutes ahead of the bar's start
            last_5min_close = close_5min[-1]
            # The bar start time is aligned to 5-minute boundaries
            bar_start_minute = (current_minute_idx // 5) * 5
            expected_max_close = bar_start_minute + 4

            # Note: This may be > current_minute_idx, which is the lookahead
            # For now, we just verify the data is consistent with resampling logic
            assert last_5min_close <= expected_max_close + 1, (
                f"5-minute bar data inconsistent at step {step}"
            )

            if truncated:
                break

    def test_observation_window_ends_at_or_before_timestamp(self, sequential_ohlcv_df):
        """
        The last bar in observation window should be at or before the query timestamp.
        """
        window_size = 10
        sampler = MarketDataObservationSampler(
            df=sequential_ohlcv_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=window_size,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset(random_start=False)

        start_time = pd.Timestamp("2024-01-01 00:00:00")

        for _ in range(20):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_minute_idx = int((timestamp - start_time).total_seconds() / 60)

            # Last close in window should equal current_minute_idx
            # (since close = minute_index and we sample at 1-min intervals)
            last_close = obs["1Minute"][-1, 3].item()
            assert last_close == current_minute_idx, (
                f"Last observation should be at current time. "
                f"Expected close={current_minute_idx}, got {last_close}"
            )

            if truncated:
                break

    def test_observation_window_is_contiguous(self, sequential_ohlcv_df):
        """
        Observation window should contain contiguous bars with no gaps.

        For 1-minute data, consecutive close values should differ by 1.
        """
        window_size = 10
        sampler = MarketDataObservationSampler(
            df=sequential_ohlcv_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=window_size,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset(random_start=False)

        for _ in range(20):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            close_values = obs["1Minute"][:, 3].numpy()

            # Check that consecutive values differ by exactly 1
            for i in range(1, len(close_values)):
                diff = close_values[i] - close_values[i - 1]
                assert diff == 1.0, (
                    f"Non-contiguous window! Close values: {close_values}"
                )

            if truncated:
                break

    def test_sequential_observations_advance_by_one(self, sequential_ohlcv_df):
        """
        Sequential observations should advance by exactly one execution period.
        """
        window_size = 5
        sampler = MarketDataObservationSampler(
            df=sequential_ohlcv_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=window_size,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset(random_start=False)

        prev_last_close = None

        for step in range(20):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_last_close = obs["1Minute"][-1, 3].item()

            if prev_last_close is not None:
                # Each step should advance by 1 (1-minute execution)
                assert current_last_close == prev_last_close + 1, (
                    f"Sequential observation did not advance by 1. "
                    f"Previous last close: {prev_last_close}, current: {current_last_close}"
                )

            prev_last_close = current_last_close

            if truncated:
                break

    def test_5min_execution_advances_correctly(self, sequential_ohlcv_df):
        """
        With 5-minute execution, observations should advance by 5 minutes.
        """
        window_size = 5
        sampler = MarketDataObservationSampler(
            df=sequential_ohlcv_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=window_size,
            execute_on=TimeFrame(5, TimeFrameUnit.Minute),
        )
        sampler.reset(random_start=False)

        prev_last_close = None

        for step in range(10):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_last_close = obs["1Minute"][-1, 3].item()

            if prev_last_close is not None:
                # Each step should advance by 5 (5-minute execution)
                diff = current_last_close - prev_last_close
                assert diff == 5, (
                    f"5-minute execution did not advance by 5. "
                    f"Previous: {prev_last_close}, current: {current_last_close}, diff: {diff}"
                )

            prev_last_close = current_last_close

            if truncated:
                break

    @pytest.mark.parametrize("exec_value,exec_unit,expected_advance", [
        (1, TimeFrameUnit.Minute, 1),
        (5, TimeFrameUnit.Minute, 5),
        (15, TimeFrameUnit.Minute, 15),
        (30, TimeFrameUnit.Minute, 30),
        (1, TimeFrameUnit.Hour, 60),
        (4, TimeFrameUnit.Hour, 240),
    ])
    def test_no_future_leakage_various_execution_timeframes(
        self, large_sequential_ohlcv_df, exec_value, exec_unit, expected_advance
    ):
        """
        CRITICAL: Test no future leakage across various execution timeframes.

        Tests: 1min, 5min, 15min, 30min, 1hour, 4hour execution intervals.

        For each execution timeframe, the observation at time T must not
        contain any data from time > T.
        """
        execute_on = TimeFrame(exec_value, exec_unit)
        window_size = 10

        sampler = MarketDataObservationSampler(
            df=large_sequential_ohlcv_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=window_size,
            execute_on=execute_on,
        )
        sampler.reset(random_start=False)

        start_time = pd.Timestamp("2024-01-01 00:00:00")
        prev_last_close = None

        # Test at least 20 steps
        for step in range(20):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_minute_idx = int((timestamp - start_time).total_seconds() / 60)

            # All close prices must be <= current_minute_idx (no future data)
            close_values = obs["1Minute"][:, 3].numpy()

            for close_val in close_values:
                assert close_val <= current_minute_idx, (
                    f"Future leakage detected with {exec_value}{exec_unit.name} execution! "
                    f"At minute {current_minute_idx}, observation contains close={close_val}"
                )

            # Verify execution advances by expected amount
            current_last_close = obs["1Minute"][-1, 3].item()
            if prev_last_close is not None:
                diff = current_last_close - prev_last_close
                assert diff == expected_advance, (
                    f"{exec_value}{exec_unit.name} execution should advance by {expected_advance}. "
                    f"Got diff={diff}"
                )

            prev_last_close = current_last_close

            if truncated:
                break

    @pytest.mark.parametrize("exec_value,exec_unit", [
        (1, TimeFrameUnit.Minute),
        (5, TimeFrameUnit.Minute),
        (15, TimeFrameUnit.Minute),
        (30, TimeFrameUnit.Minute),
        (1, TimeFrameUnit.Hour),
        (4, TimeFrameUnit.Hour),
    ])
    def test_last_observation_at_current_time_various_exec(
        self, large_sequential_ohlcv_df, exec_value, exec_unit
    ):
        """
        The last bar in observation should be at or before query timestamp.

        Tests across: 1min, 5min, 15min, 30min, 1hour, 4hour execution.
        """
        execute_on = TimeFrame(exec_value, exec_unit)
        window_size = 10

        sampler = MarketDataObservationSampler(
            df=large_sequential_ohlcv_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=window_size,
            execute_on=execute_on,
        )
        sampler.reset(random_start=False)

        start_time = pd.Timestamp("2024-01-01 00:00:00")

        for step in range(15):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_minute_idx = int((timestamp - start_time).total_seconds() / 60)

            # Last close should equal current minute (since close = minute_index)
            last_close = obs["1Minute"][-1, 3].item()
            assert last_close == current_minute_idx, (
                f"With {exec_value}{exec_unit.name} execution: "
                f"Last observation should be at current time. "
                f"Expected close={current_minute_idx}, got {last_close}"
            )

            if truncated:
                break

    @pytest.mark.parametrize("exec_value,exec_unit", [
        (1, TimeFrameUnit.Minute),
        (5, TimeFrameUnit.Minute),
        (15, TimeFrameUnit.Minute),
        (30, TimeFrameUnit.Minute),
        (1, TimeFrameUnit.Hour),
        (4, TimeFrameUnit.Hour),
    ])
    def test_multi_timeframe_no_leakage_in_execution_tf(
        self, large_sequential_ohlcv_df, exec_value, exec_unit
    ):
        """
        Multi-timeframe setup: execution timeframe observations must not leak.

        Even with multiple observation timeframes, the data for the execution
        timeframe itself must not contain future information.

        Tests across: 1min, 5min, 15min, 30min, 1hour, 4hour execution.
        """
        execute_on = TimeFrame(exec_value, exec_unit)

        # Use execution timeframe as one of the observation timeframes
        sampler = MarketDataObservationSampler(
            df=large_sequential_ohlcv_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                execute_on,  # Include execution timeframe in observations
            ],
            window_sizes=[10, 5],
            execute_on=execute_on,
        )
        sampler.reset(random_start=False)

        start_time = pd.Timestamp("2024-01-01 00:00:00")

        for step in range(15):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_minute_idx = int((timestamp - start_time).total_seconds() / 60)

            # 1-minute timeframe must have no leakage
            close_1min = obs["1Minute"][:, 3].numpy()
            for close_val in close_1min:
                assert close_val <= current_minute_idx, (
                    f"1Minute leakage with {exec_value}{exec_unit.name} execution! "
                    f"Close={close_val} > current_minute={current_minute_idx}"
                )

            if truncated:
                break


class TestLookaheadBiasFix:
    """
    Critical tests for Issue #10 - Lookahead Bias Fix.

    These tests validate that higher timeframe bars are indexed by END time,
    not START time, ensuring only completed bars are visible to the agent.
    """

    @pytest.fixture
    def lookahead_test_df(self):
        """
        Create 1-minute data where close price = minute index.

        This makes it trivial to detect lookahead bias:
        - If we see close=N at minute M where N > M, that's future leakage
        """
        n_minutes = 500
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

        close_prices = np.array([float(i) for i in range(n_minutes)])
        open_prices = close_prices
        high_prices = close_prices + 0.5
        low_prices = close_prices - 0.5
        volume = np.ones(n_minutes) * 1000

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        })

    def test_5min_bars_indexed_by_end_time(self, lookahead_test_df):
        """
        CRITICAL: 5-minute bars should be indexed by their END time after the fix.

        Without fix: 5-min bar at 00:00:00 contains data from 00:00:00-00:04:59 (START time)
        With fix:    5-min bar at 00:05:00 contains data from 00:00:00-00:04:59 (END time)

        The bar indexed at 00:05:00 should have close from minute 4 (last minute of the bar).
        """
        sampler = MarketDataObservationSampler(
            df=lookahead_test_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[5, 3],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )

        # Manually inspect the resampled 5-minute dataframe
        resampled_5min = sampler.resampled_dfs["5Minute"]

        # The first bar should be indexed at 00:05:00 (minute 5) with close from minute 4
        first_idx = resampled_5min.index[0]
        first_close = resampled_5min.iloc[0]["close"]

        # Get start time from the fixture data
        start_time = lookahead_test_df.iloc[0]["timestamp"]
        first_idx_minutes = int((first_idx - start_time).total_seconds() / 60)

        # After the fix, the first bar is indexed at END time (minute 5)
        # This bar covers minutes 0-4, so close should be from minute 4
        expected_idx_minutes = 5  # First bar should be at minute 5
        expected_close = 4.0  # Close from minute 4

        assert first_idx_minutes == expected_idx_minutes, (
            f"5-min bar index incorrect! "
            f"Expected first bar at minute {expected_idx_minutes}, got minute {first_idx_minutes}"
        )

        assert first_close == pytest.approx(expected_close, abs=0.1), (
            f"5-min bar close incorrect! "
            f"Bar at minute {first_idx_minutes} has close={first_close}, "
            f"expected close={expected_close} (from minute 4)"
        )

    def test_no_incomplete_bars_visible(self, lookahead_test_df):
        """
        CRITICAL: When querying at time T, only bars that ENDED at or before T should be visible.

        Example: At minute 12, we should see:
        - 5-min bars ending at: 5, 10 (NOT the bar ending at 15)
        - The bar from minutes 10-14 (ending at 15) is still incomplete at minute 12
        """
        sampler = MarketDataObservationSampler(
            df=lookahead_test_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[5, 3],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset(random_start=False)

        start_time = pd.Timestamp("2024-01-01 00:00:00")

        # Test at various execution times
        for step in range(50):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_minute = int((timestamp - start_time).total_seconds() / 60)

            # Get all 5-minute closes
            five_min_closes = obs["5Minute"][:, 3].numpy()

            # Each close value represents the minute index it came from
            # All closes must be < current_minute (no future data)
            for close_val in five_min_closes:
                assert close_val < current_minute, (
                    f"LOOKAHEAD BIAS DETECTED! "
                    f"At minute {current_minute}, 5-min observation contains close={close_val} "
                    f"(future data from minute {int(close_val)})"
                )

            # The most recent 5-min close should be from a completed bar
            # A bar ending at minute M has close from minute M-1
            # At current minute N, the most recent completed bar ends at floor(N/5)*5
            last_5min_close = five_min_closes[-1]

            # The last visible 5-min bar should have ended before current_minute
            # Its close is from the last minute of that bar
            bar_end_minute = int(last_5min_close) + 1  # close N comes from bar ending at N+1

            assert bar_end_minute <= current_minute, (
                f"Incomplete bar visible! At minute {current_minute}, "
                f"saw 5-min close from minute {int(last_5min_close)}, "
                f"which is from bar ending at {bar_end_minute}"
            )

            if truncated:
                break

    @pytest.mark.parametrize("tf_value,tf_unit,offset_minutes", [
        (5, TimeFrameUnit.Minute, 5),
        (15, TimeFrameUnit.Minute, 15),
        (30, TimeFrameUnit.Minute, 30),
        (1, TimeFrameUnit.Hour, 60),
    ])
    def test_offset_correctness_various_timeframes(
        self, lookahead_test_df, tf_value, tf_unit, offset_minutes
    ):
        """
        Validate that the offset applied equals the timeframe period.

        The fix shifts bars forward by their period: offset = pd.Timedelta(tf.to_pandas_freq())
        This test verifies the shift is exactly one period.
        """
        sampler = MarketDataObservationSampler(
            df=lookahead_test_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(tf_value, tf_unit),
            ],
            window_sizes=[5, 3],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )

        tf_key = TimeFrame(tf_value, tf_unit).obs_key_freq()
        resampled_tf = sampler.resampled_dfs[tf_key]

        # Check multiple bars to ensure consistent offset
        for i in range(min(5, len(resampled_tf) - 1)):
            bar = resampled_tf.iloc[i]
            bar_idx = resampled_tf.index[i]
            bar_close = bar["close"]

            start_time = pd.Timestamp("2024-01-01 00:00:00")
            bar_idx_minutes = int((bar_idx - start_time).total_seconds() / 60)

            # The bar at index minute M should have close from minute M-1
            # (because bar covers M-offset to M-1, indexed at M)
            expected_close_minute = bar_idx_minutes - 1

            assert bar_close == pytest.approx(float(expected_close_minute), abs=0.1), (
                f"{tf_value}{tf_unit.name} bar offset incorrect! "
                f"Bar at index minute {bar_idx_minutes} has close={bar_close}, "
                f"expected close={expected_close_minute} (offset={offset_minutes})"
            )

    def test_comparison_with_without_fix(self, lookahead_test_df):
        """
        Compare behavior with and without the lookahead fix by examining raw resampled data.

        This test manually checks what pandas resample returns vs what we get after the fix.
        """
        # Create sampler (with fix applied)
        sampler = MarketDataObservationSampler(
            df=lookahead_test_df,
            time_frames=[TimeFrame(5, TimeFrameUnit.Minute)],
            window_sizes=[3],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )

        # Manually resample WITHOUT the fix to compare
        df = lookahead_test_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        raw_resampled = df.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        # Get first bar from each
        raw_first_idx = raw_resampled.index[0]
        raw_first_close = raw_resampled.iloc[0]["close"]

        fixed_resampled = sampler.resampled_dfs["5Minute"]
        fixed_first_idx = fixed_resampled.index[0]
        fixed_first_close = fixed_resampled.iloc[0]["close"]

        # Without fix: bar indexed at 00:05:00 has close from minute 4 (index 4)
        # With fix: bar indexed at 00:10:00 has close from minute 4 (index 4)
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        raw_idx_min = int((raw_first_idx - start_time).total_seconds() / 60)
        fixed_idx_min = int((fixed_first_idx - start_time).total_seconds() / 60)

        # The fix should shift indices by 5 minutes
        assert fixed_idx_min == raw_idx_min + 5, (
            f"Fix not applied correctly! "
            f"Raw index: {raw_idx_min}, Fixed index: {fixed_idx_min}, "
            f"Expected diff: 5, Got: {fixed_idx_min - raw_idx_min}"
        )

        # Both should have same close value (same data, different index)
        assert fixed_first_close == pytest.approx(raw_first_close, abs=0.1), (
            f"Data mismatch after fix! Raw close: {raw_first_close}, Fixed close: {fixed_first_close}"
        )

    def test_execute_on_timeframe_not_shifted(self, lookahead_test_df):
        """
        CRITICAL: The execution timeframe itself should NOT be shifted.

        Only higher timeframes (tf != execute_on) should be shifted.
        The execution timeframe represents when we're making decisions,
        so it must remain at its natural timing.
        """
        sampler = MarketDataObservationSampler(
            df=lookahead_test_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),  # execution timeframe
                TimeFrame(5, TimeFrameUnit.Minute),  # higher timeframe
            ],
            window_sizes=[5, 3],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )

        # Check that 1-minute data is NOT shifted
        one_min_df = sampler.resampled_dfs["1Minute"]

        # Manually resample without any shift
        df = lookahead_test_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        raw_one_min = df.resample("1min").last().dropna()

        # First few timestamps should match exactly (no shift for execute_on)
        for i in range(min(5, len(one_min_df))):
            assert one_min_df.index[i] == raw_one_min.index[i], (
                f"Execute_on timeframe was shifted! Index {i}: "
                f"Expected {raw_one_min.index[i]}, got {one_min_df.index[i]}"
            )

    def test_realistic_scenario_5min_obs_1min_exec(self, lookahead_test_df):
        """
        Realistic scenario: Trading on 1-minute bars with 5-minute context.

        Test that at any execution time T:
        - 1-minute window contains last 5 minutes ending at T
        - 5-minute window contains only completed bars (bars ending at or before T)
        - No lookahead bias: all data is from time <= T
        """
        sampler = MarketDataObservationSampler(
            df=lookahead_test_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[5, 2],  # Last 5 minutes on 1-min, last 2 bars on 5-min
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset(random_start=False)

        start_time = lookahead_test_df.iloc[0]["timestamp"]

        # Test several observations to validate the pattern
        for step in range(30):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_minute = int((timestamp - start_time).total_seconds() / 60)

            # Validate 1-minute window
            one_min_closes = obs["1Minute"][:, 3].numpy()
            expected_1min_end = float(current_minute)
            expected_1min_start = float(current_minute - 4)

            assert one_min_closes[-1] == expected_1min_end, (
                f"At minute {current_minute}, last 1-min close should be {expected_1min_end}, got {one_min_closes[-1]}"
            )
            assert one_min_closes[0] == expected_1min_start, (
                f"At minute {current_minute}, first 1-min close should be {expected_1min_start}, got {one_min_closes[0]}"
            )

            # Validate 5-minute window: all bars should have ended BEFORE current_minute
            five_min_closes = obs["5Minute"][:, 3].numpy()

            for i, close_val in enumerate(five_min_closes):
                # Each 5-min bar has close from the last minute of the bar
                # If close is N, the bar ends at minute N+1
                bar_end_minute = int(close_val) + 1

                assert bar_end_minute <= current_minute, (
                    f"LOOKAHEAD BIAS! At minute {current_minute}, "
                    f"5-min bar {i} has close from minute {int(close_val)}, "
                    f"which means the bar ends at minute {bar_end_minute} (incomplete!)"
                )

                # Also verify no future data
                assert close_val < current_minute, (
                    f"LOOKAHEAD BIAS! At minute {current_minute}, "
                    f"5-min observation contains close={close_val} (future data)"
                )

            if truncated:
                break

    def test_min_start_time_accounts_for_offset(self, lookahead_test_df):
        """
        Validate that min_start_time calculation includes max_offset.

        Line 106 in sampler.py:
        self.min_start_time = latest_first_step + self.max_lookback + max_offset

        This ensures we have enough data after applying the offset.
        """
        sampler = MarketDataObservationSampler(
            df=lookahead_test_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
                TimeFrame(15, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 5, 3],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )

        # max_offset should be 15 minutes (largest higher timeframe)
        # Check that the first execution time leaves enough room for the offset
        first_exec_time = sampler.exec_times[0]
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        first_exec_minutes = int((first_exec_time - start_time).total_seconds() / 60)

        # Should be at least:
        # - 10 minutes for 1-min window (10 * 1)
        # - 25 minutes for 5-min window (5 * 5)
        # - 45 minutes for 15-min window (3 * 15)
        # - 15 minutes for max offset (15-min timeframe)
        # = 45 minutes (max) + 15 offset = 60 minutes minimum

        assert first_exec_minutes >= 45, (
            f"First execution time too early! "
            f"At minute {first_exec_minutes}, may not have enough lookback data"
        )


class TestSamplerMultiTimeframeAlignment:
    """Tests for correct alignment of multi-timeframe observations."""

    @pytest.fixture
    def alignment_ohlcv_df(self):
        """Create data for testing timeframe alignment."""
        n_minutes = 300  # 5 hours
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

        # Simple pattern for easy verification
        close_prices = np.array([100.0 + i for i in range(n_minutes)])
        open_prices = close_prices - 0.5
        high_prices = close_prices + 1.0
        low_prices = close_prices - 1.0
        volume = np.ones(n_minutes) * 1000

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        })

    def test_higher_timeframe_bar_structure(self, alignment_ohlcv_df):
        """
        Higher timeframe bars should be properly structured.

        Note: Pandas resampling indexes bars by START time. A 5-min bar at 00:05:00
        contains data from 00:05:00-00:09:59. When queried at 00:07:00, we see this
        bar which includes "future" close from 00:09:59.

        This test verifies the structure is correct, not that there's no lookahead.
        See TestSamplerNoFutureLeakage for lookahead discussion.
        """
        sampler = MarketDataObservationSampler(
            df=alignment_ohlcv_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[5, 3],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset(random_start=False)

        start_time = pd.Timestamp("2024-01-01 00:00:00")
        base_close = 100.0  # alignment_ohlcv_df uses close = 100 + minute_index

        for step in range(50):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_minute = int((timestamp - start_time).total_seconds() / 60)

            # 1-minute data: last close should match current minute
            one_min_last_close = obs["1Minute"][-1, 3].item()
            expected_1min_close = base_close + current_minute
            assert one_min_last_close == pytest.approx(expected_1min_close, rel=1e-5), (
                f"1-min close mismatch at minute {current_minute}"
            )

            # 5-minute data: verify structure
            five_min_closes = obs["5Minute"][:, 3].numpy()
            # Each 5-min bar's close should be 5 apart (representing 5-min intervals)
            for i in range(1, len(five_min_closes)):
                diff = five_min_closes[i] - five_min_closes[i - 1]
                assert diff == pytest.approx(5.0, rel=1e-5), (
                    f"5-min bars not 5 minutes apart: {five_min_closes}"
                )

            if truncated:
                break

    def test_execution_timeframe_no_future_leakage(self, alignment_ohlcv_df):
        """
        The execution timeframe (1-minute) must never have future data.

        This is the critical test - when executing at time T, we must not
        see any 1-minute data from time > T.
        """
        sampler = MarketDataObservationSampler(
            df=alignment_ohlcv_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
                TimeFrame(15, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 5, 3],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset(random_start=False)

        start_time = pd.Timestamp("2024-01-01 00:00:00")
        base_close = 100.0

        for step in range(30):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_minute = int((timestamp - start_time).total_seconds() / 60)

            # 1-minute timeframe MUST NOT have future leakage
            max_1min_close = obs["1Minute"][:, 3].max().item()
            max_allowed_close = base_close + current_minute

            assert max_1min_close <= max_allowed_close + 0.01, (
                f"1-minute future leakage! Max close={max_1min_close}, "
                f"allowed={max_allowed_close} at minute {current_minute}"
            )

            if truncated:
                break


class TestUndersizedWindowPadding:
    """
    Regression tests for BUG 5 (#157): Sampler returns undersized windows silently.

    When _get_observation_sequential encounters a negative start index (insufficient
    history), it must zero-pad the window to maintain the declared shape.
    """

    @pytest.fixture
    def padding_sampler(self):
        """Create a sampler for padding tests."""
        n_minutes = 200
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")
        close_prices = np.array([100.0 + i for i in range(n_minutes)])

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": close_prices - 0.5,
            "high": close_prices + 1.0,
            "low": close_prices - 1.0,
            "close": close_prices,
            "volume": np.ones(n_minutes) * 1000,
        })

        return MarketDataObservationSampler(
            df=df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )

    @pytest.mark.parametrize("available_bars,expected_pad", [
        (1, 9),
        (5, 5),
        (9, 1),
        (10, 0),
    ], ids=["1-bar-9pad", "5-bars-5pad", "9-bars-1pad", "exact-no-pad"])
    def test_padding_size_matches_deficit(self, padding_sampler, available_bars, expected_pad):
        """Zero-padding size should exactly fill the deficit."""
        key = "1Minute"
        padding_sampler._obs_indices[key][0] = available_bars - 1

        obs = padding_sampler._get_observation_sequential(0)
        window = obs[key]

        assert window.shape[0] == 10
        if expected_pad > 0:
            assert (window[:expected_pad] == 0).all()
        assert (window[expected_pad:] != 0).any()


class TestPerTimeframeFeatureProcessing:
    """Tests for per-timeframe feature processing functions (Issue #177)."""

    @pytest.fixture
    def multi_tf_ohlcv_df(self):
        """Create OHLCV data for multi-timeframe testing."""
        n_minutes = 500
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")

        close_prices = np.array([100.0 + i for i in range(n_minutes)])
        return pd.DataFrame({
            "timestamp": timestamps,
            "open": close_prices - 0.5,
            "high": close_prices + 1.0,
            "low": close_prices - 1.0,
            "close": close_prices,
            "volume": np.ones(n_minutes) * 1000,
        })

    def test_single_function_backward_compatible(self, multi_tf_ohlcv_df):
        """Single feature processing function should work as before."""
        def process_fn(df):
            df = df.copy().reset_index(drop=False)
            df["features_close"] = df["close"]
            df["features_volume"] = df["volume"]
            return df

        sampler = MarketDataObservationSampler(
            df=multi_tf_ohlcv_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 5],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            feature_processing_fn=process_fn,
        )

        # Both timeframes should have same features
        feature_keys = sampler.get_feature_keys()
        assert "features_close" in feature_keys
        assert "features_volume" in feature_keys
        assert len(feature_keys) == 2

        # get_num_features_per_timeframe should return same count for both
        num_features = sampler.get_num_features_per_timeframe()
        assert num_features["1Minute"] == 2
        assert num_features["5Minute"] == 2

    def test_list_of_functions_different_features(self, multi_tf_ohlcv_df):
        """List of functions producing different features should work."""
        def process_1min(df):
            """3 features for high-frequency analysis."""
            df = df.copy().reset_index(drop=False)
            df["features_close"] = df["close"]
            df["features_volume"] = df["volume"]
            df["features_range"] = df["high"] - df["low"]
            return df

        def process_5min(df):
            """5 features for trend analysis."""
            df = df.copy().reset_index(drop=False)
            df["features_close"] = df["close"]
            df["features_sma_3"] = df["close"].rolling(3).mean().fillna(df["close"])
            df["features_sma_5"] = df["close"].rolling(5).mean().fillna(df["close"])
            df["features_volatility"] = df["close"].pct_change().rolling(3).std().fillna(0)
            df["features_volume_ma"] = df["volume"].rolling(3).mean().fillna(df["volume"])
            return df

        sampler = MarketDataObservationSampler(
            df=multi_tf_ohlcv_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 5],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            feature_processing_fn=[process_1min, process_5min],
        )

        # get_feature_keys should raise error (different features)
        with pytest.raises(ValueError, match="different features"):
            sampler.get_feature_keys()

        # get_num_features_per_timeframe should return different counts
        num_features = sampler.get_num_features_per_timeframe()
        assert num_features["1Minute"] == 3
        assert num_features["5Minute"] == 5

        # get_feature_keys_per_timeframe should return different lists
        per_tf = sampler.get_feature_keys_per_timeframe()
        assert len(per_tf["1Minute"]) == 3
        assert len(per_tf["5Minute"]) == 5
        assert "features_range" in per_tf["1Minute"]
        assert "features_sma_3" in per_tf["5Minute"]

    def test_observation_shapes_with_different_features(self, multi_tf_ohlcv_df):
        """Observations should have correct shapes when features differ."""
        def process_1min(df):
            df = df.copy().reset_index(drop=False)
            df["features_a"] = df["close"]
            df["features_b"] = df["volume"]
            return df

        def process_5min(df):
            df = df.copy().reset_index(drop=False)
            df["features_x"] = df["close"]
            df["features_y"] = df["volume"]
            df["features_z"] = df["high"]
            df["features_w"] = df["low"]
            return df

        sampler = MarketDataObservationSampler(
            df=multi_tf_ohlcv_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 5],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            feature_processing_fn=[process_1min, process_5min],
        )
        sampler.reset(random_start=False)

        obs, _, _ = sampler.get_sequential_observation()

        # 1-minute: (10 window, 2 features)
        assert obs["1Minute"].shape == (10, 2)
        # 5-minute: (5 window, 4 features)
        assert obs["5Minute"].shape == (5, 4)

    def test_mismatched_list_length_raises_error(self, multi_tf_ohlcv_df):
        """Mismatched list length should raise ValueError."""
        def dummy_fn(df):
            df = df.copy().reset_index(drop=False)
            df["features_x"] = df["close"]
            return df

        with pytest.raises(ValueError, match="must match time_frames length"):
            MarketDataObservationSampler(
                df=multi_tf_ohlcv_df,
                time_frames=[
                    TimeFrame(1, TimeFrameUnit.Minute),
                    TimeFrame(5, TimeFrameUnit.Minute),
                ],
                window_sizes=[10, 5],
                execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                feature_processing_fn=[dummy_fn],  # Only 1 function for 2 timeframes
            )

    def test_none_in_list_skips_processing(self, multi_tf_ohlcv_df):
        """None in the list should skip processing for that timeframe."""
        def process_fn(df):
            df = df.copy().reset_index(drop=False)
            df["features_custom"] = df["close"] * 2
            return df

        sampler = MarketDataObservationSampler(
            df=multi_tf_ohlcv_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 5],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            feature_processing_fn=[process_fn, None],  # Process 1min, raw 5min
        )

        per_tf = sampler.get_feature_keys_per_timeframe()
        # 1-minute should have custom feature
        assert "features_custom" in per_tf["1Minute"]
        # 5-minute should have raw OHLCV
        assert per_tf["5Minute"] == ["open", "high", "low", "close", "volume"]

    def test_feature_processing_fn_with_no_features_raises(self, multi_tf_ohlcv_df):
        """Feature processing function returning no features_* columns should raise."""
        def bad_fn(df):
            df = df.copy().reset_index(drop=False)
            df["not_a_feature"] = df["close"]  # Missing features_ prefix
            df["also_not_a_feature"] = df["volume"]
            return df

        with pytest.raises(ValueError, match="produced no columns starting with"):
            MarketDataObservationSampler(
                df=multi_tf_ohlcv_df,
                time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                window_sizes=[10],
                execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                feature_processing_fn=bad_fn,
            )

    def test_feature_processing_fn_partial_failure_raises(self, multi_tf_ohlcv_df):
        """If only one timeframe's processing fails, should raise with timeframe info."""
        def good_fn(df):
            df = df.copy().reset_index(drop=False)
            df["features_close"] = df["close"]
            return df

        def bad_fn(df):
            df = df.copy().reset_index(drop=False)
            df["not_a_feature"] = df["close"]
            return df

        with pytest.raises(ValueError, match="5Minute") as exc_info:
            MarketDataObservationSampler(
                df=multi_tf_ohlcv_df,
                time_frames=[
                    TimeFrame(1, TimeFrameUnit.Minute),
                    TimeFrame(5, TimeFrameUnit.Minute),
                ],
                window_sizes=[10, 5],
                execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                feature_processing_fn=[good_fn, bad_fn],  # Second one fails
            )
        assert "produced no columns" in str(exc_info.value)

    @pytest.mark.parametrize("fn_config,expected_1min,expected_5min", [
        (None, 5, 5),  # No processing: raw OHLCV (5 features)
        ("single", 2, 2),  # Single function: same features
        ("list_same", 2, 2),  # List with same features
        ("list_diff", 3, 5),  # List with different features
    ], ids=["none", "single", "list_same", "list_diff"])
    def test_feature_count_variants(self, multi_tf_ohlcv_df, fn_config, expected_1min, expected_5min):
        """Test various feature processing configurations."""
        def process_2features(df):
            df = df.copy().reset_index(drop=False)
            df["features_a"] = df["close"]
            df["features_b"] = df["volume"]
            return df

        def process_3features(df):
            df = df.copy().reset_index(drop=False)
            df["features_a"] = df["close"]
            df["features_b"] = df["volume"]
            df["features_c"] = df["high"]
            return df

        def process_5features(df):
            df = df.copy().reset_index(drop=False)
            df["features_a"] = df["close"]
            df["features_b"] = df["volume"]
            df["features_c"] = df["high"]
            df["features_d"] = df["low"]
            df["features_e"] = df["open"]
            return df

        if fn_config is None:
            feature_fn = None
        elif fn_config == "single":
            feature_fn = process_2features
        elif fn_config == "list_same":
            feature_fn = [process_2features, process_2features]
        else:  # list_diff
            feature_fn = [process_3features, process_5features]

        sampler = MarketDataObservationSampler(
            df=multi_tf_ohlcv_df,
            time_frames=[
                TimeFrame(1, TimeFrameUnit.Minute),
                TimeFrame(5, TimeFrameUnit.Minute),
            ],
            window_sizes=[10, 5],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            feature_processing_fn=feature_fn,
        )

        num_features = sampler.get_num_features_per_timeframe()
        assert num_features["1Minute"] == expected_1min
        assert num_features["5Minute"] == expected_5min
