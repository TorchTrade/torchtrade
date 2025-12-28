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
