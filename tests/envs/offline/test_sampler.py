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
    df.fillna(0, inplace=True)
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

    @pytest.mark.parametrize("columns", [
        ["open", "high", "low", "close", "volume", "date"],
        ["ts", "o", "h", "l", "c", "v"],
        ["timestamp", "open", "high", "low", "close"],
    ], ids=["shifted-columns", "wrong-names", "missing-volume"])
    def test_sampler_raises_on_wrong_columns(self, execute_timeframe, columns):
        """Wrong columns must raise, not silently remap by position."""
        n = 200
        df = pd.DataFrame(np.random.rand(n, len(columns)), columns=columns)

        with pytest.raises(ValueError, match="missing required columns"):
            MarketDataObservationSampler(
                df=df,
                time_frames=TimeFrame(1, TimeFrameUnit.Minute),
                window_sizes=10,
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

    def test_reset_restores_step_count(
        self, sample_ohlcv_df, default_timeframes, default_window_sizes, execute_timeframe
    ):
        """Reset should restore the remaining-step count after observations are consumed."""
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


class TestAuxiliaryColumns:
    """Tests for auxiliary (non-OHLCV) column support."""

    @pytest.fixture
    def ohlcv_with_aux_df(self):
        """Create OHLCV + auxiliary columns (funding_rate, basis)."""
        n_minutes = 500
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")
        close_prices = np.array([100.0 + i * 0.1 for i in range(n_minutes)])

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": close_prices - 0.05,
            "high": close_prices + 0.1,
            "low": close_prices - 0.1,
            "close": close_prices,
            "volume": np.ones(n_minutes) * 1000,
            "funding_rate": np.sin(np.arange(n_minutes) / 50) * 0.001,
            "basis": np.cos(np.arange(n_minutes) / 30) * 0.5,
        })

    def test_auxiliary_columns_flow_through_pipeline(self, ohlcv_with_aux_df):
        """Aux columns should be accepted, resampled, and produce correct tensor shapes."""
        window_size = 10
        sampler = MarketDataObservationSampler(
            df=ohlcv_with_aux_df,
            time_frames=TimeFrame(5, TimeFrameUnit.Minute),
            window_sizes=window_size,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        # Feature count: 5 OHLCV + 2 aux = 7
        assert sampler.get_num_features_per_timeframe()["5Minute"] == 7
        # Tensor shape matches
        sampler.reset()
        obs, _, _ = sampler.get_sequential_observation()
        assert obs["5Minute"].shape == (window_size, 7)

    def test_ohlcv_positional_contract_with_shuffled_input(self, ohlcv_with_aux_df):
        """OHLCV must be first 5 columns and row[:5] slicing must return prices, not aux data."""
        # Shuffle column order — OHLCV should still come first
        shuffled = ohlcv_with_aux_df[["timestamp", "basis", "volume", "close", "funding_rate", "open", "high", "low"]]
        sampler = MarketDataObservationSampler(
            df=shuffled,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        assert list(sampler.df.columns[:5]) == ["open", "high", "low", "close", "volume"]
        # Verify row[:5] slicing returns OHLCV, not aux data
        sampler.reset()
        _, _, _, ohlcv = sampler.get_sequential_observation_with_ohlcv()
        assert ohlcv.open > 50  # Prices start at ~100, not funding_rate ~0.001
        assert ohlcv.close > 50
        assert ohlcv.volume == 1000
        # Positions 1 and 2 specifically: the vectorized envs read high/low by hardcoded
        # column index, so swapping them in ohlcv_agg would invert every SL/TP check.
        assert ohlcv.high >= ohlcv.close >= ohlcv.low

    def test_auxiliary_resampling_uses_last(self, ohlcv_with_aux_df):
        """Auxiliary columns should use 'last' aggregation when resampled."""
        df = ohlcv_with_aux_df.copy()
        df.loc[0:4, "funding_rate"] = [0.1, 0.2, 0.3, 0.4, 0.5]

        sampler = MarketDataObservationSampler(
            df=df,
            time_frames=TimeFrame(5, TimeFrameUnit.Minute),
            window_sizes=10,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        # "last" of [0.1, 0.2, 0.3, 0.4, 0.5] = 0.5
        first_funding = sampler.resampled_dfs["5Minute"]["funding_rate"].iloc[0]
        assert abs(first_funding - 0.5) < 1e-6

    def test_backward_compat_ohlcv_only(self, sample_ohlcv_df, execute_timeframe):
        """Existing OHLCV-only DataFrames should work identically."""
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            execute_on=execute_timeframe,
        )
        assert sampler.get_num_features_per_timeframe()["1Minute"] == 5

    def test_feature_processing_sees_auxiliary_columns(self, ohlcv_with_aux_df):
        """Feature processing function should have access to auxiliary columns."""
        def feature_fn(df):
            df = df.copy().reset_index(drop=False)
            for col in ["close", "volume", "funding_rate", "basis"]:
                if col in df.columns:
                    df[f"features_{col}"] = df[col]
            df.fillna(0, inplace=True)
            return df

        sampler = MarketDataObservationSampler(
            df=ohlcv_with_aux_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
            feature_processing_fn=feature_fn,
        )
        feature_keys = sampler.get_feature_keys()
        assert "features_funding_rate" in feature_keys
        assert "features_basis" in feature_keys

    def test_get_base_features_returns_only_ohlcv(self, ohlcv_with_aux_df):
        """get_base_features() must return exactly 5 OHLCV keys, not auxiliary data."""
        sampler = MarketDataObservationSampler(
            df=ohlcv_with_aux_df,
            time_frames=TimeFrame(1, TimeFrameUnit.Minute),
            window_sizes=10,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset()
        ts = sampler.exec_times[sampler._sequential_idx]
        base = sampler.get_base_features(ts)
        assert list(base.keys()) == ["open", "high", "low", "close", "volume"]
        assert base["close"] > 50  # Price, not funding_rate

    def test_sparse_auxiliary_data_no_nan_in_tensors(self):
        """Sparse aux data (NaN gaps) must be forward-filled, not leak NaN into tensors."""
        n_minutes = 500
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.date_range(start=start_time, periods=n_minutes, freq="1min")
        close_prices = np.array([100.0 + i * 0.1 for i in range(n_minutes)])

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": close_prices - 0.05,
            "high": close_prices + 0.1,
            "low": close_prices - 0.1,
            "close": close_prices,
            "volume": np.ones(n_minutes) * 1000,
            "funding_rate": np.full(n_minutes, np.nan),  # all NaN except every 60 bars
        })
        # Set funding rate only every 60 bars (simulates 1h funding on 1m data)
        df.loc[::60, "funding_rate"] = 0.001

        sampler = MarketDataObservationSampler(
            df=df,
            time_frames=TimeFrame(5, TimeFrameUnit.Minute),
            window_sizes=10,
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )
        sampler.reset()
        obs, _, _ = sampler.get_sequential_observation()
        assert not torch.isnan(obs["5Minute"]).any(), "NaN leaked into observation tensor from sparse aux data"


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

    def test_execution_bar_aggregates_its_sub_bars(self, sample_ohlcv_df):
        """Execution bars must aggregate their sub-bars, not copy the final one.

        .last() collapses high/low onto the last sub-bar, so the SL/TP and liquidation
        checks that read this range would never fire. Only a coarse execute_on can show
        this; at source frequency the two agree, which is why the bug survived.
        """
        execute_on = TimeFrame(1, TimeFrameUnit.Hour)
        sampler = MarketDataObservationSampler(
            df=sample_ohlcv_df,
            time_frames=[execute_on],
            window_sizes=[5],
            execute_on=execute_on,
        )
        sampler.reset(random_start=False)
        _, timestamp, _ = sampler.get_sequential_observation()
        base_features = sampler.get_base_features(timestamp)

        indexed = sample_ohlcv_df.set_index("timestamp")
        source = indexed[
            (indexed.index >= timestamp)
            & (indexed.index < timestamp + pd.Timedelta(execute_on.to_pandas_freq()))
        ]
        # approx, not ==: execute_base_tensor is float32 over a float64 source.
        assert base_features["open"] == pytest.approx(source["open"].iloc[0])
        assert base_features["high"] == pytest.approx(source["high"].max())
        assert base_features["low"] == pytest.approx(source["low"].min())
        assert base_features["close"] == pytest.approx(source["close"].iloc[-1])
        assert base_features["volume"] == pytest.approx(source["volume"].sum())

        # Without a real excursion beyond the last sub-bar the assertions above hold
        # under .last() too, and the test stops discriminating.
        assert source["high"].max() > source["high"].iloc[-1]
        assert source["low"].min() < source["low"].iloc[-1]

    def test_gap_bar_is_flat_rather_than_a_replay_of_the_previous_range(self, sample_ohlcv_df):
        """A bar with no source rows must not inherit the previous bar's high/low.

        Forward-filling the range would hand SL/TP the excursion the position already
        lived through one bar earlier, letting a single move stop it out twice.
        """
        execute_on = TimeFrame(15, TimeFrameUnit.Minute)
        gap_start, gap_end = pd.Timestamp("2024-01-01 03:00"), pd.Timestamp("2024-01-01 03:15")
        df = sample_ohlcv_df[
            (sample_ohlcv_df["timestamp"] < gap_start) | (sample_ohlcv_df["timestamp"] >= gap_end)
        ]

        with pytest.warns(UserWarning, match="DATA GAPS"):
            sampler = MarketDataObservationSampler(
                df=df, time_frames=[execute_on], window_sizes=[5], execute_on=execute_on,
            )

        gap = sampler.get_base_features(gap_start)
        previous = sampler.get_base_features(gap_start - pd.Timedelta("15min"))

        assert gap["close"] == pytest.approx(previous["close"])
        assert gap["open"] == pytest.approx(gap["close"])
        assert gap["high"] == pytest.approx(gap["close"])
        assert gap["low"] == pytest.approx(gap["close"])
        assert gap["volume"] == 0.0
        # Both the high and low assertions only discriminate if the bar they would have
        # been filled from sits strictly inside its own range.
        assert previous["high"] > previous["close"] > previous["low"]

    def test_bar_without_a_usable_close_keeps_its_real_range(self, sample_ohlcv_df):
        """Only bars with no source rows are flattened.

        A bar whose close is unusable still traded, and its high/low are the range SL/TP
        must see. Flattening it would re-create the too-narrow range this aggregation
        exists to fix.
        """
        execute_on = TimeFrame(15, TimeFrameUnit.Minute)
        bar_start, bar_end = pd.Timestamp("2024-01-01 03:00"), pd.Timestamp("2024-01-01 03:15")
        df = sample_ohlcv_df.copy()
        in_bar = (df["timestamp"] >= bar_start) & (df["timestamp"] < bar_end)
        df.loc[in_bar, "close"] = np.nan
        df.loc[in_bar, "low"] = 42.0  # a real excursion a stop must still see

        with pytest.warns(UserWarning, match="DATA GAPS"):
            sampler = MarketDataObservationSampler(
                df=df, time_frames=[execute_on], window_sizes=[5], execute_on=execute_on,
            )

        bar = sampler.get_base_features(bar_start)
        assert bar["low"] == pytest.approx(42.0)
        assert bar["high"] > bar["low"], "a bar that traded must not be flattened"


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

        The boundary is the DECISION INSTANT, not the bar's label. `timestamp` is the
        execution bin's start label, but the env hands money over at that bin's close --
        pinned by TestExecutionPriceConvention, verified by intercepting the executed
        price. So an observation labelled T may legitimately contain anything up to the
        end of bin T, and nothing beyond it.

        This previously bounded on the label, which made the finer window stop a full
        execution bar short of the price it was about to trade at (#282). That was not
        preventing lookahead -- the coarse window already carried that close -- it was
        withholding data the observation already contained one row up.
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
        # Last minute bar inside the bin labelled `timestamp`; the next bin is the future.
        last_minute_in_bin = int(execute_on.to_minutes()) - 1

        # Test at least 20 steps
        for step in range(20):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_minute_idx = int((timestamp - start_time).total_seconds() / 60)
            decision_instant = current_minute_idx + last_minute_in_bin

            # No close may come from after the instant the trade fills at
            close_values = obs["1Minute"][:, 3].numpy()

            for close_val in close_values:
                assert close_val <= decision_instant, (
                    f"Future leakage detected with {exec_value}{exec_unit.name} execution! "
                    f"Bin labelled minute {current_minute_idx} closes at minute "
                    f"{decision_instant}, but observation contains close={close_val}"
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
        The last bar in the observation should be the one the trade fills at.

        Tests across: 1min, 5min, 15min, 30min, 1hour, 4hour execution.

        "Current time" is the execution bin's CLOSE, not its label. `timestamp` is the
        label; the env executes at the bin's close (TestExecutionPriceConvention). For
        execute_on=1Minute the two coincide, which is why this read as correct for years
        while every coarser execute_on left the minute window a full bar behind (#282).
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
        last_minute_in_bin = int(execute_on.to_minutes()) - 1

        for step in range(15):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_minute_idx = int((timestamp - start_time).total_seconds() / 60)
            decision_instant = current_minute_idx + last_minute_in_bin

            # Last close should be the bin's closing minute (since close = minute_index)
            last_close = obs["1Minute"][-1, 3].item()
            assert last_close == decision_instant, (
                f"With {exec_value}{exec_unit.name} execution: last observation should be "
                f"the bar the trade fills at. Expected close={decision_instant}, "
                f"got {last_close}"
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

        Bounded on the execution bin's CLOSE, which is where the trade fills, not on its
        label (TestExecutionPriceConvention, #282).

        This also asserts the execute_on-keyed column, which it never did -- it checked
        only obs["1Minute"] and passed trivially, so the column the agent actually trades
        off was unguarded. Asserting both is what pins the real invariant: every
        timeframe up to execute_on ends at the same instant.
        """
        execute_on = TimeFrame(exec_value, exec_unit)
        exec_key = execute_on.obs_key_freq()

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
        last_minute_in_bin = int(execute_on.to_minutes()) - 1

        for step in range(15):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            current_minute_idx = int((timestamp - start_time).total_seconds() / 60)
            decision_instant = current_minute_idx + last_minute_in_bin

            # Neither timeframe may reach past the instant the trade fills at
            for key in ("1Minute", exec_key):
                for close_val in obs[key][:, 3].numpy():
                    assert close_val <= decision_instant, (
                        f"{key} leakage with {exec_value}{exec_unit.name} execution! "
                        f"Close={close_val} > decision instant {decision_instant}"
                    )

            # ...and they must agree about where "now" is
            assert obs["1Minute"][-1, 3].item() == obs[exec_key][-1, 3].item(), (
                f"With {exec_value}{exec_unit.name} execution the 1Minute window ends at "
                f"{obs['1Minute'][-1, 3].item()} but {exec_key} ends at "
                f"{obs[exec_key][-1, 3].item()} -- one observation, two 'now's"
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

        Only strictly coarser timeframes (tf > execute_on) are shifted. The predicate
        used to be `!= execute_on`, which compared (value, unit) and so shifted a
        60Minute frame against execute_on=1Hour -- the same duration -- leaving it a bar
        stale (#282). This test still passes under that bug, since it never constructs
        an equal-duration pair, so the predicate is spelled out here rather than left to
        the test name.

        The execution timeframe represents when we're making decisions, so it must
        remain at its natural timing.
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

        Note: pandas resampling labels bars by START time, and this class runs with
        execute_on=1Minute, so a coarser 5-min bar IS shifted to its END label before it
        can be seen -- the "future close" this note used to describe is not reachable
        here. What remains true is the narrower point: a bar aggregates its whole span,
        which is why a bar must be bounded by when it CLOSES, not when it starts.

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
    Sampler returns undersized windows silently.

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

    @pytest.mark.parametrize("scenario", ["single_tf", "multi_tf_partial"], ids=["single_tf", "multi_tf_partial"])
    def test_feature_processing_fn_with_no_features_raises(self, multi_tf_ohlcv_df, scenario):
        """Feature processing function returning no features_* columns should raise."""
        def good_fn(df):
            df = df.copy().reset_index(drop=False)
            df["features_close"] = df["close"]
            return df

        def bad_fn(df):
            df = df.copy().reset_index(drop=False)
            df["not_a_feature"] = df["close"]  # Missing features_ prefix
            return df

        if scenario == "single_tf":
            with pytest.raises(ValueError, match="produced no columns starting with"):
                MarketDataObservationSampler(
                    df=multi_tf_ohlcv_df,
                    time_frames=[TimeFrame(1, TimeFrameUnit.Minute)],
                    window_sizes=[10],
                    execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                    feature_processing_fn=bad_fn,
                )
        else:  # multi_tf_partial - second timeframe fails, error should include timeframe name
            with pytest.raises(ValueError, match="5Minute") as exc_info:
                MarketDataObservationSampler(
                    df=multi_tf_ohlcv_df,
                    time_frames=[
                        TimeFrame(1, TimeFrameUnit.Minute),
                        TimeFrame(5, TimeFrameUnit.Minute),
                    ],
                    window_sizes=[10, 5],
                    execute_on=TimeFrame(1, TimeFrameUnit.Minute),
                    feature_processing_fn=[good_fn, bad_fn],
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


class TestTimeframeSpellingIsNotObservable:
    """Two spellings of the same duration must sample the same data (#282).

    They stay distinct objects on purpose -- the observation key differs, and the parser
    warns that models trained on one will not transfer. But the DATA behind that key must
    not depend on the spelling, and it did: `if tf > execute_on` in the resample loop was
    true in both directions for equal durations, so a 60Minute frame against
    execute_on=1Hour took the anti-lookahead shift meant for coarser bars and went a
    full bar stale.
    """

    def test_same_duration_different_spelling_samples_identically(self):
        n = 60 * 40
        ts = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = np.arange(n, dtype=float) + 1000.0  # close identifies its own minute
        df = pd.DataFrame({
            "timestamp": ts, "open": close, "high": close + 0.5,
            "low": close - 0.5, "close": close, "volume": 1000.0,
        })
        execute_on = TimeFrame(1, TimeFrameUnit.Hour)

        def sample_last_close(tf):
            sampler = MarketDataObservationSampler(
                df=df.copy(), time_frames=[tf], window_sizes=[4],
                execute_on=execute_on, max_traj_length=5, seed=0,
            )
            sampler.reset()
            obs, timestamp, _ = sampler.get_sequential_observation()
            return obs[tf.obs_key_freq()][-1, 3].item(), timestamp  # col 3 == close

        as_hours = sample_last_close(TimeFrame(1, TimeFrameUnit.Hour))
        as_minutes = sample_last_close(TimeFrame(60, TimeFrameUnit.Minute))

        assert as_hours == as_minutes, (
            f"1Hour saw {as_hours}, 60Minute saw {as_minutes} -- same duration, "
            "same execute_on, so the spelling must not change the data"
        )


class TestObservationTimeframesShareOneInstant:
    """Every timeframe up to execute_on must end at the same instant (#282 defect 1).

    The env executes at the last close in obs[execute_on] -- pinned by
    tests/envs/offline/test_sequential.py::TestExecutionPriceConvention, verified by
    intercepting the executed price. So that instant is what the whole observation
    should be as-of.

    Timeframes FINER than execute_on were not: the resample loop shifts only
    tf > execute_on to END labels, and _obs_indices searches every timeframe with the
    same key -- the exec bin's START label. For the execute_on frame that lands right by
    accident, because its bin is already aggregated through the bin's end. For a finer
    frame the index has real per-bar granularity, so the same key returns the bar
    STARTING at the label: a 1Minute window sat 59 minutes behind the 1Hour window it
    was sampled beside, and behind the price the trade filled at.
    """

    @staticmethod
    def _counter_df(n=60 * 40):
        ts = pd.date_range("2024-01-01", periods=n, freq="1min")
        close = np.arange(n, dtype=float) + 1000.0  # close identifies its own minute
        return pd.DataFrame({
            "timestamp": ts, "open": close, "high": close + 0.5,
            "low": close - 0.5, "close": close, "volume": 1000.0,
        })

    def test_finer_window_ends_where_the_execute_on_window_ends(self):
        """Kept for one dimension only: a fine frame that is NOT 1-minute.

        Review showed the 1Minute params here never fired without
        TestSamplerNoFutureLeakage::test_multi_timeframe_no_leakage_in_execution_tf
        also firing -- it asserts the identical relation over six execute_on values
        instead of three. What does not exist elsewhere is a fine frame whose `value`
        exceeds execute_on's, which is the only place a `.value`-vs-duration comparison
        slip in TimeFrame.__gt__ shows up.

        Folding this into that test is not free: it parametrizes execute_on over six
        values including 5Minute, and a duplicate timeframe silently collapses in
        resampled_dfs (3 requested, 2 returned), so the 5Minute case would quietly
        degrade.
        """
        fine = TimeFrame(5, TimeFrameUnit.Minute)
        exec_tf = TimeFrame(1, TimeFrameUnit.Hour)
        sampler = MarketDataObservationSampler(
            df=self._counter_df(), time_frames=[fine, exec_tf], window_sizes=[4, 4],
            execute_on=exec_tf, max_traj_length=8, seed=0,
        )
        sampler.reset(random_start=False)

        for _ in range(5):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            fine_close = obs[fine.obs_key_freq()][-1, 3].item()
            exec_close = obs[exec_tf.obs_key_freq()][-1, 3].item()
            assert fine_close == exec_close, (
                f"at {timestamp}: {fine.obs_key_freq()} ends at close {fine_close} while "
                f"{exec_tf.obs_key_freq()} ends at {exec_close} -- the same observation "
                f"disagrees with itself about now by {exec_close - fine_close:.0f} minutes"
            )
            if truncated:
                break

    @pytest.mark.parametrize("exec_tf,fine", [
        (TimeFrame(15, TimeFrameUnit.Minute), TimeFrame(5, TimeFrameUnit.Minute)),
        (TimeFrame(15, TimeFrameUnit.Minute), TimeFrame(7, TimeFrameUnit.Minute)),
        (TimeFrame(1, TimeFrameUnit.Hour), TimeFrame(45, TimeFrameUnit.Minute)),
        (TimeFrame(1, TimeFrameUnit.Hour), TimeFrame(25, TimeFrameUnit.Minute)),
    ], ids=["5_divides_15", "7_does_not", "45_does_not", "25_does_not"])
    def test_fine_bar_straddling_the_bin_end_is_not_served(self, exec_tf, fine):
        """A fine bar may only be served once it has CLOSED at or before the bin end.

        pandas aggregates a bar across its whole span, so a bar that merely STARTS
        inside the execution bin can carry a close from after the bin. With
        execute_on=15Minute and a 7Minute frame, the bar labelled minute 70 spans
        [70, 77) and closes at 76 -- past a bin whose decision instant is 74.

        The first version of this fix bounded on "the last bar starting inside the bin"
        and leaked exactly that: 6 minutes here, 30 with 45Minute under 1Hour. Divisor
        pairs are immune because their boundaries line up and nothing straddles, which
        is why every other test in this file missed it -- they are all divisors.
        """
        sampler = MarketDataObservationSampler(
            df=self._counter_df(), time_frames=[fine, exec_tf], window_sizes=[3, 3],
            execute_on=exec_tf, max_traj_length=12, seed=0,
        )
        sampler.reset(random_start=False)
        start = pd.Timestamp("2024-01-01 00:00:00")

        for _ in range(8):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            bin_start = (timestamp - start).total_seconds() / 60
            decision_instant = bin_start + exec_tf.to_minutes() - 1
            for value in obs[fine.obs_key_freq()][:, 3].tolist():
                minute = value - 1000.0
                assert minute <= decision_instant, (
                    f"at {timestamp}: {fine.obs_key_freq()} window carries minute "
                    f"{minute:.0f}, which is {minute - decision_instant:.0f} past the "
                    f"decision instant {decision_instant:.0f}"
                )

            # Safety alone is satisfiable by any stale bound, and staleness is what
            # #282 was about. The window must end on the FRESHEST legal bar: if the
            # last legal one is [L, L+f) then the next closes at L+2f, past the bin,
            # so the lag can never reach a full fine period.
            newest = obs[fine.obs_key_freq()][-1, 3].item() - 1000.0
            assert decision_instant - newest < fine.to_minutes(), (
                f"at {timestamp}: {fine.obs_key_freq()} ends at {newest:.0f}, "
                f"{decision_instant - newest:.0f} min behind the decision instant "
                f"{decision_instant:.0f} -- a whole {fine.to_minutes():.0f}-min bar "
                f"was available and withheld"
            )
            if truncated:
                break

    def test_both_lookup_paths_agree_for_a_finer_frame(self):
        """get_observation(ts) must return what get_sequential_observation() returned.

        They are separate implementations -- one precomputes indices at init, the other
        searches at call time -- and the fine-frame rule was originally applied to only
        the first, so they disagreed by up to a full execution bar.

        Reachable, not theoretical: sequential.py and onestep.py both fall back to
        get_observation() for the terminal observation of a truncated episode, which is
        exactly what a value estimate bootstraps from.
        """
        fine = TimeFrame(1, TimeFrameUnit.Minute)
        exec_tf = TimeFrame(15, TimeFrameUnit.Minute)
        sampler = MarketDataObservationSampler(
            df=self._counter_df(), time_frames=[fine, exec_tf], window_sizes=[4, 4],
            execute_on=exec_tf, max_traj_length=10, seed=0,
        )
        sampler.reset(random_start=False)

        for _ in range(5):
            sequential, timestamp, truncated = sampler.get_sequential_observation()
            direct = sampler.get_observation(timestamp)
            for key in (fine.obs_key_freq(), exec_tf.obs_key_freq()):
                assert torch.equal(sequential[key], direct[key]), (
                    f"at {timestamp}: {key} differs between the sequential and the "
                    f"timestamp-lookup paths"
                )
            if truncated:
                break

    def test_three_timeframes_spanning_both_sides_of_execute_on(self):
        """The realistic shape from docs/guides/sampler.md, and the only config where
        the fine search key and the coarse END-shift both apply at once.

        Each side has a different correct answer, which is the point: fine and exec end
        at the decision instant, while coarse holds the last bar that has CLOSED -- it
        is still mid-bar at that instant, so showing it would be lookahead.

        The coarse side is bounded, not pinned: an exact expectation would be brittle to
        grid alignment, so a coarse frame that went several bars stale would pass here.
        Its freshness is not what this test is for -- the interaction between the two
        mechanisms is.
        """
        fine = TimeFrame(1, TimeFrameUnit.Minute)
        exec_tf = TimeFrame(15, TimeFrameUnit.Minute)
        coarse = TimeFrame(1, TimeFrameUnit.Hour)

        sampler = MarketDataObservationSampler(
            df=self._counter_df(), time_frames=[fine, exec_tf, coarse],
            window_sizes=[4, 4, 4], execute_on=exec_tf, max_traj_length=10, seed=0,
        )
        sampler.reset(random_start=False)
        start = pd.Timestamp("2024-01-01 00:00:00")

        for _ in range(6):
            obs, timestamp, truncated = sampler.get_sequential_observation()
            bin_start = (timestamp - start).total_seconds() / 60
            decision_instant = bin_start + exec_tf.to_minutes() - 1

            fine_last = obs[fine.obs_key_freq()][-1, 3].item() - 1000.0
            exec_last = obs[exec_tf.obs_key_freq()][-1, 3].item() - 1000.0
            coarse_last = obs[coarse.obs_key_freq()][-1, 3].item() - 1000.0

            assert fine_last == decision_instant == exec_last, (
                f"at {timestamp}: fine={fine_last:.0f} exec={exec_last:.0f}, both should "
                f"be the decision instant {decision_instant:.0f}"
            )
            # The coarse bar is END-labelled, so it is only shown once closed. It may
            # therefore trail, but must never reach past the decision instant.
            assert coarse_last <= decision_instant, (
                f"at {timestamp}: coarse={coarse_last:.0f} is past the decision instant "
                f"{decision_instant:.0f}"
            )
            # ...nor trail further than one coarse bar plus the exec bin it is read in.
            # Derivable, not tuned: the visible coarse bar closes at floor(B/c)*c - 1, so
            # with B a multiple of exec the lag is (B mod c) + exec < c + exec. Without
            # this, `tf < execute_on` in _fine_period_ns can be weakened to `!=` -- the
            # same spelling-vs-duration slip this branch exists to remove -- and coarse
            # frames silently go an extra bar stale with the whole suite green.
            coarse_limit = coarse.to_minutes() + exec_tf.to_minutes()
            assert decision_instant - coarse_last < coarse_limit, (
                f"at {timestamp}: coarse={coarse_last:.0f} is "
                f"{decision_instant - coarse_last:.0f} min behind the decision instant "
                f"{decision_instant:.0f}, past the {coarse_limit:.0f} min bound"
            )
            if truncated:
                break


    @pytest.mark.parametrize("exec_tf,should_raise", [
        (TimeFrame(1, TimeFrameUnit.Day), True),
        (TimeFrame(2, TimeFrameUnit.Day), True),
        (TimeFrame(1, TimeFrameUnit.Hour), False),   # Tick offsets are DST-immune
        (TimeFrame(1, TimeFrameUnit.Minute), False),
    ], ids=["1day", "2day", "1hour_ok", "1min_ok"])
    def test_tz_aware_index_refused_only_for_day_or_longer(self, exec_tf, should_raise):
        """Day-or-longer bins follow LOCAL midnight, so a DST day is 23h or 25h wide
        while the search bound uses a fixed 24h period. On a spring-forward day that
        reaches an hour past the bin's real close -- lookahead, not staleness.

        Minute and Hour resample as fixed-width Tick offsets and are immune, so the
        guard must not reject them; rejecting every tz-aware config would needlessly
        break intraday users who are not exposed to this at all.

        The guard is deliberately broader than the hazard: it rejects ANY tz-aware
        index at Day or longer, including fixed-offset zones like UTC that have no DST
        and are therefore safe. Deciding whether a given zone actually transitions
        inside the data's span is more machinery than the case is worth, and the error
        tells the caller exactly what to do instead.
        """
        n = 60 * 24 * 4
        ts = pd.date_range("2024-03-08", periods=n, freq="1min", tz="UTC")
        close = np.arange(n, dtype=float) + 1000.0
        df = pd.DataFrame({
            "timestamp": ts, "open": close, "high": close + 0.5,
            "low": close - 0.5, "close": close, "volume": 1000.0,
        })

        def build():
            return MarketDataObservationSampler(
                df=df, time_frames=[exec_tf], window_sizes=[3],
                execute_on=exec_tf, max_traj_length=5, seed=0,
            )

        if should_raise:
            with pytest.raises(ValueError, match="tz-naive"):
                build()
        else:
            build()  # must not raise


@pytest.mark.parametrize("field,value,ok", [
    ("high", 100.1, False),   # below close (100.2) but above open -- pins max, not min
    ("low", 100.1, False),    # above open (100.0) but below close -- pins min, not max
    ("high", 100.2, True),    # touching the higher of the two is well-formed
    ("low", 100.0, True),     # touching the lower of the two is well-formed
], ids=["high-below-close", "low-above-open", "high-equals-close", "low-equals-open"])
def test_malformed_ohlc_is_rejected_at_ingestion(field, value, ok):
    """A bar whose high/low do not bracket open and close is refused (#326).

    Every exit rule downstream reads the extreme to decide whether a level was touched
    and the open to price a gapped fill, so on a malformed row those two disagree and the
    engines can answer differently from one another. Validating here rather than guarding
    in each rule is the boundary-validation invariant; a guard that absorbs the nonsense
    makes it silent, which is what let ~100 tests build impossible candles unnoticed.
    """
    # open != close deliberately: with them equal, max(open, close) == min(open, close)
    # and the cells cannot tell the right operand from the wrong one -- comparing high
    # against min instead of max would pass every case.
    n = 30
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "open": [100.0] * n, "high": [100.5] * n,
        "low": [99.5] * n, "close": [100.2] * n, "volume": [1000.0] * n,
    })
    df.loc[5, field] = value

    def build():
        MarketDataObservationSampler(
            df, time_frames=[TimeFrame(1, TimeFrameUnit.Minute)], window_sizes=[5],
            execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        )

    if ok:
        build()
    else:
        with pytest.raises(ValueError, match="position 5"):
            build()
