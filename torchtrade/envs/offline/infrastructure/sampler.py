from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple, Union, Callable, Sequence
import warnings
import numpy as np
import pandas as pd
import torch

from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit, tf_to_timedelta

# Type alias for feature processing function(s)
FeatureProcessingFn = Optional[Union[Callable, Sequence[Callable]]]

# PERF: NamedTuple is faster than dict for attribute access
OHLCV = namedtuple('OHLCV', ['open', 'high', 'low', 'close', 'volume'])


class MarketDataObservationSampler:
    def __init__(
        self,
        df: pd.DataFrame,
        time_frames: Union[List[TimeFrame], TimeFrame] = TimeFrame(1, TimeFrameUnit.Minute),
        window_sizes: Union[List[int], int] = 10,
        execute_on: TimeFrame = TimeFrame(1, TimeFrameUnit.Minute),
        feature_processing_fn: FeatureProcessingFn = None,
        features_start_with: str = "features_",
        max_traj_length: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.seed = seed
        self.np_rng = np.random.default_rng(seed)

        if seed is not None:
            torch.manual_seed(seed)
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        # Ensure df has expected columns
        if list(df.columns) != required_columns:
            print("⚠️ Columns do not match the required format.")
            print("Current columns:", list(df.columns))
            print("Updating columns to:", required_columns)
            df.columns = required_columns
        else:
            print("✅ Columns already in correct format:", required_columns)

        # Make sure time_frames and window_sizes are lists of same length
        if isinstance(time_frames, TimeFrame):
            time_frames = [time_frames]
        if isinstance(window_sizes, int):
            window_sizes = [window_sizes] * len(time_frames)
        if len(window_sizes) != len(time_frames):
            raise ValueError("window_sizes must be an int or list with same length as time_frames")

        # Normalize feature_processing_fn to a list (one per timeframe)
        if feature_processing_fn is None:
            processing_fns: List[Optional[Callable]] = [None] * len(time_frames)
        elif callable(feature_processing_fn):
            # Single function: apply to all timeframes
            processing_fns = [feature_processing_fn] * len(time_frames)
        else:
            # Sequence of functions: one per timeframe
            processing_fns = list(feature_processing_fn)
            if len(processing_fns) != len(time_frames):
                raise ValueError(
                    f"feature_processing_fn list length ({len(processing_fns)}) "
                    f"must match time_frames length ({len(time_frames)})"
                )

        # Make explicit copy to avoid SettingWithCopyWarning when df is a slice
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp").sort_index()
        self.df = df

        self.time_frames = time_frames
        self.window_sizes = window_sizes
        self.max_traj_length = max_traj_length
        self.execute_on = execute_on
        self.feature_processing_fn = feature_processing_fn
        self.features_start_with = features_start_with
        # Precompute resampled OHLCV DataFrames for each timeframe
        self.resampled_dfs: Dict[str, pd.DataFrame] = {}
        first_time_stamps = []
        for tf, proc_fn in zip(time_frames, processing_fns):
            resampled = (
                self.df.resample(tf.to_pandas_freq())
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
            )

            # Fix lookahead bias: shift higher timeframe bars forward by their period
            # This ensures bars are indexed by their END time, not START time
            # Only completed bars will be visible to the agent at any execution time
            # Only shift timeframes that are HIGHER (coarser) than the execution timeframe
            if tf > execute_on:
                offset = pd.Timedelta(tf.to_pandas_freq())
                resampled.index = resampled.index + offset

            if proc_fn is not None:
                resampled = proc_fn(resampled)
                cols_to_keep = [col for col in resampled.columns if col.startswith(features_start_with)]
                cols_to_keep.append("timestamp")
                resampled = resampled[cols_to_keep]
                resampled = resampled.reset_index().set_index("timestamp")
                if "index" in resampled.columns:
                    resampled = resampled.drop(columns=["index"])

            # ensure not empty
            if len(resampled) == 0:
                raise ValueError(f"Resampled dataframe for timeframe {tf.obs_key_freq()} is empty")

            self.resampled_dfs[tf.obs_key_freq()] = resampled
            first_time_stamps.append(resampled.index.min())

        # Maximum lookback window (timedelta)
        window_durations = [tf_to_timedelta(tf) * ws for tf, ws in zip(time_frames, window_sizes)]
        self.max_lookback = max(window_durations)
        latest_first_step = max(first_time_stamps)

        # Calculate offset needed for higher timeframes (to account for END-time indexing)
        # We need extra data to ensure sufficient lookback after the offset
        higher_tf_offsets = [tf_to_timedelta(tf) for tf in time_frames if tf != execute_on]
        max_offset = max(higher_tf_offsets) if higher_tf_offsets else pd.Timedelta(0)

        # Filter execution times
        exec_times = self.df.resample(execute_on.to_pandas_freq()).first().index
        self.min_start_time = latest_first_step + self.max_lookback + max_offset
        self.exec_times = exec_times[exec_times >= self.min_start_time]
        # create base features of execution time frame (we'll keep DataFrame for column names but also build tensors)
        # Use ffill() to handle any NaN values from missing data periods
        execute_base_raw = self.df.resample(execute_on.to_pandas_freq()).last()

        # Detect and warn about data gaps
        nan_mask = execute_base_raw["close"].isna()
        if nan_mask.any():
            nan_count = nan_mask.sum()
            total_count = len(execute_base_raw)
            nan_pct = 100 * nan_count / total_count

            # Find gap regions
            nan_indices = execute_base_raw.index[nan_mask]
            if len(nan_indices) > 0:
                # Group consecutive NaN periods
                gaps = []
                gap_start = nan_indices[0]
                prev_idx = nan_indices[0]
                expected_delta = pd.Timedelta(execute_on.to_pandas_freq())

                for idx in nan_indices[1:]:
                    if idx - prev_idx > expected_delta * 2:  # New gap
                        gaps.append((gap_start, prev_idx, (prev_idx - gap_start).total_seconds() / 60))
                        gap_start = idx
                    prev_idx = idx
                gaps.append((gap_start, prev_idx, (prev_idx - gap_start).total_seconds() / 60))

                # Find largest gap
                largest_gap = max(gaps, key=lambda x: x[2])
                largest_gap_days = largest_gap[2] / (60 * 24)

                warning_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ⚠️  WARNING: SIGNIFICANT DATA GAPS DETECTED IN MARKET DATA  ⚠️              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Missing data points: {nan_count:,} / {total_count:,} ({nan_pct:.2f}%)
║  Number of gap regions: {len(gaps)}
║  Largest gap: {largest_gap_days:.1f} days ({largest_gap[0]} to {largest_gap[1]})
║
║  These gaps will be forward-filled with stale prices, which may
║  negatively impact training quality. Consider:
║    - Using a cleaner dataset
║    - Filtering to continuous time periods
║    - Removing data before/after major gaps
╚══════════════════════════════════════════════════════════════════════════════╝
"""
                warnings.warn(warning_msg, UserWarning, stacklevel=2)

        self.execute_base_features_df = execute_base_raw.ffill()[self.min_start_time:]
        if len(self.execute_base_features_df) == 0:
            raise ValueError("No execute_on base features available after min_start_time")

        self.unseen_timestamps = deque(self.exec_times)
        if len(self.exec_times) == 0:
            raise ValueError("Window duration is too large for the given dataset, no execution times found")

        self.max_steps = len(self.exec_times) - 1 if self.max_traj_length is None else min(len(self.exec_times) - 1, self.max_traj_length)
        print("Max steps:", self.max_steps)

        self._sequential_idx = 0
        self._end_idx = len(self.exec_times)
        # Pre-compute base OHLCV column order for fast dict creation
        self._base_ohlcv_keys = ["open", "high", "low", "close", "volume"]
        # PERF: Store exec_times as numpy array for fast indexing (avoid pandas overhead)
        self._exec_times_arr = self.exec_times.to_numpy()

        # PERF: Pre-compute observation indices for each execution timestamp per timeframe
        # This allows get_observation_sequential() to use direct indexing instead of searchsorted
        self._obs_indices: Dict[str, np.ndarray] = {}
        exec_ts_int64 = self.exec_times.asi8  # int64 nanoseconds

        # Convert resampled dfs to torch tensors for fast slicing.
        # Also convert timestamp indices to int64 (ns) and store as torch.long for searchsorted.
        self.torch_tensors: Dict[str, torch.FloatTensor] = {}
        self.torch_idx: Dict[str, torch.LongTensor] = {}

        for key, df_tf in self.resampled_dfs.items():
            # values -> float32
            arr = df_tf.to_numpy(dtype=np.float32)
            self.torch_tensors[key] = torch.from_numpy(arr)  # shape (N, F)

            # timestamps as int64 nanoseconds: use .asi8 on DatetimeIndex (fast)
            ts_int64 = df_tf.index.asi8  # numpy ndarray (int64)
            self.torch_idx[key] = torch.from_numpy(ts_int64).to(torch.long)  # sorted 1D long tensor

            # Use numpy searchsorted once at init instead of torch searchsorted per step
            obs_idx = np.searchsorted(ts_int64, exec_ts_int64, side='right') - 1
            self._obs_indices[key] = obs_idx

        # Execute-on base features tensor + index
        base_arr = self.execute_base_features_df.to_numpy(dtype=np.float32)
        self.execute_base_tensor = torch.from_numpy(base_arr)  # shape (M, F)
        base_ts_int64 = self.execute_base_features_df.index.asi8
        self.execute_base_idx = torch.from_numpy(base_ts_int64).to(torch.long)

        # Validate 1:1 correspondence between exec_times and execute_base_features_df
        # This ensures sequential access can use direct indexing without misalignment
        if len(self.exec_times) != len(self.execute_base_features_df):
            raise ValueError(
                f"exec_times ({len(self.exec_times)}) and execute_base_features_df "
                f"({len(self.execute_base_features_df)}) have different lengths. "
                "Sequential index access requires 1:1 row correspondence."
            )
        if not (exec_ts_int64 == base_ts_int64).all():
            raise ValueError(
                "exec_times and execute_base_features_df timestamps are not aligned. "
                "Sequential index access requires 1:1 row correspondence."
            )

    def get_random_timestamp(self, without_replacement: bool = False) -> pd.Timestamp:
        if without_replacement:
            idx = self.np_rng.choice(len(self.unseen_timestamps), size=1, replace=False)
            return self.unseen_timestamps.pop(int(idx))
        else:
            idx = self.np_rng.integers(0, len(self.exec_times))
            return self.exec_times[idx]

    def get_random_observation(self, without_replacement: bool = False) -> Tuple[Dict[str, torch.Tensor], pd.Timestamp, bool]:
        timestamp = self.get_random_timestamp(without_replacement)
        return self.get_observation(timestamp), timestamp, False

    def get_sequential_observation(self) -> Tuple[Dict[str, torch.Tensor], pd.Timestamp, bool]:
        """Get observation using index-based tracking."""
        if self._sequential_idx >= self._end_idx:
            raise ValueError("No more timestamps available. Call reset() before continuing.")

        timestamp = pd.Timestamp(self._exec_times_arr[self._sequential_idx])
        truncated = (self._sequential_idx + 1) >= self._end_idx

        obs = self._get_observation_sequential(self._sequential_idx)
        self._sequential_idx += 1

        return obs, timestamp, truncated

    def _get_observation_sequential(self, exec_idx: int) -> Dict[str, torch.Tensor]:
        """Get observation using pre-computed indices."""
        obs: Dict[str, torch.Tensor] = {}

        for tf, ws in zip(self.time_frames, self.window_sizes):
            key = tf.obs_key_freq()
            arr = self.torch_tensors[key]

            # Use pre-computed index
            idx_pos = self._obs_indices[key][exec_idx]

            start = idx_pos - ws + 1
            if start < 0:
                # Pad with zeros on the left to maintain declared window shape
                window = arr[0: idx_pos + 1]
                pad_size = ws - window.shape[0]
                padding = torch.zeros(pad_size, window.shape[1], dtype=window.dtype)
                window = torch.cat([padding, window], dim=0)
            else:
                window = arr[start: idx_pos + 1]
            obs[key] = window

        return obs

    def get_sequential_observation_with_ohlcv(self) -> Tuple[Dict[str, torch.Tensor], pd.Timestamp, bool, OHLCV]:
        """
        Get observation AND base OHLCV in one call using index-based tracking.

        Returns:
            (observation_dict, timestamp, truncated, ohlcv_namedtuple)
        """
        if self._sequential_idx >= self._end_idx:
            raise ValueError("No more timestamps available. Call reset() before continuing.")

        timestamp = pd.Timestamp(self._exec_times_arr[self._sequential_idx])

        # Check if this is the last step
        truncated = (self._sequential_idx + 1) >= self._end_idx

        obs = self._get_observation_sequential(self._sequential_idx)

        row = self.execute_base_tensor[self._sequential_idx]
        vals = row[:5].tolist()
        ohlcv = OHLCV(vals[0], vals[1], vals[2], vals[3], vals[4])
        self._sequential_idx += 1

        return obs, timestamp, truncated, ohlcv

    def get_observation(self, timestamp: pd.Timestamp) -> Dict[str, torch.Tensor]:
        """Return observation dict: { timeframe_key: tensor(shape=[ws, features]) }"""
        obs: Dict[str, torch.Tensor] = {}
        # convert timestamp to int64 ns
        ts_int = int(timestamp.value)  # pd.Timestamp.value is int64 ns
        ts_t = torch.tensor(ts_int, dtype=torch.long)

        for tf, ws in zip(self.time_frames, self.window_sizes):
            key = tf.obs_key_freq()

            arr = self.torch_tensors[key]           # (N, F) float tensor
            idx_tensor = self.torch_idx[key]        # (N,) long sorted tensor

            # pos = insertion index where ts_t would be placed (right=True)
            pos_t = torch.searchsorted(idx_tensor, ts_t, right=True)  # tensor([pos]) dtype long
            pos = int(pos_t.item())
            idx_pos = pos - 1  # last index <= timestamp

            if idx_pos < 0:
                raise ValueError(f"No resampled data exists on or before {timestamp} for timeframe {key}")

            start = idx_pos - ws + 1
            if start < 0:
                raise ValueError(
                    f"Not enough lookback data for timeframe {key}: requested {ws} bars but only {idx_pos+1} exist before {timestamp}"
                )

            # slice arr[start: idx_pos+1], inclusive on idx_pos
            window = arr[start: idx_pos + 1]  # shape (ws, features)
            if window.shape[0] != ws:
                # defensive check, should not happen
                raise ValueError(f"Window length mismatch for {key}: got {window.shape[0]}, expected {ws}")

            obs[key] = window

        return obs

    def get_max_steps(self) -> int:
        return self.max_steps

    def get_observation_keys(self) -> List[str]:
        return list(self.resampled_dfs.keys())

    def get_feature_keys(self) -> List[str]:
        """Get feature columns (raises error if timeframes have different features).

        For backward compatibility with single-function feature processing.
        If timeframes have different features, use get_feature_keys_per_timeframe() instead.

        Returns:
            List of feature column names (shared across all timeframes)

        Raises:
            ValueError: If timeframes have different feature columns
        """
        keys = self.get_observation_keys()
        columns = [list(self.resampled_dfs[k].columns) for k in keys]
        if not all(c == columns[0] for c in columns):
            raise ValueError(
                "Timeframes have different features. Use get_feature_keys_per_timeframe() instead. "
                f"Feature counts per timeframe: {self.get_num_features_per_timeframe()}"
            )
        return columns[0]

    def get_feature_keys_per_timeframe(self) -> Dict[str, List[str]]:
        """Get feature columns for each timeframe.

        Returns:
            Dict mapping timeframe key (e.g., "1Minute") to list of feature column names
        """
        return {key: list(df.columns) for key, df in self.resampled_dfs.items()}

    def get_num_features_per_timeframe(self) -> Dict[str, int]:
        """Get number of features for each timeframe.

        Returns:
            Dict mapping timeframe key (e.g., "1Minute") to number of features
        """
        return {key: len(df.columns) for key, df in self.resampled_dfs.items()}

    def reset(self, random_start: bool = False) -> int:
        """Reset using index-based tracking."""
        total_len = len(self._exec_times_arr)

        if random_start:
            if self.max_traj_length is None:
                start_idx = int(self.np_rng.integers(0, max(1, total_len)))
                self._sequential_idx = start_idx
                self._end_idx = total_len
            else:
                max_start_index = max(0, total_len - self.max_traj_length)
                start_idx = int(self.np_rng.integers(0, max_start_index + 1))
                self._sequential_idx = start_idx
                self._end_idx = min(start_idx + self.max_traj_length, total_len)
        else:
            self._sequential_idx = 0
            if self.max_traj_length is None:
                self._end_idx = total_len
            else:
                self._end_idx = min(self.max_traj_length, total_len)

        # PERF: Keep unseen_timestamps for backward compatibility but as lightweight view
        # This is only used if someone calls get_sequential_observation() directly
        self.unseen_timestamps = deque()  # Empty placeholder

        return self._end_idx - self._sequential_idx

    def get_base_features(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        Return base OHLCV for the execution timeframe as a dict:
            {"open":..., "high":..., "low":..., "close":..., "volume":...}
        Uses the most recent execution bar index <= `timestamp`.
        """
        ts_int = int(timestamp.value)
        ts_t = torch.tensor(ts_int, dtype=torch.long)

        pos_t = torch.searchsorted(self.execute_base_idx, ts_t, right=True)
        pos = int(pos_t.item())
        idx_pos = pos - 1
        if idx_pos < 0:
            raise ValueError(f"No execute_on base feature available on or before {timestamp}")

        row = self.execute_base_tensor[idx_pos]  # tensor of feature floats in same column order as DataFrame
        # Use tolist() for 14x faster batch conversion vs 5x individual .item() calls
        vals = row[:5].tolist()
        return {"open": vals[0], "high": vals[1], "low": vals[2], "close": vals[3], "volume": vals[4]}
