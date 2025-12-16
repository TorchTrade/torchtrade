
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
import torch

from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit, tf_to_timedelta

from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
import torch

from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit, tf_to_timedelta


class MarketDataObservationSampler:
    def __init__(
        self,
        df: pd.DataFrame,
        time_frames: Union[List[TimeFrame], TimeFrame] = TimeFrame(1, TimeFrameUnit.Minute),
        window_sizes: Union[List[int], int] = 10,
        execute_on: TimeFrame = TimeFrame(1, TimeFrameUnit.Minute),
        feature_processing_fn: Optional[Callable] = None,
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
        for tf in time_frames:
            resampled = (
                self.df.resample(tf.to_pandas_freq())
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
            )
            if feature_processing_fn is not None:
                resampled = feature_processing_fn(resampled)
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

        # Filter execution times
        exec_times = self.df.resample(execute_on.to_pandas_freq()).first().index
        self.min_start_time = latest_first_step + self.max_lookback
        self.exec_times = exec_times[exec_times >= self.min_start_time]
        # create base features of execution time frame (we'll keep DataFrame for column names but also build tensors)
        self.execute_base_features_df = self.df.resample(execute_on.to_pandas_freq()).last()[self.min_start_time:]
        if len(self.execute_base_features_df) == 0:
            raise ValueError("No execute_on base features available after min_start_time")

        self.unseen_timestamps = list(self.exec_times)
        if len(self.exec_times) == 0:
            raise ValueError("Window duration is too large for the given dataset, no execution times found")

        self.max_steps = len(self.exec_times) - 1 if self.max_traj_length is None else min(len(self.exec_times) - 1, self.max_traj_length)
        print("Max steps:", self.max_steps)

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

        # Execute-on base features tensor + index
        base_arr = self.execute_base_features_df.to_numpy(dtype=np.float32)
        self.execute_base_tensor = torch.from_numpy(base_arr)  # shape (M, F)
        base_ts_int64 = self.execute_base_features_df.index.asi8
        self.execute_base_idx = torch.from_numpy(base_ts_int64).to(torch.long)

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
        timestamp = self.unseen_timestamps.pop(0)
        truncated = len(self.unseen_timestamps) == 0
        return self.get_observation(timestamp), timestamp, truncated

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
        keys = self.get_observation_keys()
        columns = [list(self.resampled_dfs[k].columns) for k in keys]
        assert all(c == columns[0] for c in columns), "Not all features are similar across timeframes"
        return columns[0]

    def reset(self, random_start: bool = False) -> int:
        """Reset the unseen timestamps list and return its length."""
        if random_start:
            exec_list = list(self.exec_times)
            if self.max_traj_length is None:
                start_idx = self.np_rng.integers(0, max(1, len(exec_list)))
                self.unseen_timestamps = exec_list[start_idx:]
            else:
                max_start_index = max(0, len(exec_list) - self.max_traj_length)
                start_index = self.np_rng.integers(0, max_start_index + 1)
                self.unseen_timestamps = exec_list[start_index : start_index + self.max_traj_length]
        else:
            if self.max_traj_length is None:
                self.unseen_timestamps = list(self.exec_times)
            else:
                self.unseen_timestamps = list(self.exec_times)[: self.max_traj_length]

        # return number of unseen timestamps
        return len(self.unseen_timestamps)

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

        return {"open": row[0].item(), "high": row[1].item(), "low": row[2].item(), "close": row[3].item(), "volume": row[4].item()}


class MarketDataObservationSampler_old:
    def __init__(
        self,
        df: pd.DataFrame,
        time_frames: Union[List[TimeFrame], TimeFrame] = TimeFrame(1, TimeFrameUnit.Minute),
        window_sizes: Union[List[int], int] = 10,
        execute_on: TimeFrame = TimeFrame(1, TimeFrameUnit.Minute),
        feature_processing_fn: Optional[Callable] = None,
        features_start_with: str = "features_",
        max_traj_length: Optional[int] = None,
    ):
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        # Check if columns match
        if list(df.columns) != required_columns:
            print("⚠️ Columns do not match the required format.")
            print("Current columns:", list(df.columns))
            print("Updating columns to:", required_columns)
            df.columns = required_columns
        else:
            print("✅ Columns already in correct format:", required_columns)
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp").sort_index()
        self.df = df
        
        self.time_frames = time_frames
        self.window_sizes = window_sizes
        self.max_traj_length = max_traj_length
        self.execute_on = execute_on
        self.feature_processing_fn = feature_processing_fn
        
        # Precompute resampled OHLCV DataFrames for each timeframe
        self.resampled_dfs = {}
        first_time_stamps = []
        for tf in time_frames:
            resampled = self.df.resample(tf.to_pandas_freq()).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            if feature_processing_fn is not None:
                resampled = feature_processing_fn(resampled)
                # Keep timestamp and columns starting with "feature_"
                cols_to_keep = [col for col in resampled.columns if col.startswith(features_start_with)]
                cols_to_keep.append("timestamp")
                # Subset the dataframe
                resampled = resampled[cols_to_keep]
                resampled = resampled.reset_index().set_index("timestamp")
                if 'index' in resampled.columns:
                    resampled = resampled.drop(columns=['index'])

            self.resampled_dfs[tf.obs_key_freq()] = resampled
            first_time_stamps.append(resampled.index.min())

        
        # Maximum lookback window
        window_durations = [tf_to_timedelta(tf) * ws for tf, ws in zip(time_frames, window_sizes)]
        self.max_lookback = max(window_durations)
        latest_first_step = max(first_time_stamps)

        # Filter execution times
        # Some values might be nan drop them
        exec_times = self.df.resample(execute_on.to_pandas_freq()).first().index
        self.min_start_time = latest_first_step + self.max_lookback
        self.exec_times = exec_times[exec_times >= self.min_start_time]
        # create base features of execution time frame     
        self.execute_base_features = self.df.resample(execute_on.to_pandas_freq()).last()[self.min_start_time:]
        self.unseen_timestamps = list(self.exec_times)
        if len(self.exec_times) == 0:
            raise ValueError("Window duration is too large for the given dataset, no execution times found")
        
        self.max_steps = len(self.exec_times) -1 if self.max_traj_length is None else min(len(self.exec_times) -1, self.max_traj_length)
        
    def get_random_timestamp(self, without_replacement: bool = False)->pd.Timestamp:
        """Get a random timestamp from the dataset.
        If without_replacement is True, the timestamp is removed from the list of unseen timestamps.
        """
        if without_replacement:
            random_idx = np.random.choice(len(self.unseen_timestamps), size=1, replace=False)
            return self.unseen_timestamps.pop(random_idx.item())
        else:
            random_idx = np.random.randint(0, len(self.exec_times))
            return self.exec_times[random_idx]

    def get_random_observation(self, without_replacement: bool = False)->Tuple[np.ndarray, pd.Timestamp]:
        """Get a random observation from the dataset.
        If without_replacement is True, the timestamp is removed from the list of unseen timestamps.
        """
        timestamp = self.get_random_timestamp(without_replacement)
        return self.get_observation(timestamp), timestamp, False

    def get_sequential_observation(self)->Tuple[np.ndarray, pd.Timestamp]:
        """Get the next observation in the dataset.
        The timestamp is removed from the list of unseen timestamps.
        """
        timestamp = self.unseen_timestamps.pop(0)
        if len(self.unseen_timestamps) == 0:
            truncated = True
        else:
            truncated = False
        return self.get_observation(timestamp), timestamp, truncated

    def get_observation(self, timestamp: pd.Timestamp)->np.ndarray:
        """Get an observation from the dataset at the given timestamp."""
        obs = {}
        for tf, ws in zip(self.time_frames, self.window_sizes):
            resampled = self.resampled_dfs[tf.obs_key_freq()]
            window = resampled.loc[:timestamp].tail(ws).values
            if len(window) < ws:
                raise ValueError("Not enough data for the largest window")
            obs[tf.obs_key_freq()] = window
        return obs

    def get_max_steps(self)->int:
        return self.max_steps

    def get_observation_keys(self)->List[str]:
        return list(self.resampled_dfs.keys())

    def get_feature_keys(self)->List[str]:
        # check first if all features are the same across timeframes
        time_frame_keys = self.get_observation_keys()

        columns = [list(self.resampled_dfs[tf].columns) for tf in time_frame_keys]
        assert all(lst == columns[0] for lst in columns), "Not all features are similar across timeframes"

        return columns[0]


    def reset(self, random_start: bool = False)->None:
        """Reset the observation sampler."""
        if random_start:
            exec_times_list = list(self.exec_times)
            max_start_index = max(0, len(exec_times_list) - self.max_traj_length)
            start_index = np.random.randint(0, max_start_index + 1)  # +1 because randint is exclusive at the top
            self.unseen_timestamps = exec_times_list[start_index:]
        else:
            self.unseen_timestamps = list(self.exec_times)[:self.max_steps]
        return len(self.unseen_timestamps)

    def get_base_features(self, timestamp: pd.Timestamp)->pd.DataFrame:
        """Get the base features from the dataset at the given timestamp."""
        return self.execute_base_features.loc[timestamp]