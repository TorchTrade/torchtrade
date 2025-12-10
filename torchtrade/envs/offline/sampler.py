
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd

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
        print("Max steps:", self.max_steps)
        # convert dataframe to numpy for faster slicing
        self.np_arrays = {}
        self.np_idx = {}
        self.np_base = self.execute_base_features.to_numpy()
        self.base_idx = self.execute_base_features.index.values

        for key, df in self.resampled_dfs.items():
            self.np_arrays[key] = df.to_numpy()
            self.np_idx[key] = df.index.values

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
        current_t_stmp_np = timestamp.to_numpy()
        for tf, ws in zip(self.time_frames, self.window_sizes):
            key = tf.obs_key_freq()
            idx = self.np_idx[key]
            arr = self.np_arrays[key]
            pos = idx.searchsorted(current_t_stmp_np, side="right")
            window = arr[max(0, pos-ws):pos]
            if len(window) < ws:
                raise ValueError("Not enough data for the largest window")
            obs[key] = window
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
            self.unseen_timestamps = exec_times_list[start_index:start_index + self.max_traj_length]
        else:
            self.unseen_timestamps = list(self.exec_times)[:self.max_steps]

        assert len(self.unseen_timestamps) == self.max_steps, f"unseen_timestamps-length: {len(self.unseen_timestamps)} is not similar to max timesteps {self.max_steps}"
        return len(self.unseen_timestamps)

    def get_base_features(self, timestamp: pd.Timestamp)->pd.DataFrame:
        """Get the base features from the dataset at the given timestamp."""
        np_stamp = timestamp.to_numpy()
        pos = self.base_idx.searchsorted(np_stamp, side="left") # side left / right?
        return self.np_base[pos][3] # O, H, L, C, V -> Close = 3

        #return self.execute_base_features.loc[timestamp]


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