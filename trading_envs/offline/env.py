import pandas as pd
import numpy as np
import gymnasium as gym  # Use gymnasium (modern gym) for RL environments
from gymnasium import spaces
from enum import Enum
from typing import List

# Define TimeFrameUnit as an Enum for units like Minute, Hour
class TimeFrameUnit(Enum):
    Minute = 'T'  # Pandas freq for minutes
    Hour = 'H'    # Pandas freq for hours
    # Add more if needed, e.g., Day = 'D'

# Define TimeFrame class
class TimeFrame:
    def __init__(self, value: int, unit: TimeFrameUnit):
        self.value = value
        self.unit = unit

    def to_pandas_freq(self) -> str:
        return f"{self.value}{self.unit.value}"

# MultiTimeFrameTradingEnv: A flexible, high-performance Gym environment for RL trading
class MultiTimeFrameTradingEnv(gym.Env):
    """
    A custom Gym environment for RL-based trading using multi-timeframe observations.
    
    - Precomputes resampled OHLCV data for each timeframe to ensure high performance.
    - Observations: Stacked windows of OHLCV from multiple timeframes.
    - Actions: Discrete (e.g., 0: hold, 1: buy, 2: sell). Customize as needed.
    - Steps: Advances based on the 'execute_on' timeframe.
    - Rewards: Placeholder (e.g., based on profit). Implement your reward logic.
    
    Assumptions:
    - Input df has columns: ['timestamp' (datetime), 'open', 'high', 'low', 'close', 'volume']
    - Data is sorted by timestamp.
    - Enough historical data to fill windows (pads with zeros if insufficient).
    - No normalization in observations (add if needed, e.g., divide by last close).
    
    Usage Example:
    time_frames = [
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
        TimeFrame(15, TimeFrameUnit.Minute),
        TimeFrame(1, TimeFrameUnit.Hour),
    ]
    window_sizes = [12, 8, 8, 24]  # ~12m, 40m, 2h, 1d
    execute_on = TimeFrame(5, TimeFrameUnit.Minute)
    
    env = MultiTimeFrameTradingEnv(full_df, time_frames, window_sizes, execute_on)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        time_frames: List[TimeFrame],
        window_sizes: List[int],
        execute_on: TimeFrame,
        action_space_size: int = 3,  # e.g., hold, buy, sell
        features_per_bar: int = 5,   # OHLCV
    ):
        super(MultiTimeFrameTradingEnv, self).__init__()
        
        if len(time_frames) != len(window_sizes):
            raise ValueError("time_frames and window_sizes must have the same length")
        
        # Set index to timestamp if not already
        self.df = df.set_index('timestamp') if not isinstance(df.index, pd.DatetimeIndex) else df
        
        self.time_frames = time_frames
        self.window_sizes = window_sizes
        self.execute_on = execute_on
        self.features_per_bar = features_per_bar  # OHLCV: 5 features
        
        # Precompute resampled OHLCV DataFrames for each timeframe (high performance)
        self.resampled_dfs = []
        for tf in time_frames:
            resampled = self.df.resample(tf.to_pandas_freq()).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            self.resampled_dfs.append(resampled)
        
        # Get execution timestamps: the points where the agent acts (resampled to execute_on)
        # Use the start of each execute_on period
        self.exec_times = self.df.resample(execute_on.to_pandas_freq()).first().index
        
        # Filter exec_times to ensure we have enough data for the largest window
        max_window_duration = max(ws * tf.value for tf, ws in zip(time_frames, window_sizes))
        min_start_time = self.df.index.min() + pd.Timedelta(minutes=max_window_duration)  # Rough estimate in minutes
        self.exec_times = self.exec_times[self.exec_times >= min_start_time]
        
        self.max_steps = len(self.exec_times) - 1
        self.current_step = 0
        
        # Observation space: flattened array of all windows (ws * features_per_bar for each tf)
        obs_size = sum(ws * features_per_bar for ws in window_sizes)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        # Action space: discrete (customize as needed)
        self.action_space = spaces.Discrete(action_space_size)
        
        # Placeholder for position, balance, etc. (for reward calculation)
        self.current_position = 0  # 0: no position, 1: long, -1: short (customize)
        self.balance = 10000.0     # Initial balance (customize)
        self.current_price = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_position = 0
        self.balance = 10000.0
        self.current_price = self._get_current_price()
        return self._get_observation(), {}
    
    def step(self, action):
        # Placeholder action logic (customize based on your RL strategy)
        # Example: 0: hold, 1: buy (open long), 2: sell (close or short)
        prev_balance = self.balance
        self.current_price = self._get_current_price()
        
        if action == 1:  # Buy
            if self.current_position == 0:
                self.current_position = 1
                # Simulate buying 1 unit (customize position sizing)
                self.balance -= self.current_price  # Assume buying 1 BTC
        elif action == 2:  # Sell
            if self.current_position == 1:
                self.current_position = 0
                self.balance += self.current_price
        
        # Advance step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False  # Or implement episode truncation logic
        
        # Reward: simple profit-based (customize, e.g., Sharpe ratio, drawdown penalty)
        reward = self.balance - prev_balance
        
        # Next observation
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        return obs, reward, done, truncated, {}
    
    def _get_observation(self) -> np.ndarray:
        """Get multi-timeframe observation: stacked windows of OHLCV."""
        current_time = self.exec_times[self.current_step]
        obs_parts = []
        
        for res_df, ws in zip(self.resampled_dfs, self.window_sizes):
            # Get the last 'ws' bars up to (inclusive) current_time
            past_data = res_df[res_df.index <= current_time].tail(ws)
            
            # Pad with zeros if insufficient data (e.g., at the start)
            if len(past_data) < ws:
                padding = pd.DataFrame(np.zeros((ws - len(past_data), self.features_per_bar)),
                                       columns=['open', 'high', 'low', 'close', 'volume'])
                past_data = pd.concat([padding, past_data])
            
            # Flatten OHLCV (ws * 5)
            flat = past_data[['open', 'high', 'low', 'close', 'volume']].values.flatten()
            obs_parts.append(flat)
        
        # Concatenate all parts
        obs = np.concatenate(obs_parts).astype(np.float32)
        
        # Optional: Normalize (e.g., divide by last close across all)
        # last_close = obs[-1] if obs[-1] != 0 else 1.0  # Avoid div by zero
        # obs /= last_close
        
        return obs
    
    def _get_current_price(self) -> float:
        """Get current close price."""
        current_time = self.exec_times[self.current_step]
        return self.df.loc[current_time, 'close']  # Or use resampled if needed
    
    def render(self, mode='human'):
        # Optional: Print current step, balance, etc.
        print(f"Step: {self.current_step}, Balance: {self.balance}, Price: {self._get_current_price()}")