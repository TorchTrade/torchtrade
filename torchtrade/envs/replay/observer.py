"""Replay observer for historical data playback through live envs."""

import logging
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from torchtrade.envs.offline.infrastructure.sampler import MarketDataObservationSampler
from torchtrade.envs.utils.timeframe import TimeFrame

logger = logging.getLogger(__name__)


class ReplayObserver:
    """Replay observer that feeds historical data through the live observer interface.

    Wraps MarketDataObservationSampler to provide bar-by-bar observation playback.
    Implements the same interface as BinanceObservationClass/BybitObservationClass
    so it can be injected into any live SLTP environment.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time_frames: Union[List[TimeFrame], TimeFrame],
        window_sizes: Union[List[int], int],
        execute_on: TimeFrame,
        feature_preprocessing_fn: Optional[Callable] = None,
        executor=None,
    ):
        self.executor = executor
        self.sampler = MarketDataObservationSampler(
            df=df,
            time_frames=time_frames,
            window_sizes=window_sizes,
            execute_on=execute_on,
            feature_processing_fn=feature_preprocessing_fn,
        )
        self.sampler.reset(random_start=False)
        self._obs_keys = self.sampler.get_observation_keys()
        self.truncated = False

    def get_observations(self, return_base_ohlc: bool = False) -> Dict[str, np.ndarray]:
        """Get next observation by advancing the sampler one bar.

        Raises:
            StopIteration: When historical data is exhausted
        """
        try:
            obs, timestamp, truncated = self.sampler.get_sequential_observation()
        except ValueError:
            raise StopIteration("ReplayObserver reached end of historical data")

        self.truncated = truncated
        base = self.sampler.get_base_features(timestamp)

        if self.executor is not None:
            self.executor.advance_bar({
                "open": base["open"],
                "high": base["high"],
                "low": base["low"],
                "close": base["close"],
            })

        result = {}
        for key, tensor in obs.items():
            result[key] = tensor.numpy().astype(np.float32)

        if return_base_ohlc:
            # Only the last row is populated — live envs only access [-1, 3] (close price).
            # Earlier rows are zero-filled to match the expected (window_size, 4) shape.
            ws = self.sampler.window_sizes[0]
            base_arr = np.zeros((ws, 4), dtype=np.float32)
            base_arr[-1] = [base["open"], base["high"], base["low"], base["close"]]
            result["base_features"] = base_arr

        return result

    def get_keys(self) -> List[str]:
        """Get observation dictionary keys."""
        return self._obs_keys

    def get_features(self) -> Dict[str, List[str]]:
        """Get feature column names."""
        try:
            feature_keys = self.sampler.get_feature_keys()
        except ValueError:
            feature_keys = []
        return {
            "observation_features": [k for k in feature_keys if k.startswith("features_")],
            "original_features": ["open", "high", "low", "close", "volume"],
        }

    def reset(self):
        """Reset observer to start of data."""
        self.sampler.reset(random_start=False)
        self.truncated = False
        if self.executor is not None:
            self.executor.reset()
