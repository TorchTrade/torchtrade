"""Replay observer for historical data playback through live envs."""

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import pandas as pd

from torchtrade.envs.offline.infrastructure.sampler import MarketDataObservationSampler
from torchtrade.envs.utils.timeframe import TimeFrame


class ReplayObserver:
    """Replay observer that feeds historical data through the live observer interface.

    Wraps MarketDataObservationSampler to provide bar-by-bar observation playback.
    Implements the same interface as BinanceObservationClass/BybitObservationClass
    so it can be injected into any live SLTP environment.

    Note: Raises StopIteration when historical data is exhausted. Live envs do not
    catch this — ensure replay episodes are shorter than the available data, or wrap
    the env in a try/except StopIteration handler for long-running evaluations.
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
        self.window_sizes = self.sampler.window_sizes
        # `{timeframe}_{window}`, matching what the real observers return -- the env
        # bases prepend "market_data_" themselves. The sampler's raw keys are bare
        # timeframe names, so the env built `market_data_1Minute` where offline and live
        # both produce `market_data_1Minute_10`, and a policy trained offline could not
        # be fed the replay env without renaming its in_keys (#278). Adding the prefix
        # HERE instead produced `market_data_market_data_1Minute_10`.
        self._timeframes = self.sampler.get_observation_keys()
        self._obs_keys = [
            f"{name}_{ws}" for name, ws in zip(self._timeframes, self.window_sizes)
        ]
        self.truncated = False

    def get_observations(self, return_base_ohlc: bool = False) -> Dict[str, np.ndarray]:
        """Get next observation by advancing the sampler one bar.

        Raises:
            StopIteration: When historical data is exhausted
        """
        if self._is_exhausted():
            raise StopIteration("ReplayObserver reached end of historical data")

        obs, timestamp, truncated = self.sampler.get_sequential_observation()

        self.truncated = truncated
        base = self.sampler.get_base_features(timestamp)

        if self.executor is not None:
            self.executor.advance_bar({
                "open": base["open"],
                "high": base["high"],
                "low": base["low"],
                "close": base["close"],
            })

        # Re-keyed to the suffixed names, so the dict a policy receives matches the one
        # offline and live produce.
        suffix = dict(zip(self._timeframes, self._obs_keys))
        result = {}
        for key, tensor in obs.items():
            result[suffix.get(key, key)] = tensor.numpy().astype(np.float32)

        if return_base_ohlc:
            result["base_features"] = self._base_window(timestamp, base)

        return result

    def get_keys(self) -> List[str]:
        """Get observation dictionary keys."""
        return self._obs_keys

    def get_features(self) -> Dict[str, List[str]]:
        """The columns the sampler ACTUALLY emits, which is what the spec is built from.

        Filtering to names starting with `features_` reported an empty list whenever no
        `feature_preprocessing_fn` was given -- while the sampler still emitted the five
        raw OHLCV columns. The declared spec came out `(window, 0)` against an emitted
        `(window, 5)`, and `check_env_specs` failed with a size mismatch (#278). Offline
        counts the same columns via `get_num_features_per_timeframe`.
        """
        # No `except: []` fallback. Swallowing the per-timeframe-features error declared
        # a (window, 0) spec against a (window, 2) emission, which is the same mismatch
        # the filtering caused -- a fail-open guard in the boundary-validation sense.
        # Raising says which config is unsupported instead of shipping a broken spec.
        feature_keys = self.sampler.get_feature_keys()
        return {
            "observation_features": list(feature_keys),
            "original_features": ["open", "high", "low", "close", "volume"],
        }

    def _is_exhausted(self) -> bool:
        """Check if the sampler has no more data.

        Note: Accesses sampler internals (_sequential_idx, _end_idx).
        If MarketDataObservationSampler adds a public is_exhausted() method,
        prefer that instead.
        """
        return self.sampler._sequential_idx >= self.sampler._end_idx

    def reset(self) -> None:
        """Rewind to the start of the data, and the simulated account with it."""
        self.sampler.reset(random_start=False)
        self.truncated = False
        if self.executor is not None:
            self.executor.reset()

    def _base_window(self, timestamp, current) -> np.ndarray:
        """The last `window` execution bars of OHLC, not just the newest one.

        Only `base_arr[-1]` used to be populated, on the reasoning that live envs read
        `[-1, 3]`. But `include_base_features=True` declares a `(window, 4)` spec and puts
        the array in the OBSERVATION, so a policy consuming that key read 90% zeros --
        9 of 10 rows (#278). Real observers fill the whole window.

        The FIRST timeframe's bars, not the execution timeframe's. Live gates on
        `timeframe == self.time_frames[0]` and every venue sizes the spec off
        `window_sizes[0]`, so filling it from `execute_on` produced a window of the right
        SHAPE holding entirely different bars -- with `time_frames=[15Min, 1Min]` a
        replayed window spanned 4 minutes where live spans an hour. Worse than the
        original bug, which left the mismatched rows visibly zero; this one looks
        plausible.

        Untruncated, so a window ending at the first tradable bar has bars to look back
        at: the truncated frame starts at min_start_time, which left step 1 at 9/10 zeros
        while `market_data_*` was already full.
        """
        ws = self.window_sizes[0]
        base_arr = np.zeros((ws, 4), dtype=np.float32)
        # Through the sampler's own rule, not a re-implemented searchsorted. A frame
        # FINER than execute_on is labelled by bar start, so looking it up at the raw
        # execution label returned a window shifted back by `exec_period/tf0_period - 1`
        # bars -- stale, deterministically, whenever execute_on is coarser than
        # time_frames[0]. `_search_ts` exists for exactly this and is what every other
        # lookup in the sampler goes through.
        lookup = self.sampler._search_ts(self._timeframes[0], int(timestamp.value))
        end = int(torch.searchsorted(
            self.sampler.base_ohlc_idx,
            torch.tensor(int(lookup), dtype=torch.long),
            right=True,
        ).item())
        if end <= 0:
            return base_arr
        start = max(0, end - ws)
        window = self.sampler.base_ohlc_tensor[start:end].numpy().astype(np.float32)
        base_arr[-len(window):] = window

        # The last row is the FORMING bar, as live's is: its close is the latest trade,
        # not the last completed tf0 close. `base_features[-1, 3]` is the entry and
        # bracket price for three SLTP envs (`*/env_sltp.py: current_price = float(
        # obs["base_features"][-1, 3])`), so leaving it at the last CLOSED tf0 bar priced
        # brackets off a stale number -- measured at -1.94%/+3.07% against a configured
        # -2%/+3% on a [15Min, 1Min] layout. High and low extend to cover the trade.
        base_arr[-1, 1] = max(base_arr[-1, 1], current["high"])
        base_arr[-1, 2] = min(base_arr[-1, 2], current["low"])
        base_arr[-1, 3] = current["close"]
        return base_arr
