from collections import namedtuple
from typing import Dict, List, Optional, Tuple, Union, Callable, Sequence
import logging
import warnings
import numpy as np
import pandas as pd
import torch

from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit, tf_to_timedelta

# Type alias for feature processing function(s)
FeatureProcessingFn = Optional[Union[Callable, Sequence[Callable]]]

logger = logging.getLogger(__name__)

# PERF: NamedTuple is faster than dict for attribute access
OHLCV = namedtuple('OHLCV', ['open', 'high', 'low', 'close', 'volume'])


def _bounded(value, limit: int = 60) -> str:
    """Bound the index, the one field of the malformed-OHLC message with no natural size.

    The OHLC values are floats, whose str() is bounded by construction, so they are
    interpolated directly -- putting them through a truncating repr mangles ordinary
    prices, which is worse than the unbounded index it would be guarding against.
    """
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} chars)"


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
        required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = sorted(required_columns - set(df.columns))
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}. "
                f"Got columns: {list(df.columns)}"
            )

        # A bar whose high/low do not bracket its open and close is not a bar. Every exit
        # rule reads the extreme to decide whether a level was touched, and the scalar
        # stop-loss additionally reads the open to price a gapped fill (#280) -- so on a
        # malformed row those two disagree, and the engines can answer differently (#326).
        # Rejecting here rather than guarding in the rules is the boundary-validation
        # invariant: a guard that absorbs the nonsense makes it silent.
        # On numpy arrays rather than DataFrame.max(axis=1), which materialises a
        # two-column copy twice -- 287ms vs 7.6ms on a 2.5M-row frame, paid once per
        # ParallelEnv worker.
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        bad = (h < np.maximum(o, c)) | (l > np.minimum(o, c))
        if bad.any():
            first = int(np.argmax(bad))
            raise ValueError(
                f"Malformed OHLC in {int(bad.sum())} of {len(df)} rows. First at position "
                f"{first} (index {_bounded(df.index[first])}): open={o[first]}, "
                f"high={h[first]}, low={l[first]}, close={c[first]}. high must be >= "
                "max(open, close) and low <= min(open, close) -- if you built these bars, "
                "clamp high/low around open and close rather than drawing them off close "
                "alone."
            )

        # Canonical OHLCV-first order for self.df. The row[:5] contract itself comes from
        # ohlcv_agg's key order, since pandas orders .agg(dict) output by dict key.
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        aux_cols = [c for c in df.columns if c not in required_columns]
        df = df[["timestamp"] + ohlcv_cols + aux_cols]

        # Make sure time_frames and window_sizes are lists of same length
        if isinstance(time_frames, TimeFrame):
            time_frames = [time_frames]
        if isinstance(window_sizes, int):
            window_sizes = [window_sizes] * len(time_frames)
        if len(window_sizes) != len(time_frames):
            raise ValueError("window_sizes must be an int or list with same length as time_frames")

        # Normalize feature_processing_fn to a list (one per timeframe)
        processing_fns: List[Optional[Callable]]
        if feature_processing_fn is None:
            processing_fns = [None] * len(time_frames)
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

        # A tz-aware index anchors Day bins to LOCAL midnight, so a DST spring-forward
        # day is 23h wide while the fixed period below stays 24h -- and the bound would
        # then reach an hour past the bin's real close. That is lookahead, not
        # staleness. Nothing in this repo produces a tz-aware index, so refuse it at the
        # boundary rather than carry a silent wrong number (#282).
        if df.index.tz is not None and execute_on >= TimeFrame(1, TimeFrameUnit.Day):
            raise ValueError(
                f"execute_on={execute_on.obs_key_freq()} needs a tz-naive index: "
                "day-or-longer bins follow local midnight, so DST transitions make them "
                "23h or 25h wide and the observation window would run past the bin's "
                "close. Convert to UTC and drop the timezone."
            )

        self.time_frames = time_frames
        self.window_sizes = window_sizes
        self.max_traj_length = max_traj_length
        self.execute_on = execute_on
        self.feature_processing_fn = feature_processing_fn
        self.features_start_with = features_start_with
        # Precompute resampled DataFrames for each timeframe
        # OHLCV uses canonical aggregation; auxiliary columns use "last"
        ohlcv_agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        full_agg = {**ohlcv_agg, **{c: "last" for c in aux_cols}}

        self.resampled_dfs: Dict[str, pd.DataFrame] = {}
        first_time_stamps = []
        for tf, proc_fn in zip(time_frames, processing_fns):
            binned = self.df.resample(tf.to_pandas_freq()).agg(full_agg)
            # The complete bin grid, before empty bins are dropped. A bin's real END is
            # where the NEXT bin starts, so the END labels below are read off this grid
            # rather than computed from a bar's own label (#320).
            bin_starts = binned.index
            resampled = binned.dropna(subset=list(ohlcv_agg.keys()))
            # Forward-fill auxiliary NaN (sparse aux data persists last known value)
            if aux_cols:
                resampled[aux_cols] = resampled[aux_cols].ffill().fillna(0)

            # Before the relabel: it reads bin_starts[0], which IndexErrors on an empty
            # frame and would mask this message when a coarse frame is listed first.
            if len(resampled) == 0:
                raise ValueError(f"Resampled dataframe for timeframe {tf.obs_key_freq()} is empty")

            # Fix lookahead bias: shift higher timeframe bars forward by their period
            # This ensures bars are indexed by their END time, not START time
            # Only completed bars will be visible to the agent at any execution time
            # Only shift timeframes that are HIGHER (coarser) than the execution timeframe
            if tf > execute_on:
                # Relabel each bar to where its bin actually ends: the start of the next
                # bin on the full grid. Arithmetic on the label instead -- whether a fixed
                # +period or DatetimeIndex.shift -- is wrong on a tz-aware Day-or-longer
                # frame, because those bins follow LOCAL midnight and are 23h or 25h wide
                # across a DST transition. A fixed +1D labels a 25h bin an hour before it
                # closed, i.e. shows it early: lookahead, not staleness (#320).
                #
                # Reading the grid rather than shifting also survives the two cases that
                # make .shift unusable here: it regenerates from `bin_starts[0] + period`,
                # so a window STARTING on a transition day is off for every bar, and it
                # silently degrades to fixed arithmetic once dropna clears index.freq --
                # which any gap (a weekend, a halt) does.
                n_edges = len(bin_starts) + 1
                freq = tf.to_pandas_freq()
                if bin_starts.tz is None or tf < TimeFrame(1, TimeFrameUnit.Day):
                    edges = pd.date_range(
                        bin_starts[0], periods=n_edges, freq=freq, tz=bin_starts.tz
                    )
                else:
                    # Day-or-longer tz-aware bins follow LOCAL midnight, and in zones whose
                    # DST step lands ON midnight (Havana, Beirut, Santiago, ...) that
                    # midnight is nonexistent or ambiguous -- tz-aware date_range raises
                    # there while resample resolves it. Walk the wall clock and re-localize
                    # the way resample's own binner does, so those zones keep working.
                    # Sub-day bins must NOT take this path: they walk absolute time, and a
                    # wall-clock walk would skip an hour at spring-forward.
                    edges = pd.date_range(
                        bin_starts[0].tz_localize(None), periods=n_edges, freq=freq
                    ).tz_localize(
                        bin_starts.tz, nonexistent="shift_forward", ambiguous=True
                    )
                # rename: the END labels must keep the index's name, which the
                # feature-processing path below reset_index()es on.
                resampled.index = edges[1:][
                    bin_starts.get_indexer(resampled.index)
                ].rename(bin_starts.name)

            if proc_fn is not None:
                resampled = proc_fn(resampled)
                cols_to_keep = [col for col in resampled.columns if col.startswith(features_start_with)]
                if len(cols_to_keep) == 0:
                    raise ValueError(
                        f"Feature processing function for timeframe {tf.obs_key_freq()} "
                        f"produced no columns starting with '{features_start_with}'. "
                        f"Ensure your function adds columns like 'features_close', 'features_volume', etc."
                    )
                cols_to_keep.append("timestamp")
                resampled = resampled[cols_to_keep]
                resampled = resampled.reset_index().set_index("timestamp")
                if "index" in resampled.columns:
                    resampled = resampled.drop(columns=["index"])


            self.resampled_dfs[tf.obs_key_freq()] = resampled
            first_time_stamps.append(resampled.index.min())

        # Maximum lookback window (timedelta)
        window_durations = [tf_to_timedelta(tf) * ws for tf, ws in zip(time_frames, window_sizes)]
        self.max_lookback = max(window_durations)
        latest_first_step = max(first_time_stamps)

        # Calculate offset needed for higher timeframes (to account for END-time indexing)
        # We need extra data to ensure sufficient lookback after the offset.
        # Must be the SAME test as the shift above: reserving an offset for a timeframe
        # that was never shifted only discards leading data (#282).
        higher_tf_offsets = [tf_to_timedelta(tf) for tf in time_frames if tf > execute_on]
        max_offset = max(higher_tf_offsets) if higher_tf_offsets else pd.Timedelta(0)

        # Filter execution times
        exec_times = self.df.resample(execute_on.to_pandas_freq()).first().index
        self.min_start_time = latest_first_step + self.max_lookback + max_offset
        self.exec_times = exec_times[exec_times >= self.min_start_time]
        # SL/TP and liquidation checks read this bar's high/low, so it must span the
        # whole bar, not just the final sub-bar. Only OHLCV is ever read here (aux
        # features reach observations via resampled_dfs), and excluding aux keeps the
        # row[:5] positional contract exact rather than merely conventional.
        execute_base_raw = self.df.resample(execute_on.to_pandas_freq()).agg(ohlcv_agg)

        # Detect and warn about data gaps. ANY price field, not close alone (#353): a bar
        # with a good close and a missing open was never warned about, then forward-filled
        # -- and open is what stop_fill_price uses to price a gapped stop.
        nan_mask = execute_base_raw[["open", "high", "low", "close"]].isna().any(axis=1)
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

        execute_base_filled = execute_base_raw.ffill()
        # Forward-filling volume is defensible; forward-filling a price that FILLS ORDERS
        # is not -- stop_fill_price would fill at a price the market never traded, and
        # high/low would answer "was this level touched?" with the previous bar's range.
        #
        # PER FIELD, not per row: a row mask overwrites the fields that were PRESENT, so a
        # bar missing only its open lost its real high and low and became a flat bar on
        # which no stop can trigger -- a backtest that skips losing exits, which is worse
        # than the forward-fill it replaced. An empty bar has all three missing and still
        # collapses to close, which is the behaviour that was already correct (#353).
        for field in ("open", "high", "low"):
            execute_base_filled[field] = execute_base_filled[field].where(
                ~execute_base_raw[field].isna(), execute_base_filled["close"]
            )

        # A stale close paired with this bar's real range can violate low <= close <= high,
        # and close is what becomes current_price, the next entry, the mark and the reward.
        # Clipping keeps it inside a range the market actually traded.
        # Restore the OHLC invariant the collapse can break: a bar whose `open` is real and
        # whose `high` is missing takes close as its high, which can land BELOW open -- the
        # malformed shape the ingestion validator rejects on raw input, reintroduced from
        # the inside. Widen the extremes to span the two prices that are real, then clip
        # close, which becomes current_price, the next entry, the mark and the reward.
        # Widened from OPEN only, never from close. A red bar missing its high takes close
        # as its high and lands BELOW open -- the malformed shape the ingestion validator
        # rejects, recreated where no validator runs. `open` is a real print of THIS bar,
        # so this invents nothing; widening with a stale close would drag a real low of 110
        # down to 100 and fire a stop at 105 on a bar that never traded there.
        execute_base_filled["high"] = execute_base_filled[["high", "open"]].max(axis=1)
        execute_base_filled["low"] = execute_base_filled[["low", "open"]].min(axis=1)

        # A bar that merely lacks a usable close KEEPS its real range (existing contract --
        # discarding it recreates the too-narrow range this aggregation exists to fix), so
        # the stale close is clipped into that range rather than the range widened to it.
        execute_base_filled["close"] = execute_base_filled["close"].clip(
            lower=execute_base_filled["low"], upper=execute_base_filled["high"]
        )
        self.execute_base_features_df = execute_base_filled[self.min_start_time:]
        if len(self.execute_base_features_df) == 0:
            raise ValueError("No execute_on base features available after min_start_time")

        if len(self.exec_times) == 0:
            raise ValueError("Window duration is too large for the given dataset, no execution times found")

        self.max_steps = len(self.exec_times) - 1 if self.max_traj_length is None else min(len(self.exec_times) - 1, self.max_traj_length)
        logger.debug("Max steps: %d", self.max_steps)

        self._sequential_idx = 0
        self._end_idx = len(self.exec_times)
        # One-shot deterministic seek target for the NEXT reset(random_start=True)
        # call; set via seek(), consumed+cleared by reset(). None means "no seek
        # pending" -> reset() draws a random start index as usual.
        self._forced_start_idx: Optional[int] = None
        # PERF: Store exec_times as numpy array for fast indexing (avoid pandas overhead)
        self._exec_times_arr = self.exec_times.to_numpy()

        # PERF: Pre-compute observation indices for each execution timestamp per timeframe
        # This allows get_observation_sequential() to use direct indexing instead of searchsorted
        self._obs_indices: Dict[str, np.ndarray] = {}
        # Force nanosecond resolution for consistent int64 representation.
        # In pandas 3.0+, .asi8 returns values in the index's native resolution
        # (e.g., microseconds), but pd.Timestamp.value always returns nanoseconds.
        # Using .as_unit('ns') ensures searchsorted comparisons are consistent.
        exec_ts_int64 = self.exec_times.as_unit('ns').asi8  # int64 nanoseconds

        # Finer-than-execute_on frames are labelled by their bar's START -- only coarser
        # ones get shifted to END labels above -- so looking one up at the exec bin's
        # start label returns a bar that has barely begun (#282). See _search_ts.
        self._exec_period_ns = tf_to_timedelta(execute_on).value
        self._fine_period_ns = {
            tf.obs_key_freq(): tf_to_timedelta(tf).value
            for tf in time_frames if tf < execute_on
        }

        # Convert resampled dfs to torch tensors for fast slicing.
        # Also convert timestamp indices to int64 (ns) and store as torch.long for searchsorted.
        self.torch_tensors: Dict[str, torch.FloatTensor] = {}
        self.torch_idx: Dict[str, torch.LongTensor] = {}

        for key, df_tf in self.resampled_dfs.items():
            # values -> float32
            arr = df_tf.to_numpy(dtype=np.float32)
            self.torch_tensors[key] = torch.from_numpy(arr)  # shape (N, F)

            # timestamps as int64 nanoseconds (force ns for pandas 3.0+ compatibility)
            ts_int64 = df_tf.index.as_unit('ns').asi8  # numpy ndarray (int64)
            self.torch_idx[key] = torch.from_numpy(ts_int64).to(torch.long)  # sorted 1D long tensor

            # Use numpy searchsorted once at init instead of torch searchsorted per step
            obs_idx = np.searchsorted(
                ts_int64, self._search_ts(key, exec_ts_int64), side='right'
            ) - 1
            self._obs_indices[key] = obs_idx

        # Execute-on base features tensor + index
        base_arr = self.execute_base_features_df.to_numpy(dtype=np.float32)
        self.execute_base_tensor = torch.from_numpy(base_arr)  # shape (M, F)
        base_ts_int64 = self.execute_base_features_df.index.as_unit('ns').asi8
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


    def _search_ts(self, key: str, exec_ts_ns):
        """Where to look `key` up, given an execution bin's start label.

        Works on a scalar or an int64 array, so both lookup paths share one rule.

        A frame at or coarser than execute_on is searched at the label itself. A finer
        one is searched at the last label whose bar CLOSES at or before the bin ends.
        Not the last bar that *starts* inside the bin: pandas aggregates a bar across
        its whole span, so one starting at minute 70 with a 7-minute period carries a
        close from minute 76 -- past a bin ending at 75, and into the future.

        Assumes a tz-naive index, as every dataset in this repo is. Minute and Hour
        resample as fixed-width offsets and are DST-immune regardless, but a tz-AWARE
        index with execute_on=Day anchors bins to local midnight, making them 23h or 25h
        while _exec_period_ns stays 24h -- which would overshoot on a short day.
        """
        fine_period_ns = self._fine_period_ns.get(key)
        if fine_period_ns is None:
            return exec_ts_ns
        return exec_ts_ns + self._exec_period_ns - fine_period_ns

    def get_observation(self, timestamp: pd.Timestamp) -> Dict[str, torch.Tensor]:
        """Return observation dict: { timeframe_key: tensor(shape=[ws, features]) }

        `timestamp` must be an execution bin's START label, as `exec_times` produces and
        `get_sequential_observation` returns. A timestamp part-way through a bin is
        resolved against that bin, so a finer frame would end after the instant asked
        for -- see _search_ts.
        """
        obs: Dict[str, torch.Tensor] = {}
        # convert timestamp to int64 ns
        ts_int = int(timestamp.value)  # pd.Timestamp.value is int64 ns

        for tf, ws in zip(self.time_frames, self.window_sizes):
            key = tf.obs_key_freq()

            arr = self.torch_tensors[key]           # (N, F) float tensor
            idx_tensor = self.torch_idx[key]        # (N,) long sorted tensor

            # Same rule as the precomputed path, or a finer frame would land on a
            # different bar here than in get_sequential_observation()
            ts_t = torch.tensor(self._search_ts(key, ts_int), dtype=torch.long)

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

    def _max_organic_start_idx(self) -> int:
        """Upper bound (inclusive) of the start index reset(random_start=True) can
        draw on its own, given the current max_traj_length setting. Shared by
        reset() (to draw within range) and seek() (to validate a forced index is
        something an organic draw could have produced)."""
        total_len = len(self._exec_times_arr)
        if self.max_traj_length is None:
            return max(0, total_len - 1)
        return max(0, total_len - self.max_traj_length)

    def seek(self, index: int) -> None:
        """Force the NEXT reset(random_start=True) call to start exactly at
        `index`, instead of drawing a random start index via np_rng.

        One-shot: consumed (and cleared) by the very next reset(random_start=True)
        call; normal random sampling resumes on any subsequent reset(). Does not
        read or mutate np_rng state at all, so RNG draw order/count for callers
        that never call seek() is completely unaffected. Purely additive -
        reset(random_start=False) is unaffected entirely, and reset() for any
        caller that never calls seek() runs the exact same branches it always did.

        Args:
            index: The start index reset() should use, in the same terms as the
                `start_idx` an organic random draw would produce (i.e. this is a
                SAMPLER-side, pre-increment index -- NOT
                `OfflineTradingEnvBase._reset_idx`, which is captured one step
                later, post-increment, by get_sequential_observation()).

        Raises:
            ValueError: If `index` is outside the range an organic
                reset(random_start=True) draw could ever produce for the current
                max_traj_length setting.
        """
        max_valid = self._max_organic_start_idx()
        if not (0 <= index <= max_valid):
            raise ValueError(
                f"seek(index={index}) out of range: valid organic start index "
                f"range is [0, {max_valid}] for this sampler "
                f"(total_len={len(self._exec_times_arr)}, "
                f"max_traj_length={self.max_traj_length})."
            )
        self._forced_start_idx = index

    def reset(self, random_start: bool = False) -> int:
        """Reset using index-based tracking."""
        total_len = len(self._exec_times_arr)

        if random_start:
            if self._forced_start_idx is not None:
                start_idx = self._forced_start_idx
                self._forced_start_idx = None
            elif self.max_traj_length is None:
                start_idx = int(self.np_rng.integers(0, max(1, total_len)))
            else:
                max_start_index = self._max_organic_start_idx()
                start_idx = int(self.np_rng.integers(0, max_start_index + 1))

            self._sequential_idx = start_idx
            if self.max_traj_length is None:
                self._end_idx = total_len
            else:
                self._end_idx = min(start_idx + self.max_traj_length, total_len)
        else:
            self._sequential_idx = 0
            if self.max_traj_length is None:
                self._end_idx = total_len
            else:
                self._end_idx = min(self.max_traj_length, total_len)

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
