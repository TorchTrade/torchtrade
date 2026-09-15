"""Multi-asset observation sampler for the portfolio envs.

One unchanged MarketDataObservationSampler per asset, so the lookahead rules for coarse
and fine frames (#282, #320) exist in one place. `listed` and `tradable` ride along as aux
columns, which puts them through the same END-label relabel as the prices.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from torchtrade.envs.offline.infrastructure.sampler import MarketDataObservationSampler
from torchtrade.envs.utils.timeframe import TimeFrame, tf_to_timedelta

BAR_COLUMNS = ["timestamp", "inst_id", "open", "high", "low", "close", "volume"]
FUNDING_COLUMNS = ["timestamp", "inst_id", "funding_rate"]
_HIGH, _LOW, _CLOSE, _LISTED = 1, 2, 3, 5


def _require(df: pd.DataFrame, columns: List[str], name: str) -> pd.DataFrame:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None) if ts.dt.tz is not None else ts
    if df.duplicated(["timestamp", "inst_id"]).any():
        raise ValueError(f"{name} has duplicate (timestamp, inst_id) rows")
    return df


class PortfolioSampler:
    def __init__(
        self,
        bars: pd.DataFrame,
        time_frames: List[TimeFrame],
        window_sizes: List[int],
        execute_on: TimeFrame,
        funding: Optional[pd.DataFrame] = None,
        seed: Optional[int] = None,
    ):
        bars = _require(bars, BAR_COLUMNS, "bars")
        if bars[["open", "high", "low", "close"]].isna().any().any():
            raise ValueError("bars contain NaN prices")
        if (bars["close"] <= 0).any():
            raise ValueError("bars contain non-positive close prices")
        if "tradable" in bars.columns and bars["tradable"].isna().any():
            raise ValueError("bars have NaN in the tradable column")

        self.time_frames = time_frames
        self.np_rng = np.random.default_rng(seed)
        self.inst_ids = sorted(bars["inst_id"].unique())
        self.num_assets = n = len(self.inst_ids)
        grid = pd.DatetimeIndex(np.sort(bars["timestamp"].unique()), name="timestamp")
        exec_freq = execute_on.to_pandas_freq()

        for i, (inst, rows) in enumerate(bars.groupby("inst_id", sort=True)):
            rows = rows.set_index("timestamp")
            df = self._fill_onto_grid(rows, grid)
            sampler = MarketDataObservationSampler(
                df.reset_index(), time_frames=time_frames, window_sizes=window_sizes,
                execute_on=execute_on,
            )
            if i == 0:
                self.exec_times = sampler.exec_times
                self.num_exec = len(self.exec_times)
                if self.num_exec < 2:
                    raise ValueError("need at least two execute_on bars after warm-up")
                self._obs_idx = {k: torch.from_numpy(v).long() for k, v in sampler._obs_indices.items()}
                self._stacks: Dict[str, torch.Tensor] = {
                    k: torch.empty(t.shape[0], n, t.shape[1]) for k, t in sampler.torch_tensors.items()
                }
                self.close_exec = torch.empty(self.num_exec, n, dtype=torch.float64)
                self.tradable_exec = torch.empty(self.num_exec, n, dtype=torch.bool)
                self.delist_exec = torch.full((n,), -1, dtype=torch.long)
            elif not (
                sampler.exec_times.equals(self.exec_times)
                and all(np.array_equal(v, self._obs_idx[k].numpy()) for k, v in sampler._obs_indices.items())
            ):
                raise ValueError(f"{inst} does not share the common execution grid")

            for key, tensor in sampler.torch_tensors.items():
                self._stacks[key][:, i] = tensor
            self.close_exec[:, i] = torch.from_numpy(
                sampler.execute_base_features_df["close"].to_numpy(dtype=np.float64)
            )
            tradable = df["tradable"].resample(exec_freq).last().reindex(self.exec_times) > 0
            self.tradable_exec[:, i] = torch.from_numpy(tradable.to_numpy())
            if rows.index.max() < grid[-1] and tradable.any():
                self.delist_exec[i] = int(np.flatnonzero(tradable.to_numpy())[-1])

        self.market_data_keys: List[Tuple[str, int]] = [
            (f"market_data_{tf.obs_key_freq()}_{ws}", ws) for tf, ws in zip(time_frames, window_sizes)
        ]
        self.funding_exec = self._funding_per_step(funding, execute_on)

    @staticmethod
    def _fill_onto_grid(rows: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
        present = grid.isin(rows.index)
        df = rows[["open", "high", "low", "close", "volume"]].reindex(grid)
        close = df["close"].ffill().bfill()
        for col in ("open", "high", "low"):
            df[col] = df[col].fillna(close)
        df["close"] = close
        df["volume"] = df["volume"].fillna(0.0)
        df["listed"] = (grid >= rows.index.min()).astype(float)
        tradable = present
        if "tradable" in rows.columns:
            tradable = present & rows["tradable"].astype(bool).reindex(grid, fill_value=False).to_numpy()
        df["tradable"] = tradable.astype(float)
        return df

    def _funding_per_step(self, funding: Optional[pd.DataFrame], execute_on: TimeFrame) -> torch.Tensor:
        out = np.zeros((self.num_exec, self.num_assets))
        if funding is None:
            return torch.from_numpy(out)
        funding = _require(funding, FUNDING_COLUMNS, "funding")
        unknown = sorted(set(funding["inst_id"]) - set(self.inst_ids))
        if unknown:
            raise ValueError(f"funding has unknown inst_id: {unknown}")
        period = tf_to_timedelta(execute_on)
        fills = (self.exec_times + period).as_unit("ns").asi8
        # The appended edge bounds the last window.
        edges = np.append(fills, fills[-1] + period.value)
        stamps = pd.DatetimeIndex(funding["timestamp"]).as_unit("ns").asi8
        # Settlement s belongs to step n iff fill_n < s <= fill_{n+1}.
        step = np.searchsorted(edges, stamps, side="left") - 1
        asset = funding["inst_id"].map({inst: i for i, inst in enumerate(self.inst_ids)}).to_numpy()
        keep = (step >= 0) & (step < self.num_exec)
        # Two settlements landing in the same step must SUM, not overwrite.
        np.add.at(out, (step[keep], asset[keep]), funding["funding_rate"].to_numpy()[keep])
        return torch.from_numpy(out)

    def market_data(self, exec_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """`(B, N, W, 3)` windows of close, high, low over the window's latest close."""
        out = {}
        for tf, (key, ws) in zip(self.time_frames, self.market_data_keys):
            freq = tf.obs_key_freq()
            end = self._obs_idx[freq][exec_idx]
            rows = (end[:, None] - ws + 1 + torch.arange(ws)).clamp(min=0)
            window = self._stacks[freq][rows]                                   # (B, W, N, F)
            latest = window[:, -1:, :, _CLOSE]
            features = window[..., [_CLOSE, _HIGH, _LOW]] / latest[..., None]
            out[key] = (features * window[..., _LISTED, None]).transpose(1, 2).contiguous()
        return out

    def episode_window(
        self, u: torch.Tensor, max_traj_length: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """First and last execution index per `u` in [0, 1); one step moves one index."""
        last = self.num_exec - 1
        if max_traj_length is None:
            starts = (u * last).floor().long().clamp(max=last - 1)
            return starts, torch.full_like(starts, last)
        max_start = max(0, last - max_traj_length)
        starts = (u * (max_start + 1)).floor().long().clamp(max=max_start)
        return starts, (starts + max_traj_length).clamp(max=last)
