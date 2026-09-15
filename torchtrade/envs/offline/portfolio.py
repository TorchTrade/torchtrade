"""Offline multi-asset portfolio env: the agent sets target weights, cash first."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Bounded, Composite, Unbounded

from torchtrade.envs.core.default_rewards import log_return_reward
from torchtrade.envs.core.offline_base import TorchTradeOfflineEnv
from torchtrade.envs.core.state import PortfolioHistoryTracker
from torchtrade.envs.offline.infrastructure.portfolio_math import MONEY_DTYPE, portfolio_step
from torchtrade.envs.offline.infrastructure.portfolio_sampler import PortfolioSampler
from torchtrade.envs.utils.timeframe import TimeFrame, normalize_timeframe_config


@dataclass
class PortfolioTradingEnvConfig:
    time_frames: Union[List[Union[str, TimeFrame]], str, TimeFrame] = "1Hour"
    window_sizes: Union[List[int], int] = 50
    execute_on: Union[str, TimeFrame] = "4Hour"
    initial_cash: Union[Tuple[int, int], int, float] = 10_000
    transaction_fee: float = 0.0
    bankrupt_threshold: float = 0.1
    max_traj_length: Optional[int] = None
    random_start: bool = True
    seed: Optional[int] = 42
    max_gross: float = 1.0
    allow_short: bool = False

    def __post_init__(self):
        self.execute_on, self.time_frames, self.window_sizes = normalize_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )
        if not 0 <= self.transaction_fee < 0.25:
            raise ValueError(
                f"transaction_fee must be in [0, 0.25), got {self.transaction_fee}; "
                "the capped fee fixed point does not converge for higher fees"
            )
        if not 0 < self.max_gross <= 1:
            raise ValueError(f"max_gross must be in (0, 1], got {self.max_gross}")
        if not 0 <= self.bankrupt_threshold < 1:
            raise ValueError(f"bankrupt_threshold must be in [0, 1), got {self.bankrupt_threshold}")
        if self.max_traj_length is not None and self.max_traj_length < 1:
            raise ValueError(f"max_traj_length must be >= 1 or None, got {self.max_traj_length}")


def portfolio_specs(sampler: PortfolioSampler, config, batch: torch.Size = torch.Size()):
    """(observation_spec, action_spec) shared by the scalar and vectorized envs."""
    n = sampler.num_assets
    obs = Composite(
        portfolio_weights=Unbounded(shape=batch + (n + 1,), dtype=torch.float32),
        tradable=Binary(shape=batch + (n,), dtype=torch.float32),
        shape=batch,
    )
    for key, _, window in sampler.market_data_keys:
        obs.set(key, Unbounded(shape=batch + (n, window, 3), dtype=torch.float32))
    if config.random_start:
        obs.set("reset_index", Unbounded(shape=batch, dtype=torch.long))
        obs.set("state_index", Unbounded(shape=batch, dtype=torch.long))
    action = Bounded(
        low=-config.max_gross if config.allow_short else 0.0, high=config.max_gross,
        shape=batch + (n + 1,), dtype=torch.float32,
    )
    return obs, action


class PortfolioTradingEnv(TorchTradeOfflineEnv):
    """Allocate a portfolio across N assets and cash.

    Action: target weights `[w_cash, w_1, ..., w_N]`, normalised to `w_cash + Σ|w_i| = 1`
    (see `normalise_request`). Timing: observe bars up to n, fill at close n, value at
    close n+1. Non-tradable assets keep their holding.
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        config: PortfolioTradingEnvConfig,
        funding: Optional[pd.DataFrame] = None,
        reward_function: Optional[Callable] = None,
    ):
        self._funding = funding
        super().__init__(bars, config)
        self.reward_function = reward_function or log_return_reward
        self.inst_ids = self.sampler.inst_ids
        self.observation_spec, self.action_spec = portfolio_specs(self.sampler, config)

    def _init_sampler(self, bars, feature_preprocessing_fn):
        self.sampler = PortfolioSampler(
            bars, self.config.time_frames, self.config.window_sizes, self.config.execute_on,
            funding=self._funding, seed=self.config.seed,
        )

    def _reset_history(self):
        self.history = PortfolioHistoryTracker()

    def _get_portfolio_value(self, *args, **kwargs) -> float:
        return self.portfolio_value

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self._reset_history()
        self._reset_balance()
        u = (
            torch.tensor([self.sampler.np_rng.random()], dtype=torch.float64)
            if self.random_start else torch.zeros(1, dtype=torch.float64)
        )
        starts, ends = self.sampler.episode_window(u, self.config.max_traj_length)
        self._idx, self._end = int(starts), int(ends)
        self._reset_idx = self._idx
        self.portfolio_value = self.initial_portfolio_value
        self.drifted = torch.zeros(1, self.sampler.num_assets + 1, dtype=MONEY_DTYPE)
        self.drifted[0, 0] = 1.0
        self.history.record_step(
            self.sampler.exec_times[self._idx], self.portfolio_value, self.drifted[0].tolist()
        )
        return self._observation()

    def _observation(self) -> TensorDict:
        s = self.sampler
        td = TensorDict({
            "portfolio_weights": self.drifted[0].float(),
            "tradable": s.tradable_exec[self._idx].float(),
            **{k: v[0] for k, v in s.market_data(torch.tensor([self._idx])).items()},
        })
        if self.random_start:
            td.set("reset_index", torch.tensor(self._reset_idx, dtype=torch.long))
            td.set("state_index", torch.tensor(self._idx, dtype=torch.long))
        return td

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        s = self.sampler
        n = min(self._idx, s.num_exec - 2)  # a done env stepped again re-emits its last bar
        action = tensordict["action"].to(MONEY_DTYPE).reshape(1, -1)
        if not torch.isfinite(action).all():
            raise ValueError("action contains non-finite values")
        out = portfolio_step(
            self.drifted,
            action,
            s.tradable_exec[n : n + 1],
            (s.delist_exec == n)[None],
            (s.close_exec[n + 1] / s.close_exec[n])[None],
            s.funding_exec[n : n + 1],
            fee=self.config.transaction_fee, max_gross=self.config.max_gross,
            allow_short=self.config.allow_short,
        )
        self._idx = n + 1
        old_value = self.portfolio_value
        self.portfolio_value = old_value * out.pv_factor.item()
        self.drifted = out.drifted

        self.history.record_step(
            s.exec_times[self._idx], self.portfolio_value, out.drifted[0].tolist(),
            commission=old_value * out.commission.item(), funding=old_value * out.funding.item(),
        )
        reward = float(self.reward_function(self.history))
        self.history.rewards[-1] = reward

        terminated = (
            self.portfolio_value < self.config.bankrupt_threshold * self.initial_portfolio_value
            or self.portfolio_value <= 0
        )
        truncated = self._idx >= self._end
        td = self._observation()
        td.set("reward", torch.tensor([reward], dtype=torch.float32))
        td.set("terminated", torch.tensor([terminated]))
        td.set("truncated", torch.tensor([truncated]))
        td.set("done", torch.tensor([terminated or truncated]))
        return td

    def render_history(self, return_fig=False, plot_bh_baseline=True):
        raise NotImplementedError("PortfolioTradingEnv has no plot yet; use env.history.to_dict()")
