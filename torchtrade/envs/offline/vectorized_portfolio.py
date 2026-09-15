"""Batched PortfolioTradingEnv: num_envs lanes stepped as tensors in one _step."""

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from torchtrade.envs.core.default_rewards import batched_log_return_reward
from torchtrade.envs.offline.infrastructure.portfolio_math import MONEY_DTYPE, portfolio_step
from torchtrade.envs.offline.infrastructure.portfolio_sampler import PortfolioSampler
from torchtrade.envs.offline.portfolio import PortfolioTradingEnvConfig, portfolio_specs


@dataclass
class VectorizedPortfolioTradingEnvConfig(PortfolioTradingEnvConfig):
    num_envs: int = 64

    def __post_init__(self):
        super().__post_init__()
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")


class VectorizedPortfolioTradingEnv(EnvBase):
    """PortfolioTradingEnv over a batch; `reward_function(old_pvs, new_pvs) -> Tensor`."""

    batch_locked = True

    def __init__(
        self,
        bars: pd.DataFrame,
        config: VectorizedPortfolioTradingEnvConfig,
        funding: Optional[pd.DataFrame] = None,
        reward_function: Optional[Callable] = None,
    ):
        self.config = config
        self.reward_function = reward_function or batched_log_return_reward
        self.sampler = PortfolioSampler(
            bars, config.time_frames, config.window_sizes, config.execute_on,
            funding=funding,
        )
        self.inst_ids = self.sampler.inst_ids
        batch = torch.Size([config.num_envs])
        super().__init__(batch_size=batch)
        self.observation_spec, self.action_spec = portfolio_specs(self.sampler, config, batch)
        self.reward_spec = Unbounded(shape=batch + (1,), dtype=torch.float32)
        self.full_done_spec = Composite(
            done=Categorical(2, dtype=torch.bool, shape=batch + (1,)),
            terminated=Categorical(2, dtype=torch.bool, shape=batch + (1,)),
            truncated=Categorical(2, dtype=torch.bool, shape=batch + (1,)),
            shape=batch,
        )

        self._rng = torch.Generator()
        if config.seed is not None:
            self._rng.manual_seed(config.seed)
        else:
            self._rng.seed()

        b, n = config.num_envs, self.sampler.num_assets
        self._pvs = torch.zeros(b, dtype=MONEY_DTYPE)
        self._initial_pvs = torch.zeros(b, dtype=MONEY_DTYPE)
        self._drifted = torch.zeros(b, n + 1, dtype=MONEY_DTYPE)
        self._idx = torch.zeros(b, dtype=torch.long)
        self._end = torch.zeros(b, dtype=torch.long)
        self._starts = torch.zeros(b, dtype=torch.long)

    def _set_seed(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng.manual_seed(seed)
            torch.manual_seed(seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None and "_reset" in tensordict.keys():
            mask = tensordict["_reset"].reshape(-1).bool()
        else:
            mask = torch.ones(self.config.num_envs, dtype=torch.bool)
        k = int(mask.sum())
        if k:
            cash = self.config.initial_cash
            if isinstance(cash, (tuple, list)):
                new_cash = torch.empty(k).uniform_(float(cash[0]), float(cash[1]), generator=self._rng).to(MONEY_DTYPE)
            else:
                new_cash = torch.full((k,), float(cash), dtype=MONEY_DTYPE)
            u = (
                torch.rand(k, generator=self._rng, dtype=torch.float64)
                if self.config.random_start else torch.zeros(k, dtype=torch.float64)
            )
            starts, ends = self.sampler.episode_window(u, self.config.max_traj_length)
            self._pvs[mask] = new_cash
            self._initial_pvs[mask] = new_cash
            self._drifted[mask] = 0.0
            self._drifted[mask, 0] = 1.0
            self._starts[mask] = starts
            self._idx[mask] = starts
            self._end[mask] = ends
        return self._observation()

    def _observation(self) -> TensorDict:
        s = self.sampler
        td = TensorDict({
            "portfolio_weights": self._drifted.to(torch.float32, copy=True),
            "tradable": s.tradable_exec[self._idx].float(),
            **s.market_data(self._idx),
        }, batch_size=self.batch_size)
        if self.config.random_start:
            td.set("reset_index", self._starts.clone())
            td.set("state_index", self._idx.clone())
        return td

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        s = self.sampler
        # Keeps n + 1 in range when a done lane is stepped before its reset.
        n = self._idx.clamp(max=s.num_exec - 2)
        action = tensordict["action"].to(MONEY_DTYPE)
        if not torch.isfinite(action).all():
            raise ValueError("action contains non-finite values")
        out = portfolio_step(
            self._drifted,
            action,
            s.tradable_exec[n],
            s.delist_exec[None, :] == n[:, None],
            s.close_exec[n + 1] / s.close_exec[n],
            s.funding_exec[n],
            fee=self.config.transaction_fee, max_gross=self.config.max_gross,
            allow_short=self.config.allow_short,
        )
        new_pvs = self._pvs * out.pv_factor
        rewards = self.reward_function(self._pvs, new_pvs)
        self._pvs, self._drifted, self._idx = new_pvs, out.drifted, n + 1

        terminated = (new_pvs < self._initial_pvs * self.config.bankrupt_threshold) | (new_pvs <= 0)
        truncated = self._idx >= self._end
        td = self._observation()
        td.set("reward", rewards.unsqueeze(-1).float())
        td.set("terminated", terminated.unsqueeze(-1))
        td.set("truncated", truncated.unsqueeze(-1))
        td.set("done", (terminated | truncated).unsqueeze(-1))
        return td
