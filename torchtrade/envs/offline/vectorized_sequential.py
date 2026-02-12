"""Vectorized Sequential Trading Environment.

A batched TorchRL-compatible environment that processes N environments
in a single _step() call using pure tensor operations. No Python loops,
no IPC overhead.

Achieves orders of magnitude higher throughput than ParallelEnv for
fast environments by eliminating inter-process communication overhead.

Currently supports spot mode (leverage=1) only. Futures mode (leverage > 1,
liquidation, shorts) will be added in a follow-up.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Categorical, Composite
from torchrl.envs import EnvBase

from torchtrade.envs.offline.infrastructure.sampler import MarketDataObservationSampler
from torchtrade.envs.utils.timeframe import TimeFrame, normalize_timeframe_config
from torchtrade.envs.utils.fractional_sizing import (
    validate_action_levels,
    POSITION_TOLERANCE_PCT,
    POSITION_TOLERANCE_ABS,
)


@dataclass
class VectorizedSequentialTradingEnvConfig:
    """Configuration for vectorized sequential trading environment.

    Currently supports spot mode only (leverage=1).
    """

    num_envs: int = 64

    # Common parameters
    symbol: str = "BTC/USD"
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, TimeFrame] = "1Hour"
    initial_cash: Union[Tuple[int, int], int, float] = 10000
    transaction_fee: float = 0.0
    slippage: float = 0.0
    bankrupt_threshold: float = 0.1

    # Environment settings
    seed: Optional[int] = 42
    max_traj_length: Optional[int] = None
    random_start: bool = True

    # Action space
    action_levels: Optional[List[float]] = field(
        default_factory=lambda: [-1, 0, 1]
    )

    # Trading parameters (spot only)
    leverage: int = 1

    def __post_init__(self):
        self.execute_on, self.time_frames, self.window_sizes = (
            normalize_timeframe_config(
                self.execute_on, self.time_frames, self.window_sizes
            )
        )
        validate_action_levels(self.action_levels)
        if self.leverage != 1:
            raise ValueError(
                "VectorizedSequentialTradingEnv only supports leverage=1 (spot mode). "
                "Futures mode will be added in a follow-up."
            )
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")
        if not (0 <= self.transaction_fee <= 1):
            raise ValueError(
                f"Transaction fee must be between 0 and 1, got {self.transaction_fee}"
            )
        if not (0 <= self.slippage <= 1):
            raise ValueError(
                f"Slippage must be between 0 and 1, got {self.slippage}"
            )


class VectorizedSequentialTradingEnv(EnvBase):
    """Vectorized sequential trading environment.

    Processes N environments in a single _step() call using tensor operations.
    All state (balances, positions, step indices) is stored as (num_envs,) tensors
    and updated simultaneously via masked operations.

    This achieves 20-400x higher throughput than ParallelEnv for fast environments
    by eliminating inter-process communication overhead.

    Currently supports spot mode (leverage=1) only:
    - No liquidation mechanics
    - Negative action_levels are clipped to 0 (no shorts)
    - Distance to liquidation is always 1.0

    Args:
        df: OHLCV DataFrame for backtesting
        config: VectorizedSequentialTradingEnvConfig
        feature_preprocessing_fn: Optional function to preprocess features
    """

    batch_locked = True

    def __init__(
        self,
        df: pd.DataFrame,
        config: VectorizedSequentialTradingEnvConfig,
        feature_preprocessing_fn: Optional[Callable] = None,
    ):
        self.config = config
        self._num_envs = config.num_envs

        # Store config values
        self.transaction_fee = config.transaction_fee
        self.slippage = config.slippage
        self.bankrupt_threshold = config.bankrupt_threshold
        self.action_levels = config.action_levels

        # Action levels as tensor (clamp negatives for spot mode)
        self._action_levels_tensor = torch.tensor(
            config.action_levels, dtype=torch.float32
        ).clamp(min=0.0)

        # Fee multiplier (spot: leverage=1)
        self._fee_multiplier = 1.0 + config.transaction_fee

        # Initialize sampler (reuse existing infrastructure)
        self._sampler = MarketDataObservationSampler(
            df,
            time_frames=config.time_frames,
            window_sizes=config.window_sizes,
            execute_on=config.execute_on,
            feature_processing_fn=feature_preprocessing_fn,
            features_start_with="features_",
            max_traj_length=config.max_traj_length,
            seed=config.seed,
        )

        # Extract pre-computed data from sampler
        self._market_tensors = self._sampler.torch_tensors  # {key: (N, F)}
        self._obs_indices = self._sampler._obs_indices  # {key: np.ndarray}
        self._base_tensor = self._sampler.execute_base_tensor  # (M, F)
        self._exec_times_arr = self._sampler._exec_times_arr
        self._total_exec_times = len(self._exec_times_arr)
        self._time_frames = config.time_frames
        self._window_sizes = config.window_sizes

        # Convert obs_indices to torch tensors for gather operations
        self._obs_indices_torch = {
            key: torch.from_numpy(idx).long()
            for key, idx in self._obs_indices.items()
        }

        # Pre-compute window offset tensors for each timeframe
        self._window_offsets = {}
        for tf, ws in zip(self._time_frames, self._window_sizes):
            key = tf.obs_key_freq()
            self._window_offsets[key] = torch.arange(ws)

        # Build observation keys (same pattern as scalar env)
        self._market_data_keys = []
        for i, tf in enumerate(self._time_frames):
            key = f"market_data_{tf.obs_key_freq()}_{self._window_sizes[i]}"
            self._market_data_keys.append(key)

        # Build specs
        # For batched envs, spec shapes must start with batch_size
        num_features_per_tf = self._sampler.get_num_features_per_timeframe()
        self._account_state_names = [
            "exposure_pct",
            "position_direction",
            "unrealized_pnlpct",
            "holding_time",
            "leverage",
            "distance_to_liquidation",
        ]

        N = self._num_envs
        batch = torch.Size([N])

        # Initialize EnvBase with batch_size first
        super().__init__(batch_size=batch)

        # Observation spec (all shapes include batch dimension)
        observation_spec = Composite(shape=batch)
        observation_spec.set(
            "account_state",
            Bounded(
                low=-torch.inf,
                high=torch.inf,
                shape=batch + torch.Size([6]),
                dtype=torch.float32,
            ),
        )
        for i, tf in enumerate(self._time_frames):
            tf_key = tf.obs_key_freq()
            n_features = num_features_per_tf[tf_key]
            md_key = self._market_data_keys[i]
            observation_spec.set(
                md_key,
                Bounded(
                    low=-torch.inf,
                    high=torch.inf,
                    shape=batch + torch.Size([self._window_sizes[i], n_features]),
                    dtype=torch.float32,
                ),
            )
        self.observation_spec = observation_spec

        # Reward spec
        self.reward_spec = Bounded(
            low=-torch.inf,
            high=torch.inf,
            shape=batch + torch.Size([1]),
            dtype=torch.float32,
        )

        # Done spec
        self.full_done_spec = Composite(
            done=Categorical(2, dtype=torch.bool, shape=batch + torch.Size([1])),
            terminated=Categorical(2, dtype=torch.bool, shape=batch + torch.Size([1])),
            truncated=Categorical(2, dtype=torch.bool, shape=batch + torch.Size([1])),
            shape=batch,
        )

        # Action spec
        self.action_spec = Categorical(len(config.action_levels), shape=batch)

        # Initialize RNG
        self._rng = torch.Generator()
        if config.seed is not None:
            self._rng.manual_seed(config.seed)

        # Allocate state tensors
        self._balances = torch.zeros(N)
        self._position_sizes = torch.zeros(N)
        self._entry_prices = torch.zeros(N)
        self._hold_counters = torch.zeros(N, dtype=torch.long)
        self._prev_action_values = torch.full((N,), float("nan"))
        self._step_indices = torch.zeros(N, dtype=torch.long)
        self._end_indices = torch.zeros(N, dtype=torch.long)
        self._step_counters = torch.zeros(N, dtype=torch.long)
        self._max_traj_lengths = torch.zeros(N, dtype=torch.long)
        self._initial_pvs = torch.zeros(N)
        self._portfolio_values = torch.zeros(N)

        # Constants
        self._ones = torch.ones(N)
        self._zeros = torch.zeros(N)

    def _set_seed(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng.manual_seed(seed)
            torch.manual_seed(seed)

    def _sample_initial_cash(self, n: int) -> torch.Tensor:
        """Sample initial cash for n environments."""
        if isinstance(self.config.initial_cash, (tuple, list)):
            lo, hi = self.config.initial_cash
            return torch.empty(n).uniform_(float(lo), float(hi), generator=self._rng)
        return torch.full((n,), float(self.config.initial_cash))

    def _sample_start_indices(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample random start indices and compute end indices.

        Returns:
            (start_indices, end_indices) both of shape (n,)
        """
        total_len = self._total_exec_times

        if self.config.random_start:
            if self.config.max_traj_length is not None:
                max_start = max(0, total_len - self.config.max_traj_length)
                starts = torch.randint(
                    0, max_start + 1, (n,), generator=self._rng
                )
                ends = (starts + self.config.max_traj_length).clamp(max=total_len)
            else:
                starts = torch.randint(0, max(1, total_len), (n,), generator=self._rng)
                ends = torch.full((n,), total_len, dtype=torch.long)
        else:
            starts = torch.zeros(n, dtype=torch.long)
            if self.config.max_traj_length is not None:
                ends = torch.full(
                    (n,), min(self.config.max_traj_length, total_len), dtype=torch.long
                )
            else:
                ends = torch.full((n,), total_len, dtype=torch.long)

        return starts, ends

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset environments.

        Supports partial reset via _reset key in tensordict.
        """
        # Determine which envs to reset
        if tensordict is not None and "_reset" in tensordict.keys():
            reset_mask = tensordict["_reset"].squeeze(-1).bool()
        else:
            reset_mask = torch.ones(self._num_envs, dtype=torch.bool)

        n_reset = reset_mask.sum().item()
        if n_reset == 0:
            # Nothing to reset, return current observation
            current_prices = self._base_tensor[self._step_indices, 3]
            return self._build_observation(current_prices)

        # Reset balances
        new_cash = self._sample_initial_cash(n_reset)
        self._balances[reset_mask] = new_cash
        self._initial_pvs[reset_mask] = new_cash

        # Reset positions
        self._position_sizes[reset_mask] = 0.0
        self._entry_prices[reset_mask] = 0.0
        self._hold_counters[reset_mask] = 0
        self._prev_action_values[reset_mask] = float("nan")
        self._step_counters[reset_mask] = 0

        # Reset step indices
        starts, ends = self._sample_start_indices(n_reset)
        self._step_indices[reset_mask] = starts
        self._end_indices[reset_mask] = ends

        # Set max trajectory lengths
        if self.config.max_traj_length is not None:
            self._max_traj_lengths[reset_mask] = self.config.max_traj_length
        else:
            self._max_traj_lengths[reset_mask] = (ends - starts)

        # Update portfolio values
        self._portfolio_values[reset_mask] = new_cash

        # Build observation
        current_prices = self._base_tensor[self._step_indices, 3]
        return self._build_observation(current_prices)

    def _build_observation(self, current_prices: torch.Tensor) -> TensorDictBase:
        """Build observation TensorDict for all environments.

        Args:
            current_prices: (num_envs,) current close prices

        Returns:
            TensorDict with batch_size=(num_envs,)
        """
        N = self._num_envs

        # Account state (6 elements)
        position_values = self._position_sizes * current_prices
        pvs = self._balances + position_values
        pvs_safe = pvs.clamp(min=1e-10)

        exposure_pct = position_values.abs() / pvs_safe
        position_direction = self._position_sizes.sign()

        # Unrealized PnL % = (current - entry) / entry for longs
        has_position = self._position_sizes > 0
        entry_safe = self._entry_prices.clamp(min=1e-10)
        unrealized_pnl_pct = torch.where(
            has_position,
            (current_prices - self._entry_prices) / entry_safe,
            self._zeros,
        )

        account_state = torch.stack(
            [
                exposure_pct,
                position_direction,
                unrealized_pnl_pct,
                self._hold_counters.float(),
                self._ones,  # leverage = 1.0 (spot)
                self._ones,  # distance_to_liquidation = 1.0 (spot)
            ],
            dim=-1,
        )  # (N, 6)

        obs_data = {"account_state": account_state}

        # Market data for each timeframe
        for i, tf in enumerate(self._time_frames):
            tf_key = tf.obs_key_freq()
            ws = self._window_sizes[i]
            md_key = self._market_data_keys[i]

            # Get end indices for observation windows
            obs_idx = self._obs_indices_torch[tf_key]
            end_indices = obs_idx[self._step_indices]  # (N,)

            # Build row indices: (N, ws)
            offsets = self._window_offsets[tf_key]  # (ws,)
            row_indices = (
                end_indices.unsqueeze(1) - (ws - 1) + offsets.unsqueeze(0)
            )  # (N, ws)

            # Clamp to valid range (handles early data with insufficient lookback)
            row_indices = row_indices.clamp(min=0)

            # Gather observations: (N, ws, F)
            market_tensor = self._market_tensors[tf_key]
            obs_data[md_key] = market_tensor[row_indices]

        return TensorDict(obs_data, batch_size=[N])

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one step for all environments simultaneously."""
        N = self._num_envs

        # 1. Get actions and convert to action values
        action_indices = tensordict["action"]  # (N,)
        if action_indices.dim() > 1:
            action_indices = action_indices.squeeze(-1)
        action_values = self._action_levels_tensor[action_indices.long()]  # (N,)

        # 2. Get current prices (trade execution prices)
        trade_prices = self._base_tensor[self._step_indices, 3].clone()  # (N,)

        # 3. Apply slippage
        if self.slippage > 0:
            noise = torch.empty(N).uniform_(
                1 - self.slippage, 1 + self.slippage, generator=self._rng
            )
            trade_prices = trade_prices * noise

        # 4. Execute trades
        self._execute_trades(action_values, trade_prices)

        # 5. Advance step indices and counters
        self._step_indices += 1
        self._step_counters += 1

        # Clamp to valid range (done envs may be past the end;
        # their observation doesn't matter since they'll be auto-reset)
        self._step_indices.clamp_(max=self._total_exec_times - 1)

        # 6. Get new prices and compute portfolio values
        new_prices = self._base_tensor[self._step_indices, 3]  # (N,)
        new_pvs = self._balances + self._position_sizes * new_prices

        # 7. Compute rewards: log(new_pv / old_pv)
        old_pvs = self._portfolio_values
        # Guard against non-positive values
        safe_old = old_pvs.clamp(min=1e-10)
        safe_new = new_pvs.clamp(min=1e-10)
        rewards = torch.log(safe_new / safe_old)
        # Bankruptcy: large negative reward
        rewards = torch.where(new_pvs <= 0, torch.full_like(rewards, -10.0), rewards)

        # 8. Update stored portfolio values
        self._portfolio_values = new_pvs.clone()

        # 9. Compute termination signals
        terminated = new_pvs < (self._initial_pvs * self.bankrupt_threshold)
        truncated = (
            ((self._step_indices + 1) >= self._end_indices)
            | (self._step_counters >= self._max_traj_lengths)
        )
        done = terminated | truncated

        # 10. Build next observation
        obs_td = self._build_observation(new_prices)
        obs_td.set("reward", rewards.unsqueeze(-1))
        obs_td.set("terminated", terminated.unsqueeze(-1))
        obs_td.set("truncated", truncated.unsqueeze(-1))
        obs_td.set("done", done.unsqueeze(-1))

        return obs_td

    def _execute_trades(
        self, action_values: torch.Tensor, execution_prices: torch.Tensor
    ):
        """Execute trades for all environments using vectorized operations.

        Handles:
        - Same action optimization (hold if action unchanged)
        - Tolerance-based holding (avoid churn from small price drift)
        - Position closing and opening via masked operations
        """
        # Same action optimization (#187): if action unchanged and has position, hold
        same_action = (action_values == self._prev_action_values) & (
            self._position_sizes > 0
        )
        self._hold_counters[same_action] += 1

        # Update prev action values
        self._prev_action_values.copy_(action_values)

        # Determine which envs need to trade
        need_trade = ~same_action
        if not need_trade.any():
            return

        # Compute target positions for ALL envs (efficient batch operation)
        pvs = self._balances + self._position_sizes * execution_prices
        fraction = action_values.abs()
        capital_allocated = pvs * fraction

        # Account for fees: notional = capital / (1/leverage + fee) * leverage
        # For spot (leverage=1): notional = capital / (1 + fee)
        margin_required = capital_allocated / self._fee_multiplier
        notional = margin_required  # leverage=1
        target_sizes = torch.where(
            execution_prices > 0,
            notional / execution_prices,
            self._zeros,
        )
        # action_value == 0 → target = 0
        target_sizes = target_sizes * (action_values > 0).float()

        # Tolerance check: avoid churn from small position changes
        tolerance = (target_sizes.abs() * POSITION_TOLERANCE_PCT).clamp(
            min=POSITION_TOLERANCE_ABS
        )
        within_tol = (target_sizes - self._position_sizes).abs() < tolerance
        hold_tol = need_trade & within_tol & (self._position_sizes > 0)
        self._hold_counters[hold_tol] += 1
        need_trade = need_trade & ~hold_tol

        if not need_trade.any():
            return

        # Close existing positions that need to change
        close_mask = need_trade & (self._position_sizes > 0)
        if close_mask.any():
            # PnL = (current_price - entry_price) * position_size
            pnl = (execution_prices - self._entry_prices) * self._position_sizes
            # Fee on close notional
            close_notional = (self._position_sizes * execution_prices).abs()
            fee = close_notional * self.transaction_fee
            # Return locked margin (spot: margin = entry_price * position_size)
            margin_return = (self._position_sizes * self._entry_prices).abs()

            self._balances[close_mask] += (pnl - fee + margin_return)[close_mask]
            self._balances.clamp_(min=0.0)
            self._position_sizes[close_mask] = 0.0
            self._entry_prices[close_mask] = 0.0
            self._hold_counters[close_mask] = 0

        # Open new positions where target > 0
        open_mask = need_trade & (action_values > 0)
        if open_mask.any():
            # Recalculate target with updated balance (after closing)
            pvs_new = self._balances + self._position_sizes * execution_prices
            capital_new = pvs_new * fraction
            margin_new = capital_new / self._fee_multiplier
            notional_new = margin_new
            new_sizes = torch.where(
                execution_prices > 0,
                notional_new / execution_prices,
                self._zeros,
            )
            new_fee = notional_new * self.transaction_fee

            # Check sufficient balance
            can_afford = (margin_new + new_fee) <= self._balances
            final_open = open_mask & can_afford

            if final_open.any():
                self._balances[final_open] -= (margin_new + new_fee)[final_open]
                self._balances.clamp_(min=0.0)
                self._position_sizes[final_open] = new_sizes[final_open]
                self._entry_prices[final_open] = execution_prices[final_open]
                self._hold_counters[final_open] = 0

    def close(self):
        """Clean up resources."""
        pass
