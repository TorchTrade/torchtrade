"""Vectorized Sequential Trading Environment with Stop-Loss/Take-Profit support.

A batched TorchRL-compatible environment that processes N SLTP environments
in a single _step() call using pure tensor operations. Extends
VectorizedSequentialTradingEnv with bracket order support.

Key differences from base vectorized env:
    - SLTP timing: advance FIRST, then check triggers, then execute trades
    - 100% capital deployment (no fractional sizing)
    - SL/TP bracket orders with intrabar trigger detection
    - SL checked before TP (pessimistic bias)
    - Triggered positions close at bracket price, not market price
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
import torch
from tensordict import TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.offline.vectorized_sequential import (
    VectorizedSequentialTradingEnv,
    VectorizedSequentialTradingEnvConfig,
)
from torchtrade.envs.utils.action_maps import create_sltp_action_map

_SIDE_MAP = {"long": 1, "short": -1, "close": 2}


@dataclass
class VectorizedSequentialTradingEnvSLTPConfig(VectorizedSequentialTradingEnvConfig):
    """Configuration for vectorized sequential trading environment with SLTP support.

    Extends VectorizedSequentialTradingEnvConfig with bracket order parameters.
    """

    stoploss_levels: Union[List[float], Tuple[float, ...]] = (-0.025, -0.05, -0.1)
    takeprofit_levels: Union[List[float], Tuple[float, ...]] = (0.05, 0.1, 0.2)
    include_hold_action: bool = True
    include_close_action: bool = False

    def __post_init__(self):
        if not isinstance(self.stoploss_levels, list):
            self.stoploss_levels = list(self.stoploss_levels)
        if not isinstance(self.takeprofit_levels, list):
            self.takeprofit_levels = list(self.takeprofit_levels)

        super().__post_init__()

        for sl in self.stoploss_levels:
            if sl >= 0:
                raise ValueError(
                    f"Stop-loss levels must be negative (e.g., -0.05 for 5% loss), got {sl}"
                )
        for tp in self.takeprofit_levels:
            if tp <= 0:
                raise ValueError(
                    f"Take-profit levels must be positive (e.g., 0.1 for 10% profit), got {tp}"
                )


class VectorizedSequentialTradingEnvSLTP(VectorizedSequentialTradingEnv):
    """Vectorized sequential trading environment with stop-loss/take-profit support.

    .. warning::
        **EXPERIMENTAL**: This environment passes extensive equivalence tests
        against SequentialTradingEnvSLTP, but has not been battle-tested in
        production training runs. Use with caution and verify results against
        the scalar implementation.

    Processes N SLTP environments in a single _step() call using tensor
    operations. All state (balances, positions, SL/TP prices) is stored as
    (num_envs,) tensors and updated simultaneously via masked operations.

    Timing (different from base vectorized env):
        1. Save bar N close as trade prices (with slippage)
        2. Advance step index to bar N+1
        3. Check liquidation on bar N+1 (futures only)
        4. Check SL/TP triggers on bar N+1 (SL before TP)
        5. Execute trades at bar N price (skip triggered envs)
        6. Compute rewards from bar N+1 prices

    Args:
        df: OHLCV DataFrame for backtesting
        config: VectorizedSequentialTradingEnvSLTPConfig
        feature_preprocessing_fn: Optional function to preprocess features
    """

    batch_locked = True

    def __init__(
        self,
        df: pd.DataFrame,
        config: VectorizedSequentialTradingEnvSLTPConfig,
        feature_preprocessing_fn: Optional[Callable] = None,
    ):
        # Store SLTP config before parent init
        self.stoploss_levels = config.stoploss_levels
        self.takeprofit_levels = config.takeprofit_levels
        self.include_hold_action = config.include_hold_action
        self.include_close_action = config.include_close_action

        # Build action map
        # leverage > 1 = futures mode, which enables short bracket orders
        self.action_map = create_sltp_action_map(
            stoploss_levels=config.stoploss_levels,
            takeprofit_levels=config.takeprofit_levels,
            include_short_positions=(config.leverage > 1),
            include_hold_action=config.include_hold_action,
            include_close_action=config.include_close_action,
        )

        # Parent requires action_levels with >= 2 elements, but SLTP envs
        # don't use fractional sizing. Temporarily set dummy levels for
        # parent init, then restore. This is safe because the parent only
        # reads action_levels during __init__ to build its action spec,
        # which we override immediately after.
        original_action_levels = config.action_levels
        config.action_levels = [0.0, 1.0]

        super().__init__(df, config, feature_preprocessing_fn)

        config.action_levels = original_action_levels

        # Override action spec with SLTP action count
        num_actions = len(self.action_map)
        N = self._num_envs
        self.action_spec = Categorical(num_actions, shape=torch.Size([N]))

        # Build action lookup tensors from action map
        sides_list = []
        sl_list = []
        tp_list = []
        for i in range(num_actions):
            side, sl, tp = self.action_map[i]
            sides_list.append(_SIDE_MAP.get(side, 0))
            sl_list.append(sl if sl is not None else 0.0)
            tp_list.append(tp if tp is not None else 0.0)

        self._action_sides = torch.tensor(sides_list, dtype=torch.long)
        self._action_sl_pcts = torch.tensor(sl_list, dtype=torch.float32)
        self._action_tp_pcts = torch.tensor(tp_list, dtype=torch.float32)

        # SL/TP state tensors
        self._sl_prices = torch.zeros(N)
        self._tp_prices = torch.zeros(N)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset environments, including SL/TP state."""
        if tensordict is not None and "_reset" in tensordict.keys():
            reset_mask = tensordict["_reset"].squeeze(-1).bool()
        else:
            reset_mask = torch.ones(self._num_envs, dtype=torch.bool)

        self._sl_prices[reset_mask] = 0.0
        self._tp_prices[reset_mask] = 0.0
        return super()._reset(tensordict, **kwargs)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one step for all environments with SLTP timing.

        SLTP timing differs from base: advance FIRST, then check triggers.
        Priority: liquidation > SL/TP trigger > agent action.
        """
        N = self._num_envs
        leverage = float(self.config.leverage)

        # 1. Decode actions via tensor lookup
        action_indices = tensordict["action"]
        if action_indices.dim() > 1:
            action_indices = action_indices.squeeze(-1)
        sides = self._action_sides[action_indices.long()]
        sl_pcts = self._action_sl_pcts[action_indices.long()]
        tp_pcts = self._action_tp_pcts[action_indices.long()]

        # 2. Save bar N close as trade prices (with slippage)
        trade_prices = self._base_tensor[self._step_indices, 3].clone()
        if self.slippage > 0:
            noise = torch.empty(N).uniform_(
                1 - self.slippage, 1 + self.slippage, generator=self._rng
            )
            trade_prices = trade_prices * noise

        # 3. Advance step indices to bar N+1
        self._step_indices += 1
        self._step_counters += 1
        self._step_indices.clamp_(max=self._total_exec_times - 1)

        # 4. Get bar N+1 OHLCV for trigger checks
        new_high = self._base_tensor[self._step_indices, 1]
        new_low = self._base_tensor[self._step_indices, 2]
        new_close = self._base_tensor[self._step_indices, 3]

        # Track which envs had triggers (liquidation or SL/TP)
        triggered_mask = torch.zeros(N, dtype=torch.bool)

        # 5. Check liquidation on bar N+1 (futures only)
        if self.config.leverage > 1:
            liq_price = self._compute_liq_prices()
            long_liq = (self._position_sizes > 0) & (new_low <= liq_price)
            short_liq = (self._position_sizes < 0) & (new_high >= liq_price)
            liq_mask = long_liq | short_liq

            if liq_mask.any():
                pnl = (liq_price - self._entry_prices) * self._position_sizes
                margin_return = (
                    self._position_sizes.abs() * self._entry_prices
                ) / leverage
                fee = (self._position_sizes.abs() * liq_price) * self.transaction_fee

                self._balances = torch.where(
                    liq_mask,
                    self._balances + pnl - fee + margin_return,
                    self._balances,
                )
                self._balances.clamp_(min=0.0)
                self._position_sizes = torch.where(
                    liq_mask, self._zeros, self._position_sizes
                )
                self._entry_prices = torch.where(
                    liq_mask, self._zeros, self._entry_prices
                )
                self._hold_counters = torch.where(
                    liq_mask,
                    torch.zeros_like(self._hold_counters),
                    self._hold_counters,
                )
                # Note: SL/TP are NOT cleared on liquidation, matching scalar
                # env behavior. Stale values are harmless since position is
                # zeroed and SL/TP checks guard on has_position.
                triggered_mask = triggered_mask | liq_mask

        # 6. Check SL/TP triggers on bar N+1
        has_position = (self._position_sizes != 0) & ~triggered_mask
        has_brackets = (self._sl_prices > 0) | (self._tp_prices > 0)
        can_trigger = has_position & has_brackets

        if can_trigger.any():
            is_long = self._position_sizes > 0
            is_short = self._position_sizes < 0

            # SL checked before TP (pessimistic bias)
            long_sl = can_trigger & is_long & (self._sl_prices > 0) & (
                new_low <= self._sl_prices
            )
            short_sl = can_trigger & is_short & (self._sl_prices > 0) & (
                new_high >= self._sl_prices
            )
            sl_trigger = long_sl | short_sl

            # TP only for envs where SL didn't trigger
            remaining = can_trigger & ~sl_trigger
            long_tp = remaining & is_long & (self._tp_prices > 0) & (
                new_high >= self._tp_prices
            )
            short_tp = remaining & is_short & (self._tp_prices > 0) & (
                new_low <= self._tp_prices
            )
            tp_trigger = long_tp | short_tp

            sltp_trigger = sl_trigger | tp_trigger
            if sltp_trigger.any():
                # Close at bracket price (not market price)
                exec_price = torch.where(
                    sl_trigger, self._sl_prices, self._tp_prices
                )

                pnl = (exec_price - self._entry_prices) * self._position_sizes
                close_notional = (self._position_sizes * exec_price).abs()
                fee = close_notional * self.transaction_fee
                margin_return = (
                    self._position_sizes.abs() * self._entry_prices
                ) / leverage

                self._balances = torch.where(
                    sltp_trigger,
                    self._balances + pnl - fee + margin_return,
                    self._balances,
                )
                self._balances.clamp_(min=0.0)
                self._position_sizes = torch.where(
                    sltp_trigger, self._zeros, self._position_sizes
                )
                self._entry_prices = torch.where(
                    sltp_trigger, self._zeros, self._entry_prices
                )
                self._hold_counters = torch.where(
                    sltp_trigger,
                    torch.zeros_like(self._hold_counters),
                    self._hold_counters,
                )
                self._sl_prices = torch.where(
                    sltp_trigger, self._zeros, self._sl_prices
                )
                self._tp_prices = torch.where(
                    sltp_trigger, self._zeros, self._tp_prices
                )
                triggered_mask = triggered_mask | sltp_trigger

        # 7. Execute trades at bar N price (skip triggered envs)
        self._execute_sltp_trades(sides, sl_pcts, tp_pcts, trade_prices, triggered_mask)

        # 8. Compute rewards: log(new_pv / old_pv)
        new_pvs = self._compute_portfolio_values(new_close)
        old_pvs = self._portfolio_values
        safe_old = old_pvs.clamp(min=1e-10)
        safe_new = new_pvs.clamp(min=1e-10)
        rewards = torch.log(safe_new / safe_old)
        rewards = torch.where(new_pvs <= 0, torch.full_like(rewards, -10.0), rewards)

        self._portfolio_values = new_pvs

        # 9. Compute termination signals
        terminated = new_pvs < (self._initial_pvs * self.bankrupt_threshold)
        truncated = (
            ((self._step_indices + 1) >= self._end_indices)
            | (self._step_counters >= self._max_traj_lengths)
        )
        done = terminated | truncated

        # 10. Build observation from bar N+1
        obs_td = self._build_observation(new_close, portfolio_values=new_pvs)
        obs_td.set("reward", rewards.unsqueeze(-1))
        obs_td.set("terminated", terminated.unsqueeze(-1))
        obs_td.set("truncated", truncated.unsqueeze(-1))
        obs_td.set("done", done.unsqueeze(-1))

        return obs_td

    def _execute_sltp_trades(
        self,
        sides: torch.Tensor,
        sl_pcts: torch.Tensor,
        tp_pcts: torch.Tensor,
        trade_prices: torch.Tensor,
        triggered_mask: torch.Tensor,
    ):
        """Execute SLTP trades for all environments.

        100% capital deployment (no fractional sizing). Handles:
        - Hold: side=0 or same direction as current position
        - Close: side=2 and has position
        - Direction switch: close old, open new with brackets
        - Open from flat: open new with brackets
        """
        leverage = float(self.config.leverage)
        active = ~triggered_mask

        is_long = self._position_sizes > 0
        is_short = self._position_sizes < 0
        is_flat = self._position_sizes == 0

        # Hold: explicit hold or already in same direction
        hold_mask = active & (
            (sides == 0)
            | ((sides == 1) & is_long)
            | ((sides == -1) & is_short)
        )
        hold_with_pos = hold_mask & ~is_flat
        self._hold_counters[hold_with_pos] += 1

        # Close action (side=2) with existing position
        close_action_mask = active & (sides == 2) & ~is_flat

        # Direction switch: want opposite direction
        switch_mask = active & (
            ((sides == 1) & is_short) | ((sides == -1) & is_long)
        )

        # Open from flat: want position, currently flat
        open_from_flat = active & ((sides == 1) | (sides == -1)) & is_flat

        # Close existing positions (direction switches + close actions)
        close_mask = switch_mask | close_action_mask
        if close_mask.any():
            pnl = (trade_prices - self._entry_prices) * self._position_sizes
            close_notional = (self._position_sizes * trade_prices).abs()
            fee = close_notional * self.transaction_fee
            margin_return = (
                self._position_sizes.abs() * self._entry_prices
            ) / leverage

            self._balances[close_mask] += (pnl - fee + margin_return)[close_mask]
            self._balances.clamp_(min=0.0)
            self._position_sizes[close_mask] = 0.0
            self._entry_prices[close_mask] = 0.0
            self._hold_counters[close_mask] = 0
            # Note: SL/TP NOT cleared here, matching scalar env behavior.
            # Stale values are harmless (guarded by has_position).
            # For switches, new brackets are set in the open section below.

        # Open new positions (direction switches + open from flat)
        open_mask = switch_mask | open_from_flat
        if open_mask.any():
            # 100% capital deployment
            pvs = self._compute_portfolio_values(trade_prices)
            fee_denom = 1.0 / leverage + self.transaction_fee
            notional = pvs / fee_denom

            # Direction: +1 for long, -1 for short
            direction = torch.where(
                sides == 1, self._ones, -self._ones
            )
            new_sizes = direction * notional / trade_prices

            margin_new = notional / leverage
            new_fee = notional * self.transaction_fee

            # Float32 tolerance for can_afford check
            can_afford = (margin_new + new_fee) <= self._balances * (1 + 1e-5)
            final_open = open_mask & can_afford

            if final_open.any():
                self._balances[final_open] -= (margin_new + new_fee)[final_open]
                self._balances.clamp_(min=0.0)
                self._position_sizes[final_open] = new_sizes[final_open]
                self._entry_prices[final_open] = trade_prices[final_open]
                self._hold_counters[final_open] = 0

                # Set bracket prices: entry * (1 + pct)
                # E.g. Long entry=100, sl_pct=-0.05 → sl_price=95
                # E.g. Short entry=100, sl_pct=+0.05 → sl_price=105
                # (create_sltp_action_map already swaps pcts for shorts)
                self._sl_prices[final_open] = (
                    trade_prices * (1 + sl_pcts)
                )[final_open]
                self._tp_prices[final_open] = (
                    trade_prices * (1 + tp_pcts)
                )[final_open]
