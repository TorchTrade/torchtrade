"""Vectorized Sequential Trading Environment with Stop-Loss/Take-Profit support.

A batched TorchRL-compatible environment that processes N SLTP environments
in a single _step() call using pure tensor operations. Extends
VectorizedSequentialTradingEnv with bracket order support.

Key differences from base vectorized env:
    - SLTP timing: advance, execute the agent's trade, then check triggers
    - trade_mode-aware position sizing (fractional, notional, quantity)
    - SL/TP bracket orders with intrabar trigger detection
    - SL checked before TP (pessimistic bias)
    - Triggered positions close at bracket price, except a stop the bar gapped past,
      which fills at the open
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
import torch
from tensordict import TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.core.common import TradeMode, validate_trade_mode
from torchtrade.envs.offline.vectorized_sequential import (
    MONEY_DTYPE,
    VectorizedSequentialTradingEnv,
    VectorizedSequentialTradingEnvConfig,
)
from torchtrade.envs.utils.action_maps import create_sltp_action_map
from torchtrade.envs.utils.fractional_sizing import AFFORDABILITY_REL_TOL

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
    lock_position_until_sltp: bool = False  # If True, ignore actions while in position
    trade_mode: TradeMode = "fractional"
    position_fraction: float = 1.0
    quantity_per_trade: float = 0.001

    def __post_init__(self):
        self.trade_mode = validate_trade_mode(self.trade_mode)

        # Validate sizing parameters
        if self.trade_mode == "fractional":
            if not (0 < self.position_fraction <= 1.0):
                raise ValueError(
                    f"position_fraction must be in (0, 1.0], got {self.position_fraction}"
                )
        elif self.trade_mode in ("notional", "quantity"):
            if self.quantity_per_trade <= 0:
                raise ValueError(
                    f"quantity_per_trade must be positive, got {self.quantity_per_trade}"
                )

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
        **EXPERIMENTAL**: Not battle-tested in production training runs.

        Equivalence against SequentialTradingEnvSLTP is verified by
        tests/envs/offline/test_vec_sltp_scalar_equivalence.py across leverage,
        trade_mode and lock_position_until_sltp: every binary outcome, and money
        to within 1e-9.

    Processes N SLTP environments in a single _step() call using tensor
    operations. All state (balances, positions, SL/TP prices) is stored as
    (num_envs,) tensors and updated simultaneously via masked operations.

    Timing (different from base vectorized env):
        1. Save bar N close as trade prices (with slippage)
        2. Advance step index to bar N+1
        3. Execute trades at bar N price
        4. Check liquidation then SL/TP on bar N+1, against whatever each env holds
           after step 3
        5. Compute rewards from bar N+1 prices

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

        # Parent requires action_levels with >= 2 elements, but SLTP envs don't use
        # fractional sizing. Dummy levels for parent init, restored below -- on the
        # config, on the instance AND on the tensor the parent derives, because it copies
        # all three during its own __init__ and "we override the action spec afterwards"
        # covers only the first of them (#290).
        original_action_levels = config.action_levels
        config.action_levels = [0.0, 1.0]

        super().__init__(df, config, feature_preprocessing_fn)

        # Restored on the CONFIG and on the instance: the parent copies the dummy onto
        # self during its own __init__, so restoring only the config leaves
        # self.action_levels at the dummy forever -- anything reading it off the env, such
        # as BaseLLMActor(action_levels=env.action_levels), gets [0.0, 1.0] (#290).
        config.action_levels = original_action_levels
        self.action_levels = original_action_levels
        self._action_levels_tensor = torch.tensor(original_action_levels, dtype=MONEY_DTYPE)
        if config.leverage == 1:  # same predicate as the parent, to stay twinned
            self._action_levels_tensor = self._action_levels_tensor.clamp(min=0.0)

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
        # float64 for the level itself, not the product (torch already promotes that):
        # float32 holds 0.05 as 0.050000000745, putting the bracket at 104.999995.
        self._action_sl_pcts = torch.tensor(sl_list, dtype=MONEY_DTYPE)
        self._action_tp_pcts = torch.tensor(tp_list, dtype=MONEY_DTYPE)

        # SL/TP state tensors
        self._sl_prices = torch.zeros(N, dtype=MONEY_DTYPE)
        self._tp_prices = torch.zeros(N, dtype=MONEY_DTYPE)

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

        Same order as VectorizedSequentialTradingEnv since #281: the sampler
        advances before the checks in both.

        The agent's action fills at close(N) and bar N+1 unfolds afterwards, so the
        action runs first and the bar is applied to the resulting position. Checking
        the incoming position first instead discarded the action wherever the old
        bracket fired on N+1 (#292).
        """
        N = self._num_envs

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
            # float32 draw, for the generator-stream reason in _sample_initial_cash. No
            # cast: torch promotes the float64 price times this, bit-identically.
            noise = torch.empty(N).uniform_(
                1 - self.slippage, 1 + self.slippage, generator=self._rng
            )
            trade_prices = trade_prices * noise

        # 3. Advance step indices to bar N+1
        self._step_indices += 1
        self._step_counters += 1
        self._step_indices.clamp_(max=self._total_exec_times - 1)

        # 4. Get bar N+1 OHLCV for trigger checks
        new_open = self._base_tensor[self._step_indices, 0]
        new_high = self._base_tensor[self._step_indices, 1]
        new_low = self._base_tensor[self._step_indices, 2]
        new_close = self._base_tensor[self._step_indices, 3]

        # 5. The agent's action, at bar N's price
        self._execute_sltp_trades(sides, sl_pcts, tp_pcts, trade_prices)

        # 6. Bar N+1, against whatever each env holds after the action. Must precede the
        # portfolio values below, or reward and termination read balances that ignore it.
        self._apply_exit_checks(new_open, new_high, new_low)

        # Age straight after the last thing that can move _position_sizes -- the same
        # invariant the base env keeps by calling this after _apply_liquidation, its own
        # last position-moving step. This subclass overrides _step, so without the call
        # nothing ages the counters here and holding_time reads 0 forever (#275).
        self._advance_hold_counters()

        # 7. Compute rewards: log(new_pv / old_pv)
        new_pvs = self._compute_portfolio_values(new_close)
        old_pvs = self._portfolio_values
        safe_old = old_pvs.clamp(min=1e-10)
        safe_new = new_pvs.clamp(min=1e-10)
        rewards = torch.log(safe_new / safe_old)
        rewards = torch.where(new_pvs <= 0, torch.full_like(rewards, -10.0), rewards)

        self._portfolio_values = new_pvs

        # 8. Compute termination signals
        terminated = new_pvs < (self._initial_pvs * self.bankrupt_threshold)
        truncated = (
            ((self._step_indices + 1) >= self._end_indices)
            | (self._step_counters >= self._max_traj_lengths)
        )
        done = terminated | truncated

        # 9. Build observation from bar N+1
        obs_td = self._build_observation(new_close, portfolio_values=new_pvs)
        # reward_spec is float32.
        obs_td.set("reward", rewards.unsqueeze(-1).float())
        obs_td.set("terminated", terminated.unsqueeze(-1))
        obs_td.set("truncated", truncated.unsqueeze(-1))
        obs_td.set("done", done.unsqueeze(-1))

        return obs_td

    def _stop_reached_first_mask(
        self, new_open: torch.Tensor, new_high: torch.Tensor, new_low: torch.Tensor
    ) -> torch.Tensor:
        """Lanes whose triggered stop is crossed before liquidation (#300).

        Tensor twin of SequentialTradingEnvSLTP._stop_is_reached_first; that method
        carries the reasoning. Kept as its own tensor expression rather than routed
        through the scalar form, which would force per-lane tensor construction on a path
        that runs millions of times -- the same shape as the tensorised stop_fill_price
        below. The equivalence harness is what holds the two together.
        """
        if self.config.leverage <= 1:
            return torch.zeros_like(self._position_sizes, dtype=torch.bool)

        liq = self._compute_liq_prices()
        is_long = self._position_sizes > 0
        is_short = self._position_sizes < 0
        has_sl = self._sl_prices > 0

        sl_trigger = (is_long & has_sl & (new_low <= self._sl_prices)) | (
            is_short & has_sl & (new_high >= self._sl_prices)
        )
        stop_is_nearer = torch.where(is_long, self._sl_prices > liq, self._sl_prices < liq)
        gapped_past_liq = torch.where(is_long, new_open <= liq, new_open >= liq)
        return sl_trigger & stop_is_nearer & ~gapped_past_liq

    def _apply_exit_checks(
        self,
        new_open: torch.Tensor,
        new_high: torch.Tensor,
        new_low: torch.Tensor,
    ) -> None:
        """Close positions whose bar range hit liquidation or a bracket."""
        leverage = float(self.config.leverage)

        # Liquidation takes priority over the brackets (futures only), and shares its
        # money math with the base env. SL/TP are NOT cleared on liquidation, matching the
        # scalar env. Stale values are harmless: the trigger masks below gate on
        # can_trigger (via has_position) AND is_long/is_short, and every one of those is
        # False once _apply_liquidation has zeroed the position.
        #
        # Except where a triggered stop sits between entry and the liquidation price:
        # price cannot reach the further level without crossing the nearer one, so the
        # bracket fills first (#300). Tensor twin of the scalar
        # SequentialTradingEnvSLTP._stop_is_reached_first -- see it for why a take-profit
        # is NOT exempt. A bar that opened past liquidation is not exempt either: nothing
        # was crossed on the way, the margin was gone before the bar began.
        self._apply_liquidation(
            new_open, new_high, new_low,
            exempt_fn=lambda: self._stop_reached_first_mask(new_open, new_high, new_low),
        )

        has_position = self._position_sizes != 0
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
                # Tensorised stop_fill_price -- the scalar twin lives in sltp_helpers (#280).
                stop_fill = torch.where(
                    is_long,
                    torch.minimum(new_open, self._sl_prices),
                    torch.maximum(new_open, self._sl_prices),
                )
                exec_price = torch.where(sl_trigger, stop_fill, self._tp_prices)

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
                self._sl_prices = torch.where(
                    sltp_trigger, self._zeros, self._sl_prices
                )
                self._tp_prices = torch.where(
                    sltp_trigger, self._zeros, self._tp_prices
                )

    def _execute_sltp_trades(
        self,
        sides: torch.Tensor,
        sl_pcts: torch.Tensor,
        tp_pcts: torch.Tensor,
        trade_prices: torch.Tensor,
    ) -> None:
        """Execute SLTP trades for all environments.

        Position sizing via trade_mode (fractional/notional/quantity). Handles:
        - Hold: side=0 or same direction as current position
        - Close: side=2 and has position
        - Direction switch: close old, open new with brackets
        - Open from flat: open new with brackets
        """
        leverage = float(self.config.leverage)

        is_long = self._position_sizes > 0
        is_short = self._position_sizes < 0
        is_flat = self._position_sizes == 0

        # Position locking: force HOLD for envs with open positions
        if self.config.lock_position_until_sltp:
            has_position = ~is_flat
            if has_position.any():
                sides = sides.clone()
                sides[has_position] = 0  # Force HOLD

        # Close action (side=2) with existing position
        close_action_mask = (sides == 2) & ~is_flat

        # Direction switch: want opposite direction
        switch_mask = ((sides == 1) & is_short) | ((sides == -1) & is_long)

        # Open from flat: want position, currently flat
        open_from_flat = ((sides == 1) | (sides == -1)) & is_flat

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
            # Note: SL/TP NOT cleared here, matching scalar env behavior.
            # Stale values are harmless (guarded by has_position).
            # For switches, new brackets are set in the open section below.

        # Open new positions (direction switches + open from flat)
        open_mask = switch_mask | open_from_flat
        if open_mask.any():
            # Position sizing based on trade_mode
            if self.config.trade_mode == "fractional":
                pvs = self._compute_portfolio_values(trade_prices)
                fee_denom = 1.0 / leverage + self.transaction_fee
                notional = pvs * self.config.position_fraction / fee_denom
            elif self.config.trade_mode == "notional":
                notional = torch.full_like(trade_prices, self.config.quantity_per_trade)
            elif self.config.trade_mode == "quantity":
                notional = torch.full_like(trade_prices, self.config.quantity_per_trade) * trade_prices
            else:
                raise ValueError(f"Unsupported trade_mode={self.config.trade_mode!r}")

            # Direction: +1 for long, -1 for short
            direction = torch.where(
                sides == 1, self._ones, -self._ones
            )
            new_sizes = direction * notional / trade_prices

            margin_new = notional / leverage
            new_fee = notional * self.transaction_fee

            can_afford = (margin_new + new_fee) <= self._balances * (1 + AFFORDABILITY_REL_TOL)
            final_open = open_mask & can_afford

            if final_open.any():
                self._balances[final_open] -= (margin_new + new_fee)[final_open]
                self._balances.clamp_(min=0.0)
                self._position_sizes[final_open] = new_sizes[final_open]
                self._entry_prices[final_open] = trade_prices[final_open]

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

