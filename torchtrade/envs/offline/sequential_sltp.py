"""Sequential Trading Environment with Stop-Loss/Take-Profit support.

A TorchRL-compatible environment for algorithmic trading with bracket orders.
Supports both spot and futures trading with stop-loss and take-profit levels.

Key Features:
    - Inherits from SequentialTradingEnv (6-element account state)
    - Stop-loss and take-profit bracket order support
    - Mode-aware action space:
        * Spot: HOLD + N long positions with SL/TP combinations
        * Futures: HOLD + N long + N short positions with SL/TP combinations
    - Intrabar SL/TP trigger detection using OHLC data
    - Automatic position exit on trigger
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable

import pandas as pd
import torch
from tensordict import TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.utils.liquidation import stop_precedes_liquidation
from torchtrade.envs.offline.sequential import (
    SequentialTradingEnv,
    SequentialTradingEnvConfig,
)
from torchtrade.envs.core.common import TradeMode, validate_trade_mode, validate_position_sizing, validate_sltp_levels
from torchtrade.envs.core.state import (
    advance_hold_counter_from_size,
    binarize_action_type,
)
from torchtrade.envs.utils.sltp_helpers import (
    calculate_bracket_prices,
    stop_fill_price,
)
from torchtrade.envs.utils.action_maps import create_sltp_action_map
from torchtrade.envs.utils.fractional_sizing import AFFORDABILITY_REL_TOL


@dataclass
class SequentialTradingEnvSLTPConfig(SequentialTradingEnvConfig):
    """Configuration for sequential trading environment with SLTP support.

    Extends SequentialTradingEnvConfig with bracket order parameters.
    """
    # Stop-loss and take-profit levels (as percentages)
    stoploss_levels: Union[List[float], Tuple[float, ...]] = (-0.025, -0.05, -0.1)
    takeprofit_levels: Union[List[float], Tuple[float, ...]] = (0.05, 0.1, 0.2)

    # Action space options
    include_hold_action: bool = True  # Include HOLD action (index 0)
    include_close_action: bool = False  # Include CLOSE action (default: False for SLTP)

    # Position locking (for OneStep policy evaluation parity)
    lock_position_until_sltp: bool = False  # If True, ignore actions while in position

    # Position sizing mode
    trade_mode: TradeMode = "fractional"
    position_fraction: float = 1.0       # Used when trade_mode="fractional" (1.0 = all-in, backward compat)
    quantity_per_trade: float = 0.001     # Used when trade_mode in ("quantity", "notional")

    def __post_init__(self):
        """Validate configuration after dataclass initialization."""
        # Call parent post_init first
        super().__post_init__()

        self.trade_mode = validate_trade_mode(self.trade_mode)

        # Validate sizing parameters
        validate_position_sizing(
            self.trade_mode, self.position_fraction, self.quantity_per_trade
        )

        # Convert to lists if needed
        if not isinstance(self.stoploss_levels, list):
            self.stoploss_levels = list(self.stoploss_levels)
        if not isinstance(self.takeprofit_levels, list):
            self.takeprofit_levels = list(self.takeprofit_levels)

        # Validate SL/TP levels
        validate_sltp_levels(self.stoploss_levels, self.takeprofit_levels)


class SequentialTradingEnvSLTP(SequentialTradingEnv):
    """Sequential trading environment with stop-loss/take-profit support.

    Combines the base SequentialTradingEnv with bracket order functionality.

    Action Space Structure:
    -----------------------
    Spot Mode (3 SL levels × 3 TP levels = 10 actions):
        - Action 0: HOLD (optional, controlled by include_hold_action)
        - Actions 1-9: Long positions with (SL, TP) combinations
            * (-0.025, 0.05), (-0.025, 0.1), (-0.025, 0.2)  # SL=-2.5%
            * (-0.05, 0.05), (-0.05, 0.1), (-0.05, 0.2)     # SL=-5%
            * (-0.1, 0.05), (-0.1, 0.1), (-0.1, 0.2)        # SL=-10%

    Futures Mode (3 SL × 3 TP × 2 directions = 19 actions):
        - Action 0: HOLD (optional)
        - Actions 1-9: Long positions with (SL, TP) combinations
        - Actions 10-18: Short positions with (SL, TP) combinations

    SL/TP Trigger Behavior:
    -----------------------
    Long positions:
        - Stop-loss triggers when price <= SL level (loss protection)
        - Take-profit triggers when price >= TP level (profit taking)

    Short positions:
        - Stop-loss triggers when price >= SL level (loss protection)
        - Take-profit triggers when price <= TP level (profit taking)

    Universal Account State (inherited from SequentialTradingEnv):
    ---------------------------------------------------------------
    [exposure_pct, position_direction, unrealized_pnl_pct,
     holding_time, leverage, distance_to_liquidation]

    See SequentialTradingEnv docstring for detailed element descriptions.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: SequentialTradingEnvSLTPConfig,
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
    ):
        """Initialize the sequential SLTP environment.

        Args:
            df: OHLCV DataFrame for backtesting
            config: Environment configuration with SLTP parameters
            feature_preprocessing_fn: Optional function to preprocess features
            reward_function: Optional reward function (default: log_return_reward)
        """
        # Store SLTP configuration before initializing parent
        self.stoploss_levels = config.stoploss_levels
        self.takeprofit_levels = config.takeprofit_levels
        self.include_hold_action = config.include_hold_action
        self.include_close_action = config.include_close_action

        # Build action map for SLTP
        self.action_map = create_sltp_action_map(
            stoploss_levels=config.stoploss_levels,
            takeprofit_levels=config.takeprofit_levels,
            include_short_positions=(config.leverage > 1),  # Futures mode
            include_hold_action=config.include_hold_action,
            include_close_action=config.include_close_action,
        )

        # Temporarily set action_levels to a dummy list for parent initialization
        # This prevents the parent from failing when creating action_spec
        original_action_levels = config.action_levels
        config.action_levels = [0.0]  # Dummy value, will be overridden

        # Initialize parent class (sets up base SequentialTradingEnv)
        super().__init__(df, config, feature_preprocessing_fn, reward_function)

        # Restore original action_levels -- on the CONFIG and on the instance. The parent
        # copies the dummy onto self during __init__, and restoring only the config left
        # self.action_levels at [0.0] forever, so allows_short read False on every SLTP
        # env including 125x futures, and render_history drew those episodes as spot
        # (#290).
        config.action_levels = original_action_levels
        self.action_levels = original_action_levels

        # Override action spec with SLTP action space
        self.action_spec = Categorical(len(self.action_map))

        # Initialize SLTP-specific state
        self.stop_loss = 0.0
        self.take_profit = 0.0

        # Convert action_map to tuple for O(1) indexed lookup (performance optimization)
        self._action_tuple = tuple(self.action_map[i] for i in range(len(self.action_map)))

    def _reset_position_state(self):
        """Reset position tracking state including SLTP-specific state."""
        super()._reset_position_state()
        # Reset SLTP state
        self.stop_loss = 0.0
        self.take_profit = 0.0

    def _check_sltp_trigger(self, ohlcv: dict) -> Optional[str]:
        """Check if stop-loss or take-profit should trigger.

        Uses intrabar OHLC data to detect SL/TP triggers that may occur
        within the candle, not just at the close.

        NOTE: SL is checked before TP intentionally. When both could trigger
        within the same candle, we assume the worst case (SL). This pessimistic
        bias ensures backtesting underestimates performance, so live trading
        can only outperform the backtest.

        Args:
            ohlcv: Dictionary with keys "open", "high", "low", "close", "volume"

        Returns:
            "sl" if stop-loss triggered
            "tp" if take-profit triggered
            None if neither triggered
        """
        if self.position.position_size == 0:
            return None
        if self.stop_loss == 0.0 and self.take_profit == 0.0:
            return None

        open_price = ohlcv["open"]
        high_price = ohlcv["high"]
        low_price = ohlcv["low"]
        close_price = ohlcv["close"]

        # Stop before take-profit, deliberately (see the docstring's pessimism note).
        #
        # The extreme alone would do. #326 rejects malformed bars at ingestion, and
        # resampling preserves that -- open/high/low/close aggregate as first/max/min/last,
        # all selections from rows inside the bar, and `last` skips NaN, so the surviving
        # close always comes from a row whose own high the max already covers (checked
        # over 60 NaN-scattered frames: zero malformed resampled bars). So
        # min(open, low, close) IS low here.
        #
        # The triple stays because it states the rule without depending on that chain:
        # until #316 these three lines were forked across two engines whose only
        # disagreement was on exactly the malformed case.
        if self.position.position_size > 0:
            if self.stop_loss > 0 and min(open_price, low_price, close_price) <= self.stop_loss:
                return "sl"
            if self.take_profit > 0 and max(open_price, high_price, close_price) >= self.take_profit:
                return "tp"
        else:
            if self.stop_loss > 0 and max(open_price, high_price, close_price) >= self.stop_loss:
                return "sl"
            if self.take_profit > 0 and min(open_price, low_price, close_price) <= self.take_profit:
                return "tp"

        return None

    def _stop_is_reached_first(self, trigger: Optional[str], ohlcv: dict) -> bool:
        """Does a triggered stop sit between entry and the liquidation price? (#300)

        Liquidation otherwise outranks the bracket, and for a take-profit that is right:
        the two sit on OPPOSITE sides of entry, so a bar that reached both says nothing
        about which came first, and assuming the worse of two irreconcilable orderings is
        the only sound choice -- the same reasoning as SL-before-TP.

        A stop-loss is not that case. It sits on the SAME side as liquidation, so price
        cannot reach the further level without crossing the nearer one on the way, and the
        bar's own extreme is enough to know which. Booking a liquidation when the stop was
        crossed first is not pessimism, it is an outcome the data contradicts -- and an
        expensive one: at 10x on 10000 cash it leaves 400 where the stop leaves 5000.

        The exception is a bar that OPENED past liquidation. Nothing was crossed on the
        way there, the margin was already gone when the bar began, and no resting order
        could have worked first.

        Scope: this covers a gap at the bar BOUNDARY, which OHLC records. A hole INSIDE a
        bar does not appear in OHLC at all -- price can jump 96 to 88 without ever
        printing 95 -- and there this fills the stop where the old rule liquidated, i.e.
        trades one unresolvable case from pessimistic to optimistic. Same class as #280,
        and not visible at this resolution.

        The `trigger != "sl"` guard below is geometrically unreachable rather than
        load-bearing: if liquidation fired and the stop is nearer, price passed the stop,
        so _check_sltp_trigger has already returned "sl". It is kept because it states the
        rule, and because it is the twin of the vectorized mask's own sl_trigger term.
        """
        if trigger != "sl":
            return False

        return stop_precedes_liquidation(
            self.stop_loss, self.liquidation_price, ohlcv["open"],
            is_long=self.position.position_size > 0,
        )

    def _apply_bar_exits(self, ohlcv: dict) -> Optional[Dict]:
        """Apply a bar to the held position: liquidation first, then a bracket.

        The single home for the whole exit surface -- detection, priority, fill pricing
        and booking. OneStep re-forked all four of those (#316), which is why the gapped
        stop in #280 had to be fixed in two separate scalar places.

        Returns the trade_info of whatever fired, or None if the position survived.
        """
        trigger = self._check_sltp_trigger(ohlcv)

        if self._check_liquidation(ohlcv) and not self._stop_is_reached_first(trigger, ohlcv):
            return self._execute_liquidation(ohlcv["open"])

        if trigger is None:
            return None

        if trigger == "sl":
            execution_price = stop_fill_price(
                self.stop_loss, ohlcv["open"], is_long=self.position.position_size > 0
            )
        else:
            execution_price = self.take_profit
        return self._execute_sltp_close(execution_price, trigger)

    def _execute_sltp_close(self, execution_price: float, trigger_type: str) -> Dict:
        """Execute SL/TP triggered close.

        Args:
            execution_price: Price at which to execute (a gapped stop fills at the bar
                open, not at the bracket, so this is not the trigger price)
            trigger_type: Either "sl" or "tp"

        Returns:
            Trade info dictionary with execution details
        """
        if self.position.position_size == 0:
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # Calculate PnL
        pnl = self._calculate_unrealized_pnl(
            self.position.entry_price, execution_price, self.position.position_size
        )

        # Calculate fee and margin to return
        notional = abs(self.position.position_size * execution_price)
        fee = notional * self.transaction_fee
        # Return the margin that was locked when opening
        margin_to_return = abs(self.position.position_size * self.position.entry_price) / self.leverage

        # Update balance: add realized PnL, subtract fee, return locked margin
        self.balance += pnl - fee + margin_to_return
        self._clamp_balance()

        # Reset position and SLTP state
        self.position.position_size = 0.0
        self.position.position_value = 0.0
        self.position.entry_price = 0.0
        self.position.current_position = 0
        self.liquidation_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0

        return {
            "executed": True,
            "side": f"sltp_{trigger_type}",  # "sltp_sl" or "sltp_tp"
            "fee_paid": fee,
            "liquidated": False
        }

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step with SLTP logic.

        Order:
        1. Execute the agent's action at bar N's close
        2. Liquidation, then SL/TP, on bar N+1 against whatever position results

        The agent observes bar N and its order fills at close(N); bar N+1 unfolds
        afterwards. So the action goes first and the bar is then applied to the
        position actually held during it -- held-and-untouched, freshly opened, or
        flipped alike. Checking the incoming position first instead discarded the
        agent's order whenever the old bracket fired on N+1 (#292), leaving the
        account in a position it had asked to leave, at a price it never chose.

        The sampler is advanced before the check so triggers are evaluated against
        the first unseen bar, never the bar the agent already acted on.
        """
        self.step_counter += 1

        # Guard: if sampler was exhausted in the previous step, terminate
        # gracefully instead of letting get_sequential_observation() raise.
        if self.truncated:
            return self._build_exhaustion_response()

        # Bar N price — where the agent's action would execute
        cached_price = self._cached_base_features["close"]

        # Get desired action
        action_idx = tensordict["action"]
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        action_tuple = self._action_tuple[action_idx]
        side, sl_pct, tp_pct = action_tuple

        # Advance sampler to bar N+1
        obs_dict, base_features = self._get_observation_scaffold()
        self._cached_base_features = base_features
        new_price = base_features["close"]

        # Locking discards the agent's action by config, so the position it is holding
        # is what bar N+1 gets applied to below.
        if self.config.lock_position_until_sltp and self.position.position_size != 0:
            side, sl_pct, tp_pct = None, None, None

        trade_info = self._execute_sltp_action(side, sl_pct, tp_pct, cached_price)

        # Bar N+1, against whatever is held after the action. Must precede the portfolio
        # value and the history record below, or both read a state that ignores this exit.
        exit_info = self._apply_bar_exits(base_features)
        if exit_info is not None:
            trade_info = exit_info

        # Same canonical aging as SequentialTradingEnv (#275): one call per step off the
        # post-trade size, so no exit path can be added later without ageing correctly.
        advance_hold_counter_from_size(self.position)

        # Update position flag based on actual position size
        if trade_info["executed"]:
            if self.position.position_size > 0:
                self.position.current_position = 1  # Long
            elif self.position.position_size < 0:
                self.position.current_position = -1  # Short
            else:
                self.position.current_position = 0  # Flat

        # Build observation with UPDATED position state (no sampler advance)
        next_tensordict = self._build_observation_from_data(obs_dict, base_features)
        new_portfolio_value = self._get_portfolio_value(new_price)

        # Add coverage tracking indices (only during training with random_start)
        if self.random_start:
            self._last_state_index = self.sampler._sequential_idx
            next_tensordict.set("reset_index", torch.tensor(self._reset_idx, dtype=torch.long))
            next_tensordict.set("state_index", torch.tensor(self._last_state_index, dtype=torch.long))

        # Determine action_type and binarize action for history
        action_type = trade_info.get("side") or "hold"
        binarized_action = binarize_action_type(action_type)

        # Record step history FIRST (reward function needs updated history!)
        self.history.record_step(
            price=new_price,
            action=binarized_action,
            reward=0.0,  # Placeholder, will be set after reward calculation
            portfolio_value=new_portfolio_value,
            position=self.position.position_size,
            action_type=action_type
        )

        # Calculate reward using UPDATED history tracker
        reward = float(self.reward_function(self.history))

        # Update the reward in history
        self.history.rewards[-1] = reward

        # Check termination (bankruptcy) and truncation (time/data limit)
        terminated = self._check_termination(new_portfolio_value)
        truncated = self._check_truncation()

        next_tensordict.set("reward", torch.tensor([reward], dtype=torch.float))
        next_tensordict.set("terminated", torch.tensor([terminated], dtype=torch.bool))
        next_tensordict.set("truncated", torch.tensor([truncated], dtype=torch.bool))
        next_tensordict.set("done", torch.tensor([terminated or truncated], dtype=torch.bool))

        return next_tensordict

    def _execute_sltp_action(
        self, side: Optional[str], sl_pct: Optional[float], tp_pct: Optional[float], base_price: float
    ) -> Dict:
        """Execute action with SLTP bracket order setup.

        Args:
            side: Position side ("long", "short", "close", or None for hold)
            sl_pct: Stop-loss percentage (negative)
            tp_pct: Take-profit percentage (positive)
            base_price: Base price for execution

        Returns:
            Trade info dictionary
        """
        # HOLD action
        if side is None:
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # CLOSE action
        if side == "close":
            if self.position.position_size != 0:
                # Apply slippage
                price_noise = torch.empty(1).uniform_(1 - self.slippage, 1 + self.slippage, generator=self._rng).item()
                execution_price = base_price * price_noise
                return self._close_position(execution_price)
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # Opening new position (long or short)
        # Apply slippage
        price_noise = torch.empty(1).uniform_(1 - self.slippage, 1 + self.slippage, generator=self._rng).item()
        execution_price = base_price * price_noise

        # Check if already in same direction - if so, hold (ignore duplicate action)
        if side == "long" and self.position.position_size > 0:
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}
        if side == "short" and self.position.position_size < 0:
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # If switching direction, close existing position first
        if self.position.position_size != 0:
            self._close_position(execution_price)

        # Open new position with SLTP brackets
        return self._open_position_with_sltp(side, execution_price, sl_pct, tp_pct)

    def _open_position_with_sltp(
        self, side: str, execution_price: float, sl_pct: float, tp_pct: float
    ) -> Dict:
        """Open a new position with SL/TP bracket orders.

        Args:
            side: Position side ("long" or "short")
            execution_price: Price at which to execute (includes slippage)
            sl_pct: Stop-loss percentage
            tp_pct: Take-profit percentage

        Returns:
            Trade info dictionary
        """
        # Position sizing based on trade_mode
        if self.config.trade_mode == "fractional":
            # Fractional: use position_fraction of portfolio
            from torchtrade.envs.utils.fractional_sizing import (
                calculate_fractional_position,
                PositionCalculationParams,
            )

            if self.leverage == 1:
                action_value = self.config.position_fraction
            else:
                action_value = self.config.position_fraction if side == "long" else -self.config.position_fraction

            portfolio_value = self._get_portfolio_value(execution_price)
            params = PositionCalculationParams(
                balance=portfolio_value,
                action_value=action_value,
                current_price=execution_price,
                leverage=self.leverage,
                transaction_fee=self.transaction_fee,
            )
            position_size, notional_value, calc_side = calculate_fractional_position(params)

        elif self.config.trade_mode == "notional":
            # Notional: fixed USD per trade
            if execution_price <= 0:
                raise ValueError(f"execution_price must be positive for notional mode, got {execution_price}")
            notional_value = float(self.config.quantity_per_trade)
            position_size = notional_value / execution_price

        elif self.config.trade_mode == "quantity":
            # Quantity: fixed base-asset units per trade
            if execution_price <= 0:
                raise ValueError(f"execution_price must be positive for quantity mode, got {execution_price}")
            position_size = float(self.config.quantity_per_trade)
            notional_value = position_size * execution_price

        else:
            raise ValueError(f"Unsupported trade_mode={self.config.trade_mode!r}")

        # Calculate margin and fee
        margin_required = notional_value / self.leverage
        fee = abs(notional_value) * self.transaction_fee

        if margin_required + fee > self.balance * (1 + AFFORDABILITY_REL_TOL):
            return {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        # Deduct fee and margin
        # For spot (leverage=1): margin_required = notional_value (full cost)
        # For futures (leverage>1): margin_required = notional_value / leverage
        self.balance -= fee + margin_required
        self._clamp_balance()

        # Set position
        self.position.position_size = position_size if side == "long" else -abs(position_size)
        self.position.position_value = abs(notional_value)
        self.position.entry_price = execution_price

        # Set position direction
        if self.leverage == 1:
            self.position.current_position = 1  # Always long
        else:
            self.position.current_position = 1 if side == "long" else -1

        # Calculate liquidation price (futures only)
        self.liquidation_price = self._calculate_liquidation_price(execution_price, self.position.position_size)

        self.stop_loss, self.take_profit = calculate_bracket_prices(
            side, execution_price, sl_pct, tp_pct
        )

        return {"executed": True, "side": side, "fee_paid": fee, "liquidated": False}

    def close(self):
        """Clean up resources."""
        pass
