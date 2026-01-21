"""Sequential Futures Environment for offline training with leverage and shorting support."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
from enum import Enum

import numpy as np
from tensordict import TensorDict, TensorDictBase
import torch
from torchrl.data import Categorical
import pandas as pd
from torchtrade.envs.offline.base import TorchTradeOfflineEnv
from torchtrade.envs.timeframe import TimeFrame, TimeFrameUnit, normalize_timeframe_config
from torchtrade.envs.state import FuturesHistoryTracker
from torchtrade.envs.common import TradeMode, validate_quantity_per_trade
from torchtrade.envs.fractional_sizing import (
    build_default_action_levels,
    validate_position_sizing_mode,
    calculate_fractional_position,
    PositionCalculationParams,
    POSITION_TOLERANCE_PCT,
    POSITION_TOLERANCE_ABS,
)


class MarginType(Enum):
    """Margin type for futures trading."""
    ISOLATED = "isolated"
    CROSSED = "crossed"


@dataclass
class SeqFuturesEnvConfig:
    """Configuration for Sequential Futures Environment.

    This environment supports:
    - Long and short positions
    - Configurable leverage (1x - 125x)
    - Liquidation mechanics
    - Isolated/crossed margin
    """
    symbol: str = "BTC/USD"
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]] = "1Min"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, TimeFrame] = "1Min"

    # Initial capital settings
    initial_cash: Union[Tuple[int, int], int] = (1000, 5000)

    # Position sizing
    trade_mode: TradeMode = TradeMode.QUANTITY  # Match live environment default
    quantity_per_trade: float = 0.001  # BTC for QUANTITY, USD for NOTIONAL

    # Leverage and margin settings
    leverage: int = 1  # 1x to 125x
    margin_type: MarginType = MarginType.ISOLATED
    maintenance_margin_rate: float = 0.004  # 0.4% maintenance margin

    # Trading costs
    transaction_fee: float = 0.0004  # 0.04% typical futures fee
    slippage: float = 0.001  # 0.1% slippage
    funding_rate: float = 0.0001  # 0.01% per 8 hours (typical funding rate)
    funding_interval_hours: int = 8

    # Risk management
    bankrupt_threshold: float = 0.1  # 10% of initial balance

    # Environment settings
    seed: Optional[int] = 42
    include_base_features: bool = False
    max_traj_length: Optional[int] = None
    random_start: bool = True

    # Reward settings
    reward_scaling: float = 1.0
    reward_function: Optional[Callable] = None  # Custom reward function (uses default if None)

    # Action space configuration
    position_sizing_mode: str = "fractional"  # "fractional" (new default) or "fixed" (legacy)
    action_levels: List[float] = None  # Custom action levels, or None for defaults

    # DEPRECATED: Only used in legacy "fixed" mode
    include_hold_action: bool = True  # DEPRECATED - only used when position_sizing_mode="fixed"
    include_close_action: bool = True  # DEPRECATED - only used when position_sizing_mode="fixed"

    def __post_init__(self):
        """Normalize timeframe configuration and build action levels."""
        self.execute_on, self.time_frames, self.window_sizes = normalize_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

        validate_quantity_per_trade(self.quantity_per_trade)

        # Validate and build default action levels using shared utility
        validate_position_sizing_mode(self.position_sizing_mode)

        if self.action_levels is None:
            self.action_levels = build_default_action_levels(
                position_sizing_mode=self.position_sizing_mode,
                include_hold_action=self.include_hold_action,
                include_close_action=self.include_close_action,
                allow_short=True  # Futures allow short positions
            )


class SeqFuturesEnv(TorchTradeOfflineEnv):
    """
    Sequential Futures Environment for offline RL training.

    Supports long and short positions with leverage, similar to the
    BinanceFuturesTorchTradingEnv but for offline/backtesting use.

    Action Space (Fractional Mode - Default):
    --------------------------------------
    Actions represent the fraction of available balance to allocate to a position.
    Action values in range [-1.0, 1.0]:

    - action = -1.0: 100% short (all-in short)
    - action = -0.5: 50% short
    - action = 0.0: Market neutral (close all positions, stay in cash)
    - action = 0.5: 50% long
    - action = 1.0: 100% long (all-in long)

    Position sizing formula:
        position_size = (balance × |action| × leverage) / price

    Default action_levels: [-1.0, -0.5, 0.0, 0.5, 1.0]
    Custom levels supported: e.g., [-1, -0.3, -0.1, 0, 0.1, 0.3, 1]

    Leverage Design:
    ----------------
    Leverage is a **fixed global parameter** (not part of action space).
    This design separates risk management (leverage) from position sizing (actions):

    - Leverage = "How much risk am I willing to take?" (configuration)
    - Action = "How much of my allocation should I deploy?" (learned policy)

    **Dynamic Leverage** (not currently implemented):
    If you need the agent to dynamically choose leverage per trade, this could be
    implemented as a multi-dimensional action space:

        action_space = {
            "position_fraction": Categorical([-1, -0.5, 0, 0.5, 1]),
            "leverage_multiplier": Categorical([1, 3, 5])
        }

    However, fixed leverage is recommended for most use cases as it:
    - Simplifies learning (smaller action space)
    - Provides better risk control
    - Matches how traders typically operate
    - Makes hyperparameter tuning easier

    Account State (10 elements, matching live futures env):
    -------------------------------------------------------
    [cash, position_size, position_value, entry_price, current_price,
     unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]

    Position size is:
    - Positive for long positions
    - Negative for short positions
    - Zero for no position
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: SeqFuturesEnvConfig,
        feature_preprocessing_fn: Optional[Callable] = None,
    ):
        """
        Initialize the SeqFuturesEnv.

        Args:
            df: DataFrame with OHLCV data
            config: Environment configuration
            feature_preprocessing_fn: Optional custom preprocessing function
        """
        # Initialize base class (handles sampler, history, balance, etc.)
        super().__init__(df, config, feature_preprocessing_fn)

        # Environment-specific configuration
        self.action_levels = config.action_levels
        self.leverage = config.leverage
        self.margin_type = config.margin_type
        self.maintenance_margin_rate = config.maintenance_margin_rate

        # Validate leverage
        if not (1 <= config.leverage <= 125):
            raise ValueError("Leverage must be between 1 and 125.")

        # Define action spec
        self.action_spec = Categorical(len(self.action_levels))

        # Build observation specs with futures-specific account state
        account_state = [
            "cash", "position_size", "position_value", "entry_price", "current_price",
            "unrealized_pnlpct", "leverage", "margin_ratio", "liquidation_price", "holding_time"
        ]
        num_features = len(self.sampler.get_feature_keys())
        self._build_observation_specs(account_state, num_features)

        # Initialize futures-specific state (beyond base PositionState)
        # Note: These attributes are intentionally separate from PositionState as they are
        # specific to futures/leveraged trading and not applicable to spot/long-only environments.
        # PositionState contains universal position tracking (size, value, entry), while these
        # track futures-specific calculations (leverage-based PnL, liquidation risk).
        self.unrealized_pnl = 0.0  # Absolute unrealized PnL (calculated from leverage)
        self.unrealized_pnl_pct = 0.0  # Percentage unrealized PnL
        self.liquidation_price = 0.0  # Price at which position would be liquidated

    def _reset_history(self):
        """Reset all history tracking including position history."""
        self.history = FuturesHistoryTracker()

    def _calculate_liquidation_price(self, entry_price: float, position_size: float) -> float:
        """
        Calculate liquidation price for a position.

        For ISOLATED margin:
        - Long: liquidation_price = entry_price * (1 - 1/leverage + maintenance_margin_rate)
        - Short: liquidation_price = entry_price * (1 + 1/leverage - maintenance_margin_rate)
        """
        if position_size == 0:
            return 0.0

        margin_fraction = 1.0 / self.leverage

        if position_size > 0:
            # Long position - liquidated if price drops
            liquidation_price = entry_price * (1 - margin_fraction + self.maintenance_margin_rate)
        else:
            # Short position - liquidated if price rises
            liquidation_price = entry_price * (1 + margin_fraction - self.maintenance_margin_rate)

        return max(0, liquidation_price)

    def _calculate_margin_required(self, position_value: float) -> float:
        """Calculate initial margin required for a position."""
        return abs(position_value) / self.leverage

    def _calculate_unrealized_pnl(self, entry_price: float, current_price: float, position_size: float) -> float:
        """
        Calculate unrealized PnL.

        For long: PnL = (current_price - entry_price) * position_size
        For short: PnL = (entry_price - current_price) * abs(position_size)
        """
        if position_size == 0 or entry_price == 0:
            return 0.0

        if position_size > 0:
            # Long position
            return (current_price - entry_price) * position_size
        else:
            # Short position
            return (entry_price - current_price) * abs(position_size)

    def _calculate_unrealized_pnl_pct(self, entry_price: float, current_price: float, position_size: float) -> float:
        """Calculate unrealized PnL as a percentage."""
        if entry_price == 0:
            return 0.0

        if position_size > 0:
            # Long position
            return (current_price - entry_price) / entry_price
        elif position_size < 0:
            # Short position
            return (entry_price - current_price) / entry_price
        return 0.0

    def _check_liquidation(self, current_price: float) -> bool:
        """Check if current position should be liquidated."""
        if self.position.position_size == 0:
            return False

        if self.position.position_size > 0:
            # Long position - liquidated if price below liquidation price
            return current_price <= self.liquidation_price
        else:
            # Short position - liquidated if price above liquidation price
            return current_price >= self.liquidation_price

    def _calculate_fractional_position(
        self,
        action_value: float,
        current_price: float
    ) -> Tuple[float, float, str]:
        """Calculate position size from fractional action value.

        Uses shared utility function for consistent position sizing across all environments.

        Args:
            action_value: Action from [-1.0, 1.0] representing fraction of balance to allocate
            current_price: Current market price for position sizing calculation

        Returns:
            Tuple of (position_size, notional_value, side)
        """
        params = PositionCalculationParams(
            balance=self.balance,
            action_value=action_value,
            current_price=current_price,
            leverage=self.leverage,
            transaction_fee=self.transaction_fee,
            allow_short=True
        )
        return calculate_fractional_position(params)

    def _is_direction_switch(self, current_position: float, target_position: float) -> bool:
        """Check if switching from long to short or vice versa.

        Args:
            current_position: Current position size (positive=long, negative=short)
            target_position: Target position size (positive=long, negative=short)

        Returns:
            True if switching directions, False otherwise
        """
        return (current_position > 0 and target_position < 0) or \
               (current_position < 0 and target_position > 0)

    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        # Get market data (scaffold sets current_timestamp, truncated, and caches base features)
        obs_dict = self._get_observation_scaffold()
        current_price = self._cached_base_features["close"]

        # Calculate position value (absolute value of notional)
        self.position.position_value = abs(self.position.position_size * current_price)

        # Calculate unrealized PnL
        if self.position.position_size != 0:
            self.unrealized_pnl = self._calculate_unrealized_pnl(
                self.position.entry_price, current_price, self.position.position_size
            )
            self.unrealized_pnl_pct = self._calculate_unrealized_pnl_pct(
                self.position.entry_price, current_price, self.position.position_size
            )
        else:
            self.unrealized_pnl = 0.0
            self.unrealized_pnl_pct = 0.0

        # Calculate margin ratio
        total_balance = self.balance + self.unrealized_pnl
        margin_ratio = self.position.position_value / total_balance if total_balance > 0 else 0.0

        # Account state (10 elements, matching live futures env)
        account_state = torch.tensor([
            self.balance,
            self.position.position_size,  # Positive=long, Negative=short
            self.position.position_value,
            self.position.entry_price,
            current_price,
            self.unrealized_pnl_pct,
            float(self.leverage),
            margin_ratio,
            self.liquidation_price,
            float(self.position.hold_counter),
        ], dtype=torch.float)

        # Combine account state and market data
        obs_data = {self.account_state_key: account_state}
        obs_data.update(dict(zip(self.market_data_keys, obs_dict.values())))

        return TensorDict(obs_data, batch_size=())

    def _reset_position_state(self):
        """Reset position tracking state including futures-specific state."""
        super()._reset_position_state()
        # Reset futures-specific state
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.liquidation_price = 0.0
        self.position_history = []

    def _get_portfolio_value(self, current_price: Optional[float] = None) -> float:
        """Calculate total portfolio value including unrealized PnL for futures."""
        if current_price is None:
            if self.current_timestamp is None:
                raise RuntimeError(
                    "current_timestamp is not set. _get_portfolio_value() must be called "
                    "after _get_observation() which sets the current timestamp."
                )
            current_price = self._cached_base_features["close"]

        unrealized_pnl = self._calculate_unrealized_pnl(
            self.position.entry_price, current_price, self.position.position_size
        )
        return self.balance + unrealized_pnl

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        self.step_counter += 1

        # Store old portfolio value
        old_portfolio_value = self._get_portfolio_value()

        # Get desired action
        action_idx = tensordict.get("action", 0)
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        desired_action = self.action_levels[action_idx]

        # Get current price for liquidation check
        current_price = self.sampler.get_base_features(self.current_timestamp)["close"]

        # Check for liquidation before executing trade
        trade_info = {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        if self._check_liquidation(current_price):
            trade_info = self._execute_liquidation(current_price)
        else:
            trade_info = self._execute_trade_if_needed(desired_action)

        # Record trade action
        trade_action = 0
        if trade_info["executed"]:
            if trade_info["side"] == "long":
                trade_action = 1
            elif trade_info["side"] == "short":
                trade_action = -1

        # Get updated state
        next_tensordict = self._get_observation()
        new_portfolio_value = self._get_portfolio_value()

        # Add state_index for coverage tracking (only during training with random_start)
        if self.random_start:
            next_tensordict.set("state_index", torch.tensor(self.sampler._sequential_idx, dtype=torch.long))

        # Calculate reward
        reward = self._calculate_reward(
            old_portfolio_value, new_portfolio_value, desired_action, trade_info
        )

        # Record step history
        self.history.record_step(
            price=current_price,
            action=trade_action,
            reward=reward,
            portfolio_value=old_portfolio_value,
            position=self.position.position_size
        )

        # Check termination
        done = self._check_termination(new_portfolio_value)

        next_tensordict.set("reward", reward)
        next_tensordict.set("done", self.truncated or done)
        next_tensordict.set("truncated", self.truncated)
        next_tensordict.set("terminated", self.truncated or done)

        return next_tensordict

    def _execute_liquidation(self, current_price: float) -> Dict:
        """Execute forced liquidation of position."""
        trade_info = {
            "executed": True,
            "side": "liquidation",
            "fee_paid": 0.0,
            "liquidated": True,
        }

        # Realize the loss at liquidation price
        if self.position.position_size > 0:
            # Long position liquidated
            loss = (self.liquidation_price - self.position.entry_price) * self.position.position_size
        else:
            # Short position liquidated
            loss = (self.position.entry_price - self.liquidation_price) * abs(self.position.position_size)

        # Apply loss and fees
        liquidation_fee = abs(self.position.position_size * self.liquidation_price) * self.transaction_fee
        self.balance += loss - liquidation_fee
        trade_info["fee_paid"] = liquidation_fee

        # Reset position
        self.position.position_size = 0.0
        self.position.position_value = 0.0
        self.position.entry_price = 0.0
        self.liquidation_price = 0.0
        self.position.current_position = 0
        self.position.hold_counter = 0

        return trade_info

    def _execute_trade_if_needed(self, desired_action: float) -> Dict:
        """Execute trade if position change is needed.

        Routes to either fractional or fixed position sizing based on config.
        """
        current_price = self.sampler.get_base_features(self.current_timestamp)["close"]

        # Apply slippage
        price_noise = torch.empty(1).uniform_(1 - self.slippage, 1 + self.slippage).item()
        execution_price = current_price * price_noise

        if self.config.position_sizing_mode == "fractional":
            # NEW: Fractional position sizing
            return self._execute_fractional_action(desired_action, execution_price)
        else:
            # LEGACY: Fixed position sizing (backward compatibility)
            return self._execute_fixed_action(desired_action, current_price, price_noise)

    def _execute_fixed_action(self, desired_action: float, current_price: float, price_noise: float) -> Dict:
        """Execute action using fixed position sizing (legacy mode)."""
        trade_info = {
            "executed": False,
            "side": None,
            "fee_paid": 0.0,
            "liquidated": False,
        }

        # Action mapping: -1=short, 0=hold, 0.5=close, 1=long

        if desired_action == 0:
            # Hold - do nothing (keep any existing position open)
            pass

        elif desired_action == 0.5:
            # Close - explicitly exit position and go to cash
            if self.position.position_size != 0:
                trade_info = self._close_position(current_price, price_noise)

        elif desired_action == 1:
            # Go long
            if self.position.current_position == 1:
                # Already long - ignore duplicate action
                pass
            elif self.position.position_size < 0:
                # Close short first, then open long
                self._close_position(current_price, price_noise)
                trade_info = self._open_position("long", current_price, price_noise)
            else:
                # Open long from flat position
                trade_info = self._open_position("long", current_price, price_noise)

        elif desired_action == -1:
            # Go short
            if self.position.current_position == -1:
                # Already short - ignore duplicate action
                pass
            elif self.position.position_size > 0:
                # Close long first, then open short
                self._close_position(current_price, price_noise)
                trade_info = self._open_position("short", current_price, price_noise)
            else:
                # Open short from flat position
                trade_info = self._open_position("short", current_price, price_noise)

        # Update hold counter
        if self.position.position_size != 0:
            self.position.hold_counter += 1

        return trade_info

    def _execute_fractional_action(self, action_value: float, execution_price: float) -> Dict:
        """Execute action using fractional position sizing.

        This method handles the full lifecycle of position changes in fractional mode:
        - Opening new positions from flat
        - Closing positions to go neutral
        - Switching direction (long ↔ short)
        - Adjusting position size in same direction
        - Implicit holding (when target matches current)

        Args:
            action_value: Fractional action value in [-1.0, 1.0]
            execution_price: Price at which trade executes (includes slippage)

        Returns:
            trade_info: Dict with keys:
                - "executed" (bool): Whether trade was executed
                - "side" (str): "long", "short", "flat", or None
                - "fee_paid" (float): Transaction fee deducted
                - "liquidated" (bool): Always False for normal trades
        """
        trade_info = {
            "executed": False,
            "side": None,
            "fee_paid": 0.0,
            "liquidated": False,
        }

        # Calculate target position from action value
        target_position_size, target_notional, target_side = (
            self._calculate_fractional_position(action_value, execution_price)
        )

        # Tolerance for position comparison (0.1% of target position, or absolute minimum for very small positions)
        tolerance = max(abs(target_position_size) * POSITION_TOLERANCE_PCT, POSITION_TOLERANCE_ABS)

        # If target matches current position, do nothing (implicit HOLD)
        if abs(target_position_size - self.position.position_size) < tolerance:
            # Already at target position (implicit hold)
            if self.position.position_size != 0:
                self.position.hold_counter += 1
            return trade_info

        # Need to adjust position
        if target_position_size == 0.0:
            # Close position (go to cash)
            if self.position.position_size != 0:
                trade_info = self._close_position(execution_price, 1.0)  # No additional slippage
                trade_info["side"] = "flat"

        elif self.position.position_size == 0.0:
            # Open new position from flat
            trade_info = self._open_fractional_position(
                target_side, target_position_size, target_notional, execution_price
            )

        elif self._is_direction_switch(self.position.position_size, target_position_size):
            # Direction switch: close current, open opposite
            self._close_position(execution_price, 1.0)
            # Recalculate target position after closing (balance may have changed)
            target_position_size, target_notional, target_side = (
                self._calculate_fractional_position(action_value, execution_price)
            )
            trade_info = self._open_fractional_position(
                target_side, target_position_size, target_notional, execution_price
            )

        else:
            # Same direction, adjust size
            trade_info = self._adjust_position_size(
                target_position_size, target_notional, execution_price
            )

        # Update hold counter
        if self.position.position_size != 0:
            self.position.hold_counter += 1

        return trade_info

    def _open_fractional_position(
        self,
        side: str,
        position_size: float,
        notional_value: float,
        execution_price: float
    ) -> Dict:
        """Open a new fractional position from flat.

        Args:
            side: Position direction: "long" or "short"
            position_size: Target position size (quantity in base currency)
            notional_value: Position value in quote currency
            execution_price: Execution price (with slippage applied)

        Returns:
            trade_info: Dict with execution details including fees and success status
        """
        trade_info = {
            "executed": True,
            "side": side,
            "fee_paid": 0.0,
            "liquidated": False,
        }

        # Calculate margin and fee
        margin_required = notional_value / self.leverage
        fee = abs(notional_value) * self.transaction_fee

        # Check if sufficient balance
        if margin_required + fee > self.balance:
            trade_info["executed"] = False
            return trade_info

        # Deduct fee and margin
        self.balance -= fee
        trade_info["fee_paid"] = fee

        # Set position
        self.position.position_size = position_size
        self.position.position_value = abs(notional_value)
        self.position.entry_price = execution_price
        self.position.current_position = 1 if side == "long" else -1
        self.position.hold_counter = 0

        # Calculate liquidation price
        self.liquidation_price = self._calculate_liquidation_price(
            execution_price, position_size
        )

        return trade_info

    def _adjust_position_size(
        self,
        target_position_size: float,
        target_notional: float,
        execution_price: float
    ) -> Dict:
        """Adjust existing position size (same direction).

        Only trades the difference between current and target position.
        For example, going from 1.0 (100% long) to 0.5 (50% long) only sells 50% of the position.

        Args:
            target_position_size: Target position size
            target_notional: Target notional value
            execution_price: Execution price

        Returns:
            trade_info: Dict with execution details
        """
        trade_info = {
            "executed": True,
            "side": None,
            "fee_paid": 0.0,
            "liquidated": False,
        }

        # Calculate the difference
        delta_position = target_position_size - self.position.position_size
        delta_notional = abs(delta_position * execution_price)

        if abs(delta_position) < POSITION_TOLERANCE_ABS:  # Negligible difference
            trade_info["executed"] = False
            return trade_info

        if abs(target_position_size) > abs(self.position.position_size):
            # Increasing position size - add to position
            trade_info["side"] = "long" if delta_position > 0 else "short"

            # Calculate fee
            fee = delta_notional * self.transaction_fee

            # Check if sufficient balance
            margin_required = delta_notional / self.leverage
            if margin_required + fee > self.balance:
                trade_info["executed"] = False
                return trade_info

            # Execute trade
            self.balance -= fee
            trade_info["fee_paid"] = fee

            # Update position (keeping weighted average entry price)
            current_value = abs(self.position.position_size * self.position.entry_price)
            new_value = delta_notional
            total_value = current_value + new_value

            # Weighted average entry price
            if total_value > 0:
                self.position.entry_price = (
                    (self.position.entry_price * current_value + execution_price * new_value) / total_value
                )

            self.position.position_size = target_position_size
            self.position.position_value = abs(target_notional)

            # Recalculate liquidation price with new average entry
            self.liquidation_price = self._calculate_liquidation_price(
                self.position.entry_price, self.position.position_size
            )

        else:
            # Decreasing position size - partial close
            fraction_to_close = 1.0 - (abs(target_position_size) / abs(self.position.position_size))
            trade_info["side"] = "close_partial"

            # Calculate PnL on the portion being closed
            pnl = self._calculate_unrealized_pnl(
                self.position.entry_price,
                execution_price,
                self.position.position_size * fraction_to_close
            )

            # Calculate fee on the portion being closed
            close_notional = abs(self.position.position_size * fraction_to_close * execution_price)
            fee = close_notional * self.transaction_fee

            # Update balance
            self.balance += pnl - fee
            trade_info["fee_paid"] = fee

            # Update position (entry price stays the same for partial close)
            self.position.position_size = target_position_size
            self.position.position_value = abs(target_notional)

            # Liquidation price stays the same (same entry price, just smaller size)
            self.liquidation_price = self._calculate_liquidation_price(
                self.position.entry_price, self.position.position_size
            )

        return trade_info

    def _open_position(self, side: str, current_price: float, price_noise: float) -> Dict:
        """Open a new position."""
        trade_info = {
            "executed": True,
            "side": side,
            "fee_paid": 0.0,
            "liquidated": False,
        }

        execution_price = current_price * price_noise

        # Calculate position size using trade mode
        position_qty, notional_value = self._calculate_position_quantity(execution_price)

        # Calculate margin required
        margin_required = self._calculate_margin_required(notional_value)

        # Calculate fee
        fee = notional_value * self.transaction_fee

        if margin_required + fee > self.balance:
            # Not enough balance to open position
            trade_info["executed"] = False
            return trade_info

        # Execute trade
        self.balance -= fee
        trade_info["fee_paid"] = fee

        if side == "long":
            self.position.position_size = position_qty
            self.position.current_position = 1
        else:  # short
            self.position.position_size = -position_qty
            self.position.current_position = -1

        self.position.entry_price = execution_price
        self.position.position_value = notional_value
        self.liquidation_price = self._calculate_liquidation_price(
            execution_price, self.position.position_size
        )
        self.position.hold_counter = 0

        return trade_info

    def _close_position(self, current_price: float, price_noise: float) -> Dict:
        """Close existing position."""
        trade_info = {
            "executed": True,
            "side": "close",
            "fee_paid": 0.0,
            "liquidated": False,
        }

        if self.position.position_size == 0:
            trade_info["executed"] = False
            return trade_info

        execution_price = current_price * price_noise

        # Calculate PnL
        pnl = self._calculate_unrealized_pnl(
            self.position.entry_price, execution_price, self.position.position_size
        )

        # Calculate fee
        notional = abs(self.position.position_size * execution_price)
        fee = notional * self.transaction_fee

        # Update balance
        self.balance += pnl - fee
        trade_info["fee_paid"] = fee

        # Reset position
        self.position.position_size = 0.0
        self.position.position_value = 0.0
        self.position.entry_price = 0.0
        self.liquidation_price = 0.0
        self.position.current_position = 0
        self.position.hold_counter = 0

        return trade_info

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold or self.step_counter >= self.max_steps

    def close(self):
        """Clean up resources."""
        pass

    def get_metrics(self) -> Dict[str, float]:
        """
        Compute episode metrics from history.

        Returns:
            Dictionary containing:
                - total_return: Total portfolio return
                - sharpe_ratio: Annualized Sharpe ratio
                - sortino_ratio: Annualized Sortino ratio
                - calmar_ratio: Calmar ratio (return / max drawdown)
                - max_drawdown: Maximum drawdown (negative value)
                - max_dd_duration: Maximum drawdown duration in periods
                - num_trades: Number of trades executed
                - win_rate (reward>0): Percentage of profitable periods
                - avg_win: Average win amount
                - avg_loss: Average loss amount
                - profit_factor: Ratio of total wins to total losses
        """
        from torchtrade.metrics import compute_all_metrics
        from torchtrade.envs.offline.utils import compute_periods_per_year_crypto

        # Convert histories to tensors
        portfolio_values = torch.tensor(self.history.portfolio_values, dtype=torch.float32)
        rewards = torch.tensor(self.history.rewards, dtype=torch.float32)

        # Compute periods per year for annualization
        periods_per_year = compute_periods_per_year_crypto(
            self.execute_on_unit,
            self.execute_on_value
        )

        # Use shared metrics computation function
        return compute_all_metrics(
            portfolio_values=portfolio_values,
            rewards=rewards,
            action_history=self.history.actions,
            periods_per_year=periods_per_year,
        )

    def render_history(self, return_fig=False):
        """Render the history of the environment."""
        history_dict = self.history.to_dict()
        price_history = history_dict['base_prices']
        time_indices = list(range(len(price_history)))
        action_history = history_dict['actions']
        portfolio_value_history = history_dict['portfolio_values']
        position_history = history_dict['positions']

        # Calculate buy-and-hold balance
        initial_balance = portfolio_value_history[0]
        initial_price = price_history[0]
        units_held = initial_balance / initial_price
        buy_and_hold_balance = [units_held * price for price in price_history]

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 2, 1]})
        ax1, ax2, ax3 = axes

        # Plot price history
        ax1.plot(time_indices, price_history, label='Price History', color='blue', linewidth=1.5)

        # Plot buy/sell actions
        long_indices = [i for i, action in enumerate(action_history) if action == 1]
        long_prices = [price_history[i] for i in long_indices]
        short_indices = [i for i, action in enumerate(action_history) if action == -1]
        short_prices = [price_history[i] for i in short_indices]

        ax1.scatter(long_indices, long_prices, marker='^', color='green', s=80, label='Long', zorder=5)
        ax1.scatter(short_indices, short_prices, marker='v', color='red', s=80, label='Short', zorder=5)

        ax1.set_ylabel('Price (USD)')
        ax1.set_title('Price History with Long/Short Actions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot portfolio value
        ax2.plot(time_indices, portfolio_value_history, label='Portfolio Value', color='green', linewidth=1.5)
        ax2.plot(time_indices, buy_and_hold_balance, label='Buy and Hold', color='purple', linestyle='--', linewidth=1.5)
        ax2.set_ylabel('Portfolio Value (USD)')
        ax2.set_title('Portfolio Value vs Buy and Hold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot position
        ax3.fill_between(time_indices, position_history, 0,
                         where=[p > 0 for p in position_history], color='green', alpha=0.3, label='Long')
        ax3.fill_between(time_indices, position_history, 0,
                         where=[p < 0 for p in position_history], color='red', alpha=0.3, label='Short')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Time (Index)')
        ax3.set_ylabel('Position Size')
        ax3.set_title('Position History')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()


if __name__ == "__main__":
    import pandas as pd

    time_frames = [
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
    ]
    window_sizes = [12, 8]
    execute_on = TimeFrame(5, TimeFrameUnit.Minute)

    # Load sample data
    df = pd.read_csv(
        "/home/sebastian/Documents/TorchTrade/torchrl_alpaca_env/torchtrade/data/binance_spot_1m_cleaned/btcusdt_spot_1m_12_2024_to_09_2025.csv"
    )

    config = SeqFuturesEnvConfig(
        symbol="BTC/USD",
        time_frames=time_frames,
        window_sizes=window_sizes,
        execute_on=execute_on,
        leverage=10,
        transaction_fee=0.0004,
        slippage=0.001,
        bankrupt_threshold=0.1,
    )
    env = SeqFuturesEnv(df, config)

    td = env.reset()
    for i in range(env.max_steps):
        action = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
        td.set("action", action)
        td = env.step(td)
        td = td["next"]
        portfolio_value = td["account_state"][0].item() + td["account_state"][2].item()
        print(f"Step {i}: Action={action}, Portfolio=${portfolio_value:.2f}")
        if td["done"]:
            print(f"Episode done at step {i}")
            break

    env.render_history()
