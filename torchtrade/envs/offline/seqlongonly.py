from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensordict import TensorDict, TensorDictBase
import torch
from torchrl.data import Categorical
import pandas as pd

from torchtrade.envs.offline.base import TorchTradeOfflineEnv
from torchtrade.envs.timeframe import TimeFrame, TimeFrameUnit, normalize_timeframe_config
from torchtrade.envs.fractional_sizing import (
    build_default_action_levels,
    validate_position_sizing_mode,
    calculate_fractional_position,
    PositionCalculationParams,
    POSITION_TOLERANCE_PCT,
    POSITION_TOLERANCE_ABS,
)

@dataclass
class SeqLongOnlyEnvConfig:
    symbol: str = "BTC/USD"
    time_frames: Union[List[TimeFrame], TimeFrame] = TimeFrame(1, TimeFrameUnit.Minute)
    window_sizes: Union[List[int], int] = 10
    execute_on: TimeFrame = TimeFrame(1, TimeFrameUnit.Minute) # On which timeframe to execute trades
    initial_cash: Union[List[int], int] = (1000, 5000)
    transaction_fee: float = 0.025
    bankrupt_threshold: float = 0.1  # 10% of initial balance
    slippage: float = 0.01
    seed: Optional[int] = 42
    include_base_features: bool = False # Includes base features such as timestamps and ohlc to the tensordict
    max_traj_length: Optional[int] = None
    random_start: bool = True
    reward_function: Optional[Callable] = None  # Custom reward function (uses default if None)
    reward_scaling: float = 1.0

    # Action space configuration (fractional mode only)
    action_levels: List[float] = None  # Custom action levels, or None for defaults

    def __post_init__(self):
        """Normalize timeframe configuration and build action levels."""
        self.execute_on, self.time_frames, self.window_sizes = normalize_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

        # Build default action levels for fractional mode
        if self.action_levels is None:
            self.action_levels = build_default_action_levels(
                position_sizing_mode="fractional",
                include_hold_action=True,
                include_close_action=False,  # Long-only doesn't use close action
                allow_short=False  # Long-only environment
            )
        else:
            # Validate custom action levels
            if not all(-1.0 <= a <= 1.0 for a in self.action_levels):
                raise ValueError(
                    f"All action_levels must be in range [-1.0, 1.0], got {self.action_levels}"
                )
            if len(self.action_levels) != len(set(self.action_levels)):
                raise ValueError(
                    f"action_levels must not contain duplicates, got {self.action_levels}"
                )
            if len(self.action_levels) < 2:
                raise ValueError(
                    f"action_levels must contain at least 2 actions, got {len(self.action_levels)}"
                )

            # Warn if using negative actions (redundant for long-only)
            if any(a < 0 for a in self.action_levels):
                import warnings
                warnings.warn(
                    f"Long-only environment has negative action_levels {self.action_levels}. "
                    "Negative actions behave the same as action=0 (close position), adding "
                    "unnecessary redundancy. Consider using only non-negative actions like [0.0, 0.5, 1.0].",
                    UserWarning
                )

class SeqLongOnlyEnv(TorchTradeOfflineEnv):
    """Sequential long-only trading environment for backtesting.

    Action Space (Fractional Mode - Default):
    --------------------------------------
    Actions represent the fraction of available balance to allocate to the asset.
    Action values in range [0.0, 1.0]:

    - action = 0.0: Close position (go to 100% cash)
    - action > 0: Buy/increase position targeting action × balance
      - action = 0.5: Target 50% of balance invested
      - action = 1.0: Target 100% of balance invested (all-in)

    Position sizing formula:
        For positive actions: target_notional = balance × action
        For action = 0: sell entire position

    Default action_levels: [0.0, 0.5, 1.0]
    Custom levels supported: e.g., [0.0, 0.25, 0.5, 0.75, 1.0] for finer control

    Note: Negative actions are technically supported for backwards compatibility
    (they behave the same as action=0, closing the position), but are not recommended
    as they add unnecessary redundancy to the action space.

    Note: Unlike futures environments, there is no leverage parameter since this
    is spot trading (1x leverage only).

    **Dynamic Position Sizing** (not currently implemented):
    Similar to futures environments, action levels control position fractions
    rather than dynamic leverage. For more complex position sizing strategies,
    you could extend this to multi-dimensional actions, but the current design
    is recommended for simplicity and ease of learning.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: SeqLongOnlyEnvConfig,
        feature_preprocessing_fn: Optional[Callable] = None
    ):
        """
        Initialize the sequential long-only environment.

        Args:
            df: OHLCV DataFrame for backtesting
            config: Environment configuration
            feature_preprocessing_fn: Optional function to preprocess features
        """
        # Initialize base class (handles sampler, history, balance, etc.)
        super().__init__(df, config, feature_preprocessing_fn)

        # Environment-specific configuration
        self.action_levels = config.action_levels

        # Define action spec
        self.action_spec = Categorical(len(self.action_levels))

        # Build observation specs
        account_state = [
            "cash", "position_size", "position_value", "entry_price",
            "current_price", "unrealized_pnlpct", "holding_time"
        ]
        num_features = len(self.sampler.get_feature_keys())
        self._build_observation_specs(account_state, num_features)


    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        # Get market data (scaffold sets current_timestamp, truncated, and caches base features)
        obs_dict = self._get_observation_scaffold()
        current_price = self._cached_base_features["close"]

        # Update position metrics using base class helper
        self._update_position_metrics(current_price)

        # Build observation using base class helper
        account_state_values = [
            self.balance,
            self.position.position_size,
            self.position.position_value,
            self.position.entry_price,
            current_price,
            self.position.unrealized_pnlpc,
            self.position.hold_counter
        ]
        return self._build_standard_observation(obs_dict, account_state_values)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        self.step_counter += 1

        # Cache base features once for current timestamp (avoids 4+ redundant get_base_features calls)
        cached_base = self._cached_base_features
        cached_price = cached_base["close"]

        # Store old portfolio value for reward calculation
        old_portfolio_value = self._get_portfolio_value(cached_price)

        # Get desired action and current position
        desired_action = self.action_levels[tensordict.get("action", 0)]

        # Calculate and execute trade if needed (pass cached price)
        trade_info = self._execute_trade_if_needed(desired_action, cached_price)

        if trade_info["executed"]:
            self.position.current_position = 1 if trade_info["side"] == "buy" else 0

        # Get updated state (this advances timestamp and caches new base features)
        next_tensordict = self._get_observation()
        # Use newly cached base features for new portfolio value
        new_price = self._cached_base_features["close"]
        new_portfolio_value = self._get_portfolio_value(new_price)

        # Add state_index for coverage tracking (only during training with random_start)
        if self.random_start:
            next_tensordict.set("state_index", torch.tensor(self.sampler._sequential_idx, dtype=torch.long))

        # Calculate reward and check termination
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value, desired_action, trade_info)

        # Record step history
        self.history.record_step(
            price=cached_price,
            action=desired_action,
            reward=reward,
            portfolio_value=old_portfolio_value
        )

        done = self._check_termination(new_portfolio_value)
        next_tensordict.set("reward", reward)
        next_tensordict.set("done", self.truncated or done)
        next_tensordict.set("truncated", self.truncated)
        next_tensordict.set("terminated", self.truncated or done)

        return next_tensordict


    def _calculate_fractional_position(
        self,
        action_value: float,
        current_price: float
    ) -> Tuple[float, float, str]:
        """Calculate position size from fractional action value for long-only env.

        Uses shared utility function for consistent position sizing.
        For long-only: negative actions are handled as sells in _execute_fractional_action.

        Args:
            action_value: Action from [-1.0, 1.0]
            current_price: Current market price

        Returns:
            Tuple of (position_size, notional_value, side)
        """
        # For long-only, we only use the shared utility for positive actions
        # Negative actions are handled as position reductions in execution logic
        if action_value <= 0:
            return 0.0, 0.0, "flat" if action_value == 0 else "sell"

        params = PositionCalculationParams(
            balance=self.balance,
            action_value=action_value,
            current_price=current_price,
            leverage=1,  # Long-only has no leverage
            transaction_fee=self.transaction_fee,
            allow_short=False
        )
        position_size, notional_value, side = calculate_fractional_position(params)

        # For long-only, convert "long" to "buy" for clarity
        side = "buy" if side == "long" else side

        return position_size, notional_value, side

    def _execute_trade_if_needed(self, desired_position: float, base_price: float = None) -> Dict:
        """Execute trade using fractional position sizing."""
        if base_price is None:
            base_price = self.sampler.get_base_features(self.current_timestamp)["close"]

        # Apply slippage
        price_noise_factor = torch.empty(1,).uniform_(1 - self.slippage, 1 + self.slippage).item()
        execution_price = base_price * price_noise_factor

        # Execute fractional action
        return self._execute_fractional_action(desired_position, execution_price)

    def _execute_fractional_action(self, action_value: float, execution_price: float) -> Dict:
        """Execute action using fractional position sizing.

        Args:
            action_value: Fractional action value in [-1.0, 1.0]
            execution_price: Price at which trade executes (includes slippage)

        Returns:
            trade_info: Dict with execution details
        """
        trade_info = {
            "executed": False,
            "amount": 0,
            "side": None,
            "success": None,
            "price_noise": 0.0,
            "fee_paid": 0.0
        }

        # Calculate target position from action value
        target_position_size, target_notional, target_side = (
            self._calculate_fractional_position(action_value, execution_price)
        )

        # Tolerance for position comparison (0.1% of target position, or absolute minimum for very small positions)
        tolerance = max(abs(target_position_size) * POSITION_TOLERANCE_PCT, POSITION_TOLERANCE_ABS)

        # If target matches current position, do nothing (implicit HOLD)
        if abs(target_position_size - self.position.position_size) < tolerance:
            if self.position.position_size > 0:
                self.position.hold_counter += 1
            return trade_info

        # Execute trade
        if target_side == "flat" or target_side == "sell":
            # Sell/close position
            if self.position.position_size > 0:
                sell_amount = self.position.position_size
                proceeds = sell_amount * execution_price
                fee_paid = proceeds * self.transaction_fee
                self.balance += proceeds - fee_paid

                # Reset position
                self.position.position_size = 0.0
                self.position.position_value = 0.0
                self.position.entry_price = 0.0
                self.position.unrealized_pnlpc = 0.0
                self.position.current_position = 0.0
                self.position.hold_counter = 0

                trade_info.update({
                    "executed": True,
                    "amount": sell_amount,
                    "side": "sell",
                    "success": True,
                    "fee_paid": fee_paid
                })

        elif target_side == "buy":
            # Buy position
            # Calculate actual capital to use (accounting for fees)
            capital_to_use = target_notional
            fee_paid = capital_to_use * self.transaction_fee
            total_cost = capital_to_use + fee_paid

            if total_cost > self.balance:
                # Not enough balance - cap at what we can afford
                capital_to_use = self.balance / (1 + self.transaction_fee)
                fee_paid = capital_to_use * self.transaction_fee
                total_cost = capital_to_use + fee_paid

            if total_cost < self.balance * 0.01:  # Minimum trade threshold
                return trade_info

            # Execute buy
            self.balance -= total_cost
            position_qty = capital_to_use / execution_price

            self.position.position_size = position_qty
            self.position.entry_price = execution_price
            self.position.position_value = position_qty * execution_price
            self.position.unrealized_pnlpc = 0.0
            self.position.current_position = 1.0
            self.position.hold_counter = 0

            trade_info.update({
                "executed": True,
                "amount": capital_to_use,
                "side": "buy",
                "success": True,
                "fee_paid": fee_paid
            })

        # Update hold counter
        if self.position.position_size > 0:
            self.position.hold_counter += 1

        return trade_info

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold or self.step_counter >= self.max_steps

    def close(self):
        """Clean up resources."""