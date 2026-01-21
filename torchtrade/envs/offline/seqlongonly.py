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

    # Action space configuration
    position_sizing_mode: str = "fractional"  # "fractional" (new default) or "fixed" (legacy)
    action_levels: List[float] = None  # Custom action levels, or None for defaults

    # DEPRECATED: Only used in legacy "fixed" mode
    include_hold_action: bool = True  # DEPRECATED - only used when position_sizing_mode="fixed"

    def __post_init__(self):
        """Normalize timeframe configuration and build action levels."""
        self.execute_on, self.time_frames, self.window_sizes = normalize_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

        # Set default action levels based on position sizing mode
        if self.action_levels is None:
            if self.position_sizing_mode == "fractional":
                # New default: fractional sizing with neutral at 0
                self.action_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
            else:
                # Legacy mode: maintain backward compatibility
                if self.include_hold_action:
                    self.action_levels = [-1.0, 0.0, 1.0]  # Sell-all, Hold, Buy-all
                else:
                    self.action_levels = [-1.0, 1.0]  # Sell-all, Buy-all

        # Validate position_sizing_mode
        if self.position_sizing_mode not in ["fractional", "fixed"]:
            raise ValueError(f"position_sizing_mode must be 'fractional' or 'fixed', got '{self.position_sizing_mode}'")

class SeqLongOnlyEnv(TorchTradeOfflineEnv):
    """Sequential long-only trading environment for backtesting.

    Supports 3 discrete actions: Sell-all (-1), Hold (0), Buy-all (1).
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

        For long-only environments:
        - Positive actions allocate cash to long positions
        - Negative actions reduce positions (sell)
        - Zero closes all positions (go to cash)

        Args:
            action_value: Action from [-1.0, 1.0] representing allocation
                         - Positive: fraction of cash to allocate to long
                         - Negative: fraction of position to sell
                         - Zero: close all (go to cash)
            current_price: Current market price

        Returns:
            Tuple of (position_size, notional_value, side):
            - position_size: Quantity in base currency (always >= 0 for long-only)
            - notional_value: Value in quote currency (USD)
            - side: "buy", "sell", or "flat"
        """
        # Handle neutral case
        if action_value == 0.0:
            return 0.0, 0.0, "flat"

        if action_value > 0:
            # Buy: allocate fraction of cash, accounting for fees
            # Formula: cost + fee = balance * fraction
            # Where: fee = cost * fee_rate
            # So: cost * (1 + fee_rate) = balance * fraction
            # Therefore: cost = (balance * fraction) / (1 + fee_rate)

            fraction = abs(action_value)
            capital_allocated = self.balance * fraction
            fee_multiplier = 1 + self.transaction_fee
            capital_after_fee = capital_allocated / fee_multiplier

            # Calculate position size (no leverage for spot trading)
            position_qty = capital_after_fee / current_price

            return position_qty, capital_after_fee, "buy"

        else:
            # Sell: reduce position by fraction
            # For now, simplified: negative action = sell all
            # (Could be extended to partial sells in the future)
            if self.position.position_size > 0:
                position_qty = 0.0  # Target flat
                notional_value = self.position.position_size * current_price
                return position_qty, notional_value, "sell"
            else:
                # No position to sell
                return 0.0, 0.0, "flat"

    def _execute_trade_if_needed(self, desired_position: float, base_price: float = None) -> Dict:
        """Execute trade if position change is needed.

        Routes to either fractional or fixed position sizing based on config.
        """
        if base_price is None:
            base_price = self.sampler.get_base_features(self.current_timestamp)["close"]

        # Apply slippage
        price_noise_factor = torch.empty(1,).uniform_(1 - self.slippage, 1 + self.slippage).item()
        execution_price = base_price * price_noise_factor

        if self.config.position_sizing_mode == "fractional":
            # NEW: Fractional position sizing
            return self._execute_fractional_action(desired_position, execution_price)
        else:
            # LEGACY: Fixed position sizing (backward compatibility)
            return self._execute_fixed_action(desired_position, base_price, price_noise_factor)

    def _execute_fixed_action(self, desired_position: float, base_price: float, price_noise_factor: float) -> Dict:
        """Execute action using fixed position sizing (legacy mode)."""
        trade_info = {"executed": False, "amount": 0, "side": None, "success": None, "price_noise": 0.0, "fee_paid": 0.0}

        # No action requested (hold)
        if desired_position == 0:
            if self.position.position_size > 0:
                self.position.hold_counter += 1
            return trade_info

        # Already at desired position
        if desired_position == self.position.current_position:
            if self.position.position_size > 0:
                self.position.hold_counter += 1
            return trade_info

        # Determine trade details
        side = "buy" if desired_position > 0 else "sell"
        amount = self._calculate_trade_amount(side)

        execution_price = base_price * price_noise_factor

        if side == "buy":
            fee_paid = amount * self.transaction_fee
            effective_amount = amount - fee_paid
            self.balance -= amount
            self.position.position_size = round(effective_amount / execution_price, 3)
            self.position.entry_price = execution_price
            self.position.hold_counter = 0
            self.position.position_value = round(self.position.position_size * execution_price, 3)
            self.position.unrealized_pnlpc = 0.0
            self.position.current_position = 1.0

        else:
            # Sell all available position
            sell_amount = self.position.position_size
            # Calculate proceeds and fee based on noisy execution price
            proceeds = sell_amount * execution_price
            fee_paid = proceeds * self.transaction_fee
            self.balance += round(proceeds - fee_paid, 3)
            self.position.position_size = 0.0
            self.position.hold_counter = 0
            self.position.position_value = 0.0
            self.position.unrealized_pnlpc = 0.0
            self.position.entry_price = 0.0
            self.position.current_position = 0.0

        trade_info.update({
            "executed": True,
            "amount": amount if side == "buy" else sell_amount,
            "side": side,
            "success": True,
            "price_noise": price_noise_factor,
            "fee_paid": fee_paid
        })

        return trade_info

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

        # Tolerance for position comparison (0.1% of target position, or absolute 0.0001 for very small positions)
        tolerance = max(abs(target_position_size) * 0.001, 0.0001)

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

    def _calculate_trade_amount(self, side: str) -> float:
        """Calculate the dollar amount to trade."""
        
        if side == "buy":
            # add some noise as we probably wont buy to the exact price
            amount = self.balance - np.random.uniform(0, self.balance * 0.015)
            return amount
        else:
            # sell all available
            return -1

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold or self.step_counter >= self.max_steps

    def close(self):
        """Clean up resources."""

    
    def render_history(self, return_fig=False):
        """Render the history of the environment."""

        history_dict = self.history.to_dict()
        price_history = history_dict['base_prices']
        time_indices = list(range(len(price_history)))
        action_history = history_dict['actions']
        reward_history = history_dict['rewards']
        portfolio_value_history = history_dict['portfolio_values']

        # Calculate buy-and-hold balance
        initial_balance = portfolio_value_history[0]  # Starting balance
        initial_price = price_history[0]      # Price at time 0
        units_held = initial_balance / initial_price  # Number of units bought at t=0
        buy_and_hold_balance = [units_held * price for price in price_history]  # Value of units over time

        # Create subplots: price history on top, balance history on bottom
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 2]})

        # Plot price history on the top subplot
        ax1.plot(time_indices, price_history, label='Price History', color='blue', linewidth=2)

        # Plot buy/sell actions on the price history
        buy_indices = [i for i, action in enumerate(action_history) if action == 1]
        buy_prices = [price_history[i] for i in buy_indices]
        sell_indices = [i for i, action in enumerate(action_history) if action == -1]
        sell_prices = [price_history[i] for i in sell_indices]

        ax1.scatter(buy_indices, buy_prices, marker='^', color='green', s=100, label='Buy (1)')
        ax1.scatter(sell_indices, sell_prices, marker='v', color='red', s=100, label='Sell (-1)')

        # Customize price plot
        ax1.set_ylabel('Price (USD)')
        ax1.set_title('Price History with Buy/Sell Actions')
        ax1.legend()
        ax1.grid(True)

        # Plot balance history and buy-and-hold on the bottom subplot
        ax2.plot(time_indices, portfolio_value_history, label='Portfolio Value History', color='green', linestyle='-', linewidth=2)
        ax2.plot(time_indices, buy_and_hold_balance, label='Buy and Hold', color='purple', linestyle='-', linewidth=2)

        # Customize balance plot
        ax2.set_xlabel('Time (Index)' if not isinstance(time_indices[0], (str, datetime)) else 'Time')
        ax2.set_ylabel('Portfolio Value (USD)')
        ax2.set_title('Portfolio Value History vs Buy and Hold')
        ax2.legend()
        ax2.grid(True)

        # If timestamps are provided, format the x-axis
        if isinstance(time_indices[0], (str, datetime)):
            fig.autofmt_xdate()  # Rotate and format timestamps for readability

        # Adjust layout to prevent overlap
        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()
