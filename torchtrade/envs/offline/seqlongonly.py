from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union, Callable

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
    include_hold_action: bool = True  # Include HOLD action (0.0) in action space
    max_traj_length: Optional[int] = None
    random_start: bool = True
    reward_function: Optional[Callable] = None  # Custom reward function (uses default if None)
    reward_scaling: float = 1.0
    action_levels: List[float] = None  # Built in __post_init__ based on include_hold_action

    def __post_init__(self):
        """Normalize timeframe configuration and build action levels."""
        self.execute_on, self.time_frames, self.window_sizes = normalize_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

        if self.action_levels is None:
            if self.include_hold_action:
                self.action_levels = [-1.0, 0.0, 1.0]  # Sell-all, Hold, Buy-all
            else:
                self.action_levels = [-1.0, 1.0]  # Sell-all, Buy-all
        else:
            # User provided custom action_levels - include_hold_action is ignored
            import warnings
            if not self.include_hold_action and 0.0 in self.action_levels:
                warnings.warn(
                    "Custom action_levels provided with include_hold_action=False, but action_levels "
                    "contains 0.0 (hold action). The custom action_levels will be used as-is. "
                    "Consider removing 0.0 from action_levels or setting include_hold_action=True.",
                    UserWarning
                )

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


    def _execute_trade_if_needed(self, desired_position: float, base_price: float = None) -> Dict:
        """Execute trade if position change is needed."""
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

        # Get base price and apply noise to simulate slippage
        if base_price is None:
            base_price = self.sampler.get_base_features(self.current_timestamp)["close"]
        # Apply ±5% noise to the price to simulate market slippage
        price_noise_factor = torch.empty(1,).uniform_(1 - self.slippage, 1 + self.slippage).item()
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