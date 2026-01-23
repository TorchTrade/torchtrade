import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from warnings import warn
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
from tensordict import TensorDict, TensorDictBase
import torch
from torchrl.data import Categorical
import pandas as pd
import datasets
from torchtrade.envs.core.offline_base import TorchTradeOfflineEnv
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit, tf_to_timedelta, normalize_timeframe_config
from torchtrade.envs.offline.infrastructure.utils import build_sltp_action_map

@dataclass
class SeqLongOnlySLTPEnvConfig:
    symbol: str = "BTC/USD"
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]] = "1Min"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, TimeFrame] = "1Min"  # On which timeframe to execute trades
    initial_cash: Union[List[int], int] = (1000, 5000)
    transaction_fee: float = 0.025
    stoploss_levels: Union[List[float], float] = (-0.025, -0.05, -0.1)
    takeprofit_levels: Union[List[float], float] = (0.05, 0.1, 0.2)
    bankrupt_threshold: float = 0.1  # 10% of initial balance
    slippage: float = 0.01
    seed: Optional[int] = 42
    include_base_features: bool = False # Includes base features such as timestamps and ohlc to the tensordict
    max_traj_length: Optional[int] = None
    random_start: bool = True
    reward_function: Optional[Callable] = None  # Custom reward function (uses default if None)
    reward_scaling: float = 1.0
    include_hold_action: bool = True  # Include HOLD action (index 0) in action space
    include_close_action: bool = False  # Include CLOSE action for manual position exit (default: False for SLTP)

    def __post_init__(self):
        self.execute_on, self.time_frames, self.window_sizes = normalize_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

class SeqLongOnlySLTPEnv(TorchTradeOfflineEnv):
    """Sequential long-only trading environment with stop-loss/take-profit support.

    Supports combinatorial action space with configurable SL/TP levels.
    """

    def __init__(self, df: pd.DataFrame, config: SeqLongOnlySLTPEnvConfig, feature_preprocessing_fn: Optional[Callable] = None):
        """
        Initialize the sequential long-only SLTP environment.

        Args:
            df: OHLCV DataFrame for backtesting
            config: Environment configuration
            feature_preprocessing_fn: Optional function to preprocess features
        """
        # Initialize base class (handles sampler, history, balance, transaction fee validation, etc.)
        super().__init__(df, config, feature_preprocessing_fn)

        # Environment-specific configuration
        self.action_levels = [0.0, 1.0]  # Do-Nothing, Buy-all
        self.stoploss_levels = config.stoploss_levels
        self.takeprofit_levels = config.takeprofit_levels

        # Define action and observation spaces
        self.action_map = build_sltp_action_map(
            self.stoploss_levels,
            self.takeprofit_levels,
            include_hold_action=config.include_hold_action,
            include_close_action=config.include_close_action,
            include_short_positions=False
        )
        self.action_spec = Categorical(len(self.action_map))

        # Build observation specs
        account_state = [
            "cash", "position_size", "position_value", "entry_price",
            "current_price", "unrealized_pnlpct", "holding_time"
        ]
        num_features = len(self.sampler.get_feature_keys())
        self._build_observation_specs(account_state, num_features)

        # Initialize SLTP-specific state
        self.stop_loss = 0.0
        self.take_profit = 0.0


    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        # Get market data (scaffold sets current_timestamp, truncated, and caches base features)
        obs_dict = self._get_observation_scaffold()
        current_price = self._cached_base_features["close"]

        # Update position metrics using base class helper
        self._update_position_metrics(current_price)

        # Build and return observation using base class helper
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

    def _reset_position_state(self):
        """Reset position tracking state including SLTP-specific state."""
        super()._reset_position_state()
        self.stop_loss = 0.0
        self.take_profit = 0.0

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        self.step_counter += 1

        # Cache base features once for current timestamp (avoids 4+ redundant get_base_features calls)
        cached_base = self._cached_base_features
        cached_price = cached_base["close"]

        # Store old portfolio value for reward calculation
        old_portfolio_value = self._get_portfolio_value(cached_price)

        # Get desired action and current position
        action = tensordict["action"]
        action_tuple = self.action_map[action.item()]

        # Calculate and execute trade if needed (pass cached base features)
        trade_info = self._execute_trade_if_needed(action_tuple, cached_base)
        trade_action = 0
        if trade_info["executed"]:
            trade_action = 1 if trade_info["side"] == "buy" else -1
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
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value, action_tuple, trade_info)

        # Record step history
        self.history.record_step(
            price=cached_price,
            action=trade_action,
            reward=reward,
            portfolio_value=old_portfolio_value
        )

        done = self._check_termination(new_portfolio_value)
        next_tensordict.set("reward", reward)
        next_tensordict.set("done", self.truncated or done)
        next_tensordict.set("truncated", self.truncated)
        next_tensordict.set("terminated", self.truncated or done)
        return next_tensordict

    def trigger_sell(self, trade_info, execution_price):

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
            "amount": sell_amount,
            "side": "sell",
            "success": True,
            "price_noise": 0.0,
            "fee_paid": fee_paid
        })
        return trade_info

    def _execute_trade_if_needed(self, action_tuple, ohlcv_base_values: dict = None) -> Dict:
        """Execute trade if position change is needed."""
        trade_info = {"executed": False, "amount": 0, "side": None, "success": None, "price_noise": 0.0, "fee_paid": 0.0}

        if ohlcv_base_values is None:
            ohlcv_base_values = self.sampler.get_base_features(self.current_timestamp)
        # Test if stop loss or take profit are triggered
        if self.position.position_size > 0 and self.stop_loss > 0 and self.take_profit > 0:
            open_price = ohlcv_base_values["open"]
            close_price = ohlcv_base_values["close"]
            high_price = ohlcv_base_values["high"]
            low_price = ohlcv_base_values["low"]
            if open_price < self.stop_loss:
                return self.trigger_sell(trade_info, open_price)
            elif low_price < self.stop_loss:
                return self.trigger_sell(trade_info, low_price)
            elif high_price > self.take_profit:
                return self.trigger_sell(trade_info, high_price)
            elif close_price < self.stop_loss:
                return self.trigger_sell(trade_info, close_price)
            elif close_price > self.take_profit:
                return self.trigger_sell(trade_info, close_price)

        # CLOSE action - explicitly exit position
        if action_tuple == ("close", None):
            if self.position.position_size > 0:
                close_price = ohlcv_base_values["close"]
                trade_info = self.trigger_sell(trade_info, close_price)
            return trade_info

        # HOLD action - do nothing
        if action_tuple == (None, None):
            if self.position.position_size > 0:
                self.position.hold_counter += 1
            return trade_info

        # If already in position, do nothing (ignore duplicate long actions)
        if self.position.current_position == 1:
            if self.position.position_size > 0:
                self.position.hold_counter += 1
            return trade_info

        # Determine trade details - open new position
        if self.position.position_size == 0 and self.position.current_position == 0:
            side = "buy"
            amount = self._calculate_trade_amount(side)

            # Get base price and apply noise to simulate slippage
            # Apply ±5% noise to the price to simulate market slippage
            price_noise_factor = torch.empty(1,).uniform_(1 - self.slippage, 1 + self.slippage).item()
            execution_price = ohlcv_base_values["close"] * price_noise_factor

            fee_paid = amount * self.transaction_fee
            effective_amount = amount - fee_paid
            self.balance -= amount
            self.position.position_size = round(effective_amount / execution_price, 3)
            self.position.entry_price = execution_price
            self.position.hold_counter = 0
            self.position.position_value = round(self.position.position_size * execution_price, 3)
            self.position.unrealized_pnlpc = 0.0
            self.position.current_position = 1.0
            stop_loss_pct, take_profit_pct = action_tuple
            self.stop_loss = execution_price * (1 + stop_loss_pct)
            self.take_profit = execution_price * (1 + take_profit_pct)
                
            trade_info.update({
                "executed": True,
                "amount": amount,
                "side": side,
                "success": True,
                "price_noise": price_noise_factor,
                "fee_paid": fee_paid
            })
            
            return trade_info
        else:
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

if __name__ == "__main__":
    import pandas as pd
    import os

    time_frames=[
        TimeFrame(15, TimeFrameUnit.Minute),
    ]
    window_sizes=[32]  # ~12m, 40m, 2h, 1d
    execute_on=TimeFrame(15, TimeFrameUnit.Minute) # Try 15min

    df = datasets.load_dataset("Torch-Trade/btcusdt_spot_1m_03_2023_to_12_2025")
    df = df["train"].to_pandas()

    config = SeqLongOnlySLTPEnvConfig(
        symbol="BTC/USD",
        time_frames=time_frames,
        window_sizes=window_sizes,
        execute_on=execute_on,
        include_base_features=False,
        slippage=0.01,
        transaction_fee=0.0025,
        bankrupt_threshold=0.1,
    )
    env = SeqLongOnlySLTPEnv(df, config)

    td = env.reset()
    for i in range(env.max_steps):
        action  =  env.action_spec.sample()
        td.set("action", action)
        print(action)
        td = env.step(td)
        td = td["next"]
        print(i, " -- ", td["account_state"][0]+ td["account_state"][2])
        if td["done"]:
            print(td["done"])
            break
    env.render_history()
