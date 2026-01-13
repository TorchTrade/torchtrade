import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from warnings import warn
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
from torchtrade.envs.offline.sampler import MarketDataObservationSampler
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs import EnvBase
import torch
from torchrl.data import Bounded, MultiCategorical, Categorical
import pandas as pd
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit, tf_to_timedelta, InitialBalanceSampler, build_sltp_action_map, parse_timeframe_string
from torchtrade.envs.reward import build_reward_context, default_log_return, validate_reward_function

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

    def __post_init__(self):
        # Convert execute_on string to TimeFrame
        if isinstance(self.execute_on, str):
            self.execute_on = parse_timeframe_string(self.execute_on)

        # Normalize time_frames to list
        if not isinstance(self.time_frames, list):
            self.time_frames = [self.time_frames]

        # Convert all string timeframes to TimeFrame objects
        self.time_frames = [
            parse_timeframe_string(tf) if isinstance(tf, str) else tf
            for tf in self.time_frames
        ]

        # Normalize window_sizes to list
        if isinstance(self.window_sizes, int):
            self.window_sizes = [self.window_sizes] * len(self.time_frames)

        # Validate lengths match
        if len(self.window_sizes) != len(self.time_frames):
            raise ValueError(
                f"window_sizes length ({len(self.window_sizes)}) must match "
                f"time_frames length ({len(self.time_frames)})"
            )

class SeqLongOnlySLTPEnv(EnvBase):
    def __init__(self, df: pd.DataFrame, config: SeqLongOnlySLTPEnvConfig, feature_preprocessing_fn: Optional[Callable] = None):
        self.action_levels = [0.0, 1.0]  #Do-Nothing, Buy-all
        self.config = config

        # Validate custom reward function signature if provided
        if config.reward_function is not None:
            validate_reward_function(config.reward_function)

        self.transaction_fee = config.transaction_fee
        self.slippage = config.slippage
        if not (0 <= config.transaction_fee <= 1):
            raise ValueError("Transaction fee must be between 0 and 1 (e.g., 0.025 for 2.5%).")
        if not (0 <= config.slippage <= 1):
            raise ValueError("Slippage must be between 0 and 1 (e.g., 0.05 for 5%).")

        self.sampler = MarketDataObservationSampler(
            df,
            time_frames=config.time_frames,
            window_sizes=config.window_sizes,
            execute_on=config.execute_on,
            feature_processing_fn=feature_preprocessing_fn,
            features_start_with="features_",
            max_traj_length=config.max_traj_length,
        )
        self.random_start = config.random_start
        self.max_traj_length = config.max_traj_length
        # Execute trades on the specified time frame
        self.execute_on_value = config.execute_on.value
        self.execute_on_unit = config.execute_on.unit.value

        # reset settings
        self.initial_cash = config.initial_cash
        self.initial_cash_sampler = InitialBalanceSampler(config.initial_cash, config.seed)
        self.position_hold_counter = 0

        # action levels
        self.stoploss_levels = config.stoploss_levels
        self.takeprofit_levels = config.takeprofit_levels

        # Define action and observation spaces sell, hold (do nothing), buy
        self.action_map = build_sltp_action_map(
            self.stoploss_levels,
            self.takeprofit_levels,
            include_hold_action=config.include_hold_action,
            include_short_positions=False
        )
        self.action_spec = Categorical(len(self.action_map))

        # Get the number of features from the observer
        market_data_keys = self.sampler.get_observation_keys()
        num_features = len(self.sampler.get_feature_keys())

        # Observation space includes market data features and current position info
        self.observation_spec = CompositeSpec(shape=())
           
        self.market_data_key = "market_data"
        self.account_state_key = "account_state"

        # Account state spec: [cash, position_size, position_value, entry_price, current_price, unrealized_pnlpct, holding_time]
        self.account_state = ["cash", "position_size", "position_value", "entry_price", "current_price", "unrealized_pnlpct", "holding_time"]
        account_state_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(len(self.account_state),), dtype=torch.float)
        self.market_data_keys = []
        for i, market_data_name in enumerate(market_data_keys):
            market_data_key = "market_data_" + market_data_name + "_" + str(config.window_sizes[i])
            market_data_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(config.window_sizes[i], num_features), dtype=torch.float)
            self.observation_spec.set(market_data_key, market_data_spec)
            self.market_data_keys.append(market_data_key)
        self.observation_spec.set(self.account_state_key, account_state_spec)

        # Add coverage tracking indices to observation spec (only when random_start=True)
        if self.random_start:
            from torchrl.data.tensor_specs import Unbounded
            # reset_index: tracks episode start position diversity
            self.observation_spec.set(
                "reset_index",
                Unbounded(shape=(), dtype=torch.long)
            )
            # state_index: tracks all timesteps visited during episodes
            self.observation_spec.set(
                "state_index",
                Unbounded(shape=(), dtype=torch.long)
            )

        self.reward_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(1,), dtype=torch.float)
        self.max_steps = self.sampler.get_max_steps()
        self.step_counter = 0

        self.base_price_history = []
        self.action_history = []
        self.reward_history = []
        self.portfolio_value_history = []
        super().__init__()


    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        # Get market data
        obs_dict, self.current_timestamp, self.truncated = self.sampler.get_sequential_observation()

        # Cache base features for the new timestamp (avoids redundant calls)
        self._cached_base_features = self.sampler.get_base_features(self.current_timestamp)
        current_price = self._cached_base_features["close"]
        self.position_value = round(self.position_size * current_price, 3)
        if self.position_size > 0:
            self.unrealized_pnlpc = round((current_price - self.entry_price) / self.entry_price, 4)
        else:
            self.unrealized_pnlpc = 0.0

        # Get account state
        account_state = [
            self.balance,
            self.position_size,
            self.position_value,
            self.entry_price,
            current_price,
            self.unrealized_pnlpc,
            self.position_hold_counter
        ]
        account_state = torch.tensor(account_state, dtype=torch.float)

        obs_data = {self.account_state_key: account_state}
        obs_data.update(dict(zip(self.market_data_keys, obs_dict.values())))
        return TensorDict(obs_data, batch_size=())

    def _calculate_reward(
        self,
        old_portfolio_value: float,
        new_portfolio_value: float,
        action: float,
        trade_info: Dict,
    ) -> float:
        """
        Calculate reward using custom or default function.

        If config.reward_function is provided, uses that custom function.
        Otherwise, uses the default log return reward.

        Args:
            old_portfolio_value: Portfolio value before action
            new_portfolio_value: Portfolio value after action
            action: Action taken
            trade_info: Dictionary with trade execution details

        Returns:
            Reward value (float)
        """
        # Use custom reward function if provided
        if self.config.reward_function is not None:
            # Compute buy & hold value if terminal
            is_terminal = self.step_counter >= self.max_traj_length - 1
            buy_and_hold_value = None
            if is_terminal and len(self.base_price_history) > 0:
                buy_and_hold_value = (
                    self.initial_portfolio_value / self.base_price_history[0]
                ) * self.base_price_history[-1]

            ctx = build_reward_context(
                self,
                old_portfolio_value,
                new_portfolio_value,
                action,
                trade_info,
                portfolio_value_history=self.portfolio_value_history,
                action_history=self.action_history,
                reward_history=self.reward_history,
                base_price_history=self.base_price_history,
                initial_portfolio_value=self.initial_portfolio_value,
                buy_and_hold_value=buy_and_hold_value,
            )
            return float(self.config.reward_function(ctx)) * self.config.reward_scaling

        # Otherwise use default log return (no context needed)
        return default_log_return(old_portfolio_value, new_portfolio_value) * self.config.reward_scaling



    def _get_portfolio_value(self, current_price: float = None) -> float:
        """Calculate total portfolio value."""
        if current_price is None:
            current_price = self.sampler.get_base_features(self.current_timestamp)["close"]
        return self.balance + self.position_size * current_price


    def _set_seed(self, seed: int):
        """Set the seed for the environment."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        else:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment."""

        self.base_price_history = []
        self.action_history = []
        self.reward_history = []
        self.portfolio_value_history = []

        max_episode_steps = self.sampler.reset(random_start=self.random_start)
        self.max_traj_length = max_episode_steps # overwrite as we might execute on different time frame so actual step might differ
        initial_portfolio_value = self.initial_cash_sampler.sample()
        self.balance = initial_portfolio_value
        self.initial_portfolio_value = initial_portfolio_value
        self.position_hold_counter = 0
        self.current_position = 0.0
        self.position_value = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnlpc = 0.0
        self.step_counter = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0

        obs = self._get_observation()

        # Add coverage tracking indices (only during training with random_start)
        if self.random_start:
            obs.set("reset_index", torch.tensor(self.sampler._sequential_idx, dtype=torch.long))
            obs.set("state_index", torch.tensor(self.sampler._sequential_idx, dtype=torch.long))

        return obs

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
            self.current_position = 1 if trade_info["side"] == "buy" else 0
        self.action_history.append(trade_action)
        self.base_price_history.append(cached_price)
        self.portfolio_value_history.append(old_portfolio_value)

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
        self.reward_history.append(reward)

        done = self._check_termination(new_portfolio_value)
        next_tensordict.set("reward", reward)
        next_tensordict.set("done", self.truncated or done)
        next_tensordict.set("truncated", self.truncated)
        next_tensordict.set("terminated", self.truncated or done)
        return next_tensordict

    def trigger_sell(self, trade_info, execution_price):

        # Sell all available position
        sell_amount = self.position_size
        # Calculate proceeds and fee based on noisy execution price
        proceeds = sell_amount * execution_price
        fee_paid = proceeds * self.transaction_fee
        self.balance += round(proceeds - fee_paid, 3)
        self.position_size = 0.0
        self.position_hold_counter = 0
        self.position_value = 0.0
        self.unrealized_pnlpc = 0.0
        self.entry_price = 0.0
        self.current_position = 0.0
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
        if self.position_size > 0 and self.stop_loss > 0 and self.take_profit > 0:
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

        # If holding position or no change in position, do nothing
        if self.current_position == 1 or action_tuple == (None, None):
            # Compute unrealized PnL, add hold counter update last_portfolio_value
            if self.position_size > 0:
                self.position_hold_counter += 1

            return trade_info
        
        # Determine trade details
        if self.position_size == 0 and self.current_position == 0 and action_tuple != (None, None):
            side = "buy"
            amount = self._calculate_trade_amount(side)

            # Get base price and apply noise to simulate slippage
            # Apply ±5% noise to the price to simulate market slippage
            price_noise_factor = torch.empty(1,).uniform_(1 - self.slippage, 1 + self.slippage).item()
            execution_price = ohlcv_base_values["close"] * price_noise_factor

            fee_paid = amount * self.transaction_fee
            effective_amount = amount - fee_paid
            self.balance -= amount
            self.position_size = round(effective_amount / execution_price, 3)
            self.entry_price = execution_price
            self.position_hold_counter = 0
            self.position_value = round(self.position_size * execution_price, 3)
            self.unrealized_pnlpc = 0.0
            self.current_position = 1.0
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

    
    def render_history(self, return_fig=False):
        """Render the history of the environment."""
        
        price_history = self.base_price_history
        time_indices = list(range(len(price_history)))
        action_history = self.action_history
        reward_history = self.reward_history
        portfolio_value_history = self.portfolio_value_history

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


if __name__ == "__main__":
    import pandas as pd
    import os

    time_frames=[
        TimeFrame(15, TimeFrameUnit.Minute),
    ]
    window_sizes=[32]  # ~12m, 40m, 2h, 1d
    execute_on=TimeFrame(15, TimeFrameUnit.Minute) # Try 15min

    df = pd.read_csv("/home/sebastian/Documents/TorchTrade/torchrl_alpaca_env/torchtrade/data/binance_spot_1m_cleaned/btcusdt_spot_1m_12_2024_to_09_2025.csv")

    #df = df[0:(1440 * 7)] # 1440 minutes = 1 day

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
