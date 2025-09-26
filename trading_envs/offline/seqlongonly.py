import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from warnings import warn
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
from sampler import MarketDataObservationSampler
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs import EnvBase
import torch
from torchrl.data import Categorical, Bounded
import pandas as pd
from utils import TimeFrame, TimeFrameUnit, tf_to_timedelta

@dataclass
class SeqLongOnlyEnvConfig:
    symbol: str = "BTC/USD"
    time_frames: Union[List[TimeFrame], TimeFrame] = TimeFrame(1, TimeFrameUnit.Minute)
    window_sizes: Union[List[int], int] = 10
    execute_on: TimeFrame = TimeFrame(1, TimeFrameUnit.Minute) # On which timeframe to execute trades
    initial_cash: float = 1000
    transaction_fee: float = 0.025 
    bankrupt_threshold: float = 0.1  # 10% of initial balance
    slippage: float = 0.01
    seed: Optional[int] = 42
    include_base_features: bool = False # Includes base features such as timestamps and ohlc to the tensordict

class SeqLongOnlyEnv(EnvBase):
    def __init__(self, df: pd.DataFrame, config: SeqLongOnlyEnvConfig, feature_preprocessing_fn: Optional[Callable] = None):
        self.action_levels = [-1.0, 0.0, 1.0]  # Sell-all, Do-Nothing, Buy-all
        self.config = config
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
            features_start_with="features_"
        )

        # Execute trades on the specified time frame
        self.execute_on_value = config.execute_on.value
        self.execute_on_unit = config.execute_on.unit.value

        # reset settings 
        self.initial_portfolio_value = config.initial_cash
        self.position_hold_counter = 0

        # Define action and observation spaces sell, hold (do nothing), buy
        self.action_spec = Categorical(len(self.action_levels))

        # Get the number of features from the observer
        obs, _ = self.sampler.get_random_observation()
        market_data_keys = self.sampler.get_observation_keys()

        num_features = obs[market_data_keys[0]].shape[1]

        # Observation space includes market data features and current position info
        self.observation_spec = CompositeSpec(shape=())
           
        self.market_data_key = "market_data"
        self.account_state_key = "account_state"

        # Account state spec: [cash, portfolio_value, position_size, entry_price, unrealized_pnlpct, holding_time]
        account_state_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(6,), dtype=torch.float)
        self.market_data_keys = []
        for i, market_data_name in enumerate(market_data_keys):
            market_data_key = "market_data_" + market_data_name
            market_data_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(config.window_sizes[i], num_features), dtype=torch.float)
            self.observation_spec.set(market_data_key, market_data_spec)
            self.market_data_keys.append(market_data_key)
        self.observation_spec.set(self.account_state_key, account_state_spec)

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
        obs_dict, self.current_timestamp = self.sampler.get_sequential_observation()

        # Get account state
        account_state = [
            self.balance,
            self.position_size,
            self.position_value,
            self.entry_price,
            self.unrealized_pnlpc,
            self.position_hold_counter
        ]
        account_state = torch.tensor(account_state, dtype=torch.float)

        out_td = TensorDict({self.account_state_key: account_state}, batch_size=())
        for market_data_name, data in zip(self.market_data_keys, obs_dict.values()):
            out_td.set(market_data_name, torch.from_numpy(data))

        return out_td

    def _calculate_reward(
        self,
        old_portfolio_value: float,
        new_portfolio_value: float,
        action: float,
        trade_info: Dict,
    ) -> float:
        """Calculate the step reward.

        This function computes the reward for the agent at a single step in the environment. 
        The reward is primarily based on realized profit from executed SELL actions. 
        It can also include a small penalty if the agent attempts an invalid action 
        (e.g., trying to SELL with no position or BUY when already in position).

        Args:
            old_portfolio_value (float): Portfolio value before the action.
            new_portfolio_value (float): Portfolio value after the action.
            action (float): Action taken by the agent. For example:
                1 = BUY, -1 = SELL, 0 = HOLD
            trade_info (dict): Trade information from the Alpaca client. Expected keys:
                - "executed" (bool): Whether the trade was successfully executed.
                - Other fields as needed for trade details (e.g., price, size).

        Returns:
            float: The reward for this step, scaled by `self.config.reward_scaling`.
                Positive if realized profit was made, small negative for invalid actions,
                or 0 otherwise.
        """

        if action == -1 and trade_info["executed"]:
            # Calculate portfolio return on realized profit
            portfolio_return = (
                new_portfolio_value - old_portfolio_value
            ) / old_portfolio_value
        elif not trade_info["executed"] and action != 0:
            # small penalty if agent tries an invalid action
            portfolio_return = - 0.001
        else:
            portfolio_return = 0.0

        if self.position_hold_counter > 50:  # Penalize long holds
            hold_reward = - 0.0001 * self.position_hold_counter
        else:
            hold_reward = 0.0

        # Scale the reward
        reward = portfolio_return + hold_reward

        return reward

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
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

        self.portfolio_value_history = []
        self.action_history = []
        self.reward_history = []
        self.base_price_history = []

        self.sampler.reset()
        self.balance = self.initial_portfolio_value
        self.position_hold_counter = 0
        self.current_position = 0.0
        self.position_value = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnlpc = 0.0
        self.step_counter = 0

        return self._get_observation()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        self.step_counter += 1
        
        # Store old portfolio value for reward calculation
        old_portfolio_value = self._get_portfolio_value()
        
        # Get desired action and current position
        desired_action = self.action_levels[tensordict.get("action", 0)]
        
        # Calculate and execute trade if needed
        trade_info = self._execute_trade_if_needed(desired_action)
        self.action_history.append(desired_action if trade_info["executed"] else 0)
        self.base_price_history.append(self.sampler.get_base_features(self.current_timestamp)["close"])
        self.portfolio_value_history.append(old_portfolio_value)

        if trade_info["executed"]:
            self.current_position = 1 if trade_info["side"] == "buy" else 0

        # Get updated state
        next_tensordict = self._get_observation()
        new_portfolio_value = self._get_portfolio_value()
        
        # Calculate reward and check termination
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value, desired_action, trade_info)
        self.reward_history.append(reward)

        done = self._check_termination(new_portfolio_value)

        next_tensordict.set("reward", reward)
        next_tensordict.set("done", done)
        next_tensordict.set("truncated", False)
        next_tensordict.set("terminated", done)
        
        # TODO: Make a dashboard that shows the portfolio value and action history etc
       # _ = self._create_info_dict(new_portfolio_value, trade_info, desired_action)
        
        return next_tensordict


    def _execute_trade_if_needed(self, desired_position: float) -> Dict:
        """Execute trade if position change is needed."""
        trade_info = {"executed": False, "amount": 0, "side": None, "success": None, "price_noise": 0.0, "fee_paid": 0.0}
        
        # If holding position or no change in position, do nothing
        if desired_position == 0 or desired_position == self.current_position or (self.current_position == 1 and desired_position == 1) or (self.current_position == 0 and desired_position == -1):
            # Compute unrealized PnL, add hold counter update last_portfolio_value
            self.position_hold_counter += 1
            current_price = self.sampler.get_base_features(self.current_timestamp)["close"]
            if self.position_size > 0:
                self.unrealized_pnlpc = round((current_price - self.entry_price) / self.entry_price, 3)
            else:
                self.unrealized_pnlpc = 0.0
            self.position_value = round(self.position_size * current_price, 3)
            
            return trade_info
        
        # Determine trade details
        side = "buy" if desired_position > 0 else "sell"
        amount = self._calculate_trade_amount(side)

        # Get base price and apply noise to simulate slippage
        base_price = self.sampler.get_base_features(self.current_timestamp)["close"]
        # Apply ±5% noise to the price to simulate market slippage
        price_noise_factor = np.random.uniform(1 - self.slippage, 1 + self.slippage)
        execution_price = base_price * price_noise_factor

        if side == "buy":
            fee_paid = amount * self.transaction_fee
            effective_amount = amount - fee_paid
            self.balance -= amount
            self.position_size = round(effective_amount / execution_price, 3)
            self.entry_price = execution_price
            self.position_hold_counter = 0
            self.position_value = round(self.position_size * execution_price, 3)
            self.unrealized_pnlpc = 0.0
            self.current_position = 1.0
            
        else:
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

    
    def render_history(self):
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

        plt.show()





if __name__ == "__main__":
    import pandas as pd
    import os

    time_frames=[
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
        TimeFrame(15, TimeFrameUnit.Minute),
        TimeFrame(1, TimeFrameUnit.Hour),
    ]
    window_sizes=[12, 8, 8, 24]  # ~12m, 40m, 2h, 1d
    execute_on=TimeFrame(5, TimeFrameUnit.Minute) # Try 15min

    df = pd.read_csv("./trading_envs/data/binance_spot_1m_cleaned/btcusdt_spot_1m_12_2024_to_09_2025.csv")
    config = SeqLongOnlyEnvConfig(
        symbol="BTC/USD",
        time_frames=time_frames,
        window_sizes=window_sizes,
        execute_on=execute_on,
        include_base_features=False,
        slippage=0.01,
        transaction_fee=0.015,
        bankrupt_threshold=0.1,
    )
    env = SeqLongOnlyEnv(df, config)

    td = env.reset()
    for i in range(200):
        #action  =  env.action_spec.sample()
        action = np.random.choice([0, 1, 2], p=[0.15, 0.35, 0.5])
        td.set("action", action)
        print(action)
        td = env.step(td)
        td = td["next"]
        print(i, " -- ", td["account_state"][0]+ td["account_state"][2])
        if td["done"]:
            print(td["done"])
            break
    env.render_history()
