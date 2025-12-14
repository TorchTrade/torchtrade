import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from warnings import warn
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product

import numpy as np
from torchtrade.envs.offline.sampler import MarketDataObservationSampler
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs import EnvBase
import torch
from torchrl.data import Bounded, MultiCategorical, Categorical
import pandas as pd
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit, tf_to_timedelta
import random

def combinatory_action_map(stoploss_levels: List[float], takeprofit_levels: List[float]) -> Dict:
    action_map = {}
    # 0 = HOLD
    action_map[0] = (None, None)
    idx = 1
    for sl, tp in product(stoploss_levels, takeprofit_levels):
        action_map[idx] = (sl, tp)
        idx += 1
    return action_map

@dataclass
class SeqLongOnlySLTPEnvConfig:
    symbol: str = "BTC/USD"
    time_frames: Union[List[TimeFrame], TimeFrame] = TimeFrame(1, TimeFrameUnit.Minute)
    window_sizes: Union[List[int], int] = 10
    execute_on: TimeFrame = TimeFrame(1, TimeFrameUnit.Minute) # On which timeframe to execute trades
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

class SeqLongOnlySLTPEnv(EnvBase):
    def __init__(self, df: pd.DataFrame, config: SeqLongOnlySLTPEnvConfig, feature_preprocessing_fn: Optional[Callable] = None):
        self.action_levels = [0.0, 1.0]  #Do-Nothing, Buy-all
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
        self.position_hold_counter = 0

        # action levels
        self.stoploss_levels = config.stoploss_levels
        self.takeprofit_levels = config.takeprofit_levels

        # Define action and observation spaces sell, hold (do nothing), buy
        self.action_map = combinatory_action_map(self.stoploss_levels, self.takeprofit_levels)
        self.action_spec = Categorical(len(self.action_map))

        # Get the number of features from the observer
        market_data_keys = self.sampler.get_observation_keys()
        num_features = len(self.sampler.get_feature_keys())

        # Observation space includes market data features and current position info
        self.observation_spec = CompositeSpec(shape=())
           
        self.market_data_key = "market_data"
        self.account_state_key = "account_state"

        # Account state spec: [cash, portfolio_value, position_size, entry_price, unrealized_pnlpct, holding_time]
        account_state_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(7,), dtype=torch.float)
        self.market_data_keys = []
        for i, market_data_name in enumerate(market_data_keys):
            market_data_key = "market_data_" + market_data_name + "_" + str(config.window_sizes[i])
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
        obs_dict, self.current_timestamp, self.truncated = self.sampler.get_sequential_observation()

        current_price = self.sampler.get_base_features(self.current_timestamp)["close"]
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
        """GROK REWARD FUNCTION
        Hybrid reward:
        - Dense per-step: shaped to encourage smart entry and patient holding
        - Sparse terminal: your strong baseline (beat max of cash or buy-and-hold)
        """

        # --- Initialize running stats on first call ---
        if not hasattr(self, "returns_history"):
            self.returns_history = []      # for potential future use
            self.A = 0.0                   # DSR running mean
            self.B = 0.0001                # DSR running variance proxy (small init to avoid div0)
            self.t = 0
            self.hold_start_value = old_portfolio_value  # track entry value for unrealized

        # Daily portfolio return
        daily_return = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        self.returns_history.append(daily_return)

        # Transaction fee (normalized)
        tc = trade_info.get("fee_paid", 0.0) / old_portfolio_value if old_portfolio_value > 0 else 0.0
        # Or if you track it directly: tc = self.transaction_fee * abs(action) or similar

        # --- Dense per-step reward ---
        dense_reward = daily_return - tc  # core: profit minus cost of trading

        # Position tracking
        position_size = getattr(self, "position_size", 0.0)  # assume you have this

        if position_size > 0:
            # Unrealized PnL since entry (approximate)
            unrealized_pnl = (new_portfolio_value - self.hold_start_value) / self.hold_start_value
            if unrealized_pnl > 0:
                dense_reward += 0.0008 * unrealized_pnl  # bonus for letting winners run

            # Reset hold_start_value on new buy (if you detect entry)
            if trade_info.get("executed") and trade_info.get("side") == "buy":
                self.hold_start_value = new_portfolio_value

        # Differential Sharpe Ratio - incremental, online Sharpe update
        self.t += 1
        eta = 0.015  # adaptation rate (higher = reacts faster; good for intraday)
        old_A = self.A
        self.A += eta * (daily_return - old_A)
        self.B = (1 - eta) * self.B + eta * (daily_return - old_A) * (daily_return - self.A)

        if self.B > 1e-8:
            dsr = (self.A - old_A) / np.sqrt(self.B) if self.B > 0 else 0.0
            dense_reward += 3.0 * dsr  # scale so it matters but doesn't dominate

        # Small bonus for entering a position (overcomes fee fear slightly)
        if trade_info.get("executed") and trade_info.get("side") == "buy":
            dense_reward += 0.0005

        # Clip dense to prevent any single step from dominating
        dense_reward = np.clip(dense_reward, -0.05, 0.05)

        # --- Non-terminal: return dense guidance ---
        if self.step_counter < self.max_traj_length - 1:
            return dense_reward

        # --- Terminal: your original strong sparse reward ---
        buy_and_hold_value = (
            self.initial_portfolio_value / self.base_price_history[0]
        ) * self.base_price_history[-1]

        compare_value = max(self.initial_portfolio_value, buy_and_hold_value)
        if compare_value <= 0:
            terminal_reward = 0.0
        else:
            terminal_reward = 100 * (new_portfolio_value - compare_value) / compare_value

        # Optional: add small dense carryover, or keep pure sparse
        return terminal_reward  # or: terminal_reward + dense_reward



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

        self.base_price_history = []
        self.action_history = []
        self.reward_history = []
        self.portfolio_value_history = []

        max_episode_steps = self.sampler.reset(random_start=self.random_start)
        self.max_traj_length = max_episode_steps # overwrite as we might execute on different time frame so actual step might differ
        initial_portfolio_value = self.initial_cash if self.initial_cash is int else random.randint(self.initial_cash[0], self.initial_cash[1])
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

        return self._get_observation()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        self.step_counter += 1
        
        # Store old portfolio value for reward calculation
        old_portfolio_value = self._get_portfolio_value()
        
        # Get desired action and current position
        action = tensordict["action"]
        action_tuple = self.action_map[action.item()]
        
        # Calculate and execute trade if needed
        trade_info = self._execute_trade_if_needed(action_tuple)
        trade_action = 0
        if trade_info["executed"]:
            trade_action = 1 if trade_info["side"] == "buy" else -1
            self.current_position = 1 if trade_info["side"] == "buy" else 0
        self.action_history.append(trade_action)
        self.base_price_history.append(self.sampler.get_base_features(self.current_timestamp)["close"])
        self.portfolio_value_history.append(old_portfolio_value)           

        # Get updated state
        next_tensordict = self._get_observation()
        new_portfolio_value = self._get_portfolio_value()
        
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

    def _execute_trade_if_needed(self, action_tuple) -> Dict:
        """Execute trade if position change is needed."""
        trade_info = {"executed": False, "amount": 0, "side": None, "success": None, "price_noise": 0.0, "fee_paid": 0.0}
        
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
