import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from warnings import warn
from zoneinfo import ZoneInfo

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

        if trade_info["executed"]:
            self.current_position = 1 if trade_info["side"] == "buy" else 0

        # Get updated state
        next_tensordict = self._get_observation()
        new_portfolio_value = self._get_portfolio_value()
        
        # Calculate reward and check termination
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value, desired_action, trade_info)
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
        if desired_position == 0 or desired_position == self.current_position:
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

    # def _create_info_dict(self, portfolio_value: float, trade_info: Dict, action_value: float) -> Dict:
    #     """Create info dictionary for debugging."""
    #     portfolio_return = ((portfolio_value - self.initial_portfolio_value) / 
    #                     self.initial_portfolio_value)

    #     return {
    #         "portfolio_value": portfolio_value,
    #         "portfolio_return": portfolio_return,
    #         "cash": cash,
    #         "position_qty": position_status.qty if position_status else 0,
    #         "position_market_value": position_status.market_value if position_status else 0,
    #         "trade_executed": trade_info["executed"],
    #         "trade_amount": trade_info["amount"],
    #         "trade_success": trade_info["success"],
    #         "trade_side": trade_info["side"],
    #         "action": action_value,
    #         "trade_mode": self.trader.trade_mode,
    #     }


    def close(self):
        """Clean up resources."""





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
        
