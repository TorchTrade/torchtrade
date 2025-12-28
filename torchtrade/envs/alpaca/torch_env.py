import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from warnings import warn
from zoneinfo import ZoneInfo

import numpy as np
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from torchtrade.envs.alpaca.obs_class import AlpacaObservationClass
from torchtrade.envs.alpaca.order_executor import AlpacaOrderClass, TradeMode
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs import EnvBase
import torch
from torchrl.data import Categorical, Bounded

@dataclass
class AlpacaTradingEnvConfig:
    symbol: str = "BTC/USD"
    action_levels = [-1.0, 0.0, 1.0]  # Sell-all, Do-Nothing, Buy-all
    max_position: float = 1.0  # Maximum position size as a fraction of balance
    time_frames: Union[List[TimeFrame], TimeFrame] = TimeFrame(1, TimeFrameUnit.Minute)
    window_sizes: Union[List[int], int] = 10
    execute_on: TimeFrame = TimeFrame(1, TimeFrameUnit.Minute) # On which timeframe to execute trades
    reward_scaling: float = 1.0
    position_penalty: float = 0.0001  # Penalty for holding positions
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1  # 10% of initial balance
    paper: bool = True
    trade_mode: TradeMode = TradeMode.NOTIONAL
    seed: Optional[int] = 42
    include_base_features: bool = False # Includes base features such as timestamps and ohlc to the tensordict

class AlpacaTorchTradingEnv(EnvBase):
    def __init__(
        self,
        config: AlpacaTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        observer: Optional[AlpacaObservationClass] = None,
        trader: Optional[AlpacaOrderClass] = None,
    ):
        """
        Initialize the AlpacaTorchTradingEnv.

        Args:
            config: Environment configuration
            api_key: Alpaca API key (not required if observer and trader are provided)
            api_secret: Alpaca API secret (not required if observer and trader are provided)
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured AlpacaObservationClass for dependency injection
            trader: Optional pre-configured AlpacaOrderClass for dependency injection
        """
        self.config = config

        # Initialize Alpaca clients - use injected instances or create new ones
        self.observer = observer if observer is not None else AlpacaObservationClass(
            symbol=config.symbol,
            timeframes=config.time_frames,
            window_sizes=config.window_sizes,
            feature_preprocessing_fn=feature_preprocessing_fn,
        )

        self.trader = trader if trader is not None else AlpacaOrderClass(
            symbol=config.symbol.replace('/', ''),
            trade_mode=config.trade_mode,
            api_key=api_key,
            api_secret=api_secret,
            paper=config.paper,
        )

        # Execute trades on the specified time frame
        self.execute_on_value = config.execute_on.amount
        self.execute_on_unit = str(config.execute_on.unit)

        # reset settings 
        self.trader.close_all_positions()
        self.trader.cancel_open_orders()
        
        account = self.trader.client.get_account()
        cash = float(account.cash)
        self.initial_portfolio_value = cash
        self.position_hold_counter = 0

        self.action_levels = config.action_levels
        # Define action and observation spaces
        self.action_spec = Categorical(len(self.action_levels))
        # action levels [-1, 0, 1], categorical actions [0, 1, 2] -> 0: sell, 1: hold, 2: buy
        # self.hold_action = self.action_levels[1]
        # assert self.hold_action == 0.0, "Hold action should be 0.0, possibly action levels are not [-1, 0, 1]!"

        # Get the number of features from the observer
        num_features = self.observer.get_observations()[
            self.observer.get_keys()[0]
        ].shape[1]

        # get market data obs names
        market_data_names = self.observer.get_keys()

        # Observation space includes market data features and current position info
        # TODO: needs to be adapted for multiple timeframes
        self.observation_spec = CompositeSpec(shape=())
           
        self.market_data_key = "market_data"
        self.account_state_key = "account_state"

        # Account state spec: [cash, portfolio_value, position_size, entry_price, unrealized_pnlpct, holding_time]
        account_state_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(6,), dtype=torch.float)
        self.market_data_keys = []
        for i, market_data_name in enumerate(market_data_names):
            market_data_key = "market_data_" + market_data_name
            market_data_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(config.window_sizes[i], num_features), dtype=torch.float)
            self.observation_spec.set(market_data_key, market_data_spec)
            self.market_data_keys.append(market_data_key)
        self.observation_spec.set(self.account_state_key, account_state_spec)

        self.reward_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(1,), dtype=torch.float)

        self._reset(TensorDict({}))
        super().__init__()


    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        # Get market data
        obs_dict = self.observer.get_observations(return_base_ohlc=True if self.config.include_base_features else False)
        market_data = obs_dict[self.observer.get_keys()[0]]

        if self.config.include_base_features:
            base_features = obs_dict["base_features"][-1]
            #base_timestamps = obs_dict["base_timestamps"][-1]
            # Convert to Unix timestamps (seconds)
            #timestamps = base_timestamps.astype('datetime64[s]').astype(np.int64)
            #base_timestamps = torch.from_numpy(timestamps)
        
        market_data = [obs_dict[features_name] for features_name in self.observer.get_keys()]

        # Get account state
        status = self.trader.get_status()
        account = self.trader.client.get_account()
        cash = float(account.cash) # NOTE: should we use buying power?
        position_status = status.get("position_status", None)

        if position_status is None:
            self.position_hold_counter = 0
            position_size = 0.0
            position_value = 0.0
            entry_price = 0.0
            unrealized_pnlpc = 0.0
            holding_time = self.position_hold_counter

        else:
            self.position_hold_counter += 1
            position_size = position_status.qty
            position_value = position_status.market_value
            entry_price = position_status.avg_entry_price
            unrealized_pnlpc = position_status.unrealized_plpc
            holding_time = self.position_hold_counter

        account_state = torch.tensor(
            [cash, position_size, position_value, entry_price, unrealized_pnlpc, holding_time], dtype=torch.float
        )

        out_td = TensorDict({self.account_state_key: account_state}, batch_size=())
        for market_data_name, data in zip(self.market_data_keys, market_data):
            out_td.set(market_data_name, torch.from_numpy(data))

        if self.config.include_base_features:
            out_td.set("base_features", torch.from_numpy(base_features))
            #out_td.set("base_timestamps", base_timestamps)

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

        # Scale the reward
        reward = portfolio_return * self.config.reward_scaling

        return reward

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        status = self.trader.get_status()
        position_status = status.get("position_status", None)

        account = self.trader.client.get_account()
        self.balance = float(account.cash)

        if position_status is None:
            return self.balance
        return self.balance + position_status.market_value

    def _wait_for_next_timestamp(self) -> None:
        """Wait until the next time step based on the configured execute_on_value and execute_on_unit."""

        # Mapping units to timedelta arguments
        unit_to_timedelta = {
            "TimeFrameUnit.Minute": "minutes",
            "TimeFrameUnit.Hour": "hours",
            "TimeFrameUnit.Day": "days",
        }

        if self.execute_on_unit not in unit_to_timedelta:
            raise ValueError(f"Unsupported time unit: {self.execute_on_unit}")

        # Calculate the wait duration in timedelta
        wait_duration = timedelta(
            **{unit_to_timedelta[self.execute_on_unit]: self.execute_on_value}
        )

        # Get current time in NY timezone
        current_time = datetime.now(ZoneInfo("America/New_York"))

        # Calculate the next time step
        next_step = (current_time + wait_duration).replace(second=0, microsecond=0)

        # Wait until the target time
        while datetime.now(ZoneInfo("America/New_York")) < next_step:
            time.sleep(1)

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
        # Cancel all orders and close all positions
        self.trader.cancel_open_orders()
        #self.trader.close_all_positions() # NOTE: Not sure if we want this as we could loose money
        account = self.trader.client.get_account()
        self.balance = float(account.cash)
        self.last_portfolio_value = self.balance
        status = self.trader.get_status()
        position_status = status.get("position_status")
        self.position_hold_counter = 0
        if position_status is None:
            self.current_position = 0.0
        else:
            self.current_position = 1 if position_status.qty > 0 else 0

        # Get initial observation
        return self._get_observation()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        
        # Store old portfolio value for reward calculation
        old_portfolio_value = self._get_portfolio_value()
        
        # Get desired action and current position
        desired_action = self.action_levels[tensordict.get("action", 0)]
        
        # Calculate and execute trade if needed
        trade_info = self._execute_trade_if_needed(desired_action)

        if trade_info["executed"]:
            self.current_position = 1 if trade_info["side"] == "buy" else 0

        
        # Wait for next time step
        self._wait_for_next_timestamp()
        
        # Get updated state
        new_portfolio_value = self._get_portfolio_value()
        next_tensordict = self._get_observation()
        
        # Calculate reward and check termination
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value, desired_action, trade_info)
        done = self._check_termination(new_portfolio_value)

        next_tensordict.set("reward", reward)
        next_tensordict.set("done", done)
        next_tensordict.set("truncated", False)
        next_tensordict.set("terminated", False)
        
        # TODO: Make a dashboard that shows the portfolio value and action history etc
        _ = self._create_info_dict(new_portfolio_value, trade_info, desired_action)
        
        return next_tensordict


    def _execute_trade_if_needed(self, desired_position: float) -> Dict:
        """Execute trade if position change is needed."""
        trade_info = {"executed": False, "amount": 0, "side": None, "success": None}
        
        
        # If holding position or no change in position, do nothing
        if desired_position == 0 or desired_position == self.current_position:
            return trade_info
        
        # Determine trade details
        side = "buy" if desired_position > 0 else "sell"
        amount = self._calculate_trade_amount(side)
        
        try:
            success = self.trader.trade(side=side, amount=amount, order_type="market")
            trade_info.update({
                "executed": True,
                "amount": amount,
                "side": side,
                "success": success
            })
        except Exception as e:
            print(f"Trade failed: {side} ${amount:.2f} - {str(e)}")
            trade_info["success"] = False
        
        return trade_info

    def _calculate_trade_amount(self, side: str) -> float:
        """Calculate the dollar amount to trade."""
        if self.config.trade_mode == TradeMode.QUANTITY:
            raise NotImplementedError
        
        # NOTIONAL mode
        if side == "buy":
            return self.balance # buy with all cash we have
        else:  # sell
            # TODO: if we do fine grained trading this needs to be calculated now we sell all available
            return -1
            # status = self.trader.get_status()
            # if status.get("position_status"):
            #     position_value = (status["position_status"].qty * 
            #                     status["position_status"].current_price)
            #     return position_value # sell all we have

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        if not self.config.done_on_bankruptcy:
            return False
        
        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold

    def _create_info_dict(self, portfolio_value: float, trade_info: Dict, action_value: float) -> Dict:
        """Create info dictionary for debugging."""
        portfolio_return = ((portfolio_value - self.initial_portfolio_value) / 
                        self.initial_portfolio_value)

        account = self.trader.client.get_account()
        cash = float(account.cash)
        position_status = self.trader.get_status().get("position_status", None)
        
        return {
            "portfolio_value": portfolio_value,
            "portfolio_return": portfolio_return,
            "cash": cash,
            "position_qty": position_status.qty if position_status else 0,
            "position_market_value": position_status.market_value if position_status else 0,
            "trade_executed": trade_info["executed"],
            "trade_amount": trade_info["amount"],
            "trade_success": trade_info["success"],
            "trade_side": trade_info["side"],
            "action": action_value,
            "trade_mode": self.trader.trade_mode,
        }


    def close(self):
        """Clean up resources."""
        # Cancel all orders and close all positions
        self.trader.cancel_open_orders()
        self.trader.close_all_positions()



if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    # Load environment variables
    load_dotenv()
    # Create environment configuration
    config = AlpacaTradingEnvConfig(
        symbol="BTC/USD",
        paper=True,
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
            TimeFrame(1, TimeFrameUnit.Hour),
        ],
        window_sizes=[15, 10],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        include_base_features=True,
    )

    # Create environment
    env = AlpacaTorchTradingEnv(
        config, api_key=os.getenv("API_KEY"), api_secret=os.getenv("SECRET_KEY")
    )
    td = env.reset()
    print(td)