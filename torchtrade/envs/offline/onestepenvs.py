from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs import EnvBase
import torch
from torchrl.data import Categorical, Bounded
from utils import TimeFrame, TimeFrameUnit
from pandas import Timedelta
from sampler import MarketDataObservationSampler

from torchrl.data import (
    Composite,
    LazyMemmapStorage,
    TensorDictReplayBuffer,
    SamplerWithoutReplacement,
)
import torch
import random


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
    seed: Optional[int] = 42
    in_position_lookback: int = 10
    future_close_window: int = 10
    cash_min_max: Tuple[float, float] = (1000, 10000)
    max_quantity: float = 1.0

class OneStepTradingEnvOn(EnvBase):
    """
    Online one-step environment that takes in real-time market data and allows the agent to trade for one step. 
    """
    def __init__(self, data: pd.DataFrame, config: AlpacaTradingEnvConfig, api_key: str, api_secret: str, feature_preprocessing_fn: Optional[Callable] = None):
        self.config = config

        self.market_data_sampler = MarketDataObservationSampler(
            df=data,
            time_frames=self.config.time_frames,
            window_sizes=self.config.window_sizes,
            execute_on=self.config.execute_on,
            feature_processing_fn=feature_preprocessing_fn,
            features_start_with="features_"
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

        self.cash_min = config.cash_min_max[0]
        self.cash_max = config.cash_min_max[1]
        self.max_quantity = config.max_quantity
        self.in_position_lookback = 10
        self.future_close_window = 10

        self.action_levels = config.action_levels
        # Define action and observation spaces
        self.action_spec = Categorical(len(self.action_levels))
        # self.hold_action = self.action_levels[1]
        # assert self.hold_action == 0.0, "Hold action should be 0.0, possibly action levels are not [-1, 0, 1]!"

        # Get the number of features from the observer
        num_features = self.market_data_sampler.get_random_observation()[0].shape[1]

        # get market data obs names
        market_data_names = self.market_data_sampler.get_keys()

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

        self._reset(TensorDict({}))
        super().__init__()


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
        # Get initial observation
        self.market_data_sampler.reset()

        market_data, self.timestamp = self.market_data_sampler.get_random_observation(without_replacement=True)

        base_features = self.market_data_sampler.get_base_features(self.timestamp)

        # sample if in position or not
        pct = random.random()
        if pct < 0.5:
            self.in_position = True
            # sample entry price
            time_steps = np.random.randint(0, self.in_position_lookback)
            timedeltas = time_steps *Timedelta(minutes=self.execute_on_value)
            entry_price = self.market_data_sampler.execute_base_features.loc[self.timestamp - timedeltas].close
            # calc hold time
            hold_time = time_steps
            # sample cash
            cash = np.random.uniform(self.cash_min, self.cash_max)
            # calc unrealized pnl pct
            # NOTE: We probably dont want to give a hint for the real price of the asset sample between
            # high and low of that time step
            current_price = np.random.uniform(self.market_data_sampler.execute_base_features.loc[self.timestamp].low,
                                              self.market_data_sampler.execute_base_features.loc[self.timestamp].high)
            unrealized_pnlpc = (current_price - entry_price) / entry_price
            # sample quantity
            quantity = np.random.uniform(0, self.max_quantity)
            position_size = quantity
            position_value = quantity * current_price
            account_state = torch.tensor(
                [cash, position_size, position_value, entry_price, unrealized_pnlpc, hold_time], dtype=torch.float
            )
            self.portfolio_value = cash + position_value
        else: # not in position
            self.in_position = False
            # sample cash
            cash = np.random.uniform(self.cash_min, self.cash_max)
            account_state = torch.tensor(
                [cash, 0.0, 0.0, 0.0, 0.0, 0], dtype=torch.float
            )
            self.portfolio_value = cash

        out_td = TensorDict({self.account_state_key: account_state}, batch_size=())
        for market_data_name, data in zip(self.market_data_keys, market_data):
            out_td.set(market_data_name, torch.from_numpy(data))

        return out_td

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        
        # Get desired action and current position
        desired_action = self.action_levels[tensordict.get("action", 0)]
        account_state = tensordict.get(self.account_state_key)
        
        # Calculate and execute trade if needed
        trade_info = self._execute_trade_if_needed(desired_action, account_state)

        if trade_info["executed"]:
            self.current_position = 1 if trade_info["side"] == "buy" else 0

        
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


    def _execute_trade_if_needed(self, desired_position: float, account_state: torch.Tensor) -> torch.Tensor:
        """Execute trade if position change is needed."""

        cash, position_size, position_value, entry_price, unrealized_pnlpc, hold_time = account_state
        
        # If holding position or no change in position, do nothing
        if self.in_position:
            if desired_position == 0:
                # hold action -> do nothing
                # TODO: we need to increase hold time
                return account_state
            elif desired_position == 1:
                # TODO: we need to increase hold time
                # if we are in position we cant buy more -> do nothing
                return account_state
            else:
                # sell all
                current_close = self.market_data_sampler.execute_base_features.loc[self.timestamp].close
                amount = position_size
                new_cash = cash + current_close * amount
                new_account_state = torch.tensor(
                    [new_cash, 0.0, 0.0, 0.0, 0.0, 0], dtype=torch.float
                )
                return new_account_state

        # If not in position
        else:
            if desired_position == 0:
                # hold action -> do nothing
                return account_state
            elif desired_position == -1:
                # if we are not in position we cant sell -> do nothing
                return account_state
            else:
                # buy
                current_close = self.market_data_sampler.execute_base_features.loc[self.timestamp].close
                amount = self._calculate_trade_amount("buy")
                new_cash = cash - current_close * amount
                new_account_state = torch.tensor(
                    [new_cash, amount, amount * current_close, current_close, 0.0, 0], dtype=torch.float
                )
                return new_account_state



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

@dataclass
class OneStepTradingEnvOffConfig:
    action_levels = [-1.0, 0.0, 1.0]  # Sell-all, Do-Nothing, Buy-all
    seed: Optional[int] = 42
    storage_size: int = 10000000
    batch_size: int = 1


class OneStepTradingEnvOff(EnvBase):
    """
    Offline one-step environment that takes in a tensordict of real executed trades and allows the agent to trade for one step.
    Offline because we dont sample new positions or or compute the reward in a different way. We just allow the agent to reexperience its own data and maybe
    improve its performance.

    We could adapt the reward function to allow the agent to further improve its performance, but this is not implemented yet.
    """
    def __init__(self, tensordict: TensorDictBase, config: OneStepTradingEnvOffConfig):
        self.config = config
        self.tensordict = tensordict

        self.replay_buffer = TensorDictReplayBuffer(pin_memory=False,
                                                    prefetch=4,
                                                    #split_trajs=False,
                                                    storage=LazyMemmapStorage(config.storage_size),
                                                    batch_size=config.batch_size,
                                                    #shared=shared,
                                                    sampler=SamplerWithoutReplacement(drop_last=True), # to have fixed epochs
                                                    )

        self.replay_buffer.extend(self.tensordict)

        all_keys = self.replay_buffer.storage._storage.keys()
        self.market_keys = [key for key in all_keys if key.startswith("market_data_")]
        if "account_state" in all_keys:
            self.account_state_key = "account_state"
        else:
            raise ValueError("Account state key not found in replay buffer")


        self.action_spec = Categorical(3)
        self.observation_spec = CompositeSpec(shape=())
        for key in self.market_keys:
            self.observation_spec.set(key, Bounded(low=-torch.inf, high=torch.inf, shape=(self.replay_buffer.storage._storage[key].shape[1:]), dtype=torch.float))
        self.observation_spec.set(self.account_state_key,
                                Bounded(low=-torch.inf, high=torch.inf,
                                shape=(self.replay_buffer.storage._storage[self.account_state_key].shape[1],),
                                dtype=torch.float))

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

        super().__init__()


    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        # here we sample from the buffer and return the current state, keep the next state as information
        # keep also the old action!
        tensordict = self.replay_buffer.sample()
        self.next_state  = tensordict.pop("next")
        self.old_action = tensordict.pop("action")
        if "episode_reward" in tensordict:
            tensordict.pop("episode_reward")
        tensordict.pop("collector")
        self.old_done = tensordict.pop("done")
        self.old_terminated = tensordict.pop("terminated")
        self.old_truncated = tensordict.pop("truncated")
        return tensordict.squeeze(0) # get rid of batch dimension


    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # we receive the tensordict with a new action compute the reward and add the next state information
        old_account_state = tensordict.get(self.account_state_key)

        action = tensordict.get("action")

        if action == 1: # hold
            new_account_state = old_account_state
            new_account_state["holding_time"] += 1
            # TODO: we should update portfolio value and unrealized pnlpc but therefore we would need base features
            out_tensordict = self.next_state.clone()
            out_tensordict.pop("reward")
            out_tensordict.set("reward", 0.0)
            out_tensordict.set("done", 1)
            out_tensordict.set("terminated", 1)
            out_tensordict.set("truncated", 0)
            return out_tensordict

   


        return tensordict

    def _calculate_reward(self, old_portfolio_value: float, new_portfolio_value: float, action_value: float) -> float:
        # calculate the reward based on the portfolio value change
        return new_portfolio_value - old_portfolio_value





if __name__ == "__main__":

    data = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=4,
            #split_trajs=False,
            storage=LazyMemmapStorage(1000000),
            batch_size=1,
            #shared=shared,
            #sampler=SamplerWithoutReplacement(drop_last=True),
        )
    data.loads("/home/sebastian/Documents/TorchTrade/torchrl_alpaca_env/real_live_data/replay_buffer_random_iter3_colony2.pt")
    td = data.storage._storage.clone()
    td = td[:td["index"].max().item()]

    env = OneStepTradingEnvOff(td, OneStepTradingEnvOffConfig())
    td = env.reset()
    print(td)
    action =  env.action_spec.sample()
    next_td = env.step(td.set("action", action))
    print(next_td)