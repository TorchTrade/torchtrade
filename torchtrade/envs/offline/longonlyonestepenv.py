from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable
from warnings import warn
from itertools import product

import numpy as np
from torchtrade.envs.offline.sampler import MarketDataObservationSampler
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs import EnvBase
import torch
from torchrl.data import Bounded, Categorical
import pandas as pd
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit, tf_to_timedelta, compute_periods_per_year_crypto
import logging
import sys

log_level = logging.INFO

# Read log level from command line (e.g. DEBUG)
if len(sys.argv) > 1:
    log_level_name = sys.argv[1].upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(levelname)s:%(name)s:%(message)s"
)

logger = logging.getLogger(__name__)


def combinatory_action_map(stoploss_levels: List[float], takeprofit_levels: List[float]) -> Dict:
    action_map = {}
    # 0 = HOLD
    action_map[0] = (None, None)
    idx = 1
    for sl, tp in product(stoploss_levels, takeprofit_levels):
        action_map[idx] = (sl, tp)
        idx += 1
    return action_map

class InitialBalanceSampler:
    def __init__(self, initial_cash: Union[List[int], int], seed: Optional[int] = None):
        self.initial_cash = initial_cash
        if seed is not None:
            np.random.seed(seed)

    def sample(self) -> float:
        if isinstance(self.initial_cash, int):
            return float(self.initial_cash)
        else:
            return float(np.random.randint(self.initial_cash[0], self.initial_cash[1]))

@dataclass
class LongOnlyOneStepEnvConfig:
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

class LongOnlyOneStepEnv(EnvBase):
    def __init__(self, df: pd.DataFrame, config: LongOnlyOneStepEnvConfig, feature_preprocessing_fn: Optional[Callable] = None):
        self.config = config
        self.transaction_fee = config.transaction_fee
        self.slippage = config.slippage
        if not (0 <= config.transaction_fee <= 1):
            raise ValueError("Transaction fee must be between 0 and 1 (e.g., 0.025 for 2.5%).")
        if not (0 <= config.slippage <= 1):
            raise ValueError("Slippage must be between 0 and 1 (e.g., 0.05 for 5%).")
        self.initial_cash_sampler = InitialBalanceSampler(config.initial_cash, config.seed)
        self.episode_idx = 0
        self.sampler = MarketDataObservationSampler(
            df,
            time_frames=config.time_frames,
            window_sizes=config.window_sizes,
            execute_on=config.execute_on,
            feature_processing_fn=feature_preprocessing_fn,
            features_start_with="features_",
            max_traj_length=config.max_traj_length,
            seed=self.config.seed
        )
        self.random_start = True
        self.max_traj_length = config.max_traj_length
        # Execute trades on the specified time frame
        self.execute_on_value = config.execute_on.value
        self.execute_on_unit = config.execute_on.unit.value
        self.periods_per_year = compute_periods_per_year_crypto(self.execute_on_unit, self.execute_on_value)

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


    def _get_observation(self, initial: bool = False) -> TensorDictBase:
        """Get the current observation state."""
        # Get market
        if initial or self.current_position == 0:
            obs_dict, self.current_timestamp, self.truncated = self.sampler.get_sequential_observation()
            self.rollout_returns = []
            # Cache base features for the new timestamp
            self._cached_base_features = self.sampler.get_base_features(self.current_timestamp)
        else:
            # _rollout() updates _cached_base_features internally
            trade_info, obs_dict = self._rollout()

        # Use cached base features (avoids redundant get_base_features calls)
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
        # --- Terminal: sparse reward ---
        #reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
#        reward = torch.log(torch.tensor(new_portfolio_value, dtype=torch.float) / torch.tensor(old_portfolio_value, dtype=torch.float))
        # compute sharp ratio as return
        if len(self.rollout_returns) == 0 or action == (None, None):
            return 0.0

        # Convert list to tensor
        returns = torch.stack(self.rollout_returns)
        
        # Need at least 2 points for a valid standard deviation
        if returns.numel() < 2:
            return float(returns.sum()) # Return raw sum if we can't compute volatility

        mean_return = returns.mean()
        std_return = returns.std()
        
        # 1. Epsilon prevents NaN if std is 0
        # 2. Multiply by sqrt(N) to annualize based on the execution timeframe
        sharpe = (mean_return / (std_return + 1e-9)) * np.sqrt(self.periods_per_year)
        
        # Clip to avoid extreme gradients in RL
        return torch.clamp(sharpe, -10.0, 10.0).item()

    def _get_portfolio_value(self, current_price: float = None) -> float:
        """Calculate total portfolio value."""
        if current_price is None:
            current_price = self.sampler.get_base_features(self.current_timestamp)["close"]
        return self.balance + self.position_size * current_price

    def _set_seed(self, seed: int):
        """Set the seed for the environment."""

        self.seed = seed

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
        self.previous_portfolio_value = 0.0

        logger.debug(f"Reset environment with initial portfolio value: {initial_portfolio_value}")

        obs = self._get_observation(initial=True)

        # Add coverage tracking indices (only during training with random_start)
        if self.random_start:
            obs.set("reset_index", torch.tensor(self.sampler._sequential_idx, dtype=torch.long))
            obs.set("state_index", torch.tensor(self.sampler._sequential_idx, dtype=torch.long))

        return obs

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        self.step_counter += 1

        # Cache base features once for current timestamp (avoids redundant get_base_features calls)
        cached_price = self._cached_base_features["close"]

        # Store old portfolio value for reward calculation
        old_portfolio_value = self._get_portfolio_value(cached_price)

        # Get desired action and current position
        action = tensordict["action"]
        action_tuple = self.action_map[action.item()]
        logger.debug(f"Action: {action_tuple}")

        # Calculate and execute trade if needed (pass cached price)
        trade_info = self._execute_trade_if_needed(action_tuple, cached_price)
        if trade_info["executed"]:
            trade_action = 1 if trade_info["side"] == "buy" else -1
            self.current_position = 1 if trade_info["side"] == "buy" else 0
            logger.debug(f"Trade executed: {trade_info}")

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

        if self.truncated:
            logger.debug(f"Episode truncated after {self.step_counter} steps")
            reward = 0 # set to 0 as we do not want to reward the agent for not selling

        self.reward_history.append(reward)

        next_tensordict.set("reward", reward)
        next_tensordict.set("done", True)
        next_tensordict.set("truncated", self.truncated)
        next_tensordict.set("terminated", True)
        return next_tensordict

    def trigger_sell(self, trade_info, execution_price):
        logging.debug(f"Triggering sell at price: {execution_price}")
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

    def compute_return(self, close_price):
        current_value = self.balance + self.position_size * close_price
        current_return = torch.log(torch.tensor(current_value, dtype=torch.float) / torch.tensor(self.previous_portfolio_value, dtype=torch.float))
        self.previous_portfolio_value = current_value
        return current_return

    def _rollout(self):
        trade_info = {"executed": False, "amount": 0, "side": None, "success": None, "price_noise": 0.0, "fee_paid": 0.0}
        self.rollout_returns = []
        obs_dict = None  # Initialize to prevent UnboundLocalError

        future_rollout_steps = 1
        while not self.truncated:
            # Get next time step
            obs_dict, self.current_timestamp, self.truncated = self.sampler.get_sequential_observation()
            # Cache base features once per iteration (avoids redundant calls)
            ohlcv_base_values = self.sampler.get_base_features(self.current_timestamp)
            self._cached_base_features = ohlcv_base_values
            # Use tolist() values for faster access
            open_price = ohlcv_base_values["open"]
            close_price = ohlcv_base_values["close"]
            high_price = ohlcv_base_values["high"]
            low_price = ohlcv_base_values["low"]

            self.rollout_returns.append(self.compute_return(close_price))

            if open_price < self.stop_loss:
                logger.debug(f"Sell (sl) triggered after - on open price: {future_rollout_steps} rollout steps")
                return self.trigger_sell(trade_info, self.stop_loss), obs_dict
            elif low_price < self.stop_loss:
                logger.debug(f"Sell (sl) triggered after - on low price: {future_rollout_steps} rollout steps")
                return self.trigger_sell(trade_info, self.stop_loss), obs_dict
            elif high_price > self.take_profit:
                logger.debug(f"Sell (tp) triggered after - on high price: {future_rollout_steps} rollout steps")
                return self.trigger_sell(trade_info, self.take_profit), obs_dict
            elif close_price < self.stop_loss:
                logger.debug(f"Sell (sl) triggered after - on close price: {future_rollout_steps} rollout steps")
                return self.trigger_sell(trade_info, self.stop_loss), obs_dict
            elif close_price > self.take_profit:
                logger.debug(f"Sell (tp) triggered after - on close price: {future_rollout_steps} rollout steps")
                return self.trigger_sell(trade_info, self.take_profit), obs_dict
            future_rollout_steps += 1

        logger.debug(f"No sell triggered after: {future_rollout_steps} rollout steps")
        # If loop never executed (truncated from start), get an observation
        if obs_dict is None:
            obs_dict, self.current_timestamp, self.truncated = self.sampler.get_sequential_observation()
            self._cached_base_features = self.sampler.get_base_features(self.current_timestamp)
        return trade_info, obs_dict

    def _execute_trade_if_needed(self, action_tuple, base_price: float = None) -> Dict:
        """Execute trade if position change is needed."""
        trade_info = {"executed": False, "amount": 0, "side": None, "success": None, "price_noise": 0.0, "fee_paid": 0.0}
        logger.debug(f"Position: {self.position_size}")
        # If holding position or no change in position, do nothing
        if action_tuple == (None, None):
            # No action
            logging.debug("No action")
            return trade_info
        else:
            # execute buy
            side = "buy"
            amount = self.balance

            # Get base price and apply noise to simulate slippage
            if base_price is None:
                base_price = self.sampler.get_base_features(self.current_timestamp)["close"]
            # Apply ±5% noise to the price to simulate market slippage
            price_noise_factor = 1.0 #self.np_rng.uniform(1 - self.slippage, 1 + self.slippage)
            execution_price = base_price * price_noise_factor
            logger.debug(f"Execution price: {execution_price}")
            
            fee_paid = amount * self.transaction_fee
            logger.debug(f"Fee paid: {fee_paid}")
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
            self.previous_portfolio_value = self.balance + self.position_size * execution_price

            logger.debug(f"Stop loss: {self.stop_loss}")
            logger.debug(f"Take profit: {self.take_profit}")
                
            trade_info.update({
                "executed": True,
                "amount": amount,
                "side": side,
                "success": True,
                "price_noise": price_noise_factor,
                "fee_paid": fee_paid
            })
            
            return trade_info


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

    df = pd.read_csv("/home/sebastian/Documents/TorchTrade/torchrl_alpaca_env/torchtrade/data/binance_spot_1m_cleaned/btcusdt_spot_1m_12_2024_to_09_2025.csv")

    #df = df[0:(1440 * 7)] # 1440 minutes = 1 day

    config = LongOnlyOneStepEnvConfig(
        symbol="BTC/USD",
        time_frames=time_frames,
        window_sizes=window_sizes,
        execute_on=execute_on,
        include_base_features=False,
        slippage=0.01,
        transaction_fee=0.0025,
        bankrupt_threshold=0.1,
    )
    env = LongOnlyOneStepEnv(df, config)

    for i in range(50):
        td = env.reset()
        print("Episode: ", i)
        for j in range(env.max_steps):
            action  =  env.action_spec.sample()
            td.set("action", action)
            td = env.step(td)
            td = td["next"]
            print("Step: ", j, "Reward:", td["reward"])
            assert not torch.isnan(td["reward"])
            if td["done"]:
                print("Done:", td["done"])
                break
    #env.render_history()
