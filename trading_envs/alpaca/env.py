import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn
from zoneinfo import ZoneInfo

import gymnasium as gym
import numpy as np
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from gymnasium import spaces
from obs_class import AlpacaObservationClass
from order_executor import AlpacaOrderClass, TradeMode
from decimal import Decimal, ROUND_DOWN

@dataclass
class AlpacaTradingEnvConfig:
    symbol: str = "BTC/USD"
    initial_balance: float = 10000.0
    max_trade_amount: Optional[float] = None
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

class AlpacaTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: AlpacaTradingEnvConfig, api_key: str, api_secret: str):
        super().__init__()
        self.config = config

        # Initialize Alpaca clients
        # TODO adapt such that the observer can handle multiple timeframes in the crypto env
        self.observer = AlpacaObservationClass(
            symbol=config.symbol,
            timeframes=config.time_frames,
            window_sizes=config.window_sizes,
        )

        self.trader = AlpacaOrderClass(
            symbol=config.symbol.replace('/', ''),
            trade_mode=config.trade_mode,
            api_key=api_key,
            api_secret=api_secret,
            paper=config.paper,
            transaction_fee=0.03,
        )

        # Execute trades on the specified time frame
        self.execute_on_value = config.execute_on.amount
        self.execute_on_unit = str(config.execute_on.unit)

        # TODO: check if initial_balance is in the account
        account = self.trader.client.get_account()
        buying_power = float(account.buying_power)
        cash = float(account.cash)
        if self.config.max_trade_amount is None:
            self.max_trade_amount = self.config.max_trade_amount
        else:
            self.max_trade_amount = None

        # assert config.initial_balance <= buying_power, "Initial balance exceeds buying power"
        if config.initial_balance > buying_power:
            warn(
                f"Initial balance is lower than buying_power. Setting initial balance to {cash}"
            )
            self.config.initial_balance = min(config.initial_balance, cash)

        self.action_levels = config.action_levels
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.action_levels))

        # Get the number of features from the observer
        num_features = self.observer.get_observations()[
            self.observer.get_keys()[0]
        ].shape[1]

        # Observation space includes market data features and current position info
        # TODO: needs to be adapted for multiple timeframes
        self.observation_space = spaces.Dict(
            {
                "market_data": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(config.window_sizes[0], num_features),
                    dtype=np.float32,
                ),
                "account_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),  # [balance, position_size, position_value]
                    dtype=np.float32,
                ),
            }
        )

        self.reset()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation state."""
        # Get market data
        obs_dict = self.observer.get_observations(return_base_ohlc=True)
        market_data = obs_dict[self.observer.get_keys()[0]]

        # Get account state
        status = self.trader.get_status()
        position_status = status.get("position_status", None)

        if position_status is None:
            position_size = 0.0
            position_value = 0.0

        else:
            position_size = position_status.qty
            position_value = position_status.market_value

        account_state = np.array(
            [self.balance, position_size, position_value], dtype=np.float32
        )

        return {"market_data": market_data, "account_state": account_state}

    def _calculate_reward(
        self,
        old_portfolio_value: float,
        new_portfolio_value: float,
        position_size: float,
    ) -> float:
        """Calculate the step reward."""
        # Calculate portfolio return
        portfolio_return = (
            new_portfolio_value - old_portfolio_value
        ) / old_portfolio_value

        # Apply position penalty
        position_penalty = abs(position_size) * self.config.position_penalty

        # Scale the reward
        reward = (portfolio_return - position_penalty) * self.config.reward_scaling

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

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment."""
        # Cancel all orders and close all positions
        self.trader.cancel_open_orders()
        self.trader.close_all_positions()

        # Reset balance
        self.balance = self.config.initial_balance
        self.initial_portfolio_value = self.config.initial_balance
        self.last_portfolio_value = self.config.initial_balance

        # Get initial observation
        return self._get_observation()

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Execute one environment step."""

        # Store old portfolio value for reward calc
        old_portfolio_value = self._get_portfolio_value()

        # Map action -> desired fraction of portfolio in [-1, 1]
        desired_action = self.action_levels[action]

        # Get current position
        status = self.trader.get_status()
        position_status = status.get("position_status", None)
        current_position_size = float(position_status.qty) if position_status else 0.0

        # Calculate trade size
        trade_size = desired_action - current_position_size

        # Execute trade if necessary
        trade_amount = None
        success = None
        if trade_size > 0 and current_position_size >= 0 or trade_size < 0 and current_position_size > 0:
            side = "buy" if trade_size > 0 else "sell"
            if self.config.trade_mode == TradeMode.NOTIONAL:
                if side == "buy":
                    amount = abs(trade_size) * self.balance
                else:
                    total_position_value = (status["position_status"].qty * status["position_status"].current_price)
                    amount = abs(trade_size) * total_position_value

                if self.config.max_trade_amount is not None and side == "buy":
                    trade_amount = min(trade_amount, self.config.max_trade_amount)
            
            elif self.config.trade_mode == TradeMode.QUANTITY:
                # TODO: implement quantity trading
                warn("Quantity trading not implemented yet")
                trade_amount = abs(trade_size)

            # Execute trade
            success = None
            try:
                if amount > 10:
                    success = self.trader.trade(side=side, amount=amount, order_type="market")
                else:
                    pass
            except Exception as e:
                print(f"Trade failed: {side} {amount}")
                print(e)

        # Wait for next time step
        self._wait_for_next_timestamp()

        # Update portfolio value
        new_portfolio_value = self._get_portfolio_value()

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(
            old_portfolio_value, new_portfolio_value, desired_action
        )

        # Check if episode should end
        done = False

        if self.config.done_on_bankruptcy:
            # Check if portfolio value has dropped below bankruptcy threshold
            if (
                new_portfolio_value
                < self.config.bankrupt_threshold * self.initial_portfolio_value
            ):
                done = True

        # Store values for next step
        self.last_portfolio_value = new_portfolio_value

        # Additional info for debugging
        info = {
            "portfolio_value": new_portfolio_value,
            "portfolio_return": (new_portfolio_value - self.initial_portfolio_value)
            / self.initial_portfolio_value,
            "trade_size": trade_amount,
            "trade_success": success,
            "trade_mode": self.trader.trade_mode,
            "action": self.action_levels[action],
        }

        return observation, reward, done, info

    def render(self, mode="human"):
        # TODO implement dashboard rendering 
        """Render the environment."""
        if mode == "human":
            portfolio_value = self._get_portfolio_value()
            returns = (
                portfolio_value - self.initial_portfolio_value
            ) / self.initial_portfolio_value
            print(f"Portfolio Value: ${portfolio_value:.2f} (Return: {returns:.2%})")

    def close(self):
        """Clean up resources."""
        # Cancel all orders and close all positions
        self.trader.cancel_open_orders()
        self.trader.close_all_positions()


# Example usage:
if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Create environment configuration
    config = AlpacaTradingEnvConfig(
        symbol="BTC/USD",
        initial_balance=500.0,
        paper=True,
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
        ],
        window_sizes=[15], # 30],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
    )

    # Create environment
    env = AlpacaTradingEnv(
        config, api_key=os.getenv("API_KEY"), api_secret=os.getenv("SECRET_KEY")
    )

    # Run a simple test episode
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Random action for testing
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

        print(f"Action: {env.action_levels[action]}, Reward: {reward:.4f}")
        print(f"Info: {info}\n")

    print(f"Episode finished! Total reward: {total_reward:.4f}")
    env.close()
