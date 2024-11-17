import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import gymnasium as gym
import numpy as np
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from gymnasium import spaces
from obs_class import AlpacaObservationClass
from order_executor import AlpacaOrderClass, TradeMode


@dataclass
class TradingEnvConfig:
    symbol: str = "BTC/USD"
    initial_balance: float = 10000.0
    action_levels = [-1.0, 0.0, 1.0]  # Sell, Hold, Buy
    max_position: float = 1.0  # Maximum position size as a fraction of balance
    time_step_size: int = 3  # minutes
    history_window: int = 100  # Number of past observations to include
    reward_scaling: float = 1.0
    position_penalty: float = 0.0001  # Penalty for holding positions
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1  # 10% of initial balance
    paper: bool = True


class CryptoTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: TradingEnvConfig, api_key: str, api_secret: str):
        super().__init__()
        self.config = config

        # Initialize Alpaca clients
        # TODO adapt such that the observer can handle multiple timeframes in the crypto env
        self.observer = AlpacaObservationClass(
            symbol=config.symbol,
            timeframes=TimeFrame(config.time_step_size, TimeFrameUnit.Minute),
            window_sizes=config.history_window,
        )

        self.trader = AlpacaOrderClass(
            symbol=config.symbol,
            trade_mode=TradeMode.NOTIONAL,
            api_key=api_key,
            api_secret=api_secret,
            paper=config.paper,
        )

        self.action_levels = config.action_levels
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.action_levels))

        # Get the number of features from the observer
        num_features = self.observer.get_observations()[
            self.observer.get_keys()[0]
        ].shape[1]

        # Observation space includes market data features and current position info
        self.observation_space = spaces.Dict(
            {
                "market_data": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(config.history_window, num_features),
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

        if position_status is None:
            return self.balance
        return self.balance + position_status.market_value

    def _wait_for_next_timestamp(self) -> None:
        """Wait until the next time step."""
        # Get current time in NY timezone
        current_time = datetime.now(ZoneInfo("America/New_York"))

        # Calculate the next time step
        minutes_since_midnight = current_time.hour * 60 + current_time.minute
        next_step_minutes = (
            (minutes_since_midnight // self.config.time_step_size) + 1
        ) * self.config.time_step_size

        # Calculate the target time for the next observation
        target_time = current_time.replace(
            hour=next_step_minutes // 60,
            minute=next_step_minutes % 60,
            second=0,
            microsecond=0,
        )

        # If target time is earlier than current time, move to next day
        if target_time <= current_time:
            target_time = target_time + timedelta(days=1)

        # Wait until target time
        while datetime.now(ZoneInfo("America/New_York")) < target_time:
            time.sleep(self.config.sleep_timeout)

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
        # Store initial portfolio value for reward calculation
        old_portfolio_value = self._get_portfolio_value()

        # Convert action to trade size
        desired_position_size = self.action_levels[action]  # action_index

        # Get current position
        status = self.trader.get_status()
        position_status = status.get("position_status", None)
        current_position_size = float(position_status.qty) if position_status else 0.0

        # Calculate trade size
        trade_size = desired_position_size - current_position_size

        # Execute trade if necessary
        if abs(trade_size) > 0:
            side = "buy" if trade_size > 0 else "sell"
            trade_amount = abs(trade_size) * self.balance

            # Execute trade
            success = self.trader.trade(
                side=side, amount=trade_amount, order_type="market"
            )

            if not success:
                print(f"Trade failed: {side} {trade_amount}")

        # Update portfolio value
        new_portfolio_value = self._get_portfolio_value()

        # Calculate reward
        reward = self._calculate_reward(
            old_portfolio_value, new_portfolio_value, desired_position_size
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

        # Get new observation
        observation = self._get_observation()

        # Store values for next step
        self.last_portfolio_value = new_portfolio_value

        # Additional info for debugging
        info = {
            "portfolio_value": new_portfolio_value,
            "portfolio_return": (new_portfolio_value - self.initial_portfolio_value)
            / self.initial_portfolio_value,
            "position_size": desired_position_size,
        }

        return observation, reward, done, info

    def render(self, mode="human"):
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
    config = TradingEnvConfig(symbol="BTC/USD", initial_balance=10000.0, paper=True)

    # Create environment
    env = CryptoTradingEnv(
        config, api_key=os.getenv("API_KEY"), api_secret=os.getenv("API_SECRET")
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

        print(f"Action: {action[0]:.2f}, Reward: {reward:.4f}")
        print(f"Info: {info}\n")

    print(f"Episode finished! Total reward: {total_reward:.4f}")
    env.close()
