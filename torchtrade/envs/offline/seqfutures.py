"""Sequential Futures Environment for offline training with leverage and shorting support."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from warnings import warn
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from datetime import datetime
from enum import Enum

import numpy as np
from torchtrade.envs.offline.sampler import MarketDataObservationSampler
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs import EnvBase
import torch
from torchrl.data import Categorical, Bounded
import pandas as pd
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit, tf_to_timedelta
import random


class MarginType(Enum):
    """Margin type for futures trading."""
    ISOLATED = "isolated"
    CROSSED = "crossed"


@dataclass
class SeqFuturesEnvConfig:
    """Configuration for Sequential Futures Environment.

    This environment supports:
    - Long and short positions
    - Configurable leverage (1x - 125x)
    - Liquidation mechanics
    - Isolated/crossed margin
    """
    symbol: str = "BTC/USD"
    time_frames: Union[List[TimeFrame], TimeFrame] = field(
        default_factory=lambda: TimeFrame(1, TimeFrameUnit.Minute)
    )
    window_sizes: Union[List[int], int] = 10
    execute_on: TimeFrame = field(
        default_factory=lambda: TimeFrame(1, TimeFrameUnit.Minute)
    )

    # Initial capital settings
    initial_cash: Union[Tuple[int, int], int] = (1000, 5000)

    # Leverage and margin settings
    leverage: int = 1  # 1x to 125x
    margin_type: MarginType = MarginType.ISOLATED
    maintenance_margin_rate: float = 0.004  # 0.4% maintenance margin

    # Trading costs
    transaction_fee: float = 0.0004  # 0.04% typical futures fee
    slippage: float = 0.001  # 0.1% slippage
    funding_rate: float = 0.0001  # 0.01% per 8 hours (typical funding rate)
    funding_interval_hours: int = 8

    # Risk management
    bankrupt_threshold: float = 0.1  # 10% of initial balance
    max_position_size: float = 1.0  # Max position as fraction of balance

    # Environment settings
    seed: Optional[int] = 42
    include_base_features: bool = False
    max_traj_length: Optional[int] = None
    random_start: bool = True

    # Reward settings
    reward_scaling: float = 1.0


class SeqFuturesEnv(EnvBase):
    """
    Sequential Futures Environment for offline RL training.

    Supports long and short positions with leverage, similar to the
    BinanceFuturesTorchTradingEnv but for offline/backtesting use.

    Action Space:
    - Action 0: Go Short (or close long and open short)
    - Action 1: Hold / Close position
    - Action 2: Go Long (or close short and open long)

    Account State (10 elements, matching live futures env):
    [cash, position_size, position_value, entry_price, current_price,
     unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]

    Position size is:
    - Positive for long positions
    - Negative for short positions
    - Zero for no position
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: SeqFuturesEnvConfig,
        feature_preprocessing_fn: Optional[Callable] = None,
    ):
        """
        Initialize the SeqFuturesEnv.

        Args:
            df: DataFrame with OHLCV data
            config: Environment configuration
            feature_preprocessing_fn: Optional custom preprocessing function
        """
        self.action_levels = [-1.0, 0.0, 1.0]  # Short, Hold/Close, Long
        self.config = config
        self.transaction_fee = config.transaction_fee
        self.slippage = config.slippage
        self.leverage = config.leverage
        self.margin_type = config.margin_type
        self.maintenance_margin_rate = config.maintenance_margin_rate

        if not (0 <= config.transaction_fee <= 1):
            raise ValueError("Transaction fee must be between 0 and 1.")
        if not (0 <= config.slippage <= 1):
            raise ValueError("Slippage must be between 0 and 1.")
        if not (1 <= config.leverage <= 125):
            raise ValueError("Leverage must be between 1 and 125.")

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
        self.execute_on_value = config.execute_on.value
        self.execute_on_unit = config.execute_on.unit.value

        # Reset settings
        self.initial_cash = config.initial_cash
        self.position_hold_counter = 0

        # Define action spec
        self.action_spec = Categorical(len(self.action_levels))

        # Get the number of features from the sampler
        market_data_keys = self.sampler.get_observation_keys()
        num_features = len(self.sampler.get_feature_keys())

        # Observation space
        self.observation_spec = CompositeSpec(shape=())

        self.market_data_key = "market_data"
        self.account_state_key = "account_state"

        # Account state spec (10 elements for futures):
        # [cash, position_size, position_value, entry_price, current_price,
        #  unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]
        account_state_spec = Bounded(
            low=-torch.inf, high=torch.inf, shape=(10,), dtype=torch.float
        )

        self.market_data_keys = []
        window_sizes_list = (
            config.window_sizes
            if isinstance(config.window_sizes, list)
            else [config.window_sizes]
        )
        for i, market_data_name in enumerate(market_data_keys):
            market_data_key = f"market_data_{market_data_name}_{window_sizes_list[i]}"
            market_data_spec = Bounded(
                low=-torch.inf, high=torch.inf,
                shape=(window_sizes_list[i], num_features), dtype=torch.float
            )
            self.observation_spec.set(market_data_key, market_data_spec)
            self.market_data_keys.append(market_data_key)
        self.observation_spec.set(self.account_state_key, account_state_spec)

        self.reward_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(1,), dtype=torch.float)
        self.max_steps = self.sampler.get_max_steps()
        self.step_counter = 0

        # History tracking
        self.base_price_history = []
        self.action_history = []
        self.reward_history = []
        self.portfolio_value_history = []
        self.position_history = []

        super().__init__()

    def _calculate_liquidation_price(self, entry_price: float, position_size: float) -> float:
        """
        Calculate liquidation price for a position.

        For ISOLATED margin:
        - Long: liquidation_price = entry_price * (1 - 1/leverage + maintenance_margin_rate)
        - Short: liquidation_price = entry_price * (1 + 1/leverage - maintenance_margin_rate)
        """
        if position_size == 0:
            return 0.0

        margin_fraction = 1.0 / self.leverage

        if position_size > 0:
            # Long position - liquidated if price drops
            liquidation_price = entry_price * (1 - margin_fraction + self.maintenance_margin_rate)
        else:
            # Short position - liquidated if price rises
            liquidation_price = entry_price * (1 + margin_fraction - self.maintenance_margin_rate)

        return max(0, liquidation_price)

    def _calculate_margin_required(self, position_value: float) -> float:
        """Calculate initial margin required for a position."""
        return abs(position_value) / self.leverage

    def _calculate_unrealized_pnl(self, entry_price: float, current_price: float, position_size: float) -> float:
        """
        Calculate unrealized PnL.

        For long: PnL = (current_price - entry_price) * position_size
        For short: PnL = (entry_price - current_price) * abs(position_size)
        """
        if position_size == 0 or entry_price == 0:
            return 0.0

        if position_size > 0:
            # Long position
            return (current_price - entry_price) * position_size
        else:
            # Short position
            return (entry_price - current_price) * abs(position_size)

    def _calculate_unrealized_pnl_pct(self, entry_price: float, current_price: float, position_size: float) -> float:
        """Calculate unrealized PnL as a percentage."""
        if entry_price == 0:
            return 0.0

        if position_size > 0:
            # Long position
            return (current_price - entry_price) / entry_price
        elif position_size < 0:
            # Short position
            return (entry_price - current_price) / entry_price
        return 0.0

    def _check_liquidation(self, current_price: float) -> bool:
        """Check if current position should be liquidated."""
        if self.position_size == 0:
            return False

        if self.position_size > 0:
            # Long position - liquidated if price below liquidation price
            return current_price <= self.liquidation_price
        else:
            # Short position - liquidated if price above liquidation price
            return current_price >= self.liquidation_price

    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        # Get market data
        obs_dict, self.current_timestamp, self.truncated = self.sampler.get_sequential_observation()

        current_price = self.sampler.get_base_features(self.current_timestamp)["close"]

        # Calculate position value (absolute value of notional)
        self.position_value = abs(self.position_size * current_price)

        # Calculate unrealized PnL
        if self.position_size != 0:
            self.unrealized_pnl = self._calculate_unrealized_pnl(
                self.entry_price, current_price, self.position_size
            )
            self.unrealized_pnl_pct = self._calculate_unrealized_pnl_pct(
                self.entry_price, current_price, self.position_size
            )
        else:
            self.unrealized_pnl = 0.0
            self.unrealized_pnl_pct = 0.0

        # Calculate margin ratio
        total_balance = self.balance + self.unrealized_pnl
        margin_ratio = self.position_value / total_balance if total_balance > 0 else 0.0

        # Account state (10 elements, matching live futures env)
        account_state = [
            self.balance,
            self.position_size,  # Positive=long, Negative=short
            self.position_value,
            self.entry_price,
            current_price,
            self.unrealized_pnl_pct,
            float(self.leverage),
            margin_ratio,
            self.liquidation_price,
            float(self.position_hold_counter),
        ]
        account_state = torch.tensor(account_state, dtype=torch.float)

        obs_data = {self.account_state_key: account_state}
        obs_data.update(dict(zip(self.market_data_keys, obs_dict.values())))
        out_td = TensorDict(obs_data, batch_size=())

        return out_td

    def _calculate_reward(
        self,
        old_portfolio_value: float,
        new_portfolio_value: float,
        action: float,
        trade_info: Dict,
    ) -> float:
        """
        Calculate the step reward.

        Uses a hybrid approach:
        - Dense rewards based on portfolio return
        - Sparse terminal reward comparing to buy-and-hold
        """
        if old_portfolio_value == 0:
            return 0.0

        # Portfolio return
        portfolio_return = (new_portfolio_value - old_portfolio_value) / old_portfolio_value

        # Dense reward
        dense_reward = portfolio_return

        # Transaction cost penalty
        if trade_info.get("fee_paid", 0) > 0:
            dense_reward -= trade_info["fee_paid"] / old_portfolio_value

        # Liquidation penalty
        if trade_info.get("liquidated", False):
            dense_reward -= 0.1  # Significant penalty for liquidation

        # Clip dense reward
        dense_reward = np.clip(dense_reward, -0.1, 0.1)

        # Terminal reward
        if self.step_counter >= self.max_traj_length - 1:
            buy_and_hold_value = (
                self.initial_portfolio_value / self.base_price_history[0]
            ) * self.base_price_history[-1]

            compare_value = max(self.initial_portfolio_value, buy_and_hold_value)
            if compare_value > 0:
                terminal_reward = 100 * (new_portfolio_value - compare_value) / compare_value
                return terminal_reward

        return dense_reward * self.config.reward_scaling

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value including unrealized PnL."""
        current_price = self.sampler.get_base_features(self.current_timestamp)["close"]
        unrealized_pnl = self._calculate_unrealized_pnl(
            self.entry_price, current_price, self.position_size
        )
        return self.balance + unrealized_pnl

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
        # Reset history
        self.base_price_history = []
        self.action_history = []
        self.reward_history = []
        self.portfolio_value_history = []
        self.position_history = []

        max_episode_steps = self.sampler.reset(random_start=self.random_start)
        self.max_traj_length = max_episode_steps

        # Initialize balance
        initial_portfolio_value = (
            self.initial_cash
            if isinstance(self.initial_cash, int)
            else random.randint(self.initial_cash[0], self.initial_cash[1])
        )
        self.balance = initial_portfolio_value
        self.initial_portfolio_value = initial_portfolio_value

        # Reset position state
        self.position_hold_counter = 0
        self.current_position = 0  # -1=short, 0=none, 1=long
        self.position_size = 0.0  # Negative for short, positive for long
        self.position_value = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.liquidation_price = 0.0
        self.step_counter = 0

        return self._get_observation()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        self.step_counter += 1

        # Store old portfolio value
        old_portfolio_value = self._get_portfolio_value()

        # Get desired action
        action_idx = tensordict.get("action", 0)
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        desired_action = self.action_levels[action_idx]

        # Get current price for liquidation check
        current_price = self.sampler.get_base_features(self.current_timestamp)["close"]

        # Check for liquidation before executing trade
        trade_info = {"executed": False, "side": None, "fee_paid": 0.0, "liquidated": False}

        if self._check_liquidation(current_price):
            trade_info = self._execute_liquidation(current_price)
        else:
            trade_info = self._execute_trade_if_needed(desired_action)

        # Record history
        trade_action = 0
        if trade_info["executed"]:
            if trade_info["side"] == "long":
                trade_action = 1
            elif trade_info["side"] == "short":
                trade_action = -1
        self.action_history.append(trade_action)
        self.base_price_history.append(current_price)
        self.portfolio_value_history.append(old_portfolio_value)
        self.position_history.append(self.position_size)

        # Get updated state
        next_tensordict = self._get_observation()
        new_portfolio_value = self._get_portfolio_value()

        # Calculate reward
        reward = self._calculate_reward(
            old_portfolio_value, new_portfolio_value, desired_action, trade_info
        )
        self.reward_history.append(reward)

        # Check termination
        done = self._check_termination(new_portfolio_value)

        next_tensordict.set("reward", reward)
        next_tensordict.set("done", self.truncated or done)
        next_tensordict.set("truncated", self.truncated)
        next_tensordict.set("terminated", self.truncated or done)

        return next_tensordict

    def _execute_liquidation(self, current_price: float) -> Dict:
        """Execute forced liquidation of position."""
        trade_info = {
            "executed": True,
            "side": "liquidation",
            "fee_paid": 0.0,
            "liquidated": True,
        }

        # Realize the loss at liquidation price
        if self.position_size > 0:
            # Long position liquidated
            loss = (self.liquidation_price - self.entry_price) * self.position_size
        else:
            # Short position liquidated
            loss = (self.entry_price - self.liquidation_price) * abs(self.position_size)

        # Apply loss and fees
        liquidation_fee = abs(self.position_size * self.liquidation_price) * self.transaction_fee
        self.balance += loss - liquidation_fee
        trade_info["fee_paid"] = liquidation_fee

        # Reset position
        self.position_size = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.liquidation_price = 0.0
        self.current_position = 0
        self.position_hold_counter = 0

        return trade_info

    def _execute_trade_if_needed(self, desired_action: float) -> Dict:
        """Execute trade if position change is needed."""
        trade_info = {
            "executed": False,
            "side": None,
            "fee_paid": 0.0,
            "liquidated": False,
        }

        current_price = self.sampler.get_base_features(self.current_timestamp)["close"]

        # Apply slippage
        price_noise = torch.empty(1).uniform_(1 - self.slippage, 1 + self.slippage).item()

        # Action mapping: -1=short, 0=hold/close, 1=long

        if desired_action == 0:
            # Close any existing position
            if self.position_size != 0:
                trade_info = self._close_position(current_price, price_noise)
            else:
                # No position to close, just hold
                pass

        elif desired_action == 1:
            # Go long
            if self.position_size < 0:
                # Close short first
                self._close_position(current_price, price_noise)

            if self.position_size <= 0:
                # Open long position
                trade_info = self._open_position("long", current_price, price_noise)

        elif desired_action == -1:
            # Go short
            if self.position_size > 0:
                # Close long first
                self._close_position(current_price, price_noise)

            if self.position_size >= 0:
                # Open short position
                trade_info = self._open_position("short", current_price, price_noise)

        # Update hold counter
        if self.position_size != 0:
            self.position_hold_counter += 1

        return trade_info

    def _open_position(self, side: str, current_price: float, price_noise: float) -> Dict:
        """Open a new position."""
        trade_info = {
            "executed": True,
            "side": side,
            "fee_paid": 0.0,
            "liquidated": False,
        }

        execution_price = current_price * price_noise

        # Calculate position size based on available balance and leverage
        # Account for both margin requirement and fees:
        # margin_required + fee <= usable_balance
        # notional/leverage + notional*fee_rate <= usable_balance
        # notional <= usable_balance / (1/leverage + fee_rate)
        usable_balance = self.balance * self.config.max_position_size
        margin_plus_fee_rate = (1.0 / self.leverage) + self.transaction_fee
        max_notional = usable_balance / margin_plus_fee_rate
        notional_value = max_notional
        position_qty = notional_value / execution_price

        # Calculate margin required
        margin_required = self._calculate_margin_required(notional_value)

        # Calculate fee
        fee = notional_value * self.transaction_fee

        if margin_required + fee > self.balance:
            # Not enough balance (shouldn't happen with correct calculation, but safety check)
            trade_info["executed"] = False
            return trade_info

        # Execute trade
        self.balance -= fee
        trade_info["fee_paid"] = fee

        if side == "long":
            self.position_size = position_qty
            self.current_position = 1
        else:  # short
            self.position_size = -position_qty
            self.current_position = -1

        self.entry_price = execution_price
        self.position_value = notional_value
        self.liquidation_price = self._calculate_liquidation_price(
            execution_price, self.position_size
        )
        self.position_hold_counter = 0

        return trade_info

    def _close_position(self, current_price: float, price_noise: float) -> Dict:
        """Close existing position."""
        trade_info = {
            "executed": True,
            "side": "close",
            "fee_paid": 0.0,
            "liquidated": False,
        }

        if self.position_size == 0:
            trade_info["executed"] = False
            return trade_info

        execution_price = current_price * price_noise

        # Calculate PnL
        pnl = self._calculate_unrealized_pnl(
            self.entry_price, execution_price, self.position_size
        )

        # Calculate fee
        notional = abs(self.position_size * execution_price)
        fee = notional * self.transaction_fee

        # Update balance
        self.balance += pnl - fee
        trade_info["fee_paid"] = fee

        # Reset position
        self.position_size = 0.0
        self.position_value = 0.0
        self.entry_price = 0.0
        self.liquidation_price = 0.0
        self.current_position = 0
        self.position_hold_counter = 0

        return trade_info

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold or self.step_counter >= self.max_steps

    def close(self):
        """Clean up resources."""
        pass

    def render_history(self, return_fig=False):
        """Render the history of the environment."""
        price_history = self.base_price_history
        time_indices = list(range(len(price_history)))
        action_history = self.action_history
        portfolio_value_history = self.portfolio_value_history
        position_history = self.position_history

        # Calculate buy-and-hold balance
        initial_balance = portfolio_value_history[0]
        initial_price = price_history[0]
        units_held = initial_balance / initial_price
        buy_and_hold_balance = [units_held * price for price in price_history]

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 2, 1]})
        ax1, ax2, ax3 = axes

        # Plot price history
        ax1.plot(time_indices, price_history, label='Price History', color='blue', linewidth=1.5)

        # Plot buy/sell actions
        long_indices = [i for i, action in enumerate(action_history) if action == 1]
        long_prices = [price_history[i] for i in long_indices]
        short_indices = [i for i, action in enumerate(action_history) if action == -1]
        short_prices = [price_history[i] for i in short_indices]

        ax1.scatter(long_indices, long_prices, marker='^', color='green', s=80, label='Long', zorder=5)
        ax1.scatter(short_indices, short_prices, marker='v', color='red', s=80, label='Short', zorder=5)

        ax1.set_ylabel('Price (USD)')
        ax1.set_title('Price History with Long/Short Actions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot portfolio value
        ax2.plot(time_indices, portfolio_value_history, label='Portfolio Value', color='green', linewidth=1.5)
        ax2.plot(time_indices, buy_and_hold_balance, label='Buy and Hold', color='purple', linestyle='--', linewidth=1.5)
        ax2.set_ylabel('Portfolio Value (USD)')
        ax2.set_title('Portfolio Value vs Buy and Hold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot position
        ax3.fill_between(time_indices, position_history, 0,
                         where=[p > 0 for p in position_history], color='green', alpha=0.3, label='Long')
        ax3.fill_between(time_indices, position_history, 0,
                         where=[p < 0 for p in position_history], color='red', alpha=0.3, label='Short')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Time (Index)')
        ax3.set_ylabel('Position Size')
        ax3.set_title('Position History')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()


if __name__ == "__main__":
    import pandas as pd

    time_frames = [
        TimeFrame(1, TimeFrameUnit.Minute),
        TimeFrame(5, TimeFrameUnit.Minute),
    ]
    window_sizes = [12, 8]
    execute_on = TimeFrame(5, TimeFrameUnit.Minute)

    # Load sample data
    df = pd.read_csv(
        "/home/sebastian/Documents/TorchTrade/torchrl_alpaca_env/torchtrade/data/binance_spot_1m_cleaned/btcusdt_spot_1m_12_2024_to_09_2025.csv"
    )

    config = SeqFuturesEnvConfig(
        symbol="BTC/USD",
        time_frames=time_frames,
        window_sizes=window_sizes,
        execute_on=execute_on,
        leverage=10,
        transaction_fee=0.0004,
        slippage=0.001,
        bankrupt_threshold=0.1,
    )
    env = SeqFuturesEnv(df, config)

    td = env.reset()
    for i in range(env.max_steps):
        action = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
        td.set("action", action)
        td = env.step(td)
        td = td["next"]
        portfolio_value = td["account_state"][0].item() + td["account_state"][2].item()
        print(f"Step {i}: Action={action}, Portfolio=${portfolio_value:.2f}")
        if td["done"]:
            print(f"Episode done at step {i}")
            break

    env.render_history()
