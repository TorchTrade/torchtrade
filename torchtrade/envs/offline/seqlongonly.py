from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
from tensordict import TensorDict, TensorDictBase
import torch
from torchrl.data import Categorical
import pandas as pd

from torchtrade.envs.offline.base import TorchTradeOfflineEnv
from torchtrade.envs.offline.utils import TimeFrame, TimeFrameUnit, normalize_timeframe_config

@dataclass
class SeqLongOnlyEnvConfig:
    symbol: str = "BTC/USD"
    time_frames: Union[List[TimeFrame], TimeFrame] = TimeFrame(1, TimeFrameUnit.Minute)
    window_sizes: Union[List[int], int] = 10
    execute_on: TimeFrame = TimeFrame(1, TimeFrameUnit.Minute) # On which timeframe to execute trades
    initial_cash: Union[List[int], int] = (1000, 5000)
    transaction_fee: float = 0.025
    bankrupt_threshold: float = 0.1  # 10% of initial balance
    slippage: float = 0.01
    seed: Optional[int] = 42
    include_base_features: bool = False # Includes base features such as timestamps and ohlc to the tensordict
    max_traj_length: Optional[int] = None
    random_start: bool = True
    reward_function: Optional[Callable] = None  # Custom reward function (uses default if None)
    reward_scaling: float = 1.0

    def __post_init__(self):
        """Normalize timeframe configuration to list format."""
        self.execute_on, self.time_frames, self.window_sizes = normalize_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

class SeqLongOnlyEnv(TorchTradeOfflineEnv):
    """Sequential long-only trading environment for backtesting.

    Supports 3 discrete actions: Sell-all (-1), Hold (0), Buy-all (1).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: SeqLongOnlyEnvConfig,
        feature_preprocessing_fn: Optional[Callable] = None
    ):
        """
        Initialize the sequential long-only environment.

        Args:
            df: OHLCV DataFrame for backtesting
            config: Environment configuration
            feature_preprocessing_fn: Optional function to preprocess features
        """
        # Initialize base class (handles sampler, history, balance, etc.)
        super().__init__(df, config, feature_preprocessing_fn)

        # Environment-specific configuration
        self.action_levels = [-1.0, 0.0, 1.0]  # Sell-all, Do-Nothing, Buy-all

        # Define action spec
        self.action_spec = Categorical(len(self.action_levels))

        # Build observation specs
        account_state = [
            "cash", "position_size", "position_value", "entry_price",
            "current_price", "unrealized_pnlpct", "holding_time"
        ]
        num_features = len(self.sampler.get_feature_keys())
        self._build_observation_specs(account_state, num_features)


    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        # Get market data (scaffold sets current_timestamp, truncated, and caches base features)
        obs_dict = self._get_observation_scaffold()
        current_price = self._cached_base_features["close"]

        # Update position value and unrealized PnL
        self.position_value = round(self.position_size * current_price, 3)
        if self.position_size > 0:
            self.unrealized_pnlpc = round(
                (current_price - self.entry_price) / self.entry_price, 4
            )
        else:
            self.unrealized_pnlpc = 0.0

        # Build account state
        account_state = torch.tensor([
            self.balance,
            self.position_size,
            self.position_value,
            self.entry_price,
            current_price,
            self.unrealized_pnlpc,
            self.position_hold_counter
        ], dtype=torch.float)

        # Combine account state and market data
        obs_data = {self.account_state_key: account_state}
        obs_data.update(dict(zip(self.market_data_keys, obs_dict.values())))

        return TensorDict(obs_data, batch_size=())

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        self.step_counter += 1

        # Cache base features once for current timestamp (avoids 4+ redundant get_base_features calls)
        cached_base = self._cached_base_features
        cached_price = cached_base["close"]

        # Store old portfolio value for reward calculation
        old_portfolio_value = self._get_portfolio_value(cached_price)

        # Get desired action and current position
        desired_action = self.action_levels[tensordict.get("action", 0)]

        # Calculate and execute trade if needed (pass cached price)
        trade_info = self._execute_trade_if_needed(desired_action, cached_price)
        self.action_history.append(desired_action if trade_info["executed"] else 0)
        self.base_price_history.append(cached_price)
        self.portfolio_value_history.append(old_portfolio_value)

        if trade_info["executed"]:
            self.current_position = 1 if trade_info["side"] == "buy" else 0

        # Get updated state (this advances timestamp and caches new base features)
        next_tensordict = self._get_observation()
        # Use newly cached base features for new portfolio value
        new_price = self._cached_base_features["close"]
        new_portfolio_value = self._get_portfolio_value(new_price)

        # Add state_index for coverage tracking (only during training with random_start)
        if self.random_start:
            next_tensordict.set("state_index", torch.tensor(self.sampler._sequential_idx, dtype=torch.long))

        # Calculate reward and check termination
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value, desired_action, trade_info)
        self.reward_history.append(reward)

        done = self._check_termination(new_portfolio_value)
        next_tensordict.set("reward", reward)
        next_tensordict.set("done", self.truncated or done)
        next_tensordict.set("truncated", self.truncated)
        next_tensordict.set("terminated", self.truncated or done)

        return next_tensordict


    def _execute_trade_if_needed(self, desired_position: float, base_price: float = None) -> Dict:
        """Execute trade if position change is needed."""
        trade_info = {"executed": False, "amount": 0, "side": None, "success": None, "price_noise": 0.0, "fee_paid": 0.0}

        # No action requested (hold)
        if desired_position == 0:
            if self.position_size > 0:
                self.position_hold_counter += 1
            return trade_info

        # Already at desired position
        if desired_position == self.current_position:
            if self.position_size > 0:
                self.position_hold_counter += 1
            return trade_info

        # Determine trade details
        side = "buy" if desired_position > 0 else "sell"
        amount = self._calculate_trade_amount(side)

        # Get base price and apply noise to simulate slippage
        if base_price is None:
            base_price = self.sampler.get_base_features(self.current_timestamp)["close"]
        # Apply ±5% noise to the price to simulate market slippage
        price_noise_factor = torch.empty(1,).uniform_(1 - self.slippage, 1 + self.slippage).item()
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


import ta
import pandas as pd
import numpy as np
def custom_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess OHLCV dataframe with engineered features for RL trading.

    Expected columns: ["open", "high", "low", "close", "volume"]
    Index can be datetime or integer.
    """

    df = df.copy().reset_index(drop=False)

    # --- Basic features ---
    # Log returns
    df["features_return_log"] = np.log(df["close"]).diff()

    # Rolling volatility (5-period)
    df["features_volatility"] = df["features_return_log"].rolling(window=5).std()

    # ATR (14) normalized
    df["features_atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range() / df["close"]

    # --- Momentum & trend ---
    ema_12 = ta.trend.EMAIndicator(close=df["close"], window=12).ema_indicator()
    ema_24 = ta.trend.EMAIndicator(close=df["close"], window=24).ema_indicator()
    df["features_ema_12"] = ema_12
    df["features_ema_24"] = ema_24
    df["features_ema_slope"] = ema_12.diff()

    macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["features_macd_hist"] = macd.macd_diff()

    df["features_rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    # --- Volatility bands ---
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["features_bb_pct"] = bb.bollinger_pband()

    # --- Volume / flow ---
    df["features_volume_z"] = (
        (df["volume"] - df["volume"].rolling(20).mean()) /
        df["volume"].rolling(20).std()
    )
    df["features_vwap_dev"] = df["close"] - (
        (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    )

    # --- Candle structure ---
    df["features_body_ratio"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    df["features_upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (
        df["high"] - df["low"] + 1e-9
    )
    df["features_lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (
        df["high"] - df["low"] + 1e-9
    )

    # Drop rows with NaN from indicators
    #df.dropna(inplace=True)
    df.fillna(0, inplace=True)


    return df

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

    df = pd.read_csv("/home/sebastian/Documents/TorchTrade/torchrl_alpaca_env/torchtrade/data/binance_spot_1m_cleaned/btcusdt_spot_1m_12_2024_to_09_2025.csv")

    #df = df[0:(1440 * 7)] # 1440 minutes = 1 day

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
    env = SeqLongOnlyEnv(df, config, feature_preprocessing_fn=custom_preprocessing)

    td = env.reset()
    for i in range(env.max_steps):
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
