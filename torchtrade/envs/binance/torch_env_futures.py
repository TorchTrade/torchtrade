import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from warnings import warn

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs import EnvBase
from torchrl.data import Categorical, Bounded

from torchtrade.envs.binance.obs_class import BinanceObservationClass
from torchtrade.envs.binance.futures_order_executor import (
    BinanceFuturesOrderClass,
    TradeMode,
    MarginType,
)
from torchtrade.envs.reward import RewardContext, default_reward_function
from dotenv import load_dotenv

load_dotenv()



@dataclass
class BinanceFuturesTradingEnvConfig:
    """Configuration for Binance Futures Trading Environment."""

    symbol: str = "BTCUSDT"
    # Action levels: -1 = short, 0 = close/hold, 1 = long
    action_levels: List[float] = field(default_factory=lambda: [-1.0, 0.0, 1.0])
    max_position: float = 1.0  # Maximum position size as fraction of balance

    # Timeframes and windows
    intervals: Union[List[str], str] = field(default_factory=lambda: ["1m"])
    window_sizes: Union[List[int], int] = 10
    execute_on: str = "1m"  # Interval for trade execution timing

    # Trading parameters
    leverage: int = 1  # Leverage (1-125)
    margin_type: MarginType = MarginType.ISOLATED
    quantity_per_trade: float = 0.001  # Base quantity per trade

    # Reward settings
    reward_scaling: float = 1.0
    position_penalty: float = 0.0001

    # Termination settings
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1  # 10% of initial balance

    # Environment settings
    demo: bool = True  # Use demo/testnet for paper trading
    seed: Optional[int] = 42
    include_base_features: bool = False
    close_position_on_reset: bool = False  # Whether to close positions on env.reset()
    reward_function: Optional[Callable] = None  # Custom reward function (uses default if None)


# Interval to seconds mapping for waiting
INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


class BinanceFuturesTorchTradingEnv(EnvBase):
    """
    TorchRL environment for Binance Futures trading.

    Supports:
    - Long and short positions
    - Configurable leverage
    - Multiple timeframe observations
    - Demo (paper) trading via Binance testnet

    Action Space (default):
    - Action 0: Go Short (or increase short position)
    - Action 1: Close position / Hold (do nothing)
    - Action 2: Go Long (or increase long position)

    Account State (10 elements):
    [cash, position_size, position_value, entry_price, current_price,
     unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]
    """

    def __init__(
        self,
        config: BinanceFuturesTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        observer: Optional[BinanceObservationClass] = None,
        trader: Optional[BinanceFuturesOrderClass] = None,
    ):
        """
        Initialize the BinanceFuturesTorchTradingEnv.

        Args:
            config: Environment configuration
            api_key: Binance API key
            api_secret: Binance API secret
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured BinanceObservationClass
            trader: Optional pre-configured BinanceFuturesOrderClass
        """
        self.config = config

        # Normalize intervals to list
        intervals = config.intervals if isinstance(config.intervals, list) else [config.intervals]
        window_sizes = config.window_sizes if isinstance(config.window_sizes, list) else [config.window_sizes]

        # Initialize observer
        self.observer = observer if observer is not None else BinanceObservationClass(
            symbol=config.symbol,
            intervals=intervals,
            window_sizes=window_sizes,
            feature_preprocessing_fn=feature_preprocessing_fn,
            demo=config.demo,
        )

        # Initialize trader
        self.trader = trader if trader is not None else BinanceFuturesOrderClass(
            symbol=config.symbol,
            trade_mode=TradeMode.QUANTITY,
            api_key=api_key,
            api_secret=api_secret,
            demo=config.demo,
            leverage=config.leverage,
            margin_type=config.margin_type,
        )

        # Execute interval
        self.execute_on = config.execute_on
        self.execute_interval_seconds = INTERVAL_SECONDS.get(config.execute_on, 60)

        # Reset settings
        self.trader.cancel_open_orders()
        self.trader.close_position()

        balance = self.trader.get_account_balance()
        self.initial_portfolio_value = balance.get("total_wallet_balance", 0)
        self.position_hold_counter = 0

        self.action_levels = config.action_levels
        # Define action spec
        self.action_spec = Categorical(len(self.action_levels))

        # Get the number of features from the observer
        obs = self.observer.get_observations()
        first_key = self.observer.get_keys()[0]
        num_features = obs[first_key].shape[1]

        # Get market data obs names
        market_data_names = self.observer.get_keys()

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
        for i, market_data_name in enumerate(market_data_names):
            market_data_key = "market_data_" + market_data_name
            ws = window_sizes[i] if isinstance(window_sizes, list) else window_sizes
            market_data_spec = Bounded(
                low=-torch.inf, high=torch.inf,
                shape=(ws, num_features), dtype=torch.float
            )
            self.observation_spec.set(market_data_key, market_data_spec)
            self.market_data_keys.append(market_data_key)

        self.observation_spec.set(self.account_state_key, account_state_spec)
        self.reward_spec = Bounded(low=-torch.inf, high=torch.inf, shape=(1,), dtype=torch.float)

        self._reset(TensorDict({}))
        super().__init__()

    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        # Get market data
        obs_dict = self.observer.get_observations(
            return_base_ohlc=self.config.include_base_features
        )

        if self.config.include_base_features:
            base_features = obs_dict.get("base_features")

        market_data = [obs_dict[features_name] for features_name in self.observer.get_keys()]

        # Get account state
        status = self.trader.get_status()
        balance = self.trader.get_account_balance()

        cash = balance.get("available_balance", 0)
        total_balance = balance.get("total_wallet_balance", 0)

        position_status = status.get("position_status", None)

        if position_status is None:
            self.position_hold_counter = 0
            position_size = 0.0
            position_value = 0.0
            entry_price = 0.0
            current_price = self.trader.get_mark_price()
            unrealized_pnl_pct = 0.0
            leverage = float(self.config.leverage)
            margin_ratio = 0.0
            liquidation_price = 0.0
            holding_time = 0.0
        else:
            self.position_hold_counter += 1
            position_size = position_status.qty  # Positive=long, Negative=short
            position_value = abs(position_status.notional_value)
            entry_price = position_status.entry_price
            current_price = position_status.mark_price
            unrealized_pnl_pct = position_status.unrealized_pnl_pct
            leverage = float(position_status.leverage)
            # Calculate margin ratio (position value / total balance)
            margin_ratio = position_value / total_balance if total_balance > 0 else 0.0
            liquidation_price = position_status.liquidation_price
            holding_time = float(self.position_hold_counter)

        account_state = torch.tensor(
            [
                cash,
                position_size,
                position_value,
                entry_price,
                current_price,
                unrealized_pnl_pct,
                leverage,
                margin_ratio,
                liquidation_price,
                holding_time,
            ],
            dtype=torch.float,
        )

        out_td = TensorDict({self.account_state_key: account_state}, batch_size=())
        for market_data_name, data in zip(self.market_data_keys, market_data):
            out_td.set(market_data_name, torch.from_numpy(data))

        if self.config.include_base_features and base_features is not None:
            out_td.set("base_features", torch.from_numpy(base_features))

        return out_td

    def _build_reward_context(
        self,
        old_portfolio_value: float,
        new_portfolio_value: float,
        action: float,
        trade_info: Dict,
    ) -> RewardContext:
        """Build RewardContext from current state for custom reward functions."""
        # Get current account and position state
        position_status = self.trader.get_position_status()

        if position_status is None:
            cash = self.trader.get_account_balance().get("available_balance", 0)
            position_size = 0.0
            position_value = 0.0
            entry_price = 0.0
            current_price = 0.0
            unrealized_pnl_pct = 0.0
            leverage = 0.0
            margin_ratio = 0.0
            liquidation_price = 0.0
        else:
            balance = self.trader.get_account_balance()
            cash = balance.get("available_balance", 0)
            position_size = position_status.position_amount
            position_value = abs(position_status.position_amount * position_status.entry_price)
            entry_price = position_status.entry_price
            current_price = position_status.mark_price
            unrealized_pnl_pct = position_status.unrealized_pnl_pct
            leverage = float(position_status.leverage)
            total_balance = balance.get("total_margin_balance", 0)
            margin_ratio = position_value / total_balance if total_balance > 0 else 0.0
            liquidation_price = position_status.liquidation_price

        # Map action to side string
        trade_side = "hold"
        if action == 1:
            trade_side = "long"
        elif action == -1:
            trade_side = "short"
        elif action == 0 and trade_info.get("closed_position"):
            trade_side = "close"

        return RewardContext(
            old_portfolio_value=old_portfolio_value,
            new_portfolio_value=new_portfolio_value,
            action=int(action),
            current_step=0,  # Live environment doesn't track steps
            max_steps=1,  # Live environment is continuous
            trade_executed=trade_info.get("executed", False),
            trade_side=trade_side,
            fee_paid=0.0,  # Not tracked in this environment
            slippage_amount=0.0,  # Not tracked in this environment
            cash=cash,
            position_size=position_size,
            position_value=position_value,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl_pct=unrealized_pnl_pct,
            holding_time=self.position_hold_counter,
            portfolio_value_history=[],  # Not tracked in live environment
            action_history=[],  # Not tracked in live environment
            reward_history=[],  # Not tracked in live environment
            base_price_history=[],  # Not tracked in live environment
            liquidated=False,  # Not tracked in this environment
            leverage=leverage,
            margin_ratio=margin_ratio,
            liquidation_price=liquidation_price,
            initial_portfolio_value=old_portfolio_value,  # Approximate
            buy_and_hold_value=None,  # Not applicable for live trading
        )

    def _calculate_reward(
        self,
        old_portfolio_value: float,
        new_portfolio_value: float,
        action: float,
        trade_info: Dict,
    ) -> float:
        """
        Calculate reward using custom or default function.

        If config.reward_function is provided, uses that custom function.
        Otherwise, uses the default log return reward.

        Args:
            old_portfolio_value: Portfolio value before action
            new_portfolio_value: Portfolio value after action
            action: Action taken (-1=short, 0=close, 1=long)
            trade_info: Dictionary with trade execution details

        Returns:
            Reward value (float), scaled by config.reward_scaling
        """
        # Use custom reward function if provided
        if self.config.reward_function is not None:
            ctx = self._build_reward_context(
                old_portfolio_value,
                new_portfolio_value,
                action,
                trade_info
            )
            return float(self.config.reward_function(ctx)) * self.config.reward_scaling

        # Otherwise use default log return
        reward = default_reward_function(
            self._build_reward_context(
                old_portfolio_value,
                new_portfolio_value,
                action,
                trade_info
            )
        )
        return reward * self.config.reward_scaling

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        balance = self.trader.get_account_balance()
        return balance.get("total_margin_balance", 0)

    def _wait_for_next_timestamp(self) -> None:
        """Wait until the next time step based on the configured execute_on interval."""
        current_time = datetime.now()

        # Calculate next execution time (round up to next interval)
        seconds = self.execute_interval_seconds
        next_step = current_time + timedelta(seconds=seconds)
        next_step = next_step.replace(second=0, microsecond=0)

        # Wait until target time with a single sleep call
        sleep_seconds = (next_step - datetime.now()).total_seconds()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

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
        # Cancel all orders
        self.trader.cancel_open_orders()

        # Optionally close positions on reset (configurable)
        if self.config.close_position_on_reset:
            self.trader.close_position()

        balance = self.trader.get_account_balance()
        self.balance = balance.get("available_balance", 0)
        self.last_portfolio_value = self._get_portfolio_value()

        status = self.trader.get_status()
        position_status = status.get("position_status")
        self.position_hold_counter = 0

        if position_status is None:
            self.current_position = 0  # No position
        else:
            self.current_position = 1 if position_status.qty > 0 else -1 if position_status.qty < 0 else 0

        return self._get_observation()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""
        # Store old portfolio value
        old_portfolio_value = self._get_portfolio_value()

        # Get desired action
        action_idx = tensordict.get("action", 0)
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        desired_action = self.action_levels[action_idx]

        # Execute trade
        trade_info = self._execute_trade_if_needed(desired_action)

        if trade_info["executed"]:
            if trade_info["side"] == "BUY":
                self.current_position = 1
            elif trade_info["side"] == "SELL":
                self.current_position = -1
            elif trade_info["closed_position"]:
                self.current_position = 0

        # Wait for next time step
        self._wait_for_next_timestamp()

        # Get updated state
        new_portfolio_value = self._get_portfolio_value()
        next_tensordict = self._get_observation()

        # Calculate reward and check termination
        reward = self._calculate_reward(
            old_portfolio_value, new_portfolio_value, desired_action, trade_info
        )
        done = self._check_termination(new_portfolio_value)

        next_tensordict.set("reward", torch.tensor([reward], dtype=torch.float))
        next_tensordict.set("done", torch.tensor([done], dtype=torch.bool))
        next_tensordict.set("truncated", torch.tensor([False], dtype=torch.bool))
        next_tensordict.set("terminated", torch.tensor([done], dtype=torch.bool))

        return next_tensordict

    def _execute_trade_if_needed(self, desired_action: float) -> Dict:
        """Execute trade if position change is needed."""
        trade_info = {
            "executed": False,
            "quantity": 0,
            "side": None,
            "success": None,
            "closed_position": False,
        }

        # Action mapping:
        # -1 = Go short
        #  0 = Close position / hold
        #  1 = Go long

        status = self.trader.get_status()
        position = status.get("position_status")
        current_qty = position.qty if position else 0

        if desired_action == 0:
            # Close position if we have one
            if current_qty != 0:
                success = self.trader.close_position()
                trade_info.update({
                    "executed": True,
                    "quantity": abs(current_qty),
                    "side": "CLOSE",
                    "success": success,
                    "closed_position": True,
                })
        elif desired_action == 1:
            # Go long
            if current_qty <= 0:
                # Close short first if exists
                if current_qty < 0:
                    self.trader.close_position()

                # Open long
                success = self.trader.trade(
                    side="BUY",
                    quantity=self.config.quantity_per_trade,
                    order_type="market",
                )
                trade_info.update({
                    "executed": True,
                    "quantity": self.config.quantity_per_trade,
                    "side": "BUY",
                    "success": success,
                })
        elif desired_action == -1:
            # Go short
            if current_qty >= 0:
                # Close long first if exists
                if current_qty > 0:
                    self.trader.close_position()

                # Open short
                success = self.trader.trade(
                    side="SELL",
                    quantity=self.config.quantity_per_trade,
                    order_type="market",
                )
                trade_info.update({
                    "executed": True,
                    "quantity": self.config.quantity_per_trade,
                    "side": "SELL",
                    "success": success,
                })

        return trade_info

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        if not self.config.done_on_bankruptcy:
            return False

        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold

    def close(self):
        """Clean up resources."""
        self.trader.cancel_open_orders()
        # Optionally close positions - commented out for safety
        # self.trader.close_all_positions()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Create environment configuration
    config = BinanceFuturesTradingEnvConfig(
        symbol="BTCUSDT",
        demo=True,
        intervals=["1m", "5m"],
        window_sizes=[10, 10],
        execute_on="1m",
        leverage=5,
        quantity_per_trade=0.002,  # ~$190 notional to meet $100 minimum
        include_base_features=False,
    )

    # Create environment
    env = BinanceFuturesTorchTradingEnv(
        config,
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_SECRET", ""),
    )

    td = env.reset()
    print("Reset observation:")
    print(td)
    for i in range(5):
        action = env.action_spec.rand()
        td = TensorDict({"action": action}, batch_size=())
        next_td = env.step(td)
        print(f"Step {i+1}: Action={action.item()}, Reward={next_td['next', 'reward'].item():.6f}")
        if next_td["next", "done"].item():
            print("Episode terminated")
            break
