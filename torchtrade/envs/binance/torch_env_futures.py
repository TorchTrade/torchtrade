from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Callable

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.binance.obs_class import BinanceObservationClass
from torchtrade.envs.binance.futures_order_executor import (
    BinanceFuturesOrderClass,
    TradeMode,
    MarginType,
)
from torchtrade.envs.binance.base import BinanceBaseTorchTradingEnv


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
    trade_mode: TradeMode = TradeMode.QUANTITY

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


class BinanceFuturesTorchTradingEnv(BinanceBaseTorchTradingEnv):
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
        # Initialize base class (handles observer/trader, obs specs, portfolio value, etc.)
        super().__init__(config, api_key, api_secret, feature_preprocessing_fn, observer, trader)

        # Define action space (environment-specific)
        self.action_levels = config.action_levels
        self.action_spec = Categorical(len(self.action_levels))

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
                self.position.current_position = 1
            elif trade_info["side"] == "SELL":
                self.position.current_position = -1
            elif trade_info["closed_position"]:
                self.position.current_position = 0

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
