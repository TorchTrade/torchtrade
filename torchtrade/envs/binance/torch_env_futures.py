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

        # Get current price and position from trader status (avoids redundant observation call)
        status = self.trader.get_status()
        position_status = status.get("position_status", None)
        if position_status:
            current_price = position_status.mark_price
            position_size = position_status.qty
        else:
            current_price = self.trader.get_mark_price()
            position_size = 0.0

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

        # Update position hold counter
        if self.position.current_position != 0:
            self.position.hold_counter += 1
        else:
            self.position.hold_counter = 0

        # Get updated state
        new_portfolio_value = self._get_portfolio_value()
        next_tensordict = self._get_observation()

        # Calculate reward and check termination
        reward = self._calculate_reward(
            old_portfolio_value, new_portfolio_value, desired_action, trade_info
        )
        done = self._check_termination(new_portfolio_value)

        # Record step history
        self.history.record_step(
            price=current_price,
            action=desired_action,
            reward=reward,
            portfolio_value=old_portfolio_value,
            position=position_size
        )

        next_tensordict.set("reward", torch.tensor([reward], dtype=torch.float))
        next_tensordict.set("done", torch.tensor([done], dtype=torch.bool))
        next_tensordict.set("truncated", torch.tensor([False], dtype=torch.bool))
        next_tensordict.set("terminated", torch.tensor([done], dtype=torch.bool))

        return next_tensordict

    def _get_current_position_quantity(self) -> float:
        """Get current position quantity from trader status."""
        status = self.trader.get_status()
        position = status.get("position_status")
        return position.qty if position is not None else 0.0

    def _create_trade_info(self, executed=False, **kwargs) -> Dict:
        """Create trade info dictionary with defaults."""
        info = {
            "executed": executed,
            "quantity": 0,
            "side": None,
            "success": None,
            "closed_position": False,
        }
        info.update(kwargs)
        return info

    def _execute_market_order(self, side: str, quantity: float) -> Dict:
        """Execute a market order with error handling."""
        try:
            success = self.trader.trade(
                side=side,
                quantity=quantity,
                order_type="market",
            )
            return self._create_trade_info(
                executed=True,
                quantity=quantity,
                side=side,
                success=success,
            )
        except Exception as e:
            logger.error(f"{side} trade failed for {self.config.symbol}: quantity={quantity}, error={e}")
            return self._create_trade_info(executed=True, success=False)

    def _handle_close_action(self, current_qty: float) -> Dict:
        """Handle close position action."""
        if current_qty == 0:
            return self._create_trade_info(executed=False)

        success = self.trader.close_position()

        return self._create_trade_info(
            executed=True,
            quantity=abs(current_qty),
            side="CLOSE",
            success=success,
            closed_position=True,
        )

    def _handle_long_action(self, current_qty: float) -> Dict:
        """Handle go long action."""
        # Close short position if necessary
        if current_qty < 0:
            self.trader.close_position()

        # Only execute if not already long
        if current_qty > 0:
            return self._create_trade_info(executed=False)

        return self._execute_market_order("BUY", self.config.quantity_per_trade)

    def _handle_short_action(self, current_qty: float) -> Dict:
        """Handle go short action."""
        # Close long position if necessary
        if current_qty > 0:
            self.trader.close_position()

        # Only execute if not already short
        if current_qty < 0:
            return self._create_trade_info(executed=False)

        return self._execute_market_order("SELL", self.config.quantity_per_trade)

    def _execute_trade_if_needed(self, desired_action: float) -> Dict:
        """
        Execute trade if position change is needed.

        Args:
            desired_action: Action level (-1.0 = short, 0.0 = close/hold, 1.0 = long)

        Returns:
            Dict with trade execution info
        """
        current_qty = self._get_current_position_quantity()

        if desired_action == 0:
            return self._handle_close_action(current_qty)
        elif desired_action == 1:
            return self._handle_long_action(current_qty)
        elif desired_action == -1:
            return self._handle_short_action(current_qty)
        else:
            return self._create_trade_info(executed=False)

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
