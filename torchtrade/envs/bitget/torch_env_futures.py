from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Dict
import logging

import torch
from tensordict import TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.bitget.obs_class import BitgetObservationClass
from torchtrade.envs.bitget.futures_order_executor import (
    BitgetFuturesOrderClass,
    TradeMode,
    MarginMode,
)
from torchtrade.envs.bitget.base import BitgetBaseTorchTradingEnv


logger = logging.getLogger(__name__)


@dataclass
class BitgetFuturesTradingEnvConfig:
    """Configuration for Bitget Futures Trading Environment."""

    symbol: str = "BTCUSDT"
    action_levels: List[float] = None  # Default set in __post_init__

    # Timeframes and windows
    intervals: Union[List[str], str] = "1m"
    window_sizes: Union[List[int], int] = 10
    execute_on: str = "1m"  # Interval for trade execution timing

    # Trading parameters
    product_type: str = "SUMCBL"  # SUMCBL=testnet, UMCBL=production
    leverage: int = 1  # Leverage (1-125)
    margin_mode: MarginMode = MarginMode.ISOLATED
    quantity_per_trade: float = 0.001  # Base quantity per trade
    trade_mode: TradeMode = TradeMode.QUANTITY

    # Reward settings
    reward_scaling: float = 1.0
    position_penalty: float = 0.0001  # Penalty for holding positions

    # Termination settings
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1  # 10% of initial balance

    # Environment settings
    demo: bool = True  # Use testnet for demo
    seed: Optional[int] = 42
    include_base_features: bool = False
    close_position_on_reset: bool = False  # Whether to close positions on env.reset()
    reward_function: Optional[Callable] = None  # Custom reward function (uses default if None)

    def __post_init__(self):
        # Set default action levels if not provided
        if self.action_levels is None:
            self.action_levels = [-1.0, 0.0, 1.0]  # short, close/hold, long

        # Normalize to lists
        if isinstance(self.intervals, str):
            self.intervals = [self.intervals]
        if isinstance(self.window_sizes, int):
            self.window_sizes = [self.window_sizes]


class BitgetFuturesTorchTradingEnv(BitgetBaseTorchTradingEnv):
    """
    Bitget Futures trading environment with 3-action discrete space.

    This environment supports long and short positions with a simple action space:
    - Action 0 (-1.0): Go SHORT (or close if in position)
    - Action 1 (0.0): HOLD / CLOSE position
    - Action 2 (1.0): Go LONG (or close if in position)

    The environment uses market orders for execution and supports configurable leverage.

    Account State (10 elements):
    [cash, position_size, position_value, entry_price, current_price,
     unrealized_pnl_pct, leverage, margin_ratio, liquidation_price, holding_time]
    """

    def __init__(
        self,
        config: BitgetFuturesTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        observer: Optional[BitgetObservationClass] = None,
        trader: Optional[BitgetFuturesOrderClass] = None,
    ):
        """
        Initialize the BitgetFuturesTorchTradingEnv.

        Args:
            config: Environment configuration
            api_key: Bitget API key
            api_secret: Bitget API secret
            api_passphrase: Bitget API passphrase (required!)
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured BitgetObservationClass
            trader: Optional pre-configured BitgetFuturesOrderClass
        """
        # Initialize base class (handles observer/trader, obs specs, portfolio value, etc.)
        super().__init__(config, api_key, api_secret, api_passphrase, feature_preprocessing_fn, observer, trader)

        # Define action space (environment-specific)
        self.action_levels = config.action_levels
        self.action_spec = Categorical(len(self.action_levels))

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""

        # Store old portfolio value for reward calculation
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

        # Get desired action level
        action_idx = tensordict.get("action", 0)
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        desired_action = self.action_levels[action_idx]

        # Calculate and execute trade if needed
        trade_info = self._execute_trade_if_needed(desired_action)

        if trade_info["executed"]:
            if trade_info["side"] == "buy":
                self.current_position = 1  # Long
            elif trade_info["side"] == "sell" and trade_info.get("closed_position"):
                self.current_position = 0  # Closed
            elif trade_info["side"] == "sell":
                self.current_position = -1  # Short

        # Wait for next time step
        self._wait_for_next_timestamp()

        # Get updated state
        new_portfolio_value = self._get_portfolio_value()
        next_tensordict = self._get_observation()

        # Calculate reward and check termination
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value, desired_action, trade_info)
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
            logger.error(f"{side.capitalize()} trade failed for {self.config.symbol}: quantity={quantity}, error={e}")
            return self._create_trade_info(executed=True, success=False)

    def _handle_close_action(self, current_qty: float) -> Dict:
        """Handle close position action."""
        if current_qty == 0:
            return self._create_trade_info(executed=False)

        success = self.trader.close_position()
        side = "sell" if current_qty > 0 else "buy"
        self.current_position = 0

        return self._create_trade_info(
            executed=True,
            quantity=abs(current_qty),
            side=side,
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

        return self._execute_market_order("buy", self.config.quantity_per_trade)

    def _handle_short_action(self, current_qty: float) -> Dict:
        """Handle go short action."""
        # Close long position if necessary
        if current_qty > 0:
            self.trader.close_position()

        # Only execute if not already short
        if current_qty < 0:
            return self._create_trade_info(executed=False)

        return self._execute_market_order("sell", self.config.quantity_per_trade)

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

    print("Testing BitgetFuturesTorchTradingEnv...")

    # Create environment configuration
    config = BitgetFuturesTradingEnvConfig(
        symbol="BTCUSDT",
        demo=True,
        intervals=["1m"],
        window_sizes=[10],
        execute_on="1m",
        leverage=5,
        quantity_per_trade=0.002,  # Adjust based on Bitget minimums
        include_base_features=False,
    )

    try:
        # Create environment
        env = BitgetFuturesTorchTradingEnv(
            config,
            api_key=os.getenv("BITGET_API_KEY", ""),
            api_secret=os.getenv("BITGET_SECRET", ""),
            api_passphrase=os.getenv("BITGET_PASSPHRASE", ""),
        )

        print(f"✓ Environment created")
        print(f"  Action space size: {env.action_spec.n}")
        print(f"  Action levels: {env.action_levels}")

        # Test reset
        print("\n✓ Testing reset...")
        td = env.reset()
        print(f"  Observation keys: {list(td.keys())}")
        print(f"  Account state shape: {td['account_state'].shape}")

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
