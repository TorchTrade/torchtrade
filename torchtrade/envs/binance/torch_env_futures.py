from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Callable
import logging
import warnings

import torch

logger = logging.getLogger(__name__)
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.timeframe import TimeFrame, TimeFrameUnit
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
    max_position: float = 1.0  # Maximum position size as fraction of balance

    # Timeframes and windows
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]] = "1Min"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, TimeFrame] = "1Min"  # Timeframe for trade execution timing

    # Trading parameters
    leverage: int = 1  # Leverage (1-125)
    margin_type: MarginType = MarginType.ISOLATED

    # Action space configuration
    position_sizing_mode: str = "fractional"  # "fractional" (new default) or "fixed" (legacy)
    action_levels: List[float] = None  # Custom action levels, or None for defaults

    # DEPRECATED: Only used in legacy "fixed" mode
    quantity_per_trade: float = 0.001  # DEPRECATED - only used when position_sizing_mode="fixed"
    trade_mode: TradeMode = TradeMode.QUANTITY  # DEPRECATED - only used in fixed mode
    include_hold_action: bool = True  # DEPRECATED - only used when position_sizing_mode="fixed"

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

    def __post_init__(self):
        """Normalize timeframe configuration and build action levels."""
        from torchtrade.envs.binance.utils import normalize_binance_timeframe_config

        self.execute_on, self.time_frames, self.window_sizes = normalize_binance_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

        # Set default action levels based on position sizing mode
        if self.action_levels is None:
            if self.position_sizing_mode == "fractional":
                # New default: fractional sizing with neutral at 0
                self.action_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
            else:
                # Legacy mode: maintain backward compatibility
                if self.include_hold_action:
                    self.action_levels = [-1.0, 0.0, 1.0]  # Short, Hold, Long
                else:
                    self.action_levels = [-1.0, 1.0]  # Short, Long

        # Validate position_sizing_mode
        if self.position_sizing_mode not in ["fractional", "fixed"]:
            raise ValueError(f"position_sizing_mode must be 'fractional' or 'fixed', got '{self.position_sizing_mode}'")


class BinanceFuturesTorchTradingEnv(BinanceBaseTorchTradingEnv):
    """
    TorchRL environment for Binance Futures live trading.

    Supports:
    - Long and short positions
    - Configurable leverage (1x-125x)
    - Multiple timeframe observations
    - Demo (paper) trading via Binance testnet
    - Query-first pattern for reliable position tracking

    Action Space (Fractional Mode - Default):
    --------------------------------------
    Actions represent the fraction of available balance to allocate to a position.
    Action values in range [-1.0, 1.0]:

    - action = -1.0: 100% short (all-in short)
    - action = -0.5: 50% short
    - action = 0.0: Market neutral (close all positions, stay in cash)
    - action = 0.5: 50% long
    - action = 1.0: 100% long (all-in long)

    Position sizing formula:
        position_size = (balance × |action| × leverage) / price
        (rounded to exchange step size)

    Default action_levels: [-1.0, -0.5, 0.0, 0.5, 1.0]
    Custom levels supported: e.g., [-1, -0.3, -0.1, 0, 0.1, 0.3, 1]

    Leverage Design:
    ----------------
    Leverage is a **fixed global parameter** (not part of action space).
    See SeqFuturesEnv documentation for rationale on fixed vs dynamic leverage.

    **Dynamic Leverage** (not currently implemented):
    Could be implemented as multi-dimensional actions if needed, but fixed
    leverage is recommended for most use cases.

    Account State (10 elements):
    ---------------------------
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

    def _get_symbol_info(self) -> Dict:
        """Get exchange symbol information for precision and lot size."""
        try:
            exchange_info = self.trader.client.futures_exchange_info()
            for symbol in exchange_info['symbols']:
                if symbol['symbol'] == self.config.symbol:
                    return symbol
            raise ValueError(f"Symbol {self.config.symbol} not found in exchange info")
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            # Return defaults if exchange query fails
            return {
                'filters': [
                    {'filterType': 'LOT_SIZE', 'stepSize': '0.001'},
                    {'filterType': 'MIN_NOTIONAL', 'notional': '100'}
                ]
            }

    def _get_step_size(self) -> float:
        """Get the step size (lot size) for the trading symbol."""
        symbol_info = self._get_symbol_info()
        for filter_item in symbol_info.get('filters', []):
            if filter_item['filterType'] == 'LOT_SIZE':
                return float(filter_item['stepSize'])
        return 0.001  # Default fallback

    def _get_min_notional(self) -> float:
        """Get the minimum notional value for orders."""
        symbol_info = self._get_symbol_info()
        for filter_item in symbol_info.get('filters', []):
            if filter_item['filterType'] == 'MIN_NOTIONAL':
                return float(filter_item.get('notional', 100))
        return 100.0  # Default fallback

    def _round_to_step_size(self, quantity: float) -> float:
        """Round quantity to exchange step size."""
        step_size = self._get_step_size()
        if step_size == 0:
            return quantity
        return round(quantity / step_size) * step_size

    def _calculate_fractional_position(
        self,
        action_value: float,
        current_price: float
    ) -> tuple[float, float, str]:
        """Calculate position size from fractional action value for live trading.

        This implementation queries actual balance from exchange and accounts
        for exchange rounding constraints.

        Args:
            action_value: Action from [-1.0, 1.0] representing fraction of balance
            current_price: Current market price

        Returns:
            Tuple of (position_size, notional_value, side):
            - position_size: Quantity rounded to exchange step size
            - notional_value: Absolute value in quote currency
            - side: "long", "short", or "flat"
        """
        # Handle neutral case
        if action_value == 0.0:
            return 0.0, 0.0, "flat"

        # Query actual balance from exchange
        balance_info = self.trader.get_account_balance()
        available_balance = balance_info.get('available_balance', 0.0)

        if available_balance <= 0:
            logger.warning("No available balance for fractional position sizing")
            return 0.0, 0.0, "flat"

        # Calculate fraction and direction
        fraction = abs(action_value)
        direction = 1 if action_value > 0 else -1

        # Allocate fraction of balance, accounting for fees
        capital_allocated = available_balance * fraction
        fee_rate = 0.0004  # Binance futures maker/taker fee
        fee_multiplier = 1 + (self.config.leverage * fee_rate)
        margin_required = capital_allocated / fee_multiplier

        # Calculate notional value with leverage
        notional_value = margin_required * self.config.leverage

        # Check minimum notional
        min_notional = self._get_min_notional()
        if notional_value < min_notional:
            logger.warning(f"Notional {notional_value} below minimum {min_notional}")
            return 0.0, 0.0, "flat"

        # Calculate position size
        position_qty = notional_value / current_price

        # Round to exchange step size
        position_qty = self._round_to_step_size(position_qty)

        # Apply direction
        position_size = position_qty * direction
        side = "long" if direction > 0 else "short"

        return position_size, notional_value, side

    def _execute_fractional_action(self, action_value: float) -> Dict:
        """Execute action using fractional position sizing with query-first pattern.

        This implementation:
        1. Queries actual position from exchange (source of truth)
        2. Calculates target based on actual balance
        3. Rounds to exchange constraints
        4. Only trades the delta
        5. Uses exchange close_position() API for flat

        Args:
            action_value: Fractional action value in [-1.0, 1.0]

        Returns:
            trade_info: Dict with execution details
        """
        # 1. Query actual position from exchange (source of truth)
        current_qty = self._get_current_position_quantity()
        current_price = self.trader.get_mark_price()

        # 2. Special case: Close to flat
        if action_value == 0.0:
            if abs(current_qty) > 0:
                return self._handle_close_action(current_qty)
            return self._create_trade_info(executed=False)

        # 3. Calculate target position
        target_qty, target_notional, target_side = self._calculate_fractional_position(
            action_value, current_price
        )

        # 4. Check if target is achievable
        if target_qty == 0.0:
            return self._create_trade_info(executed=False)

        # 5. Calculate delta
        delta = target_qty - current_qty

        # 6. Check if delta is significant enough to trade
        step_size = self._get_step_size()
        if abs(delta) < step_size:
            return self._create_trade_info(executed=False)  # Already close enough

        # 7. Round delta to step size
        delta = self._round_to_step_size(delta)

        # 8. Determine trade direction and execute
        if (current_qty > 0 and target_qty < 0) or (current_qty < 0 and target_qty > 0):
            # Direction switch: close current, then open opposite
            close_info = self._handle_close_action(current_qty)
            if not close_info["executed"]:
                return close_info

            # Open new position in opposite direction
            side = "BUY" if target_qty > 0 else "SELL"
            return self._execute_market_order(side, abs(target_qty))

        elif delta > 0:
            # Increasing position (or opening long from flat)
            return self._execute_market_order("BUY", abs(delta))

        elif delta < 0:
            # Decreasing position (or opening short from flat)
            return self._execute_market_order("SELL", abs(delta))

        return self._create_trade_info(executed=False)

    def _execute_trade_if_needed(self, desired_action: float) -> Dict:
        """
        Execute trade if position change is needed.

        Routes to fractional or fixed position sizing based on config.

        Args:
            desired_action: Action level

        Returns:
            Dict with trade execution info
        """
        if self.config.position_sizing_mode == "fractional":
            # NEW: Fractional position sizing
            return self._execute_fractional_action(desired_action)
        else:
            # LEGACY: Fixed position sizing
            return self._execute_fixed_action(desired_action)

    def _execute_fixed_action(self, desired_action: float) -> Dict:
        """Execute action using fixed position sizing (legacy mode)."""
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
