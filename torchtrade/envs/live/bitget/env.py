from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Dict
import logging

import torch
from tensordict import TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.live.bitget.observation import BitgetObservationClass
from torchtrade.envs.live.bitget.order_executor import (
    BitgetFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.bitget.base import BitgetBaseTorchTradingEnv
from torchtrade.envs.utils.fractional_sizing import (
    calculate_fractional_position,
    PositionCalculationParams,
)


logger = logging.getLogger(__name__)


@dataclass
class BitgetFuturesTradingEnvConfig:
    """Configuration for Bitget Futures Trading Environment."""

    symbol: str = "BTCUSDT"

    # Timeframes and windows
    time_frames: Union[List[Union[str, "TimeFrame"]], Union[str, "TimeFrame"]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, "TimeFrame"] = "1Hour"  # Timeframe for trade execution timing

    # Trading parameters
    product_type: str = "USDT-FUTURES"  # V2 API: USDT-FUTURES, COIN-FUTURES, USDC-FUTURES
    leverage: int = 1  # Leverage (1-125)
    margin_mode: MarginMode = MarginMode.ISOLATED
    position_mode: PositionMode = PositionMode.ONE_WAY  # ONE_WAY or HEDGE

    # Action space configuration
    action_levels: List[float] = None  # Custom action levels, or None for defaults

    # Reward settings
    position_penalty: float = 0.0001  # Penalty for holding positions

    # Termination settings
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1  # 10% of initial balance

    # Environment settings
    demo: bool = True  # Use testnet for demo
    seed: Optional[int] = 42
    include_base_features: bool = False
    close_position_on_reset: bool = False  # Whether to close positions on env.reset()

    def __post_init__(self):
        # Normalize timeframes using utility function
        from torchtrade.envs.live.bitget.utils import normalize_bitget_timeframe_config
        self.execute_on, self.time_frames, self.window_sizes = normalize_bitget_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

        # Build default action levels for fractional mode
        if self.action_levels is None:
            self.action_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]  # Standard fractional with long/short


class BitgetFuturesTorchTradingEnv(BitgetBaseTorchTradingEnv):
    """
    TorchRL environment for Bitget Futures live trading.

    Supports:
    - Long and short positions
    - Configurable leverage (1x-125x)
    - Multiple timeframe observations
    - Demo (testnet) trading
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
        config: BitgetFuturesTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
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
            reward_function: Optional reward function (default: log_return_reward)
            observer: Optional pre-configured BitgetObservationClass
            trader: Optional pre-configured BitgetFuturesOrderClass
        """
        # Initialize base class (handles observer/trader, obs specs, portfolio value, etc.)
        super().__init__(config, api_key, api_secret, api_passphrase, feature_preprocessing_fn, observer, trader)

        # Set reward function (default to log return reward)
        from torchtrade.envs.core.default_rewards import log_return_reward
        self.reward_function = reward_function or log_return_reward

        # Define action space (environment-specific)
        self.action_levels = config.action_levels
        self.action_spec = Categorical(len(self.action_levels))

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""

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

        if trade_info["executed"] and trade_info.get("success") is not False:
            if trade_info["side"] == "buy":
                self.position.current_position = 1  # Long
            elif trade_info["side"] == "sell" and trade_info.get("closed_position"):
                self.position.current_position = 0  # Closed
            elif trade_info["side"] == "sell":
                self.position.current_position = -1  # Short
            self.position.current_action_level = desired_action

        # Wait for next time step
        self._wait_for_next_timestamp()

        # Get updated state
        new_portfolio_value = self._get_portfolio_value()
        next_tensordict = self._get_observation()

        # Record step history FIRST (reward function needs updated history!)
        self.history.record_step(
            price=current_price,
            action=desired_action,
            reward=0.0,  # Placeholder, will be set after reward calculation
            portfolio_value=new_portfolio_value,
            position=position_size
        )

        # Calculate reward using UPDATED history tracker
        reward = float(self.reward_function(self.history))

        # Update the reward in history
        self.history.rewards[-1] = reward

        # Check termination
        done = self._check_termination(new_portfolio_value)

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
            return self._create_trade_info(executed=False, success=False)

    def _handle_close_action(self, current_qty: float) -> Dict:
        """Handle close position action."""
        if current_qty == 0:
            return self._create_trade_info(executed=False)

        try:
            success = self.trader.close_position()
        except Exception as e:
            logger.error(f"Close position failed for {self.config.symbol}: {e}")
            return self._create_trade_info(executed=False, success=False)

        side = "sell" if current_qty > 0 else "buy"

        if success:
            self.position.current_position = 0

        return self._create_trade_info(
            executed=True,
            quantity=abs(current_qty),
            side=side,
            success=success,
            closed_position=True,
        )


    def _calculate_fractional_position(self, action_value: float, current_price: float) -> tuple[float, float, str]:
        """Calculate target position size from fractional action.

        Uses shared utility function for consistent position sizing across all environments.
        This fixes the fee calculation bug in the previous implementation.

        Args:
            action_value: Action from [-1.0, 1.0] representing fraction of balance
            current_price: Current market price

        Returns:
            Tuple of (position_size, notional_value, side):
            - position_size: Target position quantity (positive=long, negative=short, 0=flat)
            - notional_value: Absolute value in quote currency
            - side: "long", "short", or "flat"
        """
        if action_value == 0.0:
            return 0.0, 0.0, "flat"

        # Get actual balance from exchange
        # Use total_margin_balance (not available_balance) so the target reflects
        # the full portfolio, including margin already locked in open positions.
        balance_info = self.trader.get_account_balance()
        total_balance = balance_info.get('total_margin_balance', 0.0)

        if total_balance <= 0:
            logger.warning("No balance for fractional position sizing")
            return 0.0, 0.0, "flat"

        # Use shared utility for core position calculation
        # Reserve 2% buffer for exchange maintenance margin requirements
        effective_balance = total_balance * 0.98
        fee_rate = 0.0002  # Bitget futures maker/taker fee
        params = PositionCalculationParams(
            balance=effective_balance,
            action_value=action_value,
            current_price=current_price,
            leverage=self.config.leverage,
            transaction_fee=fee_rate,
        )
        position_size, notional_value, side = calculate_fractional_position(params)

        return position_size, notional_value, side

    def _execute_fractional_action(self, action_value: float) -> Dict:
        """Execute action using fractional position sizing.

        Args:
            action_value: Fractional action value in [-1.0, 1.0]

        Returns:
            trade_info: Dict with execution details
        """
        # Get current position and price from exchange
        current_qty = self._get_current_position_quantity()
        current_price = self.trader.get_mark_price()

        # Special case: Close to flat
        if action_value == 0.0:
            if abs(current_qty) > 0:
                return self._handle_close_action(current_qty)
            else:
                return self._create_trade_info(executed=False)

        # Calculate target position
        target_qty, _, _ = self._calculate_fractional_position(action_value, current_price)

        # Calculate delta (what we need to trade)
        delta_qty = target_qty - current_qty

        # Tolerance check
        # TODO: Query actual minimum quantity from Bitget exchange info instead of hardcoding
        # Consider adding similar methods to Binance's _get_symbol_info(), _get_step_size(), _get_min_notional()
        min_qty = 0.001  # Minimum tradeable quantity (hardcoded)
        if abs(delta_qty) < min_qty:
            # Already at target
            return self._create_trade_info(executed=False)

        # Determine side and amount
        if delta_qty > 0:
            # Need to buy
            side = "buy"
            amount = abs(delta_qty)
        else:
            # Need to sell
            side = "sell"
            amount = abs(delta_qty)

        # Floor to step size to avoid exceeding available margin
        step_size = min_qty  # TODO: Query actual step size from Bitget exchange info
        amount = int(amount / step_size) * step_size

        if amount < min_qty:
            return self._create_trade_info(executed=False)

        # Execute market order
        return self._execute_market_order(side, amount)

    def _execute_trade_if_needed(self, desired_action: float) -> Dict:
        """Execute trade based on desired action value.

        Skips execution if already in the requested position direction.

        Args:
            desired_action: Fractional action value in [-1.0, 1.0]

        Returns:
            trade_info: Dict with execution details
        """
        if desired_action == self.position.current_action_level:
            return self._create_trade_info(executed=False)

        return self._execute_fractional_action(desired_action)


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
        symbol="BTC/USDT:USDT",  # CCXT perpetual swap format
        demo=True,
        time_frames=["1m"],
        window_sizes=[10],
        execute_on="1m",
        leverage=5,
        action_levels=[-1.0, -0.5, 0.0, 0.5, 1.0],  # Fractional position sizing
        include_base_features=False,
    )

    try:
        # Create environment
        env = BitgetFuturesTorchTradingEnv(
            config,
            api_key=os.getenv("BITGETACCESSAPIKEY", ""),
            api_secret=os.getenv("BITGETSECRETKEY", ""),
            api_passphrase=os.getenv("BITGETPASSPHRASE", ""),
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
