from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable
import logging

import torch

logger = logging.getLogger(__name__)
from torchtrade.envs.utils.timeframe import TimeFrame, TimeFrameUnit
from torchtrade.envs.live.alpaca.utils import normalize_alpaca_timeframe_config
from torchtrade.envs.live.alpaca.observation import AlpacaObservationClass
from torchtrade.envs.live.alpaca.order_executor import AlpacaOrderClass, TradeMode
from tensordict import TensorDictBase
from torchrl.data import Categorical
from torchtrade.envs.live.alpaca.base import AlpacaBaseTorchTradingEnv

@dataclass
class AlpacaTradingEnvConfig:
    symbol: str = "BTC/USD"
    action_levels: Optional[List[float]] = None  # Custom action levels (defaults based on mode)
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, TimeFrame] = "1Hour"  # On which timeframe to execute trades
    position_penalty: float = 0.0001  # Penalty for holding positions
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1  # 10% of initial balance
    paper: bool = True
    trade_mode: TradeMode = "notional"
    seed: Optional[int] = 42
    include_base_features: bool = False # Includes base features such as timestamps and ohlc to the tensordict

    def __post_init__(self):
        self.execute_on, self.time_frames, self.window_sizes = normalize_alpaca_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

        # Build default action levels for fractional mode
        if self.action_levels is None:
            self.action_levels = [0.0, 0.5, 1.0]  # Long-only fractional

class AlpacaTorchTradingEnv(AlpacaBaseTorchTradingEnv):
    """
    TorchRL environment for Alpaca live spot trading (long-only).

    Supports:
    - Long positions only (no shorting)
    - Paper trading via Alpaca paper account
    - Multiple timeframe observations
    - Query-first pattern for reliable position tracking

    Action Space (Fractional Mode - Default):
    --------------------------------------
    Actions represent the fraction of total portfolio to have invested in the asset.
    Action values in range [0.0, 1.0] (long-only, no shorting):

    - action = 0.0: 0% invested (all cash, no position)
    - action = 0.5: 50% invested (half portfolio in asset, half cash)
    - action = 1.0: 100% invested (all-in, no cash)

    Position sizing formula:
        target_position_value = total_portfolio × action
        position_qty = target_position_value / price

    Default action_levels: [0.0, 0.5, 1.0]
    Custom levels supported: e.g., [0, 0.25, 0.5, 0.75, 1.0]

    Note: This is spot trading (no leverage). Unlike futures environments,
    there is no leverage parameter (implicitly 1x).

    **Dynamic Position Sizing** (not currently implemented):
    Similar to futures environments, action levels control position fractions.
    For more complex strategies, multi-dimensional actions could be added,
    but the current design is recommended for simplicity.
    """

    def __init__(
        self,
        config: AlpacaTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        observer: Optional[AlpacaObservationClass] = None,
        trader: Optional[AlpacaOrderClass] = None,
    ):
        """
        Initialize the AlpacaTorchTradingEnv.

        Args:
            config: Environment configuration
            api_key: Alpaca API key (not required if observer and trader are provided)
            api_secret: Alpaca API secret (not required if observer and trader are provided)
            feature_preprocessing_fn: Optional custom preprocessing function
            reward_function: Optional reward function (default: log_return_reward)
            observer: Optional pre-configured AlpacaObservationClass for dependency injection
            trader: Optional pre-configured AlpacaOrderClass for dependency injection
        """
        # Initialize base class (handles observer/trader, obs specs, portfolio value, etc.)
        super().__init__(config, api_key, api_secret, feature_preprocessing_fn, observer, trader)

        # Set reward function
        from torchtrade.envs.core.default_rewards import log_return_reward
        self.reward_function = reward_function or log_return_reward

        # Define action space (environment-specific)
        self.action_levels = config.action_levels
        self.action_spec = Categorical(len(self.action_levels))

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one environment step."""

        # Get desired action and current position
        desired_action = self.action_levels[tensordict.get("action", 0)]

        # Get current price
        status = self.trader.get_status()
        position_status = status.get("position_status", None)
        current_price = self._get_current_price(position_status)

        # Calculate and execute trade if needed
        trade_info = self._execute_trade_if_needed(desired_action)

        if trade_info["executed"]:
            self.position.current_position = 1 if trade_info["side"] == "buy" else 0

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

        # Record step history FIRST (reward function needs updated history!)
        self.history.record_step(
            price=current_price,
            action=desired_action,
            reward=0.0,  # Placeholder, will be set after reward calculation
            portfolio_value=new_portfolio_value
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


    def _execute_trade_if_needed(self, desired_action: float) -> Dict:
        """Execute trade based on desired action value.

        Skips execution if already in the requested position direction.

        Args:
            desired_action: Action value (interpretation depends on mode)

        Returns:
            trade_info: Dict with execution details
        """
        # Skip if already in the requested position
        current_pos = self.position.current_position
        if desired_action > 0 and current_pos == 1:
            return self._create_trade_info(executed=False)
        if desired_action == 0 and current_pos == 0:
            return self._create_trade_info(executed=False)

        return self._execute_fractional_action(desired_action)

    def _get_current_price(self, position_status=None) -> float:
        """Get current market price with fallback chain.

        Tries multiple sources in order:
        1. Position status (if provided or fetched)
        2. Trader's current_price attribute (for mocks)
        3. Observer's get_current_price() (fetches from market data)

        Args:
            position_status: Optional position status to avoid redundant queries

        Returns:
            Current price, or 0.0 if unavailable
        """
        # Try position status first
        if position_status is None:
            position_status = self.trader.get_status().get("position_status", None)

        current_price = position_status.current_price if position_status else 0.0

        # Fallback 1: trader's current_price attribute (for mocks)
        if current_price <= 0 and hasattr(self.trader, 'current_price'):
            current_price = self.trader.current_price

        # Fallback 2: fetch from market data
        if current_price <= 0:
            try:
                current_price = self.observer.get_current_price()
                logger.info(f"Fetched current price from market data: {current_price}")
            except Exception as e:
                logger.warning(f"Could not fetch current price: {e}")

        return current_price

    def _calculate_fractional_position(
        self, action_value: float, current_price: float
    ) -> tuple[float, float]:
        """Calculate target position value from fractional action.

        Note: Alpaca uses a portfolio allocation model that differs from the shared
        fractional_sizing utility. The shared utility is designed for balance allocation
        with leverage (action represents fraction of balance for new position), while
        Alpaca uses portfolio allocation (action represents target % of total portfolio).
        This is more appropriate for spot long-only trading without leverage.

        Args:
            action_value: Action from [0.0, 1.0] representing fraction to invest
            current_price: Current market price

        Returns:
            (target_value, target_qty): Target position value in USD and quantity
        """
        # For long-only: action represents fraction of total portfolio to have invested
        # action=0.0 → 0% invested (all cash)
        # action=1.0 → 100% invested (all in position)

        # Get actual balance from exchange (source of truth)
        account = self.trader.client.get_account()
        cash = float(account.cash)

        # Get current position value
        position_status = self.trader.get_status().get("position_status", None)
        current_position_value = float(position_status.market_value) if position_status else 0.0

        # Total portfolio value
        total_portfolio = cash + current_position_value

        # Validate portfolio value
        if total_portfolio <= 0:
            logger.warning("No portfolio value available for fractional position sizing")
            return 0.0, 0.0

        # Target position value based on action
        target_position_value = total_portfolio * action_value

        # Convert to quantity
        target_qty = target_position_value / current_price if current_price > 0 else 0.0

        return target_position_value, target_qty

    def _execute_fractional_action(self, action_value: float) -> Dict:
        """Execute action using fractional position sizing.

        Args:
            action_value: Fractional action value in [0.0, 1.0]

        Returns:
            trade_info: Dict with execution details
        """
        trade_info = {"executed": False, "amount": 0, "side": None, "success": None}

        # Get current position and price from exchange
        position_status = self.trader.get_status().get("position_status", None)
        current_qty = float(position_status.qty) if position_status else 0.0
        current_price = self._get_current_price(position_status)

        if current_price <= 0:
            logger.error(
                f"Cannot execute trade: invalid or missing price data. "
                f"price={current_price}, has_position={position_status is not None}, "
                f"has_trader_price={hasattr(self.trader, 'current_price')}"
            )
            return trade_info

        # Calculate target position
        target_value, target_qty = self._calculate_fractional_position(action_value, current_price)

        # Calculate delta (what we need to trade)
        delta_qty = target_qty - current_qty

        # Tolerance check to avoid tiny trades
        min_trade_value = 1.0  # $1 minimum trade
        delta_value = abs(delta_qty * current_price)

        if delta_value < min_trade_value:
            # Already at target position (within tolerance)
            return trade_info

        # Determine trade side and amount
        if delta_qty > 0:
            # Need to buy
            side = "buy"
            amount = delta_qty * current_price  # Dollar amount to buy
        else:
            # Need to sell
            side = "sell"
            if action_value == 0.0:
                # Special case: close entire position
                amount = -1  # Alpaca's special value for "sell all"
            else:
                # Partial sell
                amount = abs(delta_qty)  # Quantity to sell

        # Execute trade
        try:
            success = self.trader.trade(side=side, amount=amount, order_type="market")
            trade_info.update({
                "executed": True,
                "amount": amount,
                "side": side,
                "success": success
            })
        except Exception as e:
            logger.error(
                f"Trade execution failed: {side} ${amount:.2f} - {str(e)}",
                exc_info=True
            )
            trade_info["success"] = False

        return trade_info

    def _calculate_trade_amount(self, side: str) -> float:
        """Calculate the dollar amount to trade (not used in fractional mode)."""
        raise NotImplementedError("_calculate_trade_amount is not used in fractional mode")

    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        if not self.config.done_on_bankruptcy:
            return False

        bankruptcy_threshold = self.config.bankrupt_threshold * self.initial_portfolio_value
        return portfolio_value < bankruptcy_threshold

    def _create_info_dict(self, portfolio_value: float, trade_info: Dict, action_value: float) -> Dict:
        """Create info dictionary for debugging."""
        portfolio_return = ((portfolio_value - self.initial_portfolio_value) / 
                        self.initial_portfolio_value)

        account = self.trader.client.get_account()
        cash = float(account.cash)
        position_status = self.trader.get_status().get("position_status", None)
        
        return {
            "portfolio_value": portfolio_value,
            "portfolio_return": portfolio_return,
            "cash": cash,
            "position_qty": position_status.qty if position_status else 0,
            "position_market_value": position_status.market_value if position_status else 0,
            "trade_executed": trade_info["executed"],
            "trade_amount": trade_info["amount"],
            "trade_success": trade_info["success"],
            "trade_side": trade_info["side"],
            "action": action_value,
            "trade_mode": self.trader.trade_mode,
        }




if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    # Load environment variables
    load_dotenv()
    # Create environment configuration
    config = AlpacaTradingEnvConfig(
        symbol="BTC/USD",
        paper=True,
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
            TimeFrame(1, TimeFrameUnit.Hour),
        ],
        window_sizes=[15, 10],
        execute_on=TimeFrame(1, TimeFrameUnit.Minute),
        include_base_features=True,
    )

    # Create environment
    env = AlpacaTorchTradingEnv(
        config, api_key=os.getenv("API_KEY"), api_secret=os.getenv("SECRET_KEY")
    )
    td = env.reset()
    print(td)