import math
from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Dict

import torch
from tensordict import TensorDictBase
from torchrl.data import Categorical

from torchtrade.envs.live.bitget.observation import BitgetObservationClass
from torchtrade.envs.live.bitget.order_executor import (
    TAKER_FEE,
    BitgetFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.bitget.base import BitgetBaseTorchTradingEnv
from torchtrade.envs.core.live import (
    ObservationFailurePolicy,
)
from torchtrade.envs.utils.fractional_sizing import (
    validate_action_levels,
    calculate_fractional_position,
    PositionCalculationParams,
)



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

    # Termination settings
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1  # 10% of initial balance

    # Environment settings
    demo: bool = True  # Use testnet for demo
    seed: Optional[int] = 42
    include_base_features: bool = False
    close_position_on_init: bool = True
    close_position_on_reset: bool = False
    observation_failure_policy: ObservationFailurePolicy | str = ObservationFailurePolicy.HALT

    def __post_init__(self):
        self.observation_failure_policy = ObservationFailurePolicy(self.observation_failure_policy)
        # Normalize timeframes using utility function
        from torchtrade.envs.live.bitget.utils import normalize_bitget_timeframe_config
        self.execute_on, self.time_frames, self.window_sizes = normalize_bitget_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

        # Build default action levels for fractional mode
        if self.action_levels is None:
            self.action_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]  # Standard fractional with long/short

        validate_action_levels(self.action_levels)


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

    Account State (6 elements; the list is ACCOUNT_STATE on the exchange base class):
    ---------------------------
    [exposure_pct, position_direction, unrealized_pnlpct, holding_time,
     leverage, distance_to_liquidation]
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
        status, position_status, current_price, position_size = self._acquire_pre_trade_state()

        self._sync_position_from_exchange(position_status)

        # Get desired action level
        action_idx = tensordict.get("action", 0)
        if isinstance(action_idx, torch.Tensor):
            action_idx = action_idx.item()
        desired_action = self.action_levels[action_idx]

        # Calculate and execute trade if needed
        trade_info = self._execute_trade_if_needed(desired_action)

        self._record_position_after_trade(desired_action, trade_info)

        # Wait for next time step
        self._wait_for_next_timestamp()

        # Get updated state
        new_portfolio_value, new_price, new_qty, next_tensordict = self._acquire_post_bar_state()
        # None when the account is flat: there is no position mark to read, and
        # fetching one would add a round-trip that can halt the episode. The
        # pre-trade price is the honest fallback -- flat rows carry no PnL anyway.
        new_price = new_price if new_price is not None else current_price

        # Record step history FIRST (reward function needs updated history!)
        self.history.record_step(
            price=new_price,
            action=desired_action,
            reward=0.0,  # Placeholder, will be set after reward calculation
            portfolio_value=new_portfolio_value,
            position=new_qty
        )

        # Calculate reward using UPDATED history tracker
        reward = float(self.reward_function(self.history))

        # Update the reward in history
        self.history.rewards[-1] = reward

        # Check termination
        done = self._check_termination(new_portfolio_value)

        next_tensordict.set("reward", torch.tensor([reward], dtype=torch.float))
        next_tensordict.set("done", torch.tensor([done], dtype=torch.bool))
        next_tensordict.set("terminated", torch.tensor([done], dtype=torch.bool))

        return next_tensordict


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
        # Indexed, and `not (x > 0)`: defaulting to 0.0 turned a broken adapter into a
        # permanent silent refusal to trade, and `<= 0` lets a NaN balance through to
        # size a NaN position (#277).
        total_balance = balance_info['total_margin_balance']

        # isfinite, not `not (x > 0)`: that catches NaN but passes +inf, and an inf
        # balance sizes an inf target -- bitget's amount rounding then yields NaN and
        # hands it to create_order. Same defect this PR fixed on the baselines (#277).
        if not math.isfinite(total_balance) or total_balance <= 0:
            raise ValueError(
                f"cannot size a trade against a portfolio value of {total_balance}"
            )

        # Use shared utility for core position calculation
        # Reserve 2% buffer for exchange maintenance margin requirements
        effective_balance = total_balance * 0.98
        params = PositionCalculationParams(
            balance=effective_balance,
            action_value=action_value,
            current_price=current_price,
            leverage=self.config.leverage,
            transaction_fee=TAKER_FEE,
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
        current_price = self._current_mark_price()

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

        # Query real min-order-size from Bitget market info (not hardcoded)
        min_qty = self.trader.get_lot_size()["min_qty"]

        if abs(delta_qty) < min_qty:
            # Already at target (delta below the minimum tradeable size)
            return self._create_trade_info(executed=False, at_target=True)

        side = "buy" if delta_qty > 0 else "sell"
        # Floor to the exchange lot step (CCXT truncates -> never exceeds margin)
        amount = self.trader._round_amount(abs(delta_qty))

        if amount < min_qty:
            return self._create_trade_info(executed=False, at_target=True)

        # Execute market order
        info = self._execute_market_order(side, amount)
        info["target_qty"] = target_qty
        info["target_tol"] = min_qty
        return info

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
