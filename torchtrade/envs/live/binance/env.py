import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Callable
import logging


logger = logging.getLogger(__name__)
from torchrl.data import Categorical

from torchtrade.envs.utils.timeframe import TimeFrame
from torchtrade.envs.core.common import validate_unknown_status_budget
from torchtrade.envs.core.live import (
    ObservationFailurePolicy,
)
from torchtrade.envs.live.binance.observation import BinanceObservationClass
from torchtrade.envs.live.binance.order_executor import (
    TAKER_FEE,
    BinanceFuturesOrderClass,
    MarginMode,
)
from torchtrade.envs.live.binance.base import BinanceBaseTorchTradingEnv
from torchtrade.envs.utils.fractional_sizing import (
    validate_action_levels,
    build_default_action_levels,
    calculate_fractional_position,
    PositionCalculationParams,
)


@dataclass
class BinanceFuturesTradingEnvConfig:
    """Configuration for Binance Futures Trading Environment."""

    symbol: str = "BTCUSDT"

    # Timeframes and windows
    time_frames: Union[List[Union[str, TimeFrame]], Union[str, TimeFrame]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, TimeFrame] = "1Hour"  # Timeframe for trade execution timing

    # Trading parameters
    leverage: int = 1  # Leverage (1-125)
    margin_mode: MarginMode = MarginMode.ISOLATED

    # Action space configuration (fractional mode only)
    action_levels: List[float] = None  # Custom action levels, or None for defaults

    # Termination settings
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1  # 10% of initial balance

    # Environment settings
    demo: bool = True  # Use demo/testnet for paper trading
    seed: Optional[int] = 42
    include_base_features: bool = False
    close_position_on_init: bool = True
    close_position_on_reset: bool = False
    observation_failure_policy: ObservationFailurePolicy | str = ObservationFailurePolicy.HALT
    # Bars to ride out an unreadable venue before truncating; 0 disables (#295).
    max_unknown_status_steps: int = 0

    def __post_init__(self):
        """Normalize timeframe configuration and build action levels."""
        from torchtrade.envs.live.binance.utils import normalize_binance_timeframe_config

        self.observation_failure_policy = ObservationFailurePolicy(self.observation_failure_policy)
        validate_unknown_status_budget(self.max_unknown_status_steps)

        self.execute_on, self.time_frames, self.window_sizes = normalize_binance_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

        # Build default action levels
        if self.action_levels is None:
            self.action_levels = build_default_action_levels(
                allow_short=True  # Futures allow short positions
            )

        validate_action_levels(self.action_levels)


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

    Default action_levels: [-1, 0, 1] (short / flat / long)
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
        config: BinanceFuturesTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
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
            reward_function: Optional reward function (default: log_return_reward)
            observer: Optional pre-configured BinanceObservationClass
            trader: Optional pre-configured BinanceFuturesOrderClass
        """
        # Initialize base class (handles observer/trader, obs specs, portfolio value, etc.)
        super().__init__(config, api_key, api_secret, feature_preprocessing_fn, observer, trader)

        # Set reward function
        from torchtrade.envs.core.default_rewards import log_return_reward
        self.reward_function = reward_function or log_return_reward

        # Define action space (environment-specific)
        self.action_levels = config.action_levels
        self.action_spec = Categorical(len(self.action_levels))


    def _get_symbol_info(self) -> Dict:
        """Get exchange symbol information for precision and lot size.

        Binance-specific implementation that queries futures_exchange_info() API.
        """
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

    def _get_min_notional(self) -> float:
        """Get the minimum notional value for orders."""
        symbol_info = self._get_symbol_info()
        for filter_item in symbol_info.get('filters', []):
            if filter_item['filterType'] == 'MIN_NOTIONAL':
                return float(filter_item.get('notional', 100))
        return 100.0  # Default fallback

    def _calculate_fractional_position(
        self,
        action_value: float,
        current_price: float
    ) -> tuple[float, float, str]:
        """Calculate position size from fractional action value for live trading.

        Uses shared utility function for consistent position sizing across all environments.
        Applies exchange-specific validation and rounding constraints.

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
        # Use total_margin_balance (not available_balance) so the target reflects
        # the full portfolio, including margin already locked in open positions.
        # available_balance only shows free margin, which shrinks as positions grow,
        # causing repeated buys when the agent keeps requesting action=1.0.
        # The VERDICT is inside the closure, not just the read. `_halting` catches
        # ValueError precisely so an impossible account state becomes a halt; raising one
        # frame above it sent that straight out of `_step`. `equity == 0.0` is what a
        # venue reports while liquidating you -- the worst moment to crash rather than
        # halt under policy (#295).
        def read_balance():
            info = self.trader.get_account_balance()
            total_balance = info["total_margin_balance"]
            # isfinite, not `not (x > 0)`: that catches NaN but passes +inf, and an inf
            # balance sizes an inf target (#277). The name is load-bearing:
            # test_futures_sizing_rejects_a_non_finite_balance greps for it.
            if not math.isfinite(total_balance) or total_balance <= 0:
                raise ValueError(
                    f"cannot size a trade against a portfolio value of {total_balance}"
                )
            return info

        balance_info = self._halting(read_balance, cache_key="balance")
        total_balance = balance_info["total_margin_balance"]

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

        # Apply exchange-specific validation
        # Check minimum notional requirement
        #
        # Edge case: If calculated position is below exchange minimum, we return "flat"
        # instead of rounding up to minimum. This means:
        #   - Agent selects small action (e.g., 0.1 = 10% allocation)
        #   - Calculation results in notional < min_notional
        #   - Position is NOT opened (returns flat)
        #   - Agent receives warning in logs but no position state change
        #
        # Alternative approaches considered:
        #   1. Round up to minimum notional → Could overallocate beyond action intent
        #   2. Expose rejection in observation → Would require state schema change
        #   3. Current: Fail gracefully with warning → Simple, predictable behavior
        min_notional = self._get_min_notional()
        if notional_value < min_notional:
            logger.warning(
                f"Action {action_value} resulted in notional {notional_value:.2f} "
                f"below exchange minimum {min_notional:.2f}. Position not opened."
            )
            return 0.0, 0.0, "flat"

        # Floor through the executor, which caches the venue's step and carries the
        # epsilon. This used to re-query futures_exchange_info() per sizing decision and
        # floor with a bare int(), which shaves a whole step off exact multiples --
        # 0.29 -> 0.28 (#271). One owner of lot-size knowledge, not two.
        position_qty = self.trader.round_quantity(abs(position_size))

        # Apply direction
        direction = 1 if position_size > 0 else -1
        position_size = position_qty * direction

        return position_size, notional_value, side

    def _execute_fractional_action(
        self, action_value: float, *, current_qty: float, current_price: float,
    ) -> Dict:
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
        # Threaded from `_step`'s halted read; required, never defaulted (#295).
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
        sign = 1 if delta > 0 else -1
        delta = self.trader.round_quantity(abs(delta)) * sign
        if delta == 0:
            return self._create_trade_info(executed=False)  # Already close enough

        # 8. Check delta notional meets exchange minimum
        delta_notional = abs(delta) * current_price
        min_notional = self._get_min_notional()
        if delta_notional < min_notional:
            return self._create_trade_info(executed=False, at_target=True)  # Delta too small for exchange

        # 9. Determine trade direction and execute
        if (current_qty > 0 and target_qty < 0) or (current_qty < 0 and target_qty > 0):
            # Direction switch: close current, then open opposite
            #
            # Edge case handling:
            #   1. If close fails → Return early, don't open opposite position
            #      This prevents doubling position size if close is rejected
            #   2. If close succeeds but open fails → Agent ends up flat instead of target
            #      Trade info will show close executed=True but may not reflect open failure
            #   3. Between close and open, account balance changes (from PnL)
            #      Target calculation uses current balance which may differ
            #
            # TODO: Consider tracking partial execution state for observation
            close_info = self._handle_close_action(current_qty)
            if not close_info["executed"] or close_info.get("success") is False:
                logger.warning("Direction switch failed: unable to close current position")
                return close_info

            # Open new position in opposite direction
            side, amount = ("buy" if target_qty > 0 else "sell"), abs(target_qty)
        elif delta > 0:
            side, amount = "buy", abs(delta)          # increase, or open long from flat
        elif delta < 0:
            side, amount = "sell", abs(delta)         # decrease, or open short from flat
        else:
            return self._create_trade_info(executed=False)

        # One exit, so a new branch cannot skip the target and disable the check.
        info = self._execute_market_order(side, amount)
        info["target_qty"] = target_qty
        info["target_tol"] = min_notional / current_price
        return info

    def _execute_trade_if_needed(
        self, desired_action: float, *, current_qty: float, current_price: float,
    ) -> Dict:
        """
        Execute trade if position change is needed.

        Skips execution if already in the requested position direction.

        Args:
            desired_action: Action level

        Returns:
            Dict with trade execution info
        """
        if desired_action == self.position.current_action_level:
            return self._create_trade_info(executed=False)

        return self._execute_fractional_action(
            desired_action, current_qty=current_qty, current_price=current_price,
        )
