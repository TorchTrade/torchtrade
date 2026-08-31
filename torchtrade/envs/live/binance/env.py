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
    BinanceFuturesOrderClass,
    MarginMode,
)
from torchtrade.envs.live.binance.base import BinanceBaseTorchTradingEnv
from torchtrade.envs.utils.fractional_sizing import (
    validate_action_levels,
    build_default_action_levels,
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
        self, action_value: float, current_price: float
    ) -> tuple[float, float, str]:
        """The shared sizing, plus the two things only binance does at this stage.

        The other three venues apply their min-qty floor and lot-size rounding to the
        DELTA inside `_execute_fractional_action`; binance refuses below min-NOTIONAL and
        rounds the TARGET here. Both are real, neither is the other's bug.
        """
        position_size, notional_value, side = super()._calculate_fractional_position(
            action_value, current_price
        )
        if side == "flat":
            return position_size, notional_value, side

        # Below the venue minimum the position is NOT opened, rather than rounded up:
        # rounding up would allocate beyond what the action asked for.
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
        return position_qty * (1 if position_size > 0 else -1), notional_value, side

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

        # 8. Decide what will be SENT before validating anything, because a direction
        #    switch does not send the delta -- it closes, then opens `abs(target_qty)`.
        #    Validating the delta and submitting the target is how an order the venue
        #    rejects gets reported as executed: on a switch the signs oppose, so
        #    `|delta| = |target| + |current|` and the larger number clears a minimum the
        #    smaller one does not.
        switching = (current_qty > 0 and target_qty < 0) or (current_qty < 0 and target_qty > 0)
        if switching:
            side, amount = ("buy" if target_qty > 0 else "sell"), abs(target_qty)
        elif delta > 0:
            side, amount = "buy", abs(delta)          # increase, or open long from flat
        elif delta < 0:
            side, amount = "sell", abs(delta)         # decrease, or open short from flat
        else:
            return self._create_trade_info(executed=False)

        # 9. Validate the submitted quantity, and do it BEFORE the switch's close. After
        #    the close it is too late to refuse: the position is already gone and
        #    returning "not executed" would describe an account that just changed.
        min_notional = self._get_min_notional()
        if amount * current_price < min_notional:
            # `at_target` ONLY when not switching. It means "already there", and
            # `_record_position_after_trade` takes it as permission to write
            # `current_action_level = desired_action`. On a switch we are on the OPPOSITE
            # side at full size, so claiming it latches the level to a direction the
            # account does not hold -- the next sync sees cached and observed agree, never
            # repairs it, and the duplicate-action guard then suppresses every retry of
            # that direction for the rest of the episode, including once price makes the
            # order legal. Refusing an order is cheap; refusing it forever is not.
            return self._create_trade_info(executed=False, at_target=not switching)

        # 10. A switch closes first. If the close fails, do not open the opposite side --
        #     that would double the position rather than reverse it.
        if switching:
            close_info = self._handle_close_action(current_qty)
            if not close_info["executed"] or close_info.get("success") is False:
                logger.warning("Direction switch failed: unable to close current position")
                return close_info

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
