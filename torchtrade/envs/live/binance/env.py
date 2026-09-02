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
from torchtrade.envs.live.shared.futures_config import BaseFuturesTradingConfig
from torchtrade.envs.live.binance.utils import normalize_binance_timeframe_config


@dataclass
class BinanceFuturesTradingEnvConfig(BaseFuturesTradingConfig):
    """Configuration for Binance Futures Trading Environment."""

    margin_mode: MarginMode = MarginMode.ISOLATED

    _normalize_timeframes = staticmethod(normalize_binance_timeframe_config)



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


    def _get_min_notional(self) -> float | None:
        """The venue's floor, from the executor's cached filters. One owner, not two.

        None means the floor is not known -- the caller must refuse, not assume zero.
        """
        raw = self.trader.get_lot_size()["min_notional"]
        return None if raw is None else float(raw)

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
        # rounding up would allocate beyond what the action asked for. An UNKNOWN floor
        # refuses the same way -- this is the second of two call sites, and guarding only
        # the other one left `float(None)` raising out of the live step here (#414).
        min_notional = self._get_min_notional()
        if min_notional is None:
            logger.warning(
                f"{self.config.symbol}: the venue minimum is not known; not opening "
                f"rather than assuming there is no floor"
            )
            return 0.0, 0.0, "flat"
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
        # Query actual position from exchange (source of truth)
        # Threaded from `_step`'s halted read; required, never defaulted (#295).
        if action_value == 0.0:
            if abs(current_qty) > 0:
                return self._handle_close_action(current_qty)
            return self._create_trade_info(executed=False)

        # Calculate target position
        target_qty, target_notional, target_side = self._calculate_fractional_position(
            action_value, current_price
        )

        # Check if target is achievable
        if target_qty == 0.0:
            return self._create_trade_info(executed=False)

        # Calculate delta
        delta = target_qty - current_qty

        # Check if delta is significant enough to trade
        sign = 1 if delta > 0 else -1
        delta = self.trader.round_quantity(abs(delta)) * sign
        if delta == 0:
            return self._create_trade_info(executed=False)  # Already close enough

        # Decide what will be SENT before validating: a switch closes, then opens
        # `abs(target_qty)`, not the delta. On a switch `|delta| = |target| + |current|`,
        # so validating the delta clears a floor the submitted quantity does not (#414).
        switching = (current_qty > 0 and target_qty < 0) or (current_qty < 0 and target_qty > 0)
        if switching:
            side, amount = ("buy" if target_qty > 0 else "sell"), abs(target_qty)
        elif delta > 0:
            side, amount = "buy", abs(delta)          # increase, or open long from flat
        elif delta < 0:
            side, amount = "sell", abs(delta)         # decrease, or open short from flat
        else:
            return self._create_trade_info(executed=False)

        # Before the switch's close: after it, refusing would describe an account that
        # has already changed.
        min_notional = self._get_min_notional()
        if min_notional is None:
            logger.warning(
                f"{self.config.symbol}: the venue minimum is not known; refusing rather "
                f"than assuming there is no floor"
            )
            return self._create_trade_info(executed=False, at_target=not switching)
        if amount * current_price < min_notional:
            # `at_target` means "already there"; on a switch we hold the opposite side
            # at full size, so claiming it latches the level to a direction we do not
            # hold and the duplicate-action guard suppresses every retry (#414).
            return self._create_trade_info(executed=False, at_target=not switching)

        # If the close fails, do not open the opposite side -- that doubles rather than
        # reverses.
        if switching:
            close_info = self._handle_close_action(current_qty)
            if not close_info["executed"] or close_info.get("success") is False:
                logger.warning("Direction switch failed: unable to close current position")
                return close_info

        # One exit, so a new branch cannot skip the target and disable the check.
        info = self._execute_market_order(side, amount)
        info["target_qty"] = target_qty
        # The venue's minimum tradeable size, as the other three venues pass. It was
        # `min_notional / price`, which the removed 100.0 fallback kept non-zero; with
        # the executor's real 0.0 it becomes 0.0, and `_sync_position_from_exchange`
        # then compares against POSITION_DUST_EPS (1e-9). binance's delta is
        # lot-rounded, so every complete fill would read as a divergence, NaN the
        # action level, and re-size every bar (#414).
        info["target_tol"] = float(self.trader.get_lot_size()["min_qty"])
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
