"""Bybit Futures TorchRL trading environment with fractional position sizing."""

from torchtrade.envs.utils.precision import decimals_for_step
from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Dict

from torchrl.data import Categorical

from torchtrade.envs.live.bybit.observation import BybitObservationClass
from torchtrade.envs.live.bybit.order_executor import (
    BybitFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.bybit.base import BybitBaseTorchTradingEnv
from torchtrade.envs.core.common import validate_unknown_status_budget
from torchtrade.envs.core.live import (
    ObservationFailurePolicy,
)
from torchtrade.envs.utils.fractional_sizing import (
    validate_action_levels,
)


@dataclass
class BybitFuturesTradingEnvConfig:
    """Configuration for Bybit Futures Trading Environment."""

    symbol: str = "BTCUSDT"

    # Timeframes and windows
    time_frames: Union[List[Union[str, "TimeFrame"]], Union[str, "TimeFrame"]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, "TimeFrame"] = "1Hour"

    # Trading parameters
    leverage: int = 1
    margin_mode: MarginMode = MarginMode.ISOLATED
    position_mode: PositionMode = PositionMode.ONE_WAY

    # Action space configuration
    action_levels: List[float] = None

    # Termination settings
    done_on_bankruptcy: bool = True
    bankrupt_threshold: float = 0.1

    # Environment settings
    demo: bool = True
    seed: Optional[int] = 42
    include_base_features: bool = False
    close_position_on_init: bool = True
    close_position_on_reset: bool = False
    observation_failure_policy: ObservationFailurePolicy | str = ObservationFailurePolicy.HALT
    # Bars to ride out an unreadable venue before truncating; 0 disables (#295).
    max_unknown_status_steps: int = 0

    def __post_init__(self):
        self.observation_failure_policy = ObservationFailurePolicy(self.observation_failure_policy)
        validate_unknown_status_budget(self.max_unknown_status_steps)
        from torchtrade.envs.live.bybit.utils import normalize_bybit_timeframe_config
        self.execute_on, self.time_frames, self.window_sizes = normalize_bybit_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

        if self.action_levels is None:
            self.action_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]

        validate_action_levels(self.action_levels)


class BybitFuturesTorchTradingEnv(BybitBaseTorchTradingEnv):
    """
    TorchRL environment for Bybit Futures live trading.

    Supports:
    - Long and short positions
    - Configurable leverage (1x-100x)
    - Multiple timeframe observations
    - Demo (testnet) trading
    - Fractional position sizing

    Action Space (Fractional Mode - Default):
    - action = -1.0: 100% short (all-in short)
    - action = -0.5: 50% short
    - action = 0.0: Market neutral (close all positions)
    - action = 0.5: 50% long
    - action = 1.0: 100% long (all-in long)

    Default action_levels: [-1.0, -0.5, 0.0, 0.5, 1.0]
    """

    def __init__(
        self,
        config: BybitFuturesTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        observer: Optional[BybitObservationClass] = None,
        trader: Optional[BybitFuturesOrderClass] = None,
    ):
        super().__init__(config, api_key, api_secret, feature_preprocessing_fn, observer, trader)

        from torchtrade.envs.core.default_rewards import log_return_reward
        self.reward_function = reward_function or log_return_reward

        self.action_levels = config.action_levels
        self.action_spec = Categorical(len(self.action_levels))


    def _execute_fractional_action(
        self, action_value: float, *, current_qty: float, current_price: float,
    ) -> Dict:
        """Execute action using fractional position sizing."""
        # Threaded from `_step`'s halted read; required, never defaulted (#295).
        if action_value == 0.0:
            if abs(current_qty) > 0:
                return self._handle_close_action(current_qty)
            return self._create_trade_info(executed=False)

        target_qty, _, _ = self._calculate_fractional_position(action_value, current_price)
        delta_qty = target_qty - current_qty

        lot_size = self.trader.get_lot_size()
        min_qty = lot_size["min_qty"]
        qty_step = lot_size["qty_step"]

        if abs(delta_qty) < min_qty:
            return self._create_trade_info(executed=False, at_target=True)

        side = "buy" if delta_qty > 0 else "sell"
        # round() to avoid float artifacts (e.g. 0.003000000000003). The decimals come
        # from Decimal, not from str(): a 1e-06 step has no "." in its repr, so the old
        # string check answered 0 and rounded 0.977 up to a whole unit (#278).
        amount = round(int(abs(delta_qty) / qty_step) * qty_step, decimals_for_step(qty_step))

        if amount < min_qty:
            return self._create_trade_info(executed=False, at_target=True)

        # The venue's notional floor, on the floored quantity actually sent (#414).
        min_notional = float(lot_size["min_notional"])
        if min_notional > 0 and amount * current_price < min_notional * (1 - 1e-9):
            return self._create_trade_info(executed=False, at_target=True)

        info = self._execute_market_order(side, amount)
        info["target_qty"] = target_qty
        info["target_tol"] = lot_size["min_qty"]
        return info

    def _execute_trade_if_needed(
        self, desired_action: float, *, current_qty: float, current_price: float,
    ) -> Dict:
        """Execute trade based on desired action value."""
        return self._execute_fractional_action(
            desired_action, current_qty=current_qty, current_price=current_price,
        )
