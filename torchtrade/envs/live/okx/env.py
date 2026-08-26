"""OKX Futures TorchRL trading environment with fractional position sizing."""
import math
from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Dict

from torchrl.data import Categorical

from torchtrade.envs.live.okx.observation import OKXObservationClass
from torchtrade.envs.live.okx.order_executor import (
    TAKER_FEE,
    OKXFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.okx.base import OKXBaseTorchTradingEnv
from torchtrade.envs.core.common import validate_unknown_status_budget
from torchtrade.envs.core.live import (
    ObservationFailurePolicy,
)
from torchtrade.envs.utils.fractional_sizing import (
    validate_action_levels,
    calculate_fractional_position,
    PositionCalculationParams,
)


@dataclass
class OKXFuturesTradingEnvConfig:
    """Configuration for OKX Futures Trading Environment."""

    symbol: str = "BTC-USDT-SWAP"

    # Timeframes and windows
    time_frames: Union[List[Union[str, "TimeFrame"]], Union[str, "TimeFrame"]] = "1Hour"
    window_sizes: Union[List[int], int] = 10
    execute_on: Union[str, "TimeFrame"] = "1Hour"

    # Trading parameters
    leverage: int = 1
    margin_mode: MarginMode = MarginMode.ISOLATED
    position_mode: PositionMode = PositionMode.NET

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
        from torchtrade.envs.live.okx.utils import normalize_okx_timeframe_config
        self.execute_on, self.time_frames, self.window_sizes = normalize_okx_timeframe_config(
            self.execute_on, self.time_frames, self.window_sizes
        )

        if self.action_levels is None:
            self.action_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]

        validate_action_levels(self.action_levels)


class OKXFuturesTorchTradingEnv(OKXBaseTorchTradingEnv):
    """
    TorchRL environment for OKX Futures live trading.

    Supports:
    - Long and short positions
    - Configurable leverage (1x-125x)
    - Multiple timeframe observations
    - Demo trading
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
        config: OKXFuturesTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        observer: Optional[OKXObservationClass] = None,
        trader: Optional[OKXFuturesOrderClass] = None,
    ):
        super().__init__(config, api_key, api_secret, passphrase, feature_preprocessing_fn, observer, trader)

        from torchtrade.envs.core.default_rewards import log_return_reward
        self.reward_function = reward_function or log_return_reward

        self.action_levels = config.action_levels
        self.action_spec = Categorical(len(self.action_levels))


    def _calculate_fractional_position(self, action_value: float, current_price: float) -> tuple[float, float, str]:
        """Calculate target position size from fractional action."""
        if action_value == 0.0:
            return 0.0, 0.0, "flat"

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

        effective_balance = total_balance * 0.98
        params = PositionCalculationParams(
            balance=effective_balance,
            action_value=action_value,
            current_price=current_price,
            leverage=self.config.leverage,
            transaction_fee=TAKER_FEE,
        )
        return calculate_fractional_position(params)

    def _execute_fractional_action(
        self, action_value: float, *, current_qty: float, current_price: float,
    ) -> Dict:
        """Execute action using fractional position sizing."""
        if action_value == 0.0:
            if abs(current_qty) > 0:
                return self._handle_close_action(current_qty)
            return self._create_trade_info(executed=False)

        target_qty, _, _ = self._calculate_fractional_position(action_value, current_price)
        delta_qty = target_qty - current_qty

        lot_size = self.trader.get_lot_size()
        if abs(delta_qty) < lot_size["min_qty"]:
            return self._create_trade_info(executed=False, at_target=True)

        side = "buy" if delta_qty > 0 else "sell"
        # _format_size() in trade() handles lot-step quantization
        info = self._execute_market_order(side, abs(delta_qty))
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
