"""Bybit Futures TorchRL trading environment with Stop Loss and Take Profit."""
from torchtrade.envs.live.shared.sltp_config import BaseFuturesSLTPConfig
from dataclasses import dataclass
from typing import Optional, Callable

from torchrl.data import Categorical

from torchtrade.envs.live.bybit.observation import BybitObservationClass
from torchtrade.envs.live.bybit.order_executor import (
    BybitFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.bybit.utils import normalize_bybit_timeframe_config
from torchtrade.envs.live.bybit.base import BybitBaseTorchTradingEnv
from torchtrade.envs.utils.action_maps import create_sltp_action_map
from torchtrade.envs.utils.sltp_mixin import SLTPMixin



@dataclass
class BybitFuturesSLTPTradingEnvConfig(BaseFuturesSLTPConfig):
    """Configuration for Bybit Futures SLTP Trading Environment.

    Uses a combinatorial action space where each action represents a
    (side, stop_loss_pct, take_profit_pct) tuple for bracket orders.
    """

    # Trading parameters
    margin_mode: MarginMode = MarginMode.ISOLATED
    position_mode: PositionMode = PositionMode.ONE_WAY

    _normalize_timeframes = staticmethod(normalize_bybit_timeframe_config)


class BybitFuturesSLTPTorchTradingEnv(SLTPMixin, BybitBaseTorchTradingEnv):
    """
    Bybit Futures trading environment with Stop Loss and Take Profit action spec.

    Uses bracket orders via pybit's native takeProfit/stopLoss parameters.

    Action mapping:
        - 0: HOLD (do nothing)
        - 1..N: LONG with specific (stop_loss_pct, take_profit_pct) combination
        - N+1..M: SHORT with specific SL/TP combination (if enabled)
    """

    def __init__(
        self,
        config: BybitFuturesSLTPTradingEnvConfig,
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

        self.stoploss_levels = list(config.stoploss_levels)
        self.takeprofit_levels = list(config.takeprofit_levels)
        self.action_map = create_sltp_action_map(
            self.stoploss_levels,
            self.takeprofit_levels,
            include_short_positions=config.include_short_positions,
            include_hold_action=config.include_hold_action,
            include_close_action=config.include_close_action
        )

        self.action_spec = Categorical(len(self.action_map))

        self._reset_sltp_state()

    # `_step` already took the mark under the halt policy; never read again (#295).
    _PRICES_OFF_THREADED_MARK = True

