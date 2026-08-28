from torchtrade.envs.live.shared.sltp_config import BaseFuturesSLTPConfig
from dataclasses import dataclass
from typing import Optional, Callable


from torchrl.data import Categorical

from torchtrade.envs.live.binance.observation import BinanceObservationClass
from torchtrade.envs.live.binance.order_executor import (
    BinanceFuturesOrderClass,
    MarginMode,
)
from torchtrade.envs.live.binance.utils import normalize_binance_timeframe_config
from torchtrade.envs.live.binance.base import BinanceBaseTorchTradingEnv
from torchtrade.envs.utils.action_maps import create_sltp_action_map
from torchtrade.envs.utils.sltp_mixin import SLTPMixin


@dataclass
class BinanceFuturesSLTPTradingEnvConfig(BaseFuturesSLTPConfig):
    """Configuration for Binance Futures SLTP Trading Environment.

    This environment uses a combinatorial action space where each action
    represents a (side, stop_loss_pct, take_profit_pct) tuple for bracket orders.
    Supports both long and short positions with stop-loss/take-profit.
    """

    # Trading parameters
    margin_mode: MarginMode = MarginMode.ISOLATED

    _normalize_timeframes = staticmethod(normalize_binance_timeframe_config)


class BinanceFuturesSLTPTorchTradingEnv(SLTPMixin, BinanceBaseTorchTradingEnv):
    """
    Binance Futures trading environment with Stop Loss and Take Profit action spec.

    This environment uses bracket orders to implement stop-loss and take-profit
    functionality for futures trading. The action space is a categorical distribution
    over all combinations of (side, stop_loss, take_profit) levels plus a HOLD action.

    Action mapping:
        - 0: HOLD (do nothing)
        - 1..N: LONG with specific (stop_loss_pct, take_profit_pct) combination
        - N+1..M: SHORT with specific (stop_loss_pct, take_profit_pct) combination (if enabled)

    The environment automatically closes the position when either the stop-loss or
    take-profit is triggered by Binance's bracket order system.

    Key differences from standard BinanceFuturesTorchTradingEnv:
    - Combinatorial action space with SL/TP levels
    - Bracket orders instead of simple market orders
    - Tracks active SL/TP levels
    - Can optionally disable short positions for long-only strategies

    Account State (6 elements; the list is ACCOUNT_STATE on the exchange base class):
    [exposure_pct, position_direction, unrealized_pnlpct, holding_time,
     leverage, distance_to_liquidation]
    """

    def __init__(
        self,
        config: BinanceFuturesSLTPTradingEnvConfig,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        reward_function: Optional[Callable] = None,
        observer: Optional[BinanceObservationClass] = None,
        trader: Optional[BinanceFuturesOrderClass] = None,
    ):
        """
        Initialize the BinanceFuturesSLTPTorchTradingEnv.

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

        # Create action map from SL/TP combinations
        self.stoploss_levels = list(config.stoploss_levels)
        self.takeprofit_levels = list(config.takeprofit_levels)
        self.action_map = create_sltp_action_map(
            self.stoploss_levels,
            self.takeprofit_levels,
            include_short_positions=config.include_short_positions,
            include_hold_action=config.include_hold_action,
            include_close_action=config.include_close_action
        )

        # Categorical action spec: 0=HOLD, 1..N = (side, SL, TP) combinations
        self.action_spec = Categorical(len(self.action_map))

        # Track active SL/TP levels for current position
        self._reset_sltp_state()

