"""OKX Futures TorchRL trading environment with Stop Loss and Take Profit."""
from torchtrade.envs.live.shared.sltp_config import BaseFuturesSLTPConfig
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable
import logging

from torchrl.data import Categorical

from torchtrade.envs.live.okx.observation import OKXObservationClass
from torchtrade.envs.live.okx.order_executor import (
    OKXFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.okx.utils import normalize_okx_timeframe_config
from torchtrade.envs.live.okx.base import OKXBaseTorchTradingEnv
from torchtrade.envs.utils.action_maps import create_sltp_action_map
from torchtrade.envs.utils.sltp_mixin import SLTPMixin
from torchtrade.envs.utils.sltp_helpers import calculate_bracket_prices

logger = logging.getLogger(__name__)


@dataclass
class OKXFuturesSLTPTradingEnvConfig(BaseFuturesSLTPConfig):
    """Configuration for OKX Futures SLTP Trading Environment.

    Uses a combinatorial action space where each action represents a
    (side, stop_loss_pct, take_profit_pct) tuple for bracket orders.
    """
    symbol: str = "BTC-USDT-SWAP"

    # Trading parameters
    margin_mode: MarginMode = MarginMode.ISOLATED
    position_mode: PositionMode = PositionMode.NET

    _normalize_timeframes = staticmethod(normalize_okx_timeframe_config)


class OKXFuturesSLTPTorchTradingEnv(SLTPMixin, OKXBaseTorchTradingEnv):
    """
    OKX Futures trading environment with Stop Loss and Take Profit action spec.

    Uses bracket orders via OKX's attachAlgoOrds parameter.

    Action mapping:
        - 0: HOLD (do nothing)
        - 1..N: LONG with specific (stop_loss_pct, take_profit_pct) combination
        - N+1..M: SHORT with specific SL/TP combination (if enabled)
    """

    def __init__(
        self,
        config: OKXFuturesSLTPTradingEnvConfig,
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

        self.active_stop_loss = 0.0
        self.active_take_profit = 0.0

    # Priced off the mark `_step` threaded in, not a fresh read (#295).
    _bracket_entry_price = SLTPMixin._validated_mark_price

    def _resolve_bracket_quantity(self, current_price):
        """okx alone refuses a sub-minimum bracket instead of letting the venue reject it.

        Kept as a sizing override rather than inlined in the executor, which is what let
        okx keep a private ~115-line copy of it. Whether the other three should refuse too
        is #414's question, not this fold's.
        """
        quantity = super()._resolve_bracket_quantity(current_price)
        if quantity is None or quantity < self.trader.get_lot_size()["min_qty"]:
            return None
        return quantity

    def _dispatch_sltp_trade(self, action_tuple, current_price: float):
        # Threaded, not re-read: re-reading the mark inside the trade path bypassed the
        # halt policy, so a grace bar that priced a bracket died instead of truncating
        # (#295). binance and bitget price off a candle close and take the default.
        return self._execute_trade_if_needed(action_tuple, current_price=current_price)
