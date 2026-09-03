"""OKX Futures TorchRL trading environment with Stop Loss and Take Profit."""
from torchtrade.envs.live.shared.sltp_config import BaseFuturesSLTPConfig
from dataclasses import dataclass
from typing import Optional, Callable


from torchtrade.envs.live.okx.observation import OKXObservationClass
from torchtrade.envs.live.okx.order_executor import (
    OKXFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.okx.utils import normalize_okx_timeframe_config
from torchtrade.envs.live.okx.base import OKXBaseTorchTradingEnv
from torchtrade.envs.utils.sltp_mixin import SLTPMixin


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

        self._init_bracket_action_space(
            reward_function, include_short_positions=config.include_short_positions)

    # `_step` already took the mark under the halt policy; never read again (#295).
    _PRICES_OFF_THREADED_MARK = True

    def _resolve_bracket_quantity(self, current_price):
        """okx alone refuses a sub-minimum bracket instead of letting the venue reject it.

        Whether the other three should refuse too is #414's question, not this fold's.
        """
        quantity = super()._resolve_bracket_quantity(current_price)
        if quantity is None or quantity < self.trader.get_lot_size()["min_qty"]:
            return None
        return quantity

