"""OKX Futures TorchRL trading environment with fractional position sizing."""
from dataclasses import dataclass
from typing import Optional, Callable, Dict

from torchrl.data import Categorical

from torchtrade.envs.live.okx.observation import OKXObservationClass
from torchtrade.envs.live.okx.order_executor import (
    OKXFuturesOrderClass,
    MarginMode,
    PositionMode,
)
from torchtrade.envs.live.okx.base import OKXBaseTorchTradingEnv
from torchtrade.envs.live.shared.futures_config import BaseFuturesTradingConfig
from torchtrade.envs.live.okx.utils import normalize_okx_timeframe_config


@dataclass
class OKXFuturesTradingEnvConfig(BaseFuturesTradingConfig):
    """Configuration for OKX Futures Trading Environment."""

    symbol: str = "BTC-USDT-SWAP"
    margin_mode: MarginMode = MarginMode.ISOLATED
    position_mode: PositionMode = PositionMode.NET

    _normalize_timeframes = staticmethod(normalize_okx_timeframe_config)


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
    - action = 0.0: Market neutral (close all positions)
    - action = 1.0: 100% long (all-in long)

    Default action_levels: [-1, 0, 1]; see BaseFuturesTradingConfig.action_levels.
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
