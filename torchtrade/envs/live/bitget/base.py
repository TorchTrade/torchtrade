"""Base class for Bitget live trading environments."""

from typing import Callable, Optional

from torchtrade.envs.live.bitget.observation import BitgetObservationClass
from torchtrade.envs.live.bitget.order_executor import BitgetFuturesOrderClass
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv

class BitgetBaseTorchTradingEnv(TorchTradeFuturesLiveEnv):
    """
    Base class for Bitget trading environments.

    Provides common functionality for all Bitget environments:
    - BitgetObservationClass and BitgetFuturesOrderClass initialization
    - Observation spec construction (account state + market data)
    - Common observation gathering logic
    - Portfolio value calculation (total_margin_balance)
    - Helper methods for market data keys and account state

    Standard account state for Bitget futures environments (6 elements):
    [exposure_pct, position_direction, unrealized_pnl_pct,
     holding_time, leverage, distance_to_liquidation]

    Element definitions:
        - exposure_pct: position_value / total_margin_balance (equity incl. unrealized PnL)
        - position_direction: sign(position_size) (-1=short, 0=flat, +1=long)
        - unrealized_pnl_pct: percentage unrealized PnL from entry
        - holding_time: steps since position opened
        - leverage: 1-125x leverage multiplier
        - distance_to_liquidation: normalized distance to liquidation price

    Subclasses must implement:
    - Action space definition (different per environment)
    - _execute_trade_if_needed(): Trade execution logic
    """

    # Standard account state for Bitget futures environments (6 elements)
    # Universal state used across all TorchTrade environments for better generalization.
    ACCOUNT_STATE = [
        "exposure_pct", "position_direction", "unrealized_pnlpct",
        "holding_time", "leverage", "distance_to_liquidation"
    ]

    OBSERVER_CLS = BitgetObservationClass
    TRADER_CLS = BitgetFuturesOrderClass
    TRADER_FIRST = False

    def __init__(
        self,
        config,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        observer: Optional[BitgetObservationClass] = None,
        trader: Optional[BitgetFuturesOrderClass] = None,
    ):
        """Initialize the bitget trading environment.

        Args:
            config: Environment configuration
            api_key: API key (not required if observer and trader are provided)
            api_secret: API secret (not required if observer and trader are provided)
            api_passphrase: Bitget API passphrase (required for Bitget!)
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured observer for dependency injection
            trader: Optional pre-configured trader for dependency injection
        """
        self._feature_preprocessing_fn = feature_preprocessing_fn
        self._api_passphrase = api_passphrase
        super().__init__(
            config=config,
            api_key=api_key,
            api_secret=api_secret,
            observer=observer,
            trader=trader,
            timezone="UTC",
        )
        self._finish_futures_init()

    def _trader_kwargs(self, api_key: str, api_secret: str) -> dict:
        return {
            **super()._trader_kwargs(api_key, api_secret),
            "trade_mode": getattr(self.config, "trade_mode", None),
            "position_mode": self.config.position_mode,
            "product_type": getattr(self.config, "product_type", "USDT-FUTURES"),
            "passphrase": self._api_passphrase,
        }
