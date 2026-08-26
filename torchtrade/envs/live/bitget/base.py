"""Base class for Bitget live trading environments."""

from typing import Callable, Optional

from .order_executor import TAKER_FEE
from torchtrade.envs.live.bitget.observation import BitgetObservationClass
from torchtrade.envs.live.bitget.order_executor import BitgetFuturesOrderClass
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv

class BitgetBaseTorchTradingEnv(TorchTradeFuturesLiveEnv):
    """Base class for Bitget futures trading environments.

    Supplies the Bitget observer and order classes, and the three constructor arguments
    only Bitget takes: a passphrase, a product type and a position mode. Everything else
    comes from TorchTradeFuturesLiveEnv; `ACCOUNT_STATE` on TorchTradeLiveEnv is the
    observation contract.
    """

    TAKER_FEE = TAKER_FEE

    OBSERVER_CLS = BitgetObservationClass
    TRADER_CLS = BitgetFuturesOrderClass

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
