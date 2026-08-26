"""Base class for Bybit live trading environments."""

from typing import Callable, Optional

from torchtrade.envs.live.bybit.observation import BybitObservationClass
from torchtrade.envs.live.bybit.order_executor import (
    TAKER_FEE,
    BybitFuturesOrderClass,
)
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv

class BybitBaseTorchTradingEnv(TorchTradeFuturesLiveEnv):
    """Base class for Bybit futures trading environments.

    Supplies the Bybit observer and order classes. Bybit is the one venue whose observer
    reuses the trader's session, which is why TRADER_FIRST is set here. `ACCOUNT_STATE` on
    TorchTradeLiveEnv is the observation contract.
    """

    TAKER_FEE = TAKER_FEE

    OBSERVER_CLS = BybitObservationClass
    TRADER_CLS = BybitFuturesOrderClass
    TRADER_FIRST = True

    def __init__(
        self,
        config,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        observer: Optional[BybitObservationClass] = None,
        trader: Optional[BybitFuturesOrderClass] = None,
    ):
        """Initialize the bybit trading environment.

        Args:
            config: Environment configuration
            api_key: API key (not required if observer and trader are provided)
            api_secret: API secret (not required if observer and trader are provided)
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured observer for dependency injection
            trader: Optional pre-configured trader for dependency injection
        """
        self._feature_preprocessing_fn = feature_preprocessing_fn
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
            "position_mode": self.config.position_mode,
        }

    def _observer_kwargs(self) -> dict:
        # Reuses the trader's client, which is why TRADER_FIRST is set.
        return {
            **super()._observer_kwargs(),
            "client": getattr(self.trader, "client", None),
        }
