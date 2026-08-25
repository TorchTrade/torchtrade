"""Base class for Binance live trading environments."""

from typing import Callable, Optional

from torchtrade.envs.live.binance.observation import BinanceObservationClass
from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv

class BinanceBaseTorchTradingEnv(TorchTradeFuturesLiveEnv):
    """Base class for Binance futures trading environments.

    Supplies the Binance observer and order classes; everything else -- observation specs,
    account state, portfolio value, the trade-path helpers -- comes from
    TorchTradeFuturesLiveEnv. See `ACCOUNT_STATE` on TorchTradeLiveEnv for the observation
    contract, and `_trader_kwargs` below for what is genuinely Binance-specific.
    """


    OBSERVER_CLS = BinanceObservationClass
    TRADER_CLS = BinanceFuturesOrderClass

    def __init__(
        self,
        config,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        observer: Optional[BinanceObservationClass] = None,
        trader: Optional[BinanceFuturesOrderClass] = None,
    ):
        """Initialize Binance trading environment.

        Args:
            config: Environment configuration
            api_key: Binance API key (not required if observer and trader are provided)
            api_secret: Binance API secret (not required if observer and trader are provided)
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured BinanceObservationClass for dependency injection
            trader: Optional pre-configured BinanceFuturesOrderClass for dependency injection
        """
        self._feature_preprocessing_fn = feature_preprocessing_fn
        # Binance has no timezone parameter; it works in UTC internally.
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
        }
