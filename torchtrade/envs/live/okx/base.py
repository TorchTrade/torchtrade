"""Base class for OKX live trading environments."""

from typing import Callable, Optional

from torchtrade.envs.live.okx.observation import OKXObservationClass
from torchtrade.envs.live.okx.order_executor import OKXFuturesOrderClass
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv

class OKXBaseTorchTradingEnv(TorchTradeFuturesLiveEnv):
    """
    Base class for OKX trading environments.

    Provides common functionality for all OKX environments:
    - OKXObservationClass and OKXFuturesOrderClass initialization
    - Observation spec construction (account state + market data)
    - Common observation gathering logic
    - Portfolio value calculation

    Standard account state (6 elements):
    [exposure_pct, position_direction, unrealized_pnl_pct,
     holding_time, leverage, distance_to_liquidation]

    Subclasses must implement:
    - Action space definition
    - _execute_trade_if_needed(): Trade execution logic
    """

    ACCOUNT_STATE = [
        "exposure_pct", "position_direction", "unrealized_pnlpct",
        "holding_time", "leverage", "distance_to_liquidation"
    ]

    OBSERVER_CLS = OKXObservationClass
    TRADER_CLS = OKXFuturesOrderClass
    # Trader first, as before the fold. NOT for client sharing -- okx keeps market
    # data on a separate client -- but because both constructors talk to the venue
    # and the order decides which side effects have landed when one fails.
    TRADER_FIRST = True

    def __init__(
        self,
        config,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        observer: Optional[OKXObservationClass] = None,
        trader: Optional[OKXFuturesOrderClass] = None,
    ):
        """Initialize the okx trading environment.

        Args:
            config: Environment configuration
            api_key: API key (not required if observer and trader are provided)
            api_secret: API secret (not required if observer and trader are provided)
            passphrase: OKX API passphrase
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured observer for dependency injection
            trader: Optional pre-configured trader for dependency injection
        """
        self._feature_preprocessing_fn = feature_preprocessing_fn
        self._passphrase = passphrase
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
            "passphrase": self._passphrase,
        }
