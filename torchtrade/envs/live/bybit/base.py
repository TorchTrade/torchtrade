"""Base class for Bybit live trading environments."""

from typing import Callable, Optional

from torchtrade.envs.live.bybit.observation import BybitObservationClass
from torchtrade.envs.live.bybit.order_executor import BybitFuturesOrderClass
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv
from torchtrade.envs.core.state import (
    HistoryTracker,
)

class BybitBaseTorchTradingEnv(TorchTradeFuturesLiveEnv):
    """
    Base class for Bybit trading environments.

    Provides common functionality for all Bybit environments:
    - BybitObservationClass and BybitFuturesOrderClass initialization
    - Observation spec construction (account state + market data)
    - Common observation gathering logic
    - Portfolio value calculation (total_margin_balance)

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
