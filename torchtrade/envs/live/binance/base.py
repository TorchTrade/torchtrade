"""Base class for Binance live trading environments."""

from typing import Callable, Optional

from torchtrade.envs.live.binance.observation import BinanceObservationClass
from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv

class BinanceBaseTorchTradingEnv(TorchTradeFuturesLiveEnv):
    """
    Base class for Binance trading environments.

    Provides common functionality for all Binance environments:
    - BinanceObservationClass and BinanceFuturesOrderClass initialization
    - Observation spec construction (account state + market data)
    - Common observation gathering logic
    - Portfolio value calculation (total_margin_balance)
    - Helper methods for market data keys and account state

    Standard account state for Binance futures environments (6 elements):
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

    # Standard account state for Binance futures environments (6 elements)
    # Universal state used across all TorchTrade environments for better generalization.
    ACCOUNT_STATE = [
        "exposure_pct", "position_direction", "unrealized_pnlpct",
        "holding_time", "leverage", "distance_to_liquidation"
    ]

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
