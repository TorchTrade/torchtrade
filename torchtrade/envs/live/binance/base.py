"""Base class for Binance live trading environments."""

from typing import Callable, Optional

from torchtrade.envs.live.binance.observation import BinanceObservationClass
from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv
from torchtrade.envs.core.state import (
    HistoryTracker,
    PositionState,
)

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

    def __init__(
        self,
        config,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        observer: Optional[BinanceObservationClass] = None,
        trader: Optional[BinanceFuturesOrderClass] = None,
    ):
        """
        Initialize Binance trading environment.

        Args:
            config: Environment configuration
            api_key: Binance API key (not required if observer and trader are provided)
            api_secret: Binance API secret (not required if observer and trader are provided)
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured BinanceObservationClass for dependency injection
            trader: Optional pre-configured BinanceFuturesOrderClass for dependency injection
        """
        # Store feature preprocessing function for use in _init_trading_clients
        self._feature_preprocessing_fn = feature_preprocessing_fn

        # Initialize base class (will call _init_trading_clients)
        # Binance doesn't use timezone parameter (uses UTC internally)
        super().__init__(
            config=config,
            api_key=api_key,
            api_secret=api_secret,
            observer=observer,
            trader=trader,
            timezone="UTC"
        )

        # Extract execute timeframe and convert to seconds
        self.execute_on = config.execute_on

        # Flatten on startup for a clean state (configurable, default: True)
        self.trader.cancel_open_orders()
        if config.close_position_on_init:
            self.trader.close_position()

        self._capture_bankruptcy_baseline()

        # Build observation specs
        self._build_observation_specs()

        # Initialize position state
        self.position = PositionState()  # current_position: 0=no position, 1=long, -1=short

        # Initialize history tracking (futures environments use HistoryTracker)
        self.history = HistoryTracker()

    def _init_trading_clients(
        self,
        api_key: str,
        api_secret: str,
        observer: Optional[BinanceObservationClass],
        trader: Optional[BinanceFuturesOrderClass]
    ):
        """
        Initialize Binance observer and trader clients.

        Uses dependency injection pattern - uses provided instances or creates new ones.
        """
        # time_frames are already normalized in config.__post_init__,
        # so we can use them directly
        time_frames = self.config.time_frames
        window_sizes = self.config.window_sizes

        # Initialize observer
        self.observer = observer if observer is not None else BinanceObservationClass(
            symbol=self.config.symbol,
            time_frames=time_frames,
            window_sizes=window_sizes,
            feature_preprocessing_fn=self._feature_preprocessing_fn,
            demo=self.config.demo,
        )

        # Initialize trader
        self.trader = trader if trader is not None else BinanceFuturesOrderClass(
            symbol=self.config.symbol,
            trade_mode=self.config.trade_mode if hasattr(self.config, 'trade_mode') else None,
            api_key=api_key,
            api_secret=api_secret,
            demo=self.config.demo,
            leverage=self.config.leverage,
            margin_type=self.config.margin_type,
        )

