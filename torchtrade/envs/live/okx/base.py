"""Base class for OKX live trading environments."""

from typing import Callable, Optional

from torchtrade.envs.live.okx.observation import OKXObservationClass
from torchtrade.envs.live.okx.order_executor import OKXFuturesOrderClass
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv
from torchtrade.envs.core.state import (
    HistoryTracker,
)

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
        """
        Initialize OKX trading environment.

        Args:
            config: Environment configuration
            api_key: OKX API key
            api_secret: OKX API secret key
            passphrase: OKX API passphrase
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured OKXObservationClass
            trader: Optional pre-configured OKXFuturesOrderClass
        """
        self._feature_preprocessing_fn = feature_preprocessing_fn
        self._passphrase = passphrase

        # Initialize base class (will call _init_trading_clients)
        super().__init__(
            config=config,
            api_key=api_key,
            api_secret=api_secret,
            observer=observer,
            trader=trader,
            timezone="UTC"
        )

        # Extract execute timeframe (already normalized to TimeFrame in config.__post_init__)
        self.execute_on = config.execute_on

        # Flatten on startup for a clean state (configurable, default: True)
        self.trader.cancel_open_orders()
        if config.close_position_on_init:
            self.trader.close_position()

        self._capture_bankruptcy_baseline()

        # Build observation specs
        self._build_observation_specs()

        # Initialize history tracking
        self.history = HistoryTracker()

    def _init_trading_clients(
        self,
        api_key: str,
        api_secret: str,
        observer: Optional[OKXObservationClass],
        trader: Optional[OKXFuturesOrderClass]
    ):
        """Initialize OKX observer and trader clients."""
        time_frames = self.config.time_frames
        window_sizes = self.config.window_sizes
        demo = getattr(self.config, 'demo', True)

        # Initialize trader first (observer may reuse its client)
        self.trader = trader if trader is not None else OKXFuturesOrderClass(
            symbol=self.config.symbol,
            api_key=api_key,
            api_secret=api_secret,
            passphrase=self._passphrase,
            demo=demo,
            leverage=self.config.leverage,
            margin_mode=self.config.margin_mode,
            position_mode=self.config.position_mode,
        )

        # Initialize observer
        if observer is not None:
            self.observer = observer
        else:
            self.observer = OKXObservationClass(
                symbol=self.config.symbol,
                time_frames=time_frames,
                window_sizes=window_sizes,
                feature_preprocessing_fn=self._feature_preprocessing_fn,
                demo=demo,
            )

