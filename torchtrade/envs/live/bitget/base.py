"""Base class for Bitget live trading environments."""

from typing import Callable, Optional

from torchtrade.envs.live.bitget.observation import BitgetObservationClass
from torchtrade.envs.live.bitget.order_executor import BitgetFuturesOrderClass
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv
from torchtrade.envs.core.state import (
    HistoryTracker,
    PositionState,
)

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
        """
        Initialize Bitget trading environment.

        Args:
            config: Environment configuration
            api_key: Bitget API key (not required if observer and trader are provided)
            api_secret: Bitget API secret (not required if observer and trader are provided)
            api_passphrase: Bitget API passphrase (required for Bitget!)
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured BitgetObservationClass for dependency injection
            trader: Optional pre-configured BitgetFuturesOrderClass for dependency injection
        """
        # Store feature preprocessing function and passphrase for use in _init_trading_clients
        self._feature_preprocessing_fn = feature_preprocessing_fn
        self._api_passphrase = api_passphrase

        # Initialize base class (will call _init_trading_clients)
        # Bitget uses UTC internally
        super().__init__(
            config=config,
            api_key=api_key,
            api_secret=api_secret,
            observer=observer,
            trader=trader,
            timezone="UTC"
        )

        # Extract execute timeframe
        self.execute_on = config.execute_on
        # The execution timeframe, read by the shared bar wait.

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
        observer: Optional[BitgetObservationClass],
        trader: Optional[BitgetFuturesOrderClass]
    ):
        """
        Initialize Bitget observer and trader clients.

        Uses dependency injection pattern - uses provided instances or creates new ones.
        """
        # time_frames are already normalized in config.__post_init__,
        # so we can use them directly
        time_frames = self.config.time_frames
        window_sizes = self.config.window_sizes

        # Get product type from config (V2 API uses USDT-FUTURES instead of SUMCBL)
        product_type = getattr(self.config, 'product_type', 'USDT-FUTURES')
        demo = getattr(self.config, 'demo', True)

        # Initialize observer
        self.observer = observer if observer is not None else BitgetObservationClass(
            symbol=self.config.symbol,
            time_frames=time_frames,
            window_sizes=window_sizes,
            product_type=product_type,
            feature_preprocessing_fn=self._feature_preprocessing_fn,
            demo=demo,
        )

        # Initialize trader
        self.trader = trader if trader is not None else BitgetFuturesOrderClass(
            symbol=self.config.symbol,
            product_type=product_type,
            trade_mode=self.config.trade_mode if hasattr(self.config, 'trade_mode') else None,
            api_key=api_key,
            api_secret=api_secret,
            passphrase=self._api_passphrase,
            demo=demo,
            leverage=self.config.leverage,
            margin_mode=self.config.margin_mode,
            position_mode=self.config.position_mode if hasattr(self.config, 'position_mode') else None,
        )
