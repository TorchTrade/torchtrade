"""Base class for Bybit live trading environments."""

from typing import Callable, Optional

import torch
from torchrl.data import Unbounded
from torchrl.data.tensor_specs import Composite

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

    def __init__(
        self,
        config,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        observer: Optional[BybitObservationClass] = None,
        trader: Optional[BybitFuturesOrderClass] = None,
    ):
        """
        Initialize Bybit trading environment.

        Args:
            config: Environment configuration
            api_key: Bybit API key
            api_secret: Bybit API secret
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured BybitObservationClass
            trader: Optional pre-configured BybitFuturesOrderClass
        """
        self._feature_preprocessing_fn = feature_preprocessing_fn

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
        observer: Optional[BybitObservationClass],
        trader: Optional[BybitFuturesOrderClass]
    ):
        """Initialize Bybit observer and trader clients."""
        time_frames = self.config.time_frames
        window_sizes = self.config.window_sizes
        demo = getattr(self.config, 'demo', True)

        # Initialize trader first (observer may reuse its client)
        self.trader = trader if trader is not None else BybitFuturesOrderClass(
            symbol=self.config.symbol,
            api_key=api_key,
            api_secret=api_secret,
            demo=demo,
            leverage=self.config.leverage,
            margin_mode=self.config.margin_mode,
            position_mode=self.config.position_mode,
        )

        # Initialize observer, sharing trader's client if available
        if observer is not None:
            self.observer = observer
        else:
            shared_client = getattr(self.trader, 'client', None)
            self.observer = BybitObservationClass(
                symbol=self.config.symbol,
                time_frames=time_frames,
                window_sizes=window_sizes,
                feature_preprocessing_fn=self._feature_preprocessing_fn,
                client=shared_client,
                demo=demo,
            )

    def _build_observation_specs(self):
        """Build observation specs for account state and market data (no network calls)."""
        features_info = self.observer.get_features()
        num_features = len(features_info["observation_features"])
        market_data_names = self.observer.get_keys()

        window_sizes = self.config.window_sizes if isinstance(self.config.window_sizes, list) else [self.config.window_sizes]

        self.observation_spec = Composite(shape=())
        self.market_data_key = "market_data"
        self.account_state_key = "account_state"

        # Account state spec (6 elements)
        account_state_spec = Unbounded(shape=(len(self.ACCOUNT_STATE),), dtype=torch.float)
        self.observation_spec.set(self.account_state_key, account_state_spec)

        # Market data specs (one per interval/timeframe)
        self.market_data_keys = []
        for i, market_data_name in enumerate(market_data_names):
            market_data_key = "market_data_" + market_data_name
            ws = window_sizes[i] if i < len(window_sizes) else window_sizes[0]
            market_data_spec = Unbounded(shape=(ws, num_features), dtype=torch.float)
            self.observation_spec.set(market_data_key, market_data_spec)
            self.market_data_keys.append(market_data_key)

        self._declare_base_features_spec(window_sizes[0])
