"""Base class for Binance live trading environments."""

from abc import abstractmethod
from typing import Callable, List, Optional

import torch
from tensordict import TensorDictBase
from torchrl.data import Bounded
from torchrl.data import Composite

from torchtrade.envs.utils.timeframe import timeframe_to_seconds
from torchtrade.envs.live.binance.observation import BinanceObservationClass
from torchtrade.envs.live.binance.order_executor import BinanceFuturesOrderClass
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv
from torchtrade.envs.core.state import (
    HistoryTracker,
    PositionState,
    position_direction_from_status,
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
        self.execute_on_value = timeframe_to_seconds(config.execute_on)
        self.execute_on_unit = "seconds"  # Binance uses simple seconds-based intervals

        # Flatten on startup for a clean state (configurable, default: True)
        self.trader.cancel_open_orders()
        if config.close_position_on_init:
            self.trader.close_position()

        # Get initial portfolio value
        balance = self.trader.get_account_balance()
        self.initial_portfolio_value = balance.get("total_wallet_balance", 0)

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

    def _build_observation_specs(self):
        """Build observation specs for account state and market data."""
        # Get feature dimensions from observer
        obs = self.observer.get_observations()
        first_key = self.observer.get_keys()[0]
        num_features = obs[first_key].shape[1]
        market_data_names = self.observer.get_keys()

        # Normalize window sizes to list
        window_sizes = self.config.window_sizes if isinstance(self.config.window_sizes, list) else [self.config.window_sizes]

        # Create composite observation spec
        self.observation_spec = Composite(shape=())
        self.market_data_key = "market_data"
        self.account_state_key = "account_state"

        # Account state spec (6 elements)
        account_state_spec = Bounded(
            low=-torch.inf,
            high=torch.inf,
            shape=(len(self.ACCOUNT_STATE),),
            dtype=torch.float
        )
        self.observation_spec.set(self.account_state_key, account_state_spec)

        # Market data specs (one per interval/timeframe)
        self.market_data_keys = []
        for i, market_data_name in enumerate(market_data_names):
            market_data_key = "market_data_" + market_data_name
            ws = window_sizes[i] if i < len(window_sizes) else window_sizes[0]
            market_data_spec = Bounded(
                low=-torch.inf,
                high=torch.inf,
                shape=(ws, num_features),
                dtype=torch.float
            )
            self.observation_spec.set(market_data_key, market_data_spec)
            self.market_data_keys.append(market_data_key)

        self._declare_base_features_spec(window_sizes[0])

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment."""
        # Cancel all orders
        self.trader.cancel_open_orders()

        # Reset history tracking
        self.history.reset()

        if self.config.close_position_on_reset:
            self.trader.close_position()

        # Get current state
        balance = self.trader.get_account_balance()
        self.balance = balance.get("available_balance", 0)
        self.last_portfolio_value = self._get_portfolio_value()

        status = self.trader.get_status()
        position_status = status.get("position_status")
        self.position.hold_counter = 0

        self.position.current_position = position_direction_from_status(position_status)

        self._sync_action_level_after_reset()

        # Get initial observation. advance_hold=False: hold_counter was just zeroed
        # above; a reset must never itself count a bar (see advance_hold docstring).
        return self._get_observation(advance_hold=False)

    @abstractmethod
    def _execute_trade_if_needed(self, action) -> dict:
        """
        Execute trade if position change is needed.

        Must be implemented by subclasses as trade logic differs by action space.

        Args:
            action: Action to execute (format varies by subclass)

        Returns:
            Dict with trade execution details
        """
        raise NotImplementedError(
            "Subclasses must implement _execute_trade_if_needed()"
        )

    def get_market_data_keys(self) -> List[str]:
        """Return the list of market data keys."""
        return self.market_data_keys

    def get_account_state(self) -> List[str]:
        """Return the list of account state field names."""
        return self.ACCOUNT_STATE

    def close(self):
        """Clean up resources.

        Note: This method cancels open orders but does NOT automatically close
        positions. Closing positions is intentionally left to manual intervention
        to prevent accidental liquidation of intended positions, especially in
        live trading scenarios where automated position closure could result in
        unexpected losses or interrupt longer-term trading strategies.

        If you need to close positions on environment cleanup, call
        `env.trader.close_position()` explicitly before `env.close()`.
        """
        self.trader.cancel_open_orders()
