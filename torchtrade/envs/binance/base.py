"""Base class for Binance live trading environments."""

from abc import abstractmethod
from typing import Callable, List, Optional

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded
from torchrl.data.tensor_specs import CompositeSpec

from torchtrade.envs.timeframe import TimeFrame, TimeFrameUnit, timeframe_to_seconds
from torchtrade.envs.binance.obs_class import BinanceObservationClass
from torchtrade.envs.binance.futures_order_executor import BinanceFuturesOrderClass
from torchtrade.envs.live import TorchTradeLiveEnv
from torchtrade.envs.state import FuturesHistoryTracker, PositionState


class BinanceBaseTorchTradingEnv(TorchTradeLiveEnv):
    """
    Base class for Binance trading environments.

    Provides common functionality for all Binance environments:
    - BinanceObservationClass and BinanceFuturesOrderClass initialization
    - Observation spec construction (account state + market data)
    - Common observation gathering logic
    - Portfolio value calculation (total_margin_balance)
    - Helper methods for market data keys and account state

    Standard account state for Binance futures environments:
    ["cash", "position_size", "position_value", "entry_price", "current_price",
     "unrealized_pnlpct", "leverage", "margin_ratio", "liquidation_price", "holding_time"]

    Subclasses must implement:
    - Action space definition (different per environment)
    - _execute_trade_if_needed(): Trade execution logic
    - _check_termination(): Episode termination logic
    """

    # Standard account state for Binance futures environments (10 elements)
    # Note: This is intentionally different from Alpaca's 7-element state because futures trading
    # requires additional state: leverage, margin_ratio, and liquidation_price.
    ACCOUNT_STATE = [
        "cash", "position_size", "position_value", "entry_price", "current_price",
        "unrealized_pnlpct", "leverage", "margin_ratio", "liquidation_price", "holding_time"
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

        # Reset settings
        self.trader.cancel_open_orders()
        self.trader.close_position()

        # Get initial portfolio value
        balance = self.trader.get_account_balance()
        self.initial_portfolio_value = balance.get("total_wallet_balance", 0)

        # Build observation specs
        self._build_observation_specs()

        # Initialize position state
        self.position = PositionState()  # current_position: 0=no position, 1=long, -1=short

        # Initialize history tracking (futures environments use FuturesHistoryTracker)
        self.history = FuturesHistoryTracker()

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
        self.observation_spec = CompositeSpec(shape=())
        self.market_data_key = "market_data"
        self.account_state_key = "account_state"

        # Account state spec (10 elements for Binance futures)
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

    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        # Get market data
        obs_dict = self.observer.get_observations(
            return_base_ohlc=self.config.include_base_features
        )

        # Extract base features if requested
        if self.config.include_base_features:
            base_features = obs_dict.get("base_features")

        # Get market data for each interval
        market_data = [obs_dict[features_name] for features_name in self.observer.get_keys()]

        # Get account state from trader
        status = self.trader.get_status()
        balance = self.trader.get_account_balance()

        cash = balance.get("available_balance", 0)
        total_balance = balance.get("total_wallet_balance", 0)

        position_status = status.get("position_status", None)

        if position_status is None:
            position_size = 0.0
            position_value = 0.0
            entry_price = 0.0
            current_price = self.trader.get_mark_price()
            unrealized_pnl_pct = 0.0
            leverage = float(self.config.leverage)
            margin_ratio = 0.0
            liquidation_price = 0.0
            holding_time = 0.0
        else:
            position_size = position_status.qty  # Positive=long, Negative=short
            position_value = abs(position_status.notional_value)
            entry_price = position_status.entry_price
            current_price = position_status.mark_price
            unrealized_pnl_pct = position_status.unrealized_pnl_pct
            leverage = float(position_status.leverage)
            # Calculate margin ratio (position value / total balance)
            margin_ratio = position_value / total_balance if total_balance > 0 else 0.0
            liquidation_price = position_status.liquidation_price
            holding_time = float(self.position.hold_counter)

        # Build account state tensor (10 elements for Binance futures)
        account_state = torch.tensor(
            [
                cash,
                position_size,
                position_value,
                entry_price,
                current_price,
                unrealized_pnl_pct,
                leverage,
                margin_ratio,
                liquidation_price,
                holding_time,
            ],
            dtype=torch.float,
        )

        # Build output TensorDict
        out_td = TensorDict({self.account_state_key: account_state}, batch_size=())
        for market_data_name, data in zip(self.market_data_keys, market_data):
            out_td.set(market_data_name, torch.from_numpy(data))

        # Add base features if requested
        if self.config.include_base_features and base_features is not None:
            out_td.set("base_features", torch.from_numpy(base_features))

        return out_td

    def _get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value for Binance futures.

        Returns total_margin_balance (includes unrealized PnL).
        """
        balance = self.trader.get_account_balance()
        return balance.get("total_margin_balance", 0)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment."""
        # Cancel all orders
        self.trader.cancel_open_orders()

        # Reset history tracking
        self.history.reset()

        # Optionally close positions on reset (configurable)
        if hasattr(self.config, 'close_position_on_reset') and self.config.close_position_on_reset:
            self.trader.close_position()

        # Get current state
        balance = self.trader.get_account_balance()
        self.balance = balance.get("available_balance", 0)
        self.last_portfolio_value = self._get_portfolio_value()

        status = self.trader.get_status()
        position_status = status.get("position_status")
        self.position.hold_counter = 0

        if position_status is None:
            self.position.current_position = 0
        elif position_status.qty > 0:
            self.position.current_position = 1  # Long position
        elif position_status.qty < 0:
            self.position.current_position = -1  # Short position
        else:
            self.position.current_position = 0  # No position

        # Get initial observation
        return self._get_observation()

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

    @abstractmethod
    def _check_termination(self, portfolio_value: float) -> bool:
        """
        Check if episode should terminate.

        Must be implemented by subclasses as termination conditions may differ.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            True if episode should terminate, False otherwise
        """
        raise NotImplementedError(
            "Subclasses must implement _check_termination()"
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
