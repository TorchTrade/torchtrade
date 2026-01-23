"""Base class for Alpaca live trading environments."""

from abc import abstractmethod
from typing import Callable, List, Optional

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded
from torchrl.data.tensor_specs import CompositeSpec

from torchtrade.envs.live.alpaca.observation import AlpacaObservationClass
from torchtrade.envs.live.alpaca.order_executor import AlpacaOrderClass
from torchtrade.envs.core.live import TorchTradeLiveEnv
from torchtrade.envs.core.state import HistoryTracker, PositionState


class AlpacaBaseTorchTradingEnv(TorchTradeLiveEnv):
    """
    Base class for Alpaca trading environments.

    Provides common functionality for all Alpaca environments:
    - AlpacaObservationClass and AlpacaOrderClass initialization
    - Observation spec construction (account state + market data)
    - Common observation gathering logic
    - Portfolio value calculation (cash + position_market_value)
    - Helper methods for market data keys and account state

    Standard account state for Alpaca environments:
    ["cash", "position_size", "position_value", "entry_price",
     "current_price", "unrealized_pnlpct", "holding_time"]

    Subclasses must implement:
    - Action space definition (different for standard vs SLTP)
    - _execute_trade_if_needed(): Trade execution logic
    - _calculate_trade_amount(): Trade sizing logic
    - _check_termination(): Episode termination logic
    """

    # Standard account state for Alpaca environments (7 elements)
    # Note: Subclasses may define their own ACCOUNT_STATE if they have different state dimensions.
    # For example, futures environments have 10 elements instead of 7.
    ACCOUNT_STATE = [
        "cash", "position_size", "position_value", "entry_price",
        "current_price", "unrealized_pnlpct", "holding_time"
    ]

    def __init__(
        self,
        config,
        api_key: str = "",
        api_secret: str = "",
        feature_preprocessing_fn: Optional[Callable] = None,
        observer: Optional[AlpacaObservationClass] = None,
        trader: Optional[AlpacaOrderClass] = None,
    ):
        """
        Initialize Alpaca trading environment.

        Args:
            config: Environment configuration
            api_key: Alpaca API key (not required if observer and trader are provided)
            api_secret: Alpaca API secret (not required if observer and trader are provided)
            feature_preprocessing_fn: Optional custom preprocessing function
            observer: Optional pre-configured AlpacaObservationClass for dependency injection
            trader: Optional pre-configured AlpacaOrderClass for dependency injection
        """
        # Store feature preprocessing function for use in _init_trading_clients
        self._feature_preprocessing_fn = feature_preprocessing_fn

        # Initialize base class (will call _init_trading_clients)
        super().__init__(
            config=config,
            api_key=api_key,
            api_secret=api_secret,
            observer=observer,
            trader=trader,
            timezone="America/New_York"
        )

        # Extract execute_on timeframe
        self.execute_on_value = config.execute_on.value
        self.execute_on_unit = str(config.execute_on.unit.value)

        # Reset settings
        self.trader.close_all_positions()
        self.trader.cancel_open_orders()

        # Get initial portfolio value
        account = self.trader.client.get_account()
        cash = float(account.cash)
        self.initial_portfolio_value = cash

        # Build observation specs
        self._build_observation_specs()

        # Initialize position state
        self.position = PositionState()

        # Initialize history tracking
        self.history = HistoryTracker()

    def _init_trading_clients(
        self,
        api_key: str,
        api_secret: str,
        observer: Optional[AlpacaObservationClass],
        trader: Optional[AlpacaOrderClass]
    ):
        """
        Initialize Alpaca observer and trader clients.

        Uses dependency injection pattern - uses provided instances or creates new ones.
        """
        # Initialize observer
        self.observer = observer if observer is not None else AlpacaObservationClass(
            symbol=self.config.symbol,
            timeframes=self.config.time_frames,
            window_sizes=self.config.window_sizes,
            feature_preprocessing_fn=self._feature_preprocessing_fn,
        )

        # Initialize trader
        self.trader = trader if trader is not None else AlpacaOrderClass(
            symbol=self.config.symbol.replace('/', ''),
            trade_mode=self.config.trade_mode,
            api_key=api_key,
            api_secret=api_secret,
            paper=self.config.paper,
        )

    def _build_observation_specs(self):
        """Build observation specs for account state and market data."""
        # Get feature dimensions from observer
        num_features = self.observer.get_observations()[
            self.observer.get_keys()[0]
        ].shape[1]
        market_data_names = self.observer.get_keys()

        # Create composite observation spec
        self.observation_spec = CompositeSpec(shape=())
        self.market_data_key = "market_data"
        self.account_state_key = "account_state"

        # Account state spec
        account_state_spec = Bounded(
            low=-torch.inf,
            high=torch.inf,
            shape=(len(self.ACCOUNT_STATE),),
            dtype=torch.float
        )
        self.observation_spec.set(self.account_state_key, account_state_spec)
        self._account_state = self.ACCOUNT_STATE  # Set instance variable for introspection

        # Market data specs (one per timeframe)
        window_sizes = self.config.window_sizes if isinstance(self.config.window_sizes, list) else [self.config.window_sizes]
        for i, market_data_name in enumerate(market_data_names):
            market_data_key = "market_data_" + market_data_name
            window_size = window_sizes[i] if i < len(window_sizes) else window_sizes[0]
            market_data_spec = Bounded(
                low=-torch.inf,
                high=torch.inf,
                shape=(window_size, num_features),
                dtype=torch.float
            )
            self.observation_spec.set(market_data_key, market_data_spec)

    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        # Get market data
        obs_dict = self.observer.get_observations(
            return_base_ohlc=True if self.config.include_base_features else False
        )

        # Extract base features if requested
        if self.config.include_base_features:
            base_features = obs_dict["base_features"][-1]

        # Get market data for each timeframe
        market_data = [obs_dict[features_name] for features_name in self.observer.get_keys()]

        # Get account state from trader
        status = self.trader.get_status()
        account = self.trader.client.get_account()
        cash = float(account.cash)
        position_status = status.get("position_status", None)

        if position_status is None:
            position_size = 0.0
            position_value = 0.0
            entry_price = 0.0
            current_price = 0.0
            unrealized_pnlpc = 0.0
            holding_time = 0.0
        else:
            position_size = position_status.qty
            position_value = position_status.market_value
            entry_price = position_status.avg_entry_price
            current_price = position_status.current_price
            unrealized_pnlpc = position_status.unrealized_plpc
            holding_time = float(self.position.hold_counter)

        # Build account state tensor
        account_state = torch.tensor(
            [cash, position_size, position_value, entry_price, current_price, unrealized_pnlpc, holding_time],
            dtype=torch.float
        )

        # Build output TensorDict
        out_td = TensorDict({self.account_state_key: account_state}, batch_size=())
        for market_data_name, data in zip(self.market_data_keys, market_data):
            out_td.set(market_data_name, torch.from_numpy(data))

        # Add base features if requested
        if self.config.include_base_features:
            out_td.set("base_features", torch.from_numpy(base_features))

        return out_td

    def _get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value for Alpaca.

        Returns cash + position_market_value.
        """
        status = self.trader.get_status()
        position_status = status.get("position_status", None)

        account = self.trader.client.get_account()
        self.balance = float(account.cash)

        if position_status is None:
            return self.balance
        return self.balance + position_status.market_value

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment."""
        # Cancel all orders
        self.trader.cancel_open_orders()

        # Reset history tracking
        self.history.reset()

        # Get current state
        account = self.trader.client.get_account()
        self.balance = float(account.cash)
        self.last_portfolio_value = self.balance

        status = self.trader.get_status()
        position_status = status.get("position_status")
        self.position.hold_counter = 0

        if position_status is None:
            self.position.current_position = 0.0
        else:
            self.position.current_position = 1 if position_status.qty > 0 else 0

        # Get initial observation
        return self._get_observation()

    @abstractmethod
    def _execute_trade_if_needed(self, action) -> dict:
        """
        Execute trade if position change is needed.

        Must be implemented by subclasses as trade logic differs
        (standard 3-action vs SLTP bracket orders).

        Args:
            action: Action to execute (format varies by subclass)

        Returns:
            Dict with trade execution details
        """
        raise NotImplementedError(
            "Subclasses must implement _execute_trade_if_needed()"
        )

    @abstractmethod
    def _calculate_trade_amount(self, side: str) -> float:
        """
        Calculate the dollar amount to trade.

        Must be implemented by subclasses as sizing logic may differ.

        Args:
            side: "buy" or "sell"

        Returns:
            Dollar amount to trade (float)
        """
        raise NotImplementedError(
            "Subclasses must implement _calculate_trade_amount()"
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

    def close(self):
        """Clean up resources."""
        self.trader.cancel_open_orders()
        self.trader.close_all_positions()
