"""Base class for Bitget live trading environments."""

from abc import abstractmethod
from typing import Callable, List, Optional

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded
from torchrl.data.tensor_specs import CompositeSpec

from torchtrade.envs.bitget.obs_class import BitgetObservationClass
from torchtrade.envs.bitget.futures_order_executor import BitgetFuturesOrderClass
from torchtrade.envs.live import TorchTradeLiveEnv


# Interval to seconds mapping for waiting
INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}


class BitgetBaseTorchTradingEnv(TorchTradeLiveEnv):
    """
    Base class for Bitget trading environments.

    Provides common functionality for all Bitget environments:
    - BitgetObservationClass and BitgetFuturesOrderClass initialization
    - Observation spec construction (account state + market data)
    - Common observation gathering logic
    - Portfolio value calculation (total_margin_balance)
    - Helper methods for market data keys and account state

    Standard account state for Bitget futures environments:
    ["cash", "position_size", "position_value", "entry_price", "current_price",
     "unrealized_pnlpct", "leverage", "margin_ratio", "liquidation_price", "holding_time"]

    Subclasses must implement:
    - Action space definition (different per environment)
    - _execute_trade_if_needed(): Trade execution logic
    - _check_termination(): Episode termination logic
    """

    # Standard account state for Bitget futures environments (10 elements)
    ACCOUNT_STATE = [
        "cash", "position_size", "position_value", "entry_price", "current_price",
        "unrealized_pnlpct", "leverage", "margin_ratio", "liquidation_price", "holding_time"
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

        # Extract execute interval and convert to seconds
        self.execute_on = config.execute_on
        self.execute_on_value = INTERVAL_SECONDS.get(config.execute_on, 60)
        self.execute_on_unit = "seconds"  # Bitget uses simple seconds-based intervals

        # Reset settings
        self.trader.cancel_open_orders()
        if hasattr(config, 'close_position_on_reset') and config.close_position_on_reset:
            self.trader.close_position()

        # Get initial portfolio value
        balance = self.trader.get_account_balance()
        self.initial_portfolio_value = balance.get("total_wallet_balance", 0)

        # Build observation specs
        self._build_observation_specs()

        # Initialize current position tracking
        self.current_position = 0  # 0=no position, 1=long, -1=short

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
        # Normalize intervals to list
        intervals = self.config.intervals if isinstance(self.config.intervals, list) else [self.config.intervals]
        window_sizes = self.config.window_sizes if isinstance(self.config.window_sizes, list) else [self.config.window_sizes]

        # Get product type from config
        product_type = getattr(self.config, 'product_type', 'SUMCBL')
        demo = getattr(self.config, 'demo', True)

        # Initialize observer
        self.observer = observer if observer is not None else BitgetObservationClass(
            symbol=self.config.symbol,
            intervals=intervals,
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

        # Account state spec (10 elements for Bitget futures)
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
            self.position_hold_counter = 0
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
            self.position_hold_counter += 1
            position_size = position_status.qty  # Positive=long, Negative=short
            position_value = abs(position_status.notional_value)
            entry_price = position_status.entry_price
            current_price = position_status.mark_price
            unrealized_pnl_pct = position_status.unrealized_pnl_pct
            leverage = float(position_status.leverage)
            # Calculate margin ratio (position value / total balance)
            margin_ratio = position_value / total_balance if total_balance > 0 else 0.0
            liquidation_price = position_status.liquidation_price
            holding_time = float(self.position_hold_counter)

        # Build account state tensor (10 elements for Bitget futures)
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
        Calculate total portfolio value for Bitget futures.

        Returns total_margin_balance (includes unrealized PnL).
        """
        balance = self.trader.get_account_balance()
        return balance.get("total_margin_balance", 0)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment."""
        # Cancel all orders
        self.trader.cancel_open_orders()

        # Optionally close positions on reset (configurable)
        if hasattr(self.config, 'close_position_on_reset') and self.config.close_position_on_reset:
            self.trader.close_position()

        # Get current state
        balance = self.trader.get_account_balance()
        self.balance = balance.get("available_balance", 0)
        self.last_portfolio_value = self._get_portfolio_value()

        status = self.trader.get_status()
        position_status = status.get("position_status")
        self.position_hold_counter = 0

        if position_status is None:
            self.current_position = 0  # No position
        else:
            # Bitget: positive qty = long, negative qty = short
            self.current_position = 1 if position_status.qty > 0 else -1 if position_status.qty < 0 else 0

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
