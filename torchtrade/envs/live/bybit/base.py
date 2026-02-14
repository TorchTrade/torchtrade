"""Base class for Bybit live trading environments."""
import logging

from abc import abstractmethod
from typing import Callable, List, Optional

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded
from torchrl.data.tensor_specs import Composite

from torchtrade.envs.live.bybit.observation import BybitObservationClass
from torchtrade.envs.live.bybit.order_executor import BybitFuturesOrderClass
from torchtrade.envs.core.live import TorchTradeLiveEnv
from torchtrade.envs.core.state import HistoryTracker

logger = logging.getLogger(__name__)


class BybitBaseTorchTradingEnv(TorchTradeLiveEnv):
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
    - _check_termination(): Episode termination logic
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
        self.execute_on_value = config.execute_on.value
        self.execute_on_unit = str(config.execute_on.unit)

        # Get initial portfolio value
        balance = self.trader.get_account_balance()
        self.initial_portfolio_value = balance.get("total_wallet_balance", 0)

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

        # Initialize observer
        self.observer = observer if observer is not None else BybitObservationClass(
            symbol=self.config.symbol,
            time_frames=time_frames,
            window_sizes=window_sizes,
            feature_preprocessing_fn=self._feature_preprocessing_fn,
            demo=demo,
        )

        # Initialize trader
        self.trader = trader if trader is not None else BybitFuturesOrderClass(
            symbol=self.config.symbol,
            trade_mode=self.config.trade_mode if hasattr(self.config, 'trade_mode') else None,
            api_key=api_key,
            api_secret=api_secret,
            demo=demo,
            leverage=self.config.leverage,
            margin_mode=self.config.margin_mode,
            position_mode=self.config.position_mode if hasattr(self.config, 'position_mode') else None,
        )

    def _build_observation_specs(self):
        """Build observation specs for account state and market data."""
        obs = self.observer.get_observations()
        first_key = self.observer.get_keys()[0]
        num_features = obs[first_key].shape[1]
        market_data_names = self.observer.get_keys()

        window_sizes = self.config.window_sizes if isinstance(self.config.window_sizes, list) else [self.config.window_sizes]

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

    def _get_observation(self) -> TensorDictBase:
        """Get the current observation state."""
        obs_dict = self.observer.get_observations(
            return_base_ohlc=self.config.include_base_features
        )

        if self.config.include_base_features:
            base_features = obs_dict.get("base_features")

        market_data = [obs_dict[features_name] for features_name in self.observer.get_keys()]

        # Get account state from trader
        status = self.trader.get_status()
        balance = self.trader.get_account_balance()

        total_balance = balance.get("total_wallet_balance", 0)
        position_status = status.get("position_status", None)

        if position_status is None:
            self.position.hold_counter = 0
            position_size = 0.0
            position_value = 0.0
            current_price = self.trader.get_mark_price()
            unrealized_pnl_pct = 0.0
            leverage = float(self.config.leverage)
            liquidation_price = 0.0
            holding_time = 0.0
        else:
            self.position.hold_counter += 1
            position_size = position_status.qty
            position_value = abs(position_status.notional_value)
            current_price = position_status.mark_price
            unrealized_pnl_pct = position_status.unrealized_pnl_pct
            leverage = float(position_status.leverage)
            liquidation_price = position_status.liquidation_price
            holding_time = float(self.position.hold_counter)

        # Build 6-element account state
        exposure_pct = position_value / total_balance if total_balance > 0 else 0.0

        position_direction = float(
            1 if position_size > 0
            else -1 if position_size < 0
            else 0
        )

        if position_size == 0 or current_price == 0 or liquidation_price <= 0:
            distance_to_liquidation = 1.0
        else:
            if position_size > 0:
                distance_to_liquidation = (current_price - liquidation_price) / current_price
            else:
                distance_to_liquidation = (liquidation_price - current_price) / current_price
            distance_to_liquidation = max(0.0, distance_to_liquidation)

        account_state = torch.tensor(
            [
                exposure_pct,
                position_direction,
                unrealized_pnl_pct,
                holding_time,
                leverage,
                distance_to_liquidation,
            ],
            dtype=torch.float,
        )

        out_td = TensorDict({self.account_state_key: account_state}, batch_size=())
        for market_data_name, data in zip(self.market_data_keys, market_data):
            out_td.set(market_data_name, torch.from_numpy(data))

        if self.config.include_base_features and base_features is not None:
            out_td.set("base_features", torch.from_numpy(base_features))

        return out_td

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value (includes unrealized PnL)."""
        balance = self.trader.get_account_balance()
        return balance.get("total_margin_balance", 0)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment."""
        self.trader.cancel_open_orders()
        self.history.reset()

        if self.config.close_position_on_reset:
            self.trader.close_position()

        balance = self.trader.get_account_balance()
        self.balance = balance.get("available_balance", 0)
        self.last_portfolio_value = self._get_portfolio_value()

        status = self.trader.get_status()
        position_status = status.get("position_status")
        self.position.hold_counter = 0

        if position_status is None:
            self.position.current_position = 0
        elif position_status.qty > 0:
            self.position.current_position = 1
        elif position_status.qty < 0:
            self.position.current_position = -1
        else:
            self.position.current_position = 0

        return self._get_observation()

    @abstractmethod
    def _execute_trade_if_needed(self, action) -> dict:
        """Execute trade if position change is needed."""
        raise NotImplementedError

    @abstractmethod
    def _check_termination(self, portfolio_value: float) -> bool:
        """Check if episode should terminate."""
        raise NotImplementedError

    def get_market_data_keys(self) -> List[str]:
        """Return the list of market data keys."""
        return self.market_data_keys

    def get_account_state(self) -> List[str]:
        """Return the list of account state field names."""
        return self.ACCOUNT_STATE

    def close(self):
        """Clean up resources."""
        try:
            status = self.trader.get_status()
            if status.get("position_status") and status["position_status"].qty != 0:
                logger.warning(
                    "Closing environment with open position! "
                    "Call env.trader.close_position() before env.close() if needed."
                )
        except Exception:
            pass

        self.trader.cancel_open_orders()
