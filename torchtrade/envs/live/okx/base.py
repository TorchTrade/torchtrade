"""Base class for OKX live trading environments."""
import logging

from abc import abstractmethod
from typing import Callable, List, Optional

import torch
from tensordict import TensorDictBase
from torchrl.data import Bounded
from torchrl.data.tensor_specs import Composite

from torchtrade.envs.live.okx.observation import OKXObservationClass
from torchtrade.envs.live.okx.order_executor import OKXFuturesOrderClass
from torchtrade.envs.live.shared.futures_live_base import TorchTradeFuturesLiveEnv
from torchtrade.envs.core.state import (
    HistoryTracker,
    position_direction_from_status,
)

logger = logging.getLogger(__name__)


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
        self.execute_on_value = config.execute_on.value
        self.execute_on_unit = str(config.execute_on.unit)

        # Flatten on startup for a clean state (configurable, default: True)
        self.trader.cancel_open_orders()
        if config.close_position_on_init:
            self.trader.close_position()

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

        # Base features spec (raw OHLC from first timeframe)
        if self.config.include_base_features:
            base_ws = window_sizes[0]
            self.observation_spec.set(
                "base_features",
                Bounded(low=-torch.inf, high=torch.inf, shape=(base_ws, 4), dtype=torch.float),
            )

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """Reset the environment."""
        if not self.trader.cancel_open_orders():
            logger.warning("cancel_open_orders failed during reset; proceeding with potentially stale orders")
        self.history.reset()

        if self.config.close_position_on_reset:
            if not self.trader.close_position():
                logger.warning("close_position failed during reset; proceeding with residual exposure")

        balance = self.trader.get_account_balance()
        self.balance = balance.get("available_balance", 0)
        self.last_portfolio_value = self._get_portfolio_value()

        status = self.trader.get_status()
        position_status = status.get("position_status")
        self.position.hold_counter = 0

        self.position.current_position = position_direction_from_status(position_status)

        # No-op today (okx's _execute_trade_if_needed recomputes qty live and never reads
        # current_action_level), but keeps the field consistent so adding a duplicate-action
        # guard here can't reintroduce the silent no-op that bit bitget/binance/alpaca.
        self._sync_action_level_after_reset()

        # advance_hold=False: hold_counter was just zeroed above; a reset must
        # never itself count a bar (see advance_hold docstring).
        return self._get_observation(advance_hold=False)

    @abstractmethod
    def _execute_trade_if_needed(self, action) -> dict:
        """Execute trade if position change is needed."""
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

        try:
            self.trader.cancel_open_orders()
        except Exception as e:
            logger.error(f"Failed to cancel open orders on close(): {e}")
