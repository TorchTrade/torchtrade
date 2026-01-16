"""Base class for live trading environments."""

import logging
import time
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tensordict import TensorDictBase

from torchtrade.envs.base import TorchTradeBaseEnv
from torchtrade.envs.state import PositionState

logger = logging.getLogger(__name__)


class TorchTradeLiveEnv(TorchTradeBaseEnv):
    """
    Base class for live trading environments.

    Provides common functionality for all live trading environments:
    - Observer/trader dependency injection pattern
    - Common waiting logic (_wait_for_next_timestamp)
    - Market data observation spec construction pattern
    - Reset scaffolding

    Subclasses must implement:
    - _init_trading_clients(): Provider-specific client initialization
    - _get_portfolio_value(): Provider-specific portfolio calculation
    - _step(): Environment step logic
    """

    def __init__(
        self,
        config,
        api_key: str = "",
        api_secret: str = "",
        observer=None,
        trader=None,
        timezone: str = "America/New_York"
    ):
        """
        Initialize live trading environment.

        Args:
            config: Environment configuration
            api_key: API key for trading provider
            api_secret: API secret for trading provider
            observer: Optional pre-configured observation client (dependency injection)
            trader: Optional pre-configured trading client (dependency injection)
            timezone: Timezone for time-based operations (default: America/New_York)
        """
        # Initialize base class first
        super().__init__(config)

        # Store timezone with validation
        try:
            self.timezone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as e:
            raise ValueError(
                f"Invalid timezone: '{timezone}'. Common valid timezones include: "
                f"'America/New_York', 'America/Chicago', 'America/Los_Angeles', 'UTC', "
                f"'Europe/London', 'Asia/Tokyo'. See "
                f"https://en.wikipedia.org/wiki/List_of_tz_database_time_zones "
                f"for a complete list of valid timezone names."
            ) from e

        # Store execution timeframe
        # Note: Subclasses should ensure execute_on has 'value' and 'unit' attributes
        # For TimeFrame objects: execute_on.value (int) and execute_on.unit (TimeFrameUnit enum)
        self.execute_on_value = None
        self.execute_on_unit = None

        # Initialize trading clients (observer and trader)
        # Subclasses should override _init_trading_clients to provide provider-specific setup
        self._init_trading_clients(api_key, api_secret, observer, trader)

        # Initialize position state
        # Note: Subclasses may override this with their specific position tracking needs
        self.position = PositionState()

    @abstractmethod
    def _init_trading_clients(
        self,
        api_key: str,
        api_secret: str,
        observer,
        trader
    ):
        """
        Initialize observer and trader clients.

        Subclasses must implement this to provide provider-specific initialization:
        - Use injected observer/trader if provided
        - Otherwise create new instances with api_key/api_secret

        Should set:
        - self.observer: Observation/data client
        - self.trader: Order execution client

        Args:
            api_key: API key for provider
            api_secret: API secret for provider
            observer: Optional pre-configured observer
            trader: Optional pre-configured trader
        """
        raise NotImplementedError(
            "Subclasses must implement _init_trading_clients()"
        )

    def _wait_for_next_timestamp(self):
        """
        Wait until next time step - improved version.

        This is COMMON across all live environments - timing logic is universal.

        Uses:
        - self.execute_on_value: Number of time units
        - self.execute_on_unit: Time unit (e.g., "Minute", "Hour", "Day")
        - self.timezone: Timezone for time calculations
        """
        # Map time unit strings to timedelta kwargs
        unit_to_timedelta = {
            "TimeFrameUnit.Minute": "minutes",
            "TimeFrameUnit.Hour": "hours",
            "TimeFrameUnit.Day": "days",
            # Alpaca uses different naming
            "Minute": "minutes",
            "Hour": "hours",
            "Day": "days",
        }

        if self.execute_on_unit not in unit_to_timedelta:
            raise ValueError(
                f"Unsupported time unit: {self.execute_on_unit}. "
                f"Supported: {list(unit_to_timedelta.keys())}"
            )

        # Calculate wait duration
        wait_duration = timedelta(
            **{unit_to_timedelta[self.execute_on_unit]: self.execute_on_value}
        )

        # Calculate next step time
        current_time = datetime.now(self.timezone)
        next_step = (current_time + wait_duration).replace(second=0, microsecond=0)

        # Calculate exact sleep duration
        sleep_seconds = (next_step - datetime.now(self.timezone)).total_seconds()

        # Single sleep instead of polling loop
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    @abstractmethod
    def _get_portfolio_value(self, *args, **kwargs) -> float:
        """
        Calculate total portfolio value.

        MUST be implemented by subclasses as calculation is provider-specific.

        Examples:
        - Alpaca spot: cash + position_market_value
        - Binance futures: total_margin_balance
        - Interactive Brokers: net_liquidation_value

        Returns:
            Total portfolio value (float)
        """
        raise NotImplementedError(
            "Subclasses must implement _get_portfolio_value() "
            "as portfolio calculation is provider-specific"
        )

    @abstractmethod
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Execute one environment step.

        MUST be implemented by subclasses for provider-specific step logic.

        Args:
            tensordict: Input tensordict containing action

        Returns:
            TensorDict with next observation, reward, done flags
        """
        raise NotImplementedError(
            "Subclasses must implement _step() for environment step logic"
        )
