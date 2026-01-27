"""Base environment classes for TorchTrade."""

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch
from torchrl.data import Bounded
from torchrl.envs import EnvBase

logger = logging.getLogger(__name__)


class TorchTradeBaseEnv(EnvBase):
    """
    Base class for all TorchTrade environments.

    Provides common functionality including:
    - Config validation (transaction fees, slippage)
    - Reward calculation logic (custom vs. default)
    - Portfolio value calculation interface
    - Seed setting
    - Reward spec construction

    Subclasses must implement:
    - _get_portfolio_value(): Provider/mode-specific portfolio calculation
    """

    def __init__(self, config):
        """
        Initialize base environment with common configuration.

        Args:
            config: Environment configuration dataclass with at least:
                   - transaction_fee: float
                   - slippage: float
                   - seed: Optional[int]
                   - reward_function: Optional[Callable] (takes history tracker, returns float)
        """
        self.config = config

        # Validate custom reward function signature if provided
        reward_function = getattr(config, 'reward_function', None)
        if reward_function is not None:
            self._validate_reward_function(reward_function)

        # Validate and store transaction parameters (only for offline environments)
        if hasattr(config, 'transaction_fee') and hasattr(config, 'slippage'):
            self._validate_transaction_parameters(config)
            self.transaction_fee = config.transaction_fee
            self.slippage = config.slippage

        # Create reward spec (common across all environments)
        self.reward_spec = Bounded(
            low=-torch.inf,
            high=torch.inf,
            shape=(1,),
            dtype=torch.float
        )

        super().__init__()

    def _validate_transaction_parameters(self, config):
        """Validate transaction fee and slippage parameters.

        Args:
            config: Environment configuration

        Raises:
            ValueError: If parameters are out of valid range [0, 1]
        """
        if not (0 <= config.transaction_fee <= 1):
            raise ValueError(
                f"Transaction fee must be between 0 and 1, got {config.transaction_fee}"
            )
        if not (0 <= config.slippage <= 1):
            raise ValueError(
                f"Slippage must be between 0 and 1, got {config.slippage}"
            )

    def _validate_reward_function(self, reward_function: Any):
        """Validate that custom reward function has correct signature.

        Reward functions must accept a single parameter (history tracker) and return a float.

        Args:
            reward_function: The custom reward function to validate

        Raises:
            TypeError: If the reward function doesn't have the correct signature

        Example:
            >>> def my_reward(history) -> float:
            ...     return np.log(history.portfolio_values[-1] / history.portfolio_values[-2])
        """
        if reward_function is None:
            return

        import inspect

        sig = inspect.signature(reward_function)
        params = list(sig.parameters.values())

        if len(params) != 1:
            raise TypeError(
                f"Reward function must accept exactly 1 parameter (history tracker), "
                f"but got {len(params)} parameters: {list(sig.parameters.keys())}. "
                f"Expected signature: def reward_function(history) -> float"
            )

    @abstractmethod
    def _get_portfolio_value(self, *args, **kwargs) -> float:
        """
        Calculate total portfolio value.

        Must be implemented by subclasses as calculation is environment-specific:
        - Offline environments: balance + position_size * current_price
        - Alpaca spot: cash + position_market_value
        - Binance futures: total_margin_balance

        Returns:
            Total portfolio value (float)
        """
        raise NotImplementedError(
            "Subclasses must implement _get_portfolio_value()"
        )

    def _set_seed(self, seed: Optional[int] = None):
        """
        Set the random seed for the environment.

        Args:
            seed: Random seed to use. If None, uses config.seed
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        elif hasattr(self.config, 'seed') and self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
