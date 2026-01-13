"""Base environment classes for TorchTrade."""

from abc import abstractmethod
from typing import Dict, Optional

import numpy as np
import torch
from torchrl.data import Bounded
from torchrl.envs import EnvBase

from torchtrade.envs.reward import build_reward_context, default_log_return, validate_reward_function


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
                   - reward_function: Optional[Callable]
                   - reward_scaling: float
        """
        self.config = config

        # Validate custom reward function signature if provided
        reward_function = getattr(config, 'reward_function', None)
        if reward_function is not None:
            validate_reward_function(reward_function)

        # Validate and store transaction parameters
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

    def _calculate_reward(
        self,
        old_portfolio_value: float,
        new_portfolio_value: float,
        action: float,
        trade_info: Dict,
    ) -> float:
        """
        Calculate reward using custom or default function.

        If config.reward_function is provided, uses that custom function.
        Otherwise, uses the default log return reward.

        Args:
            old_portfolio_value: Portfolio value before action
            new_portfolio_value: Portfolio value after action
            action: Action taken
            trade_info: Dictionary with trade execution details

        Returns:
            Reward value (float)
        """
        # Use custom reward function if provided
        reward_function = getattr(self.config, 'reward_function', None)
        if reward_function is not None:
            # Compute buy & hold value if terminal
            is_terminal = self.step_counter >= self.max_traj_length - 1
            buy_and_hold_value = None

            # Only compute buy_and_hold if we have sufficient price history (need at least 2 points)
            # and first price is valid (non-zero to avoid division by zero)
            if (is_terminal and hasattr(self, 'base_price_history') and
                len(self.base_price_history) >= 2):
                first_price = self.base_price_history[0]
                if first_price > 0:
                    buy_and_hold_value = (
                        self.initial_portfolio_value / first_price
                    ) * self.base_price_history[-1]
                # else: first_price is <= 0, skip buy_and_hold calculation (silently, as this is data issue)

            # Check if this is an offline environment (has history tracking)
            # For offline envs using custom rewards, history attributes are required
            is_offline_env = hasattr(self, 'sampler') and hasattr(self, 'base_price_history')

            if is_offline_env:
                # Validate that offline environment has all required history attributes
                required_attrs = ['portfolio_value_history', 'action_history',
                                'reward_history', 'base_price_history', 'initial_portfolio_value']
                missing = [attr for attr in required_attrs if not hasattr(self, attr)]
                if missing:
                    raise RuntimeError(
                        f"Offline environment using custom reward function is missing required "
                        f"history tracking attributes: {missing}. Ensure the environment properly "
                        f"inherits from TorchTradeOfflineEnv and calls super().__init__()."
                    )

                ctx = build_reward_context(
                    self,
                    old_portfolio_value,
                    new_portfolio_value,
                    action,
                    trade_info,
                    portfolio_value_history=self.portfolio_value_history,
                    action_history=self.action_history,
                    reward_history=self.reward_history,
                    base_price_history=self.base_price_history,
                    initial_portfolio_value=self.initial_portfolio_value,
                    buy_and_hold_value=buy_and_hold_value,
                )
            else:
                # Live environments don't track history, pass empty data
                ctx = build_reward_context(
                    self,
                    old_portfolio_value,
                    new_portfolio_value,
                    action,
                    trade_info,
                    portfolio_value_history=[],
                    action_history=[],
                    reward_history=[],
                    base_price_history=[],
                    initial_portfolio_value=None,
                    buy_and_hold_value=None,
                )

            return float(self.config.reward_function(ctx)) * self.config.reward_scaling

        # Otherwise use default log return
        return default_log_return(old_portfolio_value, new_portfolio_value) * self.config.reward_scaling

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
