"""Base class for offline trading environments."""

from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDictBase
from torchrl.data import Bounded
from torchrl.data.tensor_specs import CompositeSpec, Unbounded

from torchtrade.envs.base import TorchTradeBaseEnv
from torchtrade.envs.offline.sampler import MarketDataObservationSampler
from torchtrade.envs.offline.utils import InitialBalanceSampler
from torchtrade.envs.state import HistoryTracker


class TorchTradeOfflineEnv(TorchTradeBaseEnv):
    """
    Base class for offline (backtesting) trading environments.

    Provides common functionality for all offline environments:
    - MarketDataObservationSampler initialization (single source of truth)
    - History tracking via HistoryTracker (price, action, reward, portfolio)
    - Coverage tracking indices (reset_index, state_index)
    - Reset logic scaffold
    - Market data observation spec construction
    - Common portfolio value calculation

    Subclasses must implement:
    - _step(): Environment step logic (trade execution)

    Attributes:
        history: HistoryTracker instance for episode history management
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config,
        feature_preprocessing_fn: Optional[Callable] = None
    ):
        """
        Initialize offline environment.

        Args:
            df: OHLCV DataFrame for backtesting
            config: Environment configuration with:
                   - time_frames: List[TimeFrame]
                   - window_sizes: List[int]
                   - execute_on: TimeFrame
                   - max_traj_length: Optional[int]
                   - initial_cash: Union[int, tuple]
                   - seed: Optional[int]
                   - random_start: bool
            feature_preprocessing_fn: Optional function to preprocess features
        """
        # Initialize base class first
        super().__init__(config)

        # Initialize sampler (SINGLE SOURCE OF TRUTH)
        self._init_sampler(df, feature_preprocessing_fn)

        # Initialize history tracking
        self._reset_history()

        # Initialize balance sampler
        self.initial_cash = config.initial_cash
        self.initial_cash_sampler = InitialBalanceSampler(
            config.initial_cash,
            config.seed
        )

        # Store reset settings
        self.random_start = config.random_start
        self.max_traj_length = config.max_traj_length

        # Store execution timeframe
        self.execute_on_value = config.execute_on.value
        self.execute_on_unit = config.execute_on.unit.value

        # Initialize step counter
        self.step_counter = 0
        self.max_steps = self.sampler.get_max_steps()

        # Initialize position tracking
        self.position_hold_counter = 0

        # Initialize state attributes (set to valid defaults, will be properly set in _reset)
        self.current_timestamp = None
        self.truncated = False
        self._cached_base_features = None

        # Initialize position state variables
        self.current_position = 0.0
        self.position_value = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnlpc = 0.0

    def _init_sampler(
        self,
        df: pd.DataFrame,
        feature_preprocessing_fn: Optional[Callable]
    ):
        """
        Initialize MarketDataObservationSampler.

        This is the SINGLE SOURCE OF TRUTH for sampler configuration.
        All offline environments use the same sampler initialization.

        Args:
            df: OHLCV DataFrame
            feature_preprocessing_fn: Optional feature preprocessing function
        """
        self.sampler = MarketDataObservationSampler(
            df,
            time_frames=self.config.time_frames,
            window_sizes=self.config.window_sizes,
            execute_on=self.config.execute_on,
            feature_processing_fn=feature_preprocessing_fn,
            features_start_with="features_",
            max_traj_length=self.config.max_traj_length,
        )

    def _build_observation_specs(
        self,
        account_state: List[str],
        num_features: int
    ):
        """
        Build observation specs from sampler configuration.

        Args:
            account_state: List of account state field names
            num_features: Number of features per observation window
        """
        self.observation_spec = CompositeSpec(shape=())

        # Account state spec
        self.account_state_key = "account_state"
        self.account_state = account_state
        account_state_spec = Bounded(
            low=-torch.inf,
            high=torch.inf,
            shape=(len(account_state),),
            dtype=torch.float
        )
        self.observation_spec.set(self.account_state_key, account_state_spec)

        # Market data specs (one per timeframe)
        market_data_keys = self.sampler.get_observation_keys()
        self.market_data_keys = []

        for i, market_data_name in enumerate(market_data_keys):
            market_data_key = (
                f"market_data_{market_data_name}_{self.config.window_sizes[i]}"
            )
            market_data_spec = Bounded(
                low=-torch.inf,
                high=torch.inf,
                shape=(self.config.window_sizes[i], num_features),
                dtype=torch.float
            )
            self.observation_spec.set(market_data_key, market_data_spec)
            self.market_data_keys.append(market_data_key)

        # Add coverage tracking indices (only when random_start=True)
        if self.random_start:
            # reset_index: tracks episode start position diversity
            self.observation_spec.set(
                "reset_index",
                Unbounded(shape=(), dtype=torch.long)
            )
            # state_index: tracks all timesteps visited during episodes
            self.observation_spec.set(
                "state_index",
                Unbounded(shape=(), dtype=torch.long)
            )

    def get_account_state(self) -> List[str]:
        """
        Get list of account state field names.

        Used for LLM actor integration to understand account state structure.

        Returns:
            List of account state field names (e.g., ["cash", "position_size", ...])
        """
        return self.account_state

    def get_market_data_keys(self) -> List[str]:
        """
        Get list of market data observation keys.

        Used for LLM actor integration to understand market data structure.

        Returns:
            List of market data keys (e.g., ["market_data_1Minute_10", ...])
        """
        return self.market_data_keys

    def _reset_history(self):
        """Reset all history tracking to a new HistoryTracker instance.

        Subclasses supporting futures trading should override this method
        to instantiate FuturesHistoryTracker instead:

        Example:
            def _reset_history(self):
                self.history = FuturesHistoryTracker()
        """
        self.history = HistoryTracker()

    def _reset_balance(self):
        """Reset balance to sampled initial value."""
        initial_portfolio_value = self.initial_cash_sampler.sample()
        self.balance = initial_portfolio_value
        self.initial_portfolio_value = initial_portfolio_value

    def _reset_position_state(self):
        """Reset position tracking state.

        Subclasses may override to add additional position state.
        """
        self.position_hold_counter = 0
        self.current_position = 0.0
        self.position_value = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnlpc = 0.0
        self.step_counter = 0

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """
        Reset the environment.

        Common reset logic for all offline environments:
        1. Reset history tracking
        2. Reset sampler (with optional random start)
        3. Reset balance
        4. Reset position state
        5. Get initial observation
        6. Add coverage tracking indices (if random_start)

        Returns:
            Initial observation TensorDict
        """
        # Reset history
        self._reset_history()

        # Reset sampler and get max trajectory length
        max_episode_steps = self.sampler.reset(random_start=self.random_start)

        # Validate sampler returned a valid trajectory length
        if not isinstance(max_episode_steps, (int, np.integer)):
            raise TypeError(
                f"sampler.reset() must return an integer, got {type(max_episode_steps).__name__}. "
                f"This indicates a bug in the sampler implementation."
            )

        if max_episode_steps <= 0:
            raise ValueError(
                f"sampler.reset() returned invalid max_episode_steps: {max_episode_steps}. "
                f"Must be positive. This may indicate insufficient data in the dataset."
            )

        self.max_traj_length = max_episode_steps

        # Reset balance
        self._reset_balance()

        # Reset position state
        self._reset_position_state()

        # Get initial observation
        obs = self._get_observation()

        # Add coverage tracking indices (only during training with random_start)
        if self.random_start:
            obs.set(
                "reset_index",
                torch.tensor(self.sampler._sequential_idx, dtype=torch.long)
            )
            obs.set(
                "state_index",
                torch.tensor(self.sampler._sequential_idx, dtype=torch.long)
            )

        return obs

    def _get_portfolio_value(self, current_price: Optional[float] = None) -> float:
        """
        Calculate total portfolio value for offline environments.

        Formula: balance + position_size * current_price

        Args:
            current_price: Current asset price. If None, fetches from sampler.

        Returns:
            Total portfolio value

        Raises:
            RuntimeError: If current_timestamp not set (must call _get_observation first)
            ValueError: If current_price is invalid (NaN, inf, or negative)
        """
        if current_price is None:
            if self.current_timestamp is None:
                raise RuntimeError(
                    "current_timestamp is not set. _get_portfolio_value() must be called "
                    "after _get_observation() which sets the current timestamp. "
                    "This indicates an environment implementation error."
                )
            current_price = self.sampler.get_base_features(self.current_timestamp)["close"]

        # Validate price is a valid number
        if not isinstance(current_price, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"current_price must be a number, got {type(current_price).__name__}"
            )

        if np.isnan(current_price) or np.isinf(current_price):
            raise ValueError(
                f"Invalid price: {current_price} at timestamp {self.current_timestamp}. "
                f"This indicates corrupted data in the dataset (NaN or infinity values)."
            )

        if current_price < 0:
            raise ValueError(
                f"Negative price: {current_price} at timestamp {self.current_timestamp}. "
                f"This indicates invalid data in the dataset."
            )

        return self.balance + self.position_size * current_price

    def _get_observation_scaffold(self):
        """
        Get observation data from sampler and cache state.

        Sets instance attributes:
        - self.current_timestamp: Current observation timestamp
        - self.truncated: Whether insufficient data remains to build full observation windows
          (True when approaching end of dataset, not just when completely exhausted)
        - self._cached_base_features: Cached base features for timestamp

        Returns:
            Dictionary with market data observations for each timeframe
        """
        obs_dict, self.current_timestamp, self.truncated = self.sampler.get_sequential_observation()
        self._cached_base_features = self.sampler.get_base_features(self.current_timestamp)
        return obs_dict

    def _update_position_metrics(self, current_price: float):
        """Update position value and unrealized PnL based on current price.

        This is common logic used by most environments to update position metrics
        from the current price. Sets self.position_value and self.unrealized_pnlpc.

        Args:
            current_price: Current asset price
        """
        self.position_value = round(self.position_size * current_price, 3)
        if self.position_size > 0:
            self.unrealized_pnlpc = round(
                (current_price - self.entry_price) / self.entry_price, 4
            )
        else:
            self.unrealized_pnlpc = 0.0

    def _build_standard_observation(
        self,
        obs_dict: dict,
        account_state_values: list
    ) -> TensorDictBase:
        """Build observation TensorDict from market data and account state.

        This is common logic used by most environments to assemble the final
        observation from market data and account state.

        Args:
            obs_dict: Market data observation dictionary from sampler
            account_state_values: List of account state values

        Returns:
            TensorDict with combined observation
        """
        from tensordict import TensorDict
        import torch

        account_state = torch.tensor(account_state_values, dtype=torch.float)
        obs_data = {self.account_state_key: account_state}
        obs_data.update(dict(zip(self.market_data_keys, obs_dict.values())))
        return TensorDict(obs_data, batch_size=())

    def _validate_observation(self, obs: TensorDictBase) -> None:
        """Validate observation for NaN/Inf values.

        Checks all tensors in the observation for invalid values and raises
        informative errors if found. This helps catch data corruption early
        before it causes silent failures downstream.

        Args:
            obs: Observation TensorDict to validate

        Raises:
            ValueError: If any NaN or Inf values are found in the observation

        Example:
            >>> obs = self._get_observation()
            >>> self._validate_observation(obs)  # Raises if invalid
            >>> return obs
        """
        import torch

        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    raise ValueError(
                        f"Invalid observation: NaN values found in '{key}' at timestamp {self.current_timestamp}. "
                        f"This indicates corrupted data or calculation errors. "
                        f"Check your dataset and feature preprocessing function."
                    )
                if torch.isinf(value).any():
                    raise ValueError(
                        f"Invalid observation: Inf values found in '{key}' at timestamp {self.current_timestamp}. "
                        f"This indicates corrupted data or calculation errors (possible division by zero). "
                        f"Check your dataset and feature preprocessing function."
                    )
