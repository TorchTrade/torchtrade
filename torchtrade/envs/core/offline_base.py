"""Base class for offline trading environments."""

from typing import Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensordict import TensorDictBase
from torchrl.data import Bounded
from torchrl.data import Categorical, Composite, Unbounded

from torchtrade.envs.core.base import TorchTradeBaseEnv
from torchtrade.envs.core.state import HistoryTracker, PositionState
from tensordict import TensorDict
import torch


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
        from torchtrade.envs.offline.infrastructure.utils import InitialBalanceSampler
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

        # Initialize state attributes (set to valid defaults, will be properly set in _reset)
        self.current_timestamp = None
        self.truncated = False
        self._cached_base_features = None

        # Initialize position state
        self.position = PositionState()

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
        from torchtrade.envs.offline.infrastructure.sampler import MarketDataObservationSampler
        self.sampler = MarketDataObservationSampler(
            df,
            time_frames=self.config.time_frames,
            window_sizes=self.config.window_sizes,
            execute_on=self.config.execute_on,
            feature_processing_fn=feature_preprocessing_fn,
            features_start_with="features_",
            max_traj_length=self.config.max_traj_length,
            seed=self.config.seed,
        )

    def _build_observation_specs(
        self,
        account_state: List[str],
        num_features: Union[int, Dict[str, int]]
    ):
        """
        Build observation specs from sampler configuration.

        Args:
            account_state: List of account state field names
            num_features: Number of features per observation window.
                         Can be a single int (same for all timeframes) or
                         a dict mapping timeframe keys to feature counts
                         (for per-timeframe feature dimensions).
        """
        self.observation_spec = Composite(shape=())

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

        # Validate num_features dict has all required keys
        if isinstance(num_features, dict):
            missing_keys = set(market_data_keys) - set(num_features.keys())
            if missing_keys:
                raise ValueError(
                    f"num_features dict is missing keys for timeframes: {missing_keys}. "
                    f"Expected keys: {market_data_keys}, got: {list(num_features.keys())}"
                )

        for i, market_data_name in enumerate(market_data_keys):
            market_data_key = (
                f"market_data_{market_data_name}_{self.config.window_sizes[i]}"
            )
            # Get feature count for this timeframe
            if isinstance(num_features, dict):
                tf_num_features = num_features[market_data_name]
            else:
                tf_num_features = num_features

            market_data_spec = Bounded(
                low=-torch.inf,
                high=torch.inf,
                shape=(self.config.window_sizes[i], tf_num_features),
                dtype=torch.float
            )
            self.observation_spec.set(market_data_key, market_data_spec)
            self.market_data_keys.append(market_data_key)

        # Done spec: declare truncated so check_env_specs passes
        self.full_done_spec = Composite(
            done=Categorical(2, dtype=torch.bool, shape=(1,)),
            terminated=Categorical(2, dtype=torch.bool, shape=(1,)),
            truncated=Categorical(2, dtype=torch.bool, shape=(1,)),
        )

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

        HistoryTracker supports position tracking for all environment types.
        Use the position parameter in record_step() to track position size.
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
        self.position.reset()
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

        # Record initial state to history (before any actions)
        initial_price = self._cached_base_features["close"]
        initial_portfolio_value = self._get_portfolio_value(initial_price)
        self.history.record_step(
            price=initial_price,
            action=0.0,  # No action taken yet (hold)
            reward=0.0,  # No reward yet
            portfolio_value=initial_portfolio_value,
            position=0.0,  # Starting with no position
            action_type="hold"
        )

        # Add coverage tracking indices (only during training with random_start)
        if self.random_start:
            self._reset_idx = self.sampler._sequential_idx
            obs.set(
                "reset_index",
                torch.tensor(self._reset_idx, dtype=torch.long)
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

        return self.balance + self.position.position_size * current_price

    def _get_observation_scaffold(self):
        """
        Get observation data from sampler and cache state.

        Sets instance attributes:
        - self.current_timestamp: Current observation timestamp
        - self.truncated: Whether insufficient data remains to build full observation windows
          (True when approaching end of dataset, not just when completely exhausted)

        Returns:
            Tuple of (obs_dict, base_features) where obs_dict is a dictionary with
            market data observations for each timeframe, and base_features contains
            the base OHLCV features for the current timestamp.
        """
        obs_dict, self.current_timestamp, self.truncated = self.sampler.get_sequential_observation()
        base_features = self.sampler.get_base_features(self.current_timestamp)
        return obs_dict, base_features

    def _update_position_metrics(self, current_price: float):
        """Update position value and unrealized PnL based on current price.

        This is common logic used by most environments to update position metrics
        from the current price. Sets self.position.position_value and self.position.unrealized_pnlpc.

        Args:
            current_price: Current asset price
        """
        self.position.position_value = self.position.position_size * current_price
        if self.position.position_size > 0:
            self.position.unrealized_pnlpc = (current_price - self.position.entry_price) / self.position.entry_price
        else:
            self.position.unrealized_pnlpc = 0.0

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

    def _calculate_position_quantity(self, execution_price: float):
        """
        Calculate position quantity and notional value based on trade_mode.

        This method provides a common implementation for futures environments
        that support both QUANTITY and NOTIONAL trade modes.

        Args:
            execution_price: Price at which position will be opened

        Returns:
            Tuple of (position_qty, notional_value):
            - position_qty: Number of units (positive, sign handled by caller)
            - notional_value: Market value in quote currency (always positive)

        Raises:
            AttributeError: If config doesn't have trade_mode or quantity_per_trade
            ValueError: If trade_mode is unknown, quantity_per_trade is invalid,
                       or execution_price is non-positive in NOTIONAL mode
        """
        if self.config.trade_mode == "quantity":
            position_qty = self.config.quantity_per_trade
            notional_value = position_qty * execution_price
        elif self.config.trade_mode == "notional":
            if execution_price <= 0:
                raise ValueError(
                    f"execution_price must be positive for notional mode, got {execution_price}"
                )
            notional_value = self.config.quantity_per_trade
            position_qty = notional_value / execution_price
        else:
            raise ValueError(f"Unknown trade_mode: {self.config.trade_mode}")

        return position_qty, notional_value

    def _get_action_markers(self, action_types, action_history, position_history, is_futures):
        """Return list of (indices, marker, color, label, alpha) for action scatter plots."""
        exit_types = {"close", "liquidation", "sltp_sl", "sltp_tp"}

        if is_futures:
            if action_types:
                long_idx = [i for i, a in enumerate(action_types) if a == "long"]
                short_idx = [i for i, a in enumerate(action_types) if a == "short"]
                close_long_idx, close_short_idx = [], []
                for i, a in enumerate(action_types):
                    if a in exit_types and i > 0:
                        prev = position_history[i - 1]
                        if prev > 0:
                            close_long_idx.append(i)
                        elif prev < 0:
                            close_short_idx.append(i)
            else:
                long_idx = [i for i, a in enumerate(action_history) if a == 1]
                short_idx = [i for i, a in enumerate(action_history) if a == -1]
                close_long_idx, close_short_idx = [], []

            return [
                (long_idx, '^', 'green', 'Long Open', 1.0),
                (short_idx, 'v', 'red', 'Short Open', 1.0),
                (close_long_idx, 'v', 'orange', 'Long Close', 0.7),
                (close_short_idx, '^', 'cyan', 'Short Close', 0.7),
            ]

        # Spot/long-only
        if action_types:
            buy_idx = [i for i, a in enumerate(action_types) if a in ("buy", "long")]
            sell_idx = [i for i, a in enumerate(action_types) if a == "sell"]
            close_idx = [i for i, a in enumerate(action_types) if a in exit_types]
        else:
            buy_idx = [i for i, a in enumerate(action_history) if a == 1]
            sell_idx = [i for i, a in enumerate(action_history) if a == -1]
            close_idx = []

        return [
            (buy_idx, '^', 'green', 'Buy', 1.0),
            (sell_idx, 'v', 'red', 'Sell', 1.0),
            (close_idx, 'v', 'orange', 'Position Close', 0.7),
        ]

    def render_history(self, return_fig=False, plot_bh_baseline=True):
        """
        Render the history of the environment.

        Creates visualization plots showing:
        - For all environments: Price history with actions, Portfolio value (with optional buy-and-hold baseline comparison)
        - For futures environments: Additional position history plot

        Args:
            return_fig: If True, returns the matplotlib figure instead of showing it
            plot_bh_baseline: If True, plots the buy-and-hold baseline for comparison

        Returns:
            matplotlib.figure.Figure if return_fig=True, None otherwise
        """
        history_dict = self.history.to_dict()
        price_history = history_dict['base_prices']
        time_indices = list(range(len(price_history)))
        action_history = history_dict['actions']
        portfolio_value_history = history_dict['portfolio_values']

        # Calculate buy-and-hold baseline if requested
        if plot_bh_baseline:
            initial_balance = portfolio_value_history[0]
            initial_price = price_history[0]
            units_held = initial_balance / initial_price
            buy_and_hold_balance = [units_held * price for price in price_history]
        else:
            buy_and_hold_balance = None

        # Check if this is a futures environment (supports shorting)
        is_futures = getattr(self, 'allows_short', False)
        position_history = history_dict.get('positions', [])

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 10), sharex=True,
            gridspec_kw={'height_ratios': [3, 2, 1]}
        )

        # --- Subplot 1: Price history with action markers ---
        ax1.plot(
            time_indices, price_history,
            label='Price History', color='blue', linewidth=1.5
        )

        action_types = history_dict.get('action_types', [])
        markers = self._get_action_markers(
            action_types, action_history, position_history, is_futures
        )
        for indices, marker, color, label, alpha in markers:
            if indices:
                prices = [price_history[i] for i in indices]
                ax1.scatter(
                    indices, prices,
                    marker=marker, color=color, s=80, label=label,
                    zorder=5, alpha=alpha
                )

        action_label = 'Long/Short' if is_futures else 'Buy/Sell'
        ax1.set_ylabel('Price (USD)')
        ax1.set_title(f'Price History with {action_label} Actions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Subplot 2: Portfolio value ---
        ax2.plot(
            time_indices, portfolio_value_history,
            label='Portfolio Value', color='green', linewidth=1.5
        )
        if plot_bh_baseline:
            ax2.plot(
                time_indices, buy_and_hold_balance,
                label='Buy and Hold', color='purple', linestyle='--', linewidth=1.5
            )
            ax2.set_title('Portfolio Value vs Buy and Hold')
        else:
            ax2.set_title('Portfolio Value')
        ax2.set_ylabel('Portfolio Value (USD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- Subplot 3: Exposure history ---
        if position_history:
            exposure_pct = [
                (pos * price / pv * 100) if pv > 0 else 0.0
                for pos, price, pv in zip(position_history, price_history, portfolio_value_history)
            ]
            ax3.fill_between(
                time_indices, exposure_pct, 0,
                where=[e > 0 for e in exposure_pct],
                color='green', alpha=0.3, label='Long'
            )
            if is_futures:
                ax3.fill_between(
                    time_indices, exposure_pct, 0,
                    where=[e < 0 for e in exposure_pct],
                    color='red', alpha=0.3, label='Short'
                )
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_ylabel('Exposure (%)')
            ax3.set_title('Exposure History')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        ax3.set_xlabel('Time (Index)')

        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()
