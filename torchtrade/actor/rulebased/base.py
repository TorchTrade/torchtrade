"""
Base class for rule-based trading actors.

Rule-based actors implement deterministic trading strategies using technical indicators
computed from market data. They are designed to generate expert demonstrations for
imitation learning and provide baselines for RL policy evaluation.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
from tensordict import TensorDict


class RuleBasedActor(ABC):
    """
    Abstract base class for rule-based trading actors.

    Rule-based actors implement deterministic trading strategies using
    technical indicators computed from market data. They are designed to:

    1. Generate expert demonstrations for imitation learning
    2. Provide baselines for RL policy evaluation
    3. Bootstrap RL training with prior knowledge

    All rule-based actors must implement the `select_action` method which
    takes an observation TensorDict and returns an action index.

    Args:
        market_data_keys: List of market data keys to use from observation
                         (e.g., ["market_data_1Minute_12", "market_data_5Minute_8"])
        features_order: Order of features in market data tensor
                       (e.g., ["close", "open", "high", "low", "volume"])
        action_space_size: Number of discrete actions (default: 3 for sell/hold/buy)
        debug: Whether to print debug information (default: False)
    """

    def __init__(
        self,
        market_data_keys: Optional[List[str]] = None,
        features_order: Optional[List[str]] = None,
        action_space_size: int = 3,
        debug: bool = False,
    ):
        self.market_data_keys = market_data_keys or ["market_data_1Minute_1"]
        self.features_order = features_order or ["close", "open", "high", "low", "volume"]
        self.action_space_size = action_space_size
        self.debug = debug

        # Build feature index mapping
        self.feature_idx = {feat: i for i, feat in enumerate(self.features_order)}

    def extract_market_data(self, observation: TensorDict) -> Dict[str, torch.Tensor]:
        """
        Extract and organize market data from observation TensorDict.

        Args:
            observation: TensorDict containing market data and account state

        Returns:
            Dictionary mapping timeframe names to market data tensors

        Example:
            >>> data = actor.extract_market_data(obs)
            >>> prices = data["1Minute"][:, actor.feature_idx["close"]]
        """
        market_data = {}
        for key in self.market_data_keys:
            if key in observation.keys():
                # Extract timeframe name (e.g., "market_data_5Minute_8" -> "5Minute")
                timeframe_name = key.replace("market_data_", "").rsplit("_", 1)[0]
                data = observation[key]

                # Handle batch dimension if present
                if len(data.shape) == 3:  # (batch, window, features)
                    data = data.squeeze(0)

                market_data[timeframe_name] = data

        return market_data

    def get_feature(self, data: torch.Tensor, feature_name: str) -> torch.Tensor:
        """
        Extract a specific feature from market data tensor.

        Args:
            data: Market data tensor of shape (window_size, num_features)
            feature_name: Name of feature to extract (e.g., "close", "high")

        Returns:
            1D tensor of feature values across the window

        Example:
            >>> closes = actor.get_feature(data, "close")  # Shape: (window_size,)
        """
        if feature_name not in self.feature_idx:
            raise ValueError(
                f"Feature '{feature_name}' not found. "
                f"Available features: {list(self.feature_idx.keys())}"
            )
        return data[:, self.feature_idx[feature_name]]

    @abstractmethod
    def select_action(self, observation: TensorDict) -> int:
        """
        Select an action based on the observation using rule-based strategy.

        Args:
            observation: TensorDict containing market data and account state

        Returns:
            Action index (0-based integer)

        Example:
            For 3-action space (SeqLongOnlyEnv):
                0 = Sell all
                1 = Hold (do nothing)
                2 = Buy all
        """
        pass

    def __call__(self, observation: TensorDict) -> TensorDict:
        """Make actor callable like a TorchRL policy."""
        return self.forward(observation)

    def forward(self, observation: TensorDict) -> TensorDict:
        """
        Process observation and set action in TensorDict.

        Args:
            observation: TensorDict containing market data and account state

        Returns:
            TensorDict with "action" key set
        """
        action = self.select_action(observation)

        if self.debug:
            print(f"{self.__class__.__name__} selected action: {action}")

        # Set action in observation TensorDict
        observation.set("action", torch.tensor([action], dtype=torch.long))

        return observation

    @abstractmethod
    def get_preprocessing_fn(self) -> Optional[Callable[[pd.DataFrame], pd.DataFrame]]:
        """
        Return the preprocessing function that computes required features.

        This function will be passed to the environment's feature_preprocessing_fn parameter.
        It receives the FULL resampled DataFrame for each timeframe and computes indicators
        on the complete historical data (proper lookback periods).

        Returns:
            Callable[[pd.DataFrame], pd.DataFrame] that takes a DataFrame and returns it
            with added features_* columns, or None if no preprocessing is needed.

        Example:
            >>> def preprocess(df: pd.DataFrame) -> pd.DataFrame:
            ...     df["features_rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
            ...     df["features_bb_upper"] = df["close"].rolling(20).mean() + 2*df["close"].rolling(20).std()
            ...     df.fillna(0, inplace=True)
            ...     return df
            >>> return preprocess
        """
        pass
