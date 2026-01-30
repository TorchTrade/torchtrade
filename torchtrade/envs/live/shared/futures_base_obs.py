"""Base observation class for futures exchanges."""

from abc import ABC, abstractmethod
from typing import List, Union, Callable, Dict, Optional
import numpy as np
import pandas as pd

from torchtrade.envs.utils.timeframe import TimeFrame


class BaseFuturesObservationClass(ABC):
    """
    Base observation class for fetching market data from futures exchanges.

    Provides common functionality for multi-timeframe observations with custom
    feature preprocessing. Provider-specific implementations (Bitget, Binance, etc.)
    only need to implement data fetching and parsing logic.
    """

    def __init__(
        self,
        symbol: str,
        time_frames: Union[List[TimeFrame], TimeFrame],
        window_sizes: Union[List[int], int] = 10,
        feature_preprocessing_fn: Optional[Callable] = None,
        client: Optional[object] = None,
        demo: bool = True,
    ):
        """
        Initialize the observation class.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            time_frames: Single TimeFrame or list of TimeFrame objects
            window_sizes: Single integer or list of integers specifying window sizes.
                         If a list is provided, it must have the same length as time_frames.
            feature_preprocessing_fn: Optional custom preprocessing function that takes a DataFrame
                                     and returns a DataFrame with feature columns
            client: Optional pre-configured client for dependency injection (useful for testing)
            demo: Whether to use demo/testnet environment (default: True)
        """
        # Normalize symbol (remove slash if present)
        self.symbol = symbol.replace("/", "")

        # Normalize time_frames and window_sizes to lists
        self.time_frames = [time_frames] if isinstance(time_frames, TimeFrame) else time_frames
        self.window_sizes = [window_sizes] if isinstance(window_sizes, int) else window_sizes
        self.demo = demo

        # Validate time_frames
        for tf in self.time_frames:
            self._validate_timeframe(tf)

        # Validate that time_frames and window_sizes have matching lengths
        if len(self.time_frames) != len(self.window_sizes):
            raise ValueError("time_frames and window_sizes must have the same length")

        self.feature_preprocessing_fn = feature_preprocessing_fn or self._default_preprocessing

        # Initialize client (provider-specific)
        self.client = client if client is not None else self._create_client()

    @abstractmethod
    def _create_client(self) -> object:
        """
        Create provider-specific API client.

        Returns:
            Initialized API client for the provider
        """
        pass

    @abstractmethod
    def _validate_timeframe(self, timeframe: TimeFrame) -> None:
        """
        Validate that a timeframe is supported by this provider.

        Args:
            timeframe: TimeFrame object to validate

        Raises:
            ValueError: If timeframe is not supported
        """
        pass

    @abstractmethod
    def _fetch_klines(self, symbol: str, interval: str, limit: int) -> list:
        """
        Fetch raw kline/candle data from the provider's API.

        Args:
            symbol: Trading symbol
            interval: Provider-specific interval string
            limit: Number of candles to fetch

        Returns:
            Raw kline data from the API
        """
        pass

    @abstractmethod
    def _parse_klines(self, raw_klines: list) -> pd.DataFrame:
        """
        Parse raw kline data into a standardized DataFrame.

        The returned DataFrame must have at least these columns:
        - timestamp: datetime column
        - open: float
        - high: float
        - low: float
        - close: float
        - volume: float

        Args:
            raw_klines: Raw kline data from the API

        Returns:
            DataFrame with standardized OHLCV columns
        """
        pass

    @abstractmethod
    def _convert_timeframe(self, timeframe: TimeFrame) -> str:
        """
        Convert TimeFrame to provider-specific interval format.

        Args:
            timeframe: TimeFrame object

        Returns:
            Provider-specific interval string (e.g., "1H" for Bitget, "1h" for Binance)
        """
        pass

    @abstractmethod
    def _get_default_lookback(self) -> int:
        """Get the default number of candles to fetch."""
        pass

    @abstractmethod
    def _get_timestamp_column(self) -> str:
        """
        Get the name of the primary timestamp column for base observations.

        Returns:
            Column name (e.g., "timestamp" for Bitget, "open_time" for Binance)
        """
        pass

    # Common methods (shared across all providers)

    def _default_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Default preprocessing function if none is provided."""
        df = df.copy()
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        # Generating base features (normalized)
        df["feature_close"] = df["close"].pct_change().fillna(0)
        df["feature_open"] = df["open"] / df["close"]
        df["feature_high"] = df["high"] / df["close"]
        df["feature_low"] = df["low"] / df["close"]
        df.dropna(inplace=True)
        return df

    def _get_numpy_obs(self, df: pd.DataFrame, columns: List[str] = None) -> np.ndarray:
        """Convert specified columns to numpy array."""
        if columns is None:
            columns = [col for col in df.columns if "feature" in col]
        if not columns:
            raise ValueError(f"No columns found in preprocessed DataFrame matching: {columns}")
        return np.array(df[columns], dtype=np.float32)

    def _fetch_single_timeframe(self, timeframe: TimeFrame, limit: int = None) -> pd.DataFrame:
        """Fetch and preprocess data for a single timeframe."""
        if limit is None:
            limit = self._get_default_lookback()

        # Convert TimeFrame to provider-specific interval format
        provider_interval = self._convert_timeframe(timeframe)

        try:
            # Fetch klines using provider-specific method
            raw_klines = self._fetch_klines(self.symbol, provider_interval, limit)

            # Validate we got data
            if not raw_klines or len(raw_klines) == 0:
                raise ValueError(f"No candle data returned for {self.symbol} on {timeframe.obs_key_freq()}")

            # Parse klines into standardized DataFrame
            df = self._parse_klines(raw_klines)

            # Sort by timestamp (ascending) - some exchanges return in reverse order
            timestamp_col = self._get_timestamp_column()
            df = df.sort_values(timestamp_col).reset_index(drop=True)

            return df

        except Exception as e:
            raise RuntimeError(f"Failed to fetch candles for {self.symbol} on {timeframe.obs_key_freq()}: {str(e)}")

    def get_keys(self) -> List[str]:
        """
        Get the list of keys that will be present in the observations dictionary.

        Returns:
            List of strings formatted as '{timeframe}_{window_size}'
            (e.g., '5Minute_10', '1Hour_20')
        """
        return [
            f"{tf.obs_key_freq()}_{ws}"
            for tf, ws in zip(self.time_frames, self.window_sizes)
        ]

    def get_features(self) -> Dict[str, List[str]]:
        """Returns a dictionary with the observation features and the original features."""
        def get_dummy_data(window_size: int):
            df = pd.DataFrame()
            df["open"] = np.random.rand(window_size)
            df["high"] = np.random.rand(window_size)
            df["low"] = np.random.rand(window_size)
            df["close"] = np.random.rand(window_size)
            df["volume"] = np.random.rand(window_size)
            return df

        dummy_df = self.feature_preprocessing_fn(get_dummy_data(self.window_sizes[0]))
        observation_features = [f for f in dummy_df.columns if "feature" in f]
        original_features = [f for f in dummy_df.columns if "feature" not in f]
        return {"observation_features": observation_features, "original_features": original_features}

    def get_observations(self, return_base_ohlc: bool = False) -> Dict[str, np.ndarray]:
        """
        Fetch and process observations for all specified timeframes and window sizes.

        Args:
            return_base_ohlc: If True, includes the raw OHLC data from the first timeframe
                            in the observations dictionary under the 'base_features' key.

        Returns:
            Dictionary with keys formatted as '{timeframe}_{window_size}' and numpy array values.
            If return_base_ohlc is True, includes 'base_features' and 'base_timestamps' keys.
        """
        observations = {}

        for timeframe, window_size in zip(self.time_frames, self.window_sizes):
            key = f"{timeframe.obs_key_freq()}_{window_size}"
            # Fetch extra data for preprocessing (rolling windows may need more)
            df = self._fetch_single_timeframe(timeframe, limit=window_size + 50)

            # Store base OHLC features if this is the first timeframe and return_base_ohlc is True
            if return_base_ohlc and timeframe == self.time_frames[0]:
                base_df = df.copy()
                timestamp_col = self._get_timestamp_column()
                observations['base_features'] = self._get_numpy_obs(
                    base_df,
                    columns=['open', 'high', 'low', 'close']
                )[-window_size:]
                observations['base_timestamps'] = base_df[timestamp_col].values[-window_size:]

            processed_df = self.feature_preprocessing_fn(df)
            # Apply window size
            processed_df = processed_df.iloc[-window_size:]
            observations[key] = self._get_numpy_obs(processed_df)

        return observations
