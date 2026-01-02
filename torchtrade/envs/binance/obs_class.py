from typing import List, Union, Callable, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# Binance kline intervals mapping
INTERVAL_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
    "1M": "1M",
}


class BinanceObservationClass:
    """
    Observation class for fetching market data from Binance.

    Similar to AlpacaObservationClass but uses Binance API for data fetching.
    Supports multiple timeframes and custom feature preprocessing.
    """

    def __init__(
        self,
        symbol: str,
        intervals: Union[List[str], str],
        window_sizes: Union[List[int], int] = 10,
        feature_preprocessing_fn: Optional[Callable] = None,
        client: Optional[object] = None,
        demo: bool = True,
    ):
        """
        Initialize the BinanceObservationClass.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT")
            intervals: Single interval or list of intervals (e.g., "1m", "5m", "1h")
                      Valid intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
            window_sizes: Single integer or list of integers specifying window sizes.
                         If a list is provided, it must have the same length as intervals.
            feature_preprocessing_fn: Optional custom preprocessing function that takes a DataFrame
                                     and returns a DataFrame with feature columns
            client: Optional pre-configured Binance Client for dependency injection (useful for testing)
            demo: Whether to use demo/testnet environment (default: True)
        """
        # Normalize symbol (remove slash if present)
        self.symbol = symbol.replace("/", "")

        self.intervals = [intervals] if isinstance(intervals, str) else intervals
        self.window_sizes = [window_sizes] if isinstance(window_sizes, int) else window_sizes
        self.demo = demo
        self.default_lookback = 500  # Binance default limit

        # Validate intervals
        for interval in self.intervals:
            if interval not in INTERVAL_MAP:
                raise ValueError(f"Invalid interval: {interval}. Valid intervals: {list(INTERVAL_MAP.keys())}")

        # Validate that intervals and window_sizes have matching lengths
        if len(self.intervals) != len(self.window_sizes):
            raise ValueError("intervals and window_sizes must have the same length")

        self.feature_preprocessing_fn = feature_preprocessing_fn or self._default_preprocessing

        # Initialize client if not provided
        if client is not None:
            self.client = client
        else:
            try:
                from binance.client import Client
                # For observation only, we don't need API keys
                self.client = Client()
            except ImportError:
                raise ImportError("python-binance is required. Install with: pip install python-binance")

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

    def _fetch_single_interval(self, interval: str, limit: int = None) -> pd.DataFrame:
        """Fetch and preprocess data for a single interval."""
        if limit is None:
            limit = self.default_lookback

        # Fetch klines from Binance
        klines = self.client.get_klines(
            symbol=self.symbol,
            interval=interval,
            limit=limit
        )

        # Convert to DataFrame
        # Kline format: [open_time, open, high, low, close, volume, close_time,
        #                quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert types
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        return df

    def get_keys(self) -> List[str]:
        """
        Get the list of keys that will be present in the observations dictionary.

        Returns:
            List of strings formatted as '{interval}_{window_size}'
        """
        return [
            f"{interval}_{ws}"
            for interval, ws in zip(self.intervals, self.window_sizes)
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
        Fetch and process observations for all specified intervals and window sizes.

        Args:
            return_base_ohlc: If True, includes the raw OHLC data from the first interval
                            in the observations dictionary under the 'base_features' key.

        Returns:
            Dictionary with keys formatted as '{interval}_{window_size}' and numpy array values.
            If return_base_ohlc is True, includes 'base_features' and 'base_timestamps' keys.
        """
        observations = {}

        for interval, window_size in zip(self.intervals, self.window_sizes):
            key = f"{interval}_{window_size}"
            # Fetch extra data for preprocessing (rolling windows may need more)
            df = self._fetch_single_interval(interval, limit=window_size + 50)

            # Store base OHLC features if this is the first interval and return_base_ohlc is True
            if return_base_ohlc and interval == self.intervals[0]:
                base_df = df.copy()
                observations['base_features'] = self._get_numpy_obs(
                    base_df,
                    columns=['open', 'high', 'low', 'close']
                )[-window_size:]
                observations['base_timestamps'] = base_df['open_time'].values[-window_size:]

            processed_df = self.feature_preprocessing_fn(df)
            # Apply window size
            processed_df = processed_df.iloc[-window_size:]
            observations[key] = self._get_numpy_obs(processed_df)

        return observations


# Example usage:
if __name__ == "__main__":
    # Single interval example
    print("Testing single interval...")
    window_size = 10
    observer = BinanceObservationClass(
        symbol="BTCUSDT",
        intervals="15m",
        window_sizes=window_size,
    )
    expected_keys = observer.get_keys()
    print(f"Expected keys: {expected_keys}")
    observations = observer.get_observations()

    assert set(observations.keys()) == set(expected_keys), "Keys don't match expected keys"
    assert observations[expected_keys[0]].shape == (window_size, 4), \
        f"Expected shape ({window_size}, 4) for default features, got {observations[expected_keys[0]].shape}"
    print("Single interval test passed!")

    # Example with multiple intervals and window sizes
    print("\nTesting multiple intervals...")
    window_sizes = [10, 20]
    observer = BinanceObservationClass(
        symbol="BTCUSDT",
        intervals=["15m", "1h"],
        window_sizes=window_sizes
    )

    expected_keys = observer.get_keys()
    print("Expected keys:", expected_keys)
    observations = observer.get_observations()

    assert set(observations.keys()) == set(expected_keys), "Keys don't match expected keys"
    assert len(observations) == 2, "Expected exactly 2 observations"
    print("Multiple intervals test passed!")
