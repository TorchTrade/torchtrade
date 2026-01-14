from typing import List, Union, Callable, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# Bitget interval mapping for futures klines
INTERVAL_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "1D",
    "3d": "3D",
    "1w": "1W",
}


class BitgetObservationClass:
    """
    Observation class for fetching market data from Bitget.

    Similar to BinanceObservationClass but uses Bitget API for data fetching.
    Supports multiple timeframes and custom feature preprocessing.
    """

    def __init__(
        self,
        symbol: str,
        intervals: Union[List[str], str],
        window_sizes: Union[List[int], int] = 10,
        product_type: str = "SUMCBL",  # SUMCBL for testnet, UMCBL for prod
        feature_preprocessing_fn: Optional[Callable] = None,
        client: Optional[object] = None,
        demo: bool = True,
    ):
        """
        Initialize the BitgetObservationClass.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT")
            intervals: Single interval or list of intervals (e.g., "1m", "5m", "1H")
                      Valid intervals: 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D, 3D, 1W
            window_sizes: Single integer or list of integers specifying window sizes.
                         If a list is provided, it must have the same length as intervals.
            product_type: Product type for Bitget futures (SUMCBL=testnet USDT, UMCBL=prod USDT)
            feature_preprocessing_fn: Optional custom preprocessing function that takes a DataFrame
                                     and returns a DataFrame with feature columns
            client: Optional pre-configured Bitget Client for dependency injection (useful for testing)
            demo: Whether to use demo/testnet environment (default: True)
        """
        # Normalize symbol (remove slash if present)
        self.symbol = symbol.replace("/", "")

        self.intervals = [intervals] if isinstance(intervals, str) else intervals
        self.window_sizes = [window_sizes] if isinstance(window_sizes, int) else window_sizes
        self.product_type = "SUMCBL" if demo else product_type  # Force testnet for demo
        self.demo = demo
        self.default_lookback = 200  # Bitget default limit

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
                from pybitget import Client
                # For market data, we can use public API (no keys required)
                # But pybitget requires keys, so we'll use empty strings for read-only
                self.client = Client("", "", passphrase="")
            except ImportError:
                raise ImportError("python-bitget is required. Install with: pip install python-bitget")

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

        # Map interval to Bitget format
        bitget_interval = INTERVAL_MAP[interval]

        try:
            # Bitget candles API: mix_get_candles(symbol, granularity, productType, limit=...)
            # Granularity format: "1m", "5m", "1H", "1D", etc.
            candles = self.client.mix_get_candles(
                symbol=self.symbol,
                granularity=bitget_interval,
                productType=self.product_type,
                limit=limit
            )

            # Bitget candles format (per their API docs):
            # [timestamp, open, high, low, close, volume, quote_volume]
            # Note: Timestamp is in milliseconds
            if not candles or len(candles) == 0:
                raise ValueError(f"No candle data returned for {self.symbol} on {interval}")

            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'
            ])

            # Convert types
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Sort by timestamp (ascending) - some exchanges return in reverse order
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            raise RuntimeError(f"Failed to fetch candles from Bitget for {self.symbol} on {interval}: {str(e)}")

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
                observations['base_timestamps'] = base_df['timestamp'].values[-window_size:]

            processed_df = self.feature_preprocessing_fn(df)
            # Apply window size
            processed_df = processed_df.iloc[-window_size:]
            observations[key] = self._get_numpy_obs(processed_df)

        return observations


# Example usage:
if __name__ == "__main__":
    import os

    # Test with demo/testnet (no API keys needed for public data)
    print("Testing BitgetObservationClass...")

    # Single interval example
    print("\n1. Testing single interval...")
    window_size = 10
    observer = BitgetObservationClass(
        symbol="BTCUSDT",
        intervals="1m",
        window_sizes=window_size,
        demo=True
    )
    expected_keys = observer.get_keys()
    print(f"   Expected keys: {expected_keys}")

    try:
        observations = observer.get_observations()
        assert set(observations.keys()) == set(expected_keys), "Keys don't match expected keys"
        assert observations[expected_keys[0]].shape == (window_size, 4), \
            f"Expected shape ({window_size}, 4) for default features, got {observations[expected_keys[0]].shape}"
        print("   ✓ Single interval test passed!")
    except Exception as e:
        print(f"   ✗ Single interval test failed: {e}")

    # Multiple intervals example
    print("\n2. Testing multiple intervals...")
    window_sizes = [10, 20]
    observer = BitgetObservationClass(
        symbol="BTCUSDT",
        intervals=["1m", "5m"],
        window_sizes=window_sizes,
        demo=True
    )

    expected_keys = observer.get_keys()
    print(f"   Expected keys: {expected_keys}")

    try:
        observations = observer.get_observations()
        assert set(observations.keys()) == set(expected_keys), "Keys don't match expected keys"
        assert len(observations) == 2, "Expected exactly 2 observations"
        print("   ✓ Multiple intervals test passed!")
    except Exception as e:
        print(f"   ✗ Multiple intervals test failed: {e}")

    print("\n✅ All tests completed!")
