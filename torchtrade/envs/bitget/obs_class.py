from typing import Union, List, Optional, Callable
import pandas as pd

from torchtrade.envs.futures.obs_class import BaseFuturesObservationClass


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


class BitgetObservationClass(BaseFuturesObservationClass):
    """
    Observation class for fetching market data from Bitget.

    Inherits common functionality from BaseFuturesObservationClass.
    Implements Bitget-specific API interaction for futures market data.
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
            window_sizes: Single integer or list of integers specifying window sizes
            product_type: Product type for Bitget futures (SUMCBL=testnet, UMCBL=prod)
            feature_preprocessing_fn: Optional custom preprocessing function
            client: Optional pre-configured Bitget Client for dependency injection
            demo: Whether to use demo/testnet environment (default: True)
        """
        # Store Bitget-specific attributes before calling super().__init__
        self.product_type = "SUMCBL" if demo else product_type  # Force testnet for demo

        # Call parent constructor
        super().__init__(
            symbol=symbol,
            intervals=intervals,
            window_sizes=window_sizes,
            feature_preprocessing_fn=feature_preprocessing_fn,
            client=client,
            demo=demo,
        )

    def _create_client(self) -> object:
        """Create Bitget API client."""
        try:
            from pybitget import Client
            # For market data, we can use public API (no keys required)
            # But pybitget requires keys, so we'll use empty strings for read-only
            return Client("", "", passphrase="")
        except ImportError:
            raise ImportError("python-bitget is required. Install with: pip install python-bitget")

    def _validate_interval(self, interval: str) -> None:
        """Validate that an interval is supported by Bitget."""
        if interval not in INTERVAL_MAP:
            raise ValueError(
                f"Invalid interval: {interval}. "
                f"Valid intervals: {list(INTERVAL_MAP.keys())}"
            )

    def _fetch_klines(self, symbol: str, interval: str, limit: int) -> list:
        """
        Fetch raw kline data from Bitget API.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            interval: Bitget-specific interval string (e.g., "1H", "1D")
            limit: Number of candles to fetch

        Returns:
            Raw kline data: list of [timestamp, open, high, low, close, volume, quote_volume]
        """
        candles = self.client.mix_get_candles(
            symbol=symbol,
            granularity=interval,
            productType=self.product_type,
            limit=limit
        )
        return candles

    def _parse_klines(self, raw_klines: list) -> pd.DataFrame:
        """
        Parse raw Bitget kline data into standardized DataFrame.

        Bitget candles format: [timestamp, open, high, low, close, volume, quote_volume]
        Note: Timestamp is in milliseconds

        Args:
            raw_klines: Raw kline data from Bitget API

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        df = pd.DataFrame(raw_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'
        ])

        # Convert types
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df

    def _convert_interval(self, interval: str) -> str:
        """
        Convert standard interval to Bitget-specific format.

        Args:
            interval: Standard interval (e.g., "1h", "1d")

        Returns:
            Bitget interval string (e.g., "1H", "1D")
        """
        return INTERVAL_MAP[interval]

    def _get_default_lookback(self) -> int:
        """Get the default number of candles to fetch (Bitget limit)."""
        return 200

    def _get_timestamp_column(self) -> str:
        """Get the name of the timestamp column."""
        return "timestamp"


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
