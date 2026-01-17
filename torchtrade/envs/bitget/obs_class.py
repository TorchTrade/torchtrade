from typing import Union, List, Optional, Callable
import pandas as pd

from torchtrade.envs.futures.obs_class import BaseFuturesObservationClass
from torchtrade.envs.timeframe import TimeFrame, TimeFrameUnit
from torchtrade.envs.bitget.utils import timeframe_to_bitget, BITGET_INTERVAL_MAP


class BitgetObservationClass(BaseFuturesObservationClass):
    """
    Observation class for fetching market data from Bitget.

    Inherits common functionality from BaseFuturesObservationClass.
    Implements Bitget-specific API interaction for futures market data.
    """

    def __init__(
        self,
        symbol: str,
        time_frames: Union[List[TimeFrame], TimeFrame],
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
            time_frames: Single TimeFrame or list of TimeFrame objects
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
            time_frames=time_frames,
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

    def _validate_timeframe(self, timeframe: TimeFrame) -> None:
        """Validate that a timeframe is supported by Bitget."""
        if timeframe.unit not in BITGET_INTERVAL_MAP:
            raise ValueError(
                f"Unsupported TimeFrameUnit for Bitget: {timeframe.unit}"
            )

        unit_map = BITGET_INTERVAL_MAP[timeframe.unit]
        if timeframe.value not in unit_map:
            raise ValueError(
                f"Unsupported timeframe value for Bitget: {timeframe.value}{timeframe.unit.value}. "
                f"Valid values for {timeframe.unit.value}: {list(unit_map.keys())}"
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

    def _convert_timeframe(self, timeframe: TimeFrame) -> str:
        """
        Convert TimeFrame to Bitget-specific interval format.

        Args:
            timeframe: TimeFrame object

        Returns:
            Bitget interval string (e.g., "1H", "1D")
        """
        return timeframe_to_bitget(timeframe)

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

    # Single timeframe example
    print("\n1. Testing single timeframe...")
    window_size = 10
    observer = BitgetObservationClass(
        symbol="BTCUSDT",
        time_frames=TimeFrame(1, TimeFrameUnit.Minute),
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
        print("   ✓ Single timeframe test passed!")
    except Exception as e:
        print(f"   ✗ Single timeframe test failed: {e}")

    # Multiple timeframes example
    print("\n2. Testing multiple timeframes...")
    window_sizes = [10, 20]
    observer = BitgetObservationClass(
        symbol="BTCUSDT",
        time_frames=[
            TimeFrame(1, TimeFrameUnit.Minute),
            TimeFrame(5, TimeFrameUnit.Minute),
        ],
        window_sizes=window_sizes,
        demo=True
    )

    expected_keys = observer.get_keys()
    print(f"   Expected keys: {expected_keys}")

    try:
        observations = observer.get_observations()
        assert set(observations.keys()) == set(expected_keys), "Keys don't match expected keys"
        assert len(observations) == 2, "Expected exactly 2 observations"
        print("   ✓ Multiple timeframes test passed!")
    except Exception as e:
        print(f"   ✗ Multiple timeframes test failed: {e}")

    print("\n✅ All tests completed!")
