from typing import Union, List, Optional, Callable
import pandas as pd
import ccxt

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
        product_type: str = "USDT-FUTURES",
        feature_preprocessing_fn: Optional[Callable] = None,
        client: Optional[object] = None,
        demo: bool = True,
    ):
        """
        Initialize the BitgetObservationClass.

        Args:
            symbol: The trading symbol (e.g., "BTC/USDT:USDT")
            time_frames: Single TimeFrame or list of TimeFrame objects
            window_sizes: Single integer or list of integers specifying window sizes
            product_type: Product type for Bitget V2 API (USDT-FUTURES, COIN-FUTURES, etc.)
            feature_preprocessing_fn: Optional custom preprocessing function
            client: Optional pre-configured Bitget Client for dependency injection
            demo: Whether to use demo/testnet environment (default: True)
        """
        # Normalize symbol to CCXT perpetual swap format before passing to parent
        # Expected format: "BTC/USDT:USDT"
        if "/" not in symbol:
            # Symbol is like "BTCUSDT" or "BTCUSDT:USDT"
            if ":" in symbol:
                # Symbol is like "BTCUSDT:USDT" - split and add /
                base_quote = symbol.split(":")[0]
                settle = symbol.split(":")[1]
                if base_quote.endswith("USDT"):
                    base = base_quote[:-4]
                    symbol = f"{base}/USDT:{settle}"
                else:
                    symbol = f"{base_quote}/USDT:{settle}"
            else:
                # Symbol is like "BTCUSDT" - add both / and :USDT
                if symbol.endswith("USDT"):
                    base = symbol[:-4]
                    symbol = f"{base}/USDT:USDT"
                else:
                    symbol = f"{symbol}/USDT:USDT"
        elif ":" not in symbol:
            # Symbol is like "BTC/USDT" - add :USDT for perpetual swaps
            symbol = f"{symbol}:USDT"

        # Store Bitget-specific attributes before calling super().__init__
        self.product_type = product_type
        self.demo = demo

        # Call parent constructor with normalized symbol
        super().__init__(
            symbol=symbol,
            time_frames=time_frames,
            window_sizes=window_sizes,
            feature_preprocessing_fn=feature_preprocessing_fn,
            client=client,
            demo=demo,
        )

    def _create_client(self) -> object:
        """Create Bitget API client using CCXT."""
        try:
            # For market data, we can use public API (no keys required)
            client = ccxt.bitget({
                'options': {
                    'defaultType': 'swap',  # Use futures/swap
                    'sandboxMode': self.demo,
                }
            })

            # Enable sandbox/testnet mode if demo
            if self.demo:
                client.set_sandbox_mode(True)

            return client
        except Exception as e:
            raise ImportError(f"CCXT is required. Install with: pip install ccxt. Error: {e}")

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
        Fetch raw kline data from Bitget API using CCXT.

        Args:
            symbol: Trading symbol in CCXT format (e.g., "BTC/USDT:USDT")
            interval: Bitget-specific interval string (e.g., "1H", "1D")
            limit: Number of candles to fetch

        Returns:
            Raw kline data from CCXT: list of [timestamp_ms, open, high, low, close, volume]
        """
        # Symbol should already be normalized in __init__, but double-check
        # Parent class might have modified it, so re-normalize if needed
        if "/" not in symbol or ":" not in symbol:
            # Something went wrong, re-normalize
            # Parse the symbol carefully
            if ":" in symbol:
                # Symbol is like "BTCUSDT:USDT" - split by :
                parts = symbol.split(":")
                base_quote = parts[0]  # "BTCUSDT"
                settle = parts[1] if len(parts) > 1 else "USDT"  # "USDT"

                # Now parse base_quote to get base currency
                if base_quote.endswith("USDT"):
                    base = base_quote[:-4]  # "BTC"
                    symbol = f"{base}/USDT:{settle}"
                else:
                    # Try other common quote currencies
                    for quote in ["USDC", "USD", "BUSD"]:
                        if base_quote.endswith(quote):
                            base = base_quote[:-len(quote)]
                            symbol = f"{base}/{quote}:{settle}"
                            break
                    else:
                        # Fallback - assume USDT
                        symbol = f"{base_quote}/USDT:{settle}"
            else:
                # Symbol is like "BTCUSDT" or "BTC/USDT" with no :
                if "/" in symbol:
                    # Just add :USDT
                    symbol = f"{symbol}:USDT"
                else:
                    # Parse without :
                    if symbol.endswith("USDT"):
                        base = symbol[:-4]
                        symbol = f"{base}/USDT:USDT"
                    else:
                        # Try other common quote currencies
                        for quote in ["USDC", "USD", "BUSD"]:
                            if symbol.endswith(quote):
                                base = symbol[:-len(quote)]
                                symbol = f"{base}/{quote}:USDT"
                                break
                        else:
                            # Fallback
                            symbol = f"{symbol}/USDT:USDT"

        # CCXT uses standardized timeframe strings (1m, 5m, 1h, 1d)
        # Convert Bitget format to CCXT format (already lowercase)
        interval_ccxt = interval.lower()

        candles = self.client.fetch_ohlcv(
            symbol=symbol,
            timeframe=interval_ccxt,
            limit=limit
        )
        return candles

    def _parse_klines(self, raw_klines: list) -> pd.DataFrame:
        """
        Parse raw CCXT kline data into standardized DataFrame.

        CCXT candles format: [timestamp_ms, open, high, low, close, volume]
        Note: Timestamp is in milliseconds

        Args:
            raw_klines: Raw kline data from CCXT

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        df = pd.DataFrame(raw_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
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
    print("Testing BitgetObservationClass with CCXT...")

    # Single timeframe example
    print("\n1. Testing single timeframe...")
    window_size = 10
    observer = BitgetObservationClass(
        symbol="BTC/USDT:USDT",  # CCXT perpetual swap format
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
        import traceback
        traceback.print_exc()

    # Multiple timeframes example
    print("\n2. Testing multiple timeframes...")
    window_sizes = [10, 20]
    observer = BitgetObservationClass(
        symbol="BTC/USDT:USDT",  # CCXT perpetual swap format
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
        import traceback
        traceback.print_exc()

    print("\n✅ All tests completed!")
