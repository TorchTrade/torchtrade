"""Observation class for fetching market data from Bybit using pybit."""
from typing import Union, List, Optional, Callable
import pandas as pd

from torchtrade.envs.live.shared.futures_base_obs import BaseFuturesObservationClass
from torchtrade.envs.utils.timeframe import TimeFrame
from torchtrade.envs.live.bybit.utils import timeframe_to_bybit, normalize_symbol


class BybitObservationClass(BaseFuturesObservationClass):
    """
    Observation class for fetching market data from Bybit.

    Inherits common functionality from BaseFuturesObservationClass.
    Implements Bybit-specific API interaction using pybit for futures market data.
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
        Initialize the BybitObservationClass.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT")
            time_frames: Single TimeFrame or list of TimeFrame objects
            window_sizes: Single integer or list of integers specifying window sizes
            feature_preprocessing_fn: Optional custom preprocessing function
            client: Optional pre-configured pybit HTTP client for dependency injection
            demo: Whether to use demo/testnet environment (default: True)
        """
        symbol = normalize_symbol(symbol)
        super().__init__(
            symbol=symbol,
            time_frames=time_frames,
            window_sizes=window_sizes,
            feature_preprocessing_fn=feature_preprocessing_fn,
            client=client,
            demo=demo,
        )

    def _create_client(self) -> object:
        """Create Bybit API client using pybit (public endpoints, no keys needed)."""
        from pybit.unified_trading import HTTP

        return HTTP(testnet=self.demo)

    def _validate_timeframe(self, timeframe: TimeFrame) -> None:
        """Validate that a timeframe is supported by Bybit."""
        timeframe_to_bybit(timeframe)

    def _fetch_klines(self, symbol: str, interval: str, limit: int) -> list:
        """
        Fetch raw kline data from Bybit API using pybit.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            interval: Bybit-specific interval string (e.g., "15", "60", "D")
            limit: Number of candles to fetch

        Returns:
            Raw kline data from pybit: result["list"] (reverse chronological order)
        """
        response = self.client.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
        ret_code = response.get("retCode")
        if ret_code is not None and int(ret_code) != 0:
            ret_msg = response.get("retMsg", "unknown error")
            raise RuntimeError(f"get_kline failed (retCode={ret_code}): {ret_msg}")
        return response["result"]["list"]

    def _parse_klines(self, raw_klines: list) -> pd.DataFrame:
        """
        Parse raw pybit kline data into standardized DataFrame.

        pybit kline format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        Note: pybit returns data in reverse chronological order (newest first).

        Args:
            raw_klines: Raw kline data from pybit

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(raw_klines, columns=[
            'timestamp', *ohlcv_cols, 'turnover'
        ])

        # Convert types (pybit returns strings)
        df[ohlcv_cols] = df[ohlcv_cols].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')

        # Ensure chronological order (pybit returns newest first)
        df = df.sort_values('timestamp', ascending=True, kind='stable').reset_index(drop=True)

        return df.drop(columns=['turnover'])

    def _convert_timeframe(self, timeframe: TimeFrame) -> str:
        """
        Convert TimeFrame to Bybit-specific interval format.

        Args:
            timeframe: TimeFrame object

        Returns:
            Bybit interval string (e.g., "15", "60", "D")
        """
        return timeframe_to_bybit(timeframe)

    def _get_default_lookback(self) -> int:
        """Get the default number of candles to fetch (Bybit limit)."""
        return 200

    def _get_timestamp_column(self) -> str:
        """Get the name of the timestamp column."""
        return "timestamp"
